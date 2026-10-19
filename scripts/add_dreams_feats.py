"""Add DreaMS analog-channel features (casmi/dreams_feats.py) to simulated retrieval rows.

python scripts/add_dreams_feats.py <rows_fp.parquet> <out.parquet> <pool_dir> <dreams_model.pt> <qemb_cache.npz> <shard.npz> ...
Hiding as in scripts/dump_ret_rows.py (standard mode): class 2 hides every library spectrum of the truth structure,
class 1 hides the query spectra (q_rows); a class-1 truth whose DreaMS representative is a query spectrum gets a
substitute representative (its next visible spectrum of that polarity in work/lib10.npz), as at test time. Query spectra (train.parquet rows) are embedded once and cached in <qemb_cache.npz>.
"""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.dreams_emb import DreamsEmbedder
from casmi.dreams_feats import DR_FEATS, DreamsLib, dreams_features
from casmi.retrieval import tanimoto_packed

rows_p, out, pool_dir, model, cache = sys.argv[1:6]
shards = sys.argv[6:]
D = pd.read_parquet(rows_p)
meta = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['inchikey14', 'ionization_mode'])
mkey, mpol = meta.inchikey14.values, (meta.ionization_mode.values != 'positive').astype(np.int8)
srows = np.concatenate([np.load(p)['row'] for p in shards])
DL = DreamsLib(shards, {int(r): mkey[r] for r in srows}, {int(r): mpol[r] for r in srows})
print('library', len(DL.row), flush=True)

Q = D.drop_duplicates('qid')[['qid', 'cls', 'truth', 'q_rows']].copy()
Q['rows'] = Q.q_rows.map(lambda s: [int(x) for x in s.split(',')])
need = sorted({r for rr in Q.rows for r in rr})
E = {}
if os.path.exists(cache):
    z = np.load(cache)
    E = dict(zip(z['row'].tolist(), z['emb']))
missing = [r for r in need if r not in E]
assert not missing, f'{len(missing)} query spectra not embedded: run scripts/embed_query_rows.py first'
assert 'efix' in np.load(cache).files, 'query embeddings predate the electron-mass correction'

# class-1 substitutes: when every same-polarity DreaMS representative of the truth is one of the hidden query spectra,
# the truth would vanish from the DreaMS library, which never happens for a real class-1 molecule (its representative
# is always visible). Use the truth's next visible spectrum of that polarity from the 10-per-structure library instead.
L10 = np.load('/home/user/work/lib10.npz', allow_pickle=True)
l10_n = np.diff(L10['off'])
l10 = {}
for j in np.argsort(-l10_n, kind='stable'):
    l10.setdefault((L10['key'][j], int(L10['pol'][j])), []).append(int(L10['row'][j]))
backup = {}                       # qid -> [(key, pol, train row)]
for q in Q[Q.cls == 1].itertuples():
    hr = set(q.rows)
    for p in sorted({int(mpol[r]) for r in q.rows}):
        reps = [j for j in DL.self_idx.get(q.truth, ()) if DL.pol[j] == p]
        if reps and not all(int(DL.row[j]) in hr for j in reps):
            continue
        alt = [r for r in l10.get((q.truth, p), []) if r not in hr]
        if alt:
            backup.setdefault(q.qid, []).append((q.truth, p, alt[0]))
bcache = cache.replace('.npz', '_lib.npz')       # library-side spectra: uncorrected, kept apart from query embeddings
B = {}
if os.path.exists(bcache):
    z = np.load(bcache); B = dict(zip(z['row'].tolist(), z['emb']))
brows = sorted({b[2] for v in backup.values() for b in v} - set(B))
print('class-1 queries needing a substitute representative', len(backup), 'new spectra to embed', len(brows), flush=True)
if brows:
    em = DreamsEmbedder(model, int(os.environ.get('THREADS', '4')))
    pf = pq.ParquetFile('/home/user/data/train.parquet')
    want = np.zeros(pf.metadata.num_rows, bool); want[brows] = True
    base = 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        loc = np.where(want[base:base + n])[0].tolist()
        if loc:
            t = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz']).take(loc).to_pandas()
            for j, e in zip(loc, em.embed(list(zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz)))):
                B[base + j] = e.astype(np.float16)
        base += n
    np.savez(bcache, row=np.array(list(B), np.int64), emb=np.stack(list(B.values())))

P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['key'])
fp = np.load(f'{pool_dir}/pool_fp.npy', mmap_mode='r')
key_idx = {k: i for i, k in enumerate(P.key.values)}
key_fp = lambda k: np.asarray(fp[key_idx[k]]) if k in key_idx else None
cols = {c: np.zeros(len(D)) for c in DR_FEATS}
D = D.reset_index(drop=True)
t0 = time.time()
for n_done, (qid, g) in enumerate(D.groupby('qid', sort=False)):
    q = Q[Q.qid == qid].iloc[0]
    rows = q.rows
    qe = np.stack([E[r].astype(np.float32) for r in rows]); qp = mpol[rows]
    hide_keys = {q.truth} if q.cls == 2 else set()
    hide_rows = set(rows) if q.cls == 1 else set()
    extra = [(k, p, B[r].astype(np.float32)) for k, p, r in backup.get(qid, [])]
    f = dreams_features(DL, qe, qp, P.key.values[g.pid.values], np.asarray(fp[g.pid.values]), key_fp, tanimoto_packed,
                        hide_keys=hide_keys, hide_rows=hide_rows, extra=extra)
    for c in DR_FEATS:
        cols[c][g.index.values] = f[c]
    if n_done % 500 == 0:
        print('queries', n_done, f'{time.time() - t0:.0f}s', flush=True)
for c in DR_FEATS:
    D[c] = cols[c]
D.to_parquet(out)
print('done', out, D.shape)
