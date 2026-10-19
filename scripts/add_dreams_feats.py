"""Add DreaMS analog-channel features (casmi/dreams_feats.py) to simulated retrieval rows.

python scripts/add_dreams_feats.py <rows_fp.parquet> <out.parquet> <pool_dir> <dreams_model.pt> <qemb_cache.npz> <shard.npz> ...
Hiding as in scripts/dump_ret_rows.py: class 2 hides every library spectrum of the truth structure, class 1 hides the
query spectra (q_rows). Query spectra (train.parquet rows) are embedded once and cached in <qemb_cache.npz>.
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
DL = DreamsLib(shards, {int(r): mkey[r] for r in srows}, {int(r): mpol[r] for r in srows}, None)
print('library', len(DL.row), flush=True)

Q = D.drop_duplicates('qid')[['qid', 'cls', 'truth', 'q_rows']].copy()
Q['rows'] = Q.q_rows.map(lambda s: [int(x) for x in s.split(',')])
need = sorted({r for rr in Q.rows for r in rr})
E = {}
if os.path.exists(cache):
    z = np.load(cache)
    E = dict(zip(z['row'].tolist(), z['emb']))
todo = [r for r in need if r not in E]
if todo:
    em = DreamsEmbedder(model, int(os.environ.get('THREADS', '4')))
    pf = pq.ParquetFile('/home/user/data/train.parquet')
    base, t0 = 0, time.time()
    want = set(todo)
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        loc = [i - base for i in range(base, base + n) if i in want]
        if loc:
            t = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz']).take(loc).to_pandas()
            emb = em.embed(list(zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz)))
            for j, e in zip(loc, emb):
                E[base + j] = e.astype(np.float16)
            print(f'embedded {len(E)}/{len(need)} {time.time() - t0:.0f}s', flush=True)
        base += n
    np.savez(cache, row=np.array(list(E), np.int64), emb=np.stack(list(E.values())))

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
    f = dreams_features(DL, qe, qp, P.key.values[g.pid.values], np.asarray(fp[g.pid.values]), key_fp, tanimoto_packed,
                        hide_keys=hide_keys, hide_rows=hide_rows)
    for c in DR_FEATS:
        cols[c][g.index.values] = f[c]
    if n_done % 500 == 0:
        print('queries', n_done, f'{time.time() - t0:.0f}s', flush=True)
for c in DR_FEATS:
    D[c] = cols[c]
D.to_parquet(out)
print('done', out, D.shape)
