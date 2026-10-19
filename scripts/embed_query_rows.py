"""DreaMS embeddings of the query spectra (train.parquet rows in q_rows) of simulated retrieval rows -> cache npz
(row, emb float16) used by scripts/add_dreams_feats.py.

python scripts/embed_query_rows.py <dreams_model.pt> <cache.npz> <rows.parquet> [<rows.parquet> ...]
Positive-mode enveda-180 query spectra are shifted by -0.000548579909 Da (precursor and fragments): the hidden test
carries this electron-mass calibration fix and enveda-180 in train does not (library spectra stay uncorrected, as
they are at test time). DreaMS is sensitive to it (median cosine 0.996, min 0.979 between shifted and unshifted).
"""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.dreams_emb import DreamsEmbedder
ELECTRON = 0.000548579909

model, cache = sys.argv[1:3]
need = set()
for p in sys.argv[3:]:
    q = pd.read_parquet(p, columns=['qid', 'q_rows']).drop_duplicates('qid')
    need |= {int(x) for s in q.q_rows for x in s.split(',')}
meta = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['ingest_lib', 'adduct'])
fix = (meta.ingest_lib.values == 'enveda-180') & meta.adduct.fillna('').str.endswith('+').values
E = {}
if os.path.exists(cache):
    z = np.load(cache); E = dict(zip(z['row'].tolist(), z['emb']))
    if 'efix' not in z.files:                     # caches made before the correction: re-embed the affected rows
        E = {r: e for r, e in E.items() if not fix[r]}
todo = sorted(need - set(E))
print('query spectra', len(need), 'to embed', len(todo), flush=True)
em = DreamsEmbedder(model, int(os.environ.get('THREADS', '4')))
pf = pq.ParquetFile('/home/user/data/train.parquet')
want = np.zeros(pf.metadata.num_rows, bool); want[todo] = True
base, t0, done = 0, time.time(), 0
for g in range(pf.metadata.num_row_groups):
    n = pf.metadata.row_group(g).num_rows
    loc = np.where(want[base:base + n])[0].tolist()
    if loc:
        t = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz']).take(loc).to_pandas()
        for i in range(0, len(t), 512):
            c = t.iloc[i:i + 512]
            rows_c = loc[i:i + 512]
            sh = [ELECTRON if fix[base + j] else 0.0 for j in rows_c]
            emb = em.embed([(np.asarray(m, np.float64) - d, it, p - d)
                            for m, it, p, d in zip(c.ms2_mzs, c.ms2_normalized_intensities, c.precursor_mz, sh)])
            for j, e in zip(rows_c, emb):
                E[base + j] = e.astype(np.float16)
            done += len(c)
            print(f'{done}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
        np.savez(cache, row=np.array(list(E), np.int64), emb=np.stack(list(E.values())), efix=np.array(1))
    base += n
print('done', len(E))
