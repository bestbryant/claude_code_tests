"""DreaMS embeddings of the query spectra (train.parquet rows in q_rows) of simulated retrieval rows -> cache npz
(row, emb float16) used by scripts/add_dreams_feats.py.

python scripts/embed_query_rows.py <dreams_model.pt> <cache.npz> <rows.parquet> [<rows.parquet> ...]
"""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.dreams_emb import DreamsEmbedder

model, cache = sys.argv[1:3]
need = set()
for p in sys.argv[3:]:
    q = pd.read_parquet(p, columns=['qid', 'q_rows']).drop_duplicates('qid')
    need |= {int(x) for s in q.q_rows for x in s.split(',')}
E = {}
if os.path.exists(cache):
    z = np.load(cache); E = dict(zip(z['row'].tolist(), z['emb']))
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
            emb = em.embed(list(zip(c.ms2_mzs, c.ms2_normalized_intensities, c.precursor_mz)))
            for j, e in zip(loc[i:i + 512], emb):
                E[base + j] = e.astype(np.float16)
            done += len(c)
            print(f'{done}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
        np.savez(cache, row=np.array(list(E), np.int64), emb=np.stack(list(E.values())))
    base += n
print('done', len(E))
