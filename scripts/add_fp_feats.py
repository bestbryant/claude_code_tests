"""Add fingerprint-model features (fp_cos, fp_bern + within-query rank/gap) to dumped retrieval rows.

Usage: python scripts/add_fp_feats.py <rows.parquet> <weights.pt> <out.parquet>
"""
import sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.fpscore import FPScorer
from casmi.own import fp_feats
from casmi.prep_meta import inst_family
from casmi.spectra import Store

rows, wpath, out = sys.argv[1:4]
torch.set_num_threads(1)
D = pd.read_parquet(rows)
POOL = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else '/home/user/work/pool2'
SHARD, NSH = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
D = D[D.qid % NSH == SHARD]
P = pd.read_parquet(f'{POOL}/pool.parquet')
sc = FPScorer(wpath, np.load(f'{POOL}/pool_fp.npy', mmap_mode='r'))
st = Store('/home/user/work/train_store'); meta = st.meta
parts = []
t0 = time.time()
for n, (qid, g) in enumerate(D.groupby('qid', sort=False)):
    spectra = []
    for i in map(int, g.q_rows.iloc[0].split(',')):
        mz, it = st.peaks(i); r = meta.loc[i]
        ce = r.collision_energy_ev
        spectra.append(dict(mz=mz, it=it, prec=float(r.precursor_mz), adduct=r.adduct,
                            ce=float(np.mean(ce)) if ce is not None and len(ce) else -1.0, inst=inst_family(r.instrument_type)))
    pids = g.pid.values
    cos, bern = sc.score(spectra, pids, P.smiles.values[pids])
    parts.append(fp_feats(g.copy(), cos, bern))
    if n % 500 == 0:
        print(n, f'{time.time()-t0:.0f}s', flush=True)
D = pd.concat(parts, ignore_index=True)
D.to_parquet(out)
print('done', len(D), f'{time.time()-t0:.0f}s')
