"""Multi-block fingerprints (casmi/richfp.py) for the PeakFormer training arrays of scripts/prep_fp_data.py.

python scripts/prep_pf_data.py <fp_data_dir> <pool_dir> <out_dir>
fp_data_dir: spectra.npz / structs.npz / negs.npy / negfp.npy (structs.pool_idx and the negatives index pool_dir,
the pool those arrays were built from). The negative rows of negfp.npy are the sorted unique pool rows referenced by
the window search of prep_fp_data.py; that search is repeated here and checked against negfp's Morgan bits.
Writes <out_dir>/fpx.npz: keep (raw bit ids kept), s_ptr / s_idx (structures, CSR, uint16 kept ids), n_ptr / n_idx
(negatives).
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.richfp import N_RAW, raw_onbits, to_csr

src, pool_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
T = np.load(f'{src}/structs.npz', allow_pickle=True); S = np.load(f'{src}/spectra.npz')
P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['smiles', 'mass', 'formula'])
pool_idx = T['pool_idx']
mass, form = P.mass.values, P.formula.values
used = set()
for i, pi in enumerate(pool_idx):                       # same window search as prep_fp_data.py
    M = mass[pi]; tol = max(M * 1e-5, 0.002)
    lo, hi = np.searchsorted(mass, M - tol), np.searchsorted(mass, M + tol)
    c = np.arange(lo, hi); c = c[c != pi]
    if len(c) == 0:
        continue
    same = c[form[c] == form[pi]]; other = c[form[c] != form[pi]]
    c = np.concatenate([same[np.random.default_rng(i).permutation(len(same))],
                        other[np.random.default_rng(i + 1).permutation(len(other))]])[:64]
    used.update(c.tolist())
used = np.array(sorted(used))
negfp = np.load(f'{src}/negfp.npy', mmap_mode='r')
assert len(used) == len(negfp), (len(used), len(negfp))
pfp = np.load(f'{pool_dir}/pool_fp.npy', mmap_mode='r')
chk = np.random.default_rng(0).choice(len(used), 2000, replace=False)
assert (np.unpackbits(np.asarray(negfp[chk]), axis=1)[:, :2048] == np.unpackbits(np.asarray(pfp[used[chk]]), axis=1)[:, :2048]).all()
print('negatives', len(used), 'verified against negfp', flush=True)

with Pool(int(os.environ.get('PROCS', '4'))) as p:
    s_raw = p.map(raw_onbits, P.smiles.values[pool_idx].tolist(), chunksize=2000)
    print('structures done', flush=True)
    n_raw = p.map(raw_onbits, P.smiles.values[used].tolist(), chunksize=2000)
tr_structs = np.unique(S['sidx'][S['split'] == 0])
cnt = np.bincount(np.concatenate([s_raw[i] for i in tr_structs]), minlength=N_RAW)
fr = cnt / len(tr_structs)
keep = np.where((cnt >= max(2, len(tr_structs) // 2000)) & (fr < 0.995))[0]
km = -np.ones(N_RAW, np.int64); km[keep] = np.arange(len(keep))
s_ptr, s_idx = to_csr(s_raw, km); n_ptr, n_idx = to_csr(n_raw, km)
np.savez(f'{out}/fpx.npz', keep=keep.astype(np.int32), s_ptr=s_ptr, s_idx=s_idx, n_ptr=n_ptr, n_idx=n_idx)
print('kept bits', len(keep), 'of', N_RAW, 'mean on-bits per structure', round(float(np.diff(s_ptr).mean()), 1),
      'max', int(np.diff(s_ptr).max()), flush=True)
