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
from casmi.richfp import N_RAW, raw_onbits

src, pool_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
T = np.load(f'{src}/structs.npz', allow_pickle=True); S = np.load(f'{src}/spectra.npz')
P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['smiles', 'mass', 'formula'])
pool_idx = T['pool_idx']
mass, form = P.mass.values, P.formula.to_numpy(dtype=object)        # object array: Arrow string indexing is slow
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

def raw_u16(smi):
    return raw_onbits(smi).astype(np.uint16)          # raw ids < 16,551


def csr_raw(smiles, p, chunk=20000):
    """Raw on-bit CSR (ptr, uint16 idx) of many SMILES, built chunk by chunk to bound memory."""
    ptrs, idxs, base = [np.zeros(1, np.int64)], [], 0
    for c in range(0, len(smiles), chunk):
        part = p.map(raw_u16, smiles[c:c + chunk], chunksize=500)
        l = np.array([len(a) for a in part], np.int64)
        ptrs.append(base + np.cumsum(l)); base += int(l.sum())
        idxs.append(np.concatenate(part) if len(part) else np.zeros(0, np.uint16))
        print('  ', c + len(part), 'of', len(smiles), flush=True)
    return np.concatenate(ptrs), np.concatenate(idxs)


def remap(ptr, idx, km, chunk=50_000_000):
    """Raw CSR -> kept-bit CSR (uint16), in chunks of on-bits (km: int32 raw -> kept or -1)."""
    vals, keep_n = [], np.zeros(len(ptr) - 1, np.int64)
    row_end = ptr[1:]
    for a in range(0, len(idx), chunk):
        v = km[idx[a:a + chunk]]
        ok = v >= 0
        r = np.searchsorted(row_end, np.arange(a, a + len(v))[ok], side='right')
        keep_n += np.bincount(r, minlength=len(keep_n))
        vals.append(v[ok].astype(np.uint16))
    nptr = np.zeros(len(ptr), np.int64); nptr[1:] = np.cumsum(keep_n)
    return nptr, np.concatenate(vals)


with Pool(int(os.environ.get('PROCS', '4'))) as p:
    if os.path.exists(f'{out}/s_raw.npz'):
        z = np.load(f'{out}/s_raw.npz'); sp, si = z['ptr'], z['idx']
    else:
        sp, si = csr_raw(P.smiles.values[pool_idx].tolist(), p)
        np.savez(f'{out}/s_raw.npz', ptr=sp, idx=si)
    print('structures done', 'raw on-bits per structure', round(float(np.diff(sp).mean()), 1), flush=True)
    tr_structs = np.unique(S['sidx'][S['split'] == 0])
    cnt = np.zeros(N_RAW, np.int64)
    for t in tr_structs:
        cnt[si[sp[t]:sp[t + 1]]] += 1
    fr = cnt / len(tr_structs)
    keep = np.where((cnt >= max(2, len(tr_structs) // 2000)) & (fr < 0.995))[0]
    km = -np.ones(N_RAW, np.int32); km[keep] = np.arange(len(keep))
    s_ptr, s_idx = remap(sp, si, km); del sp, si
    if os.path.exists(f'{out}/n_raw.npz'):
        z = np.load(f'{out}/n_raw.npz'); np_, ni = z['ptr'], z['idx']
    else:
        np_, ni = csr_raw(P.smiles.values[used].tolist(), p)
        np.savez(f'{out}/n_raw.npz', ptr=np_, idx=ni)
    n_ptr, n_idx = remap(np_, ni, km); del np_, ni
np.savez(f'{out}/fpx.npz', keep=keep.astype(np.int32), s_ptr=s_ptr, s_idx=s_idx, n_ptr=n_ptr, n_idx=n_idx)
print('kept bits', len(keep), 'of', N_RAW, 'mean on-bits per structure', round(float(np.diff(s_ptr).mean()), 1),
      'max', int(np.diff(s_ptr).max()), flush=True)
