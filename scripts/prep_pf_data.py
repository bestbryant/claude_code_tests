"""Multi-block fingerprints (casmi/richfp.py) for the PeakFormer training arrays of scripts/prep_fp_data.py.

python scripts/prep_pf_data.py <fp_data_dir> <pool_dir> <out_dir>
fp_data_dir: spectra.npz / structs.npz / negs.npy / negfp.npy (structs.pool_idx and the negatives index pool_dir,
the pool those arrays were built from). The negative rows of negfp.npy are the sorted unique pool rows referenced by
the window search of prep_fp_data.py; that search is repeated here and checked against negfp's Morgan bits.
Writes <out_dir>/fpx.npz: keep (raw bit ids kept), s_ptr / s_idx (structures, CSR, uint16 kept ids), n_ptr / n_idx
(negatives). Every npz (fpx and the s_raw / n_raw caches) is written to <name>.tmp and renamed (os.replace), so a crash
(the Oct 20 run hit ENOSPC in the final save) leaves no truncated file to be reused or published. Logs the input md5s,
each array's dtype / shape / md5 and the file md5; with the reference inputs (casmi26-fp-data + pool v1) asserts the
published casmi26-pf-data fpx.npz (md5 bdd0e338; np.savez output is byte-stable: fixed 1980 zip timestamps).
A run from scratch writes ~3.6 GB (s_raw 0.5, n_raw 1.3, fpx 1.8); replacing an existing fpx.npz needs 1.8 GB more
while both copies exist.
"""
import hashlib, os, shutil, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.richfp import N_RAW, raw_onbits

# reference inputs (docs/reproduce.md 1.2) and the published fpx.npz arrays (md5 of the C-order array bytes)
REF_IN = {'structs.npz': '280213bd8a11071400a7227746759cfa', 'spectra.npz': '72f6fbdea3d0afaf3f801b38f35f748e',
          'pool.parquet': '707055a9731618d21fe14922d72c7c4c'}
REF_OUT = {'keep': '6ba3e79b46c7769f53856c7102241a95', 's_ptr': '3ec74e51e7167735d19e313668022bc1',
           's_idx': '7884998ecae636c54162bfd89e53ebf9', 'n_ptr': '602fa1d7d8d2f4cb528fe4868b9f1b28',
           'n_idx': '2a12bf595ee61f8e826b1863cf9a5816'}
REF_FILE = 'bdd0e3389cc219c8db6f20043f99cd7b'


def file_md5(path, bs=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def save_npz(path, **arrs):
    """np.savez into path + '.tmp', fsync, os.replace; the partial temp file is removed on failure."""
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as f:
            np.savez(f, **arrs); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


src, pool_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
ins = {'structs.npz': file_md5(f'{src}/structs.npz'), 'spectra.npz': file_md5(f'{src}/spectra.npz'),
       'pool.parquet': file_md5(f'{pool_dir}/pool.parquet')}
ref = ins == REF_IN
print('inputs', ins, 'reference (casmi26-fp-data + pool v1)' if ref else 'NOT the reference inputs: no output check',
      flush=True)
print('free disk', round(shutil.disk_usage(out).free / 1e9, 2), 'GB in', out, flush=True)
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
        save_npz(f'{out}/s_raw.npz', ptr=sp, idx=si)
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
        save_npz(f'{out}/n_raw.npz', ptr=np_, idx=ni)
    n_ptr, n_idx = remap(np_, ni, km); del np_, ni
print('kept bits', len(keep), 'of', N_RAW, 'mean on-bits per structure', round(float(np.diff(s_ptr).mean()), 1),
      'max', int(np.diff(s_ptr).max()), flush=True)
arrs = dict(keep=keep.astype(np.int32), s_ptr=s_ptr, s_idx=s_idx, n_ptr=n_ptr, n_idx=n_idx)   # key order = file layout
md5s = {k: hashlib.md5(np.ascontiguousarray(v)).hexdigest() for k, v in arrs.items()}
for k, v in arrs.items():
    print('fpx', k, v.dtype, v.shape, md5s[k], flush=True)
save_npz(f'{out}/fpx.npz', **arrs)
fm = file_md5(f'{out}/fpx.npz')
print('wrote', f'{out}/fpx.npz', os.path.getsize(f'{out}/fpx.npz'), 'bytes, md5', fm, flush=True)
if ref:
    bad = [k for k in REF_OUT if md5s[k] != REF_OUT[k]]
    assert not bad, f'arrays differ from the published casmi26-pf-data fpx.npz: {bad}'
    assert fm == REF_FILE, f'arrays equal but file md5 {fm} != {REF_FILE} (numpy npy/zip layout changed?)'
    print('identical to the published casmi26-pf-data fpx.npz', flush=True)
