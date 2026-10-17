"""Training arrays for the spectrum<->structure dual encoder (uploaded as a private Kaggle dataset).

spectra.npz : mz (float32, flat), it (float16, flat), off (int64), prec, ce, add_id, inst_id, pol, sidx (structure
              index into structs), split (0 train, 1 val, 2 ranker-holdout)
structs.npz : fp (packed uint8: Morgan r2 2048 + MACCS 167 -> 2215 bits), mass, key, pool_idx
negs.npy    : (n_structs, 64) int32 hard negatives = indices into negfp (pool structures in the 10 ppm window,
              same formula first), -1 padded; negfp.npy packed fingerprints of every pool row referenced
Usage: python scripts/prep_fp_data.py <out_dir>
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.spectra import Store

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-', '[M-H2O-H]-',
       '[M-2H2O+H]+', '[2M+H]+', '[2M+Na]+', '[2M-H]-', '[M]+', '[M+C2H4O2-H]-']
_G = None


def maccs_bits(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(167, np.uint8)
    return np.array(list(MACCSkeys.GenMACCSKeys(m)), np.uint8)


def inst_family(s):
    s = s.lower() if isinstance(s, str) else ''
    if 'tims' in s:
        return 1
    if 'orbitrap' in s or 'q exactive' in s or 'exploris' in s or 'ft' in s or 'hcd' in s:
        return 2
    if 'tof' in s:
        return 3
    if 'trap' in s or 'it' == s:
        return 4
    return 0


if __name__ == '__main__':
    st = Store('/home/user/work/train_store'); meta = st.meta
    P = pd.read_parquet('/home/user/work/pool/pool.parquet')
    pfp = np.load('/home/user/work/pool/pool_fp.npy')
    key2pool = {k: i for i, k in enumerate(P.key.values)}
    val = set(open('/home/user/work/val_keys.txt').read().split())
    rk = set(open('/home/user/work/rank_keys.txt').read().split())
    keep = meta.adduct.isin(ADD) & meta.inchikey14.isin(key2pool.keys())
    sel = np.where(keep.values)[0]
    m = meta.iloc[sel]
    skeys = np.array(sorted(m.inchikey14.unique()))
    s_index = {k: i for i, k in enumerate(skeys)}
    pool_idx = np.array([key2pool[k] for k in skeys])
    # structure fingerprints: Morgan bits from the pool + MACCS
    with Pool(4) as p:
        mac = np.stack(p.map(maccs_bits, P.smiles.values[pool_idx].tolist(), chunksize=2000))
    morgan = np.unpackbits(pfp[pool_idx], axis=1)
    sfp = np.packbits(np.concatenate([morgan, mac], 1), axis=1)
    # spectra
    offs = st.off
    lens = (offs[sel + 1] - offs[sel])
    mz = np.concatenate([np.asarray(st.mz[offs[i]:offs[i + 1]]) for i in sel]).astype(np.float32)
    it = np.concatenate([np.asarray(st.it[offs[i]:offs[i + 1]]) for i in sel]).astype(np.float16)
    off = np.zeros(len(sel) + 1, np.int64); off[1:] = np.cumsum(lens)
    ce = m.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else -1.0).values.astype(np.float32)
    add_id = m.adduct.map({a: i for i, a in enumerate(ADD)}).values.astype(np.int8)
    inst = np.array([inst_family(s) for s in m.instrument_type.values], np.int8)
    sidx = m.inchikey14.map(s_index).values.astype(np.int32)
    split = np.where(m.inchikey14.isin(val), 1, np.where(m.inchikey14.isin(rk), 2, 0)).astype(np.int8)
    np.savez(f'{out}/spectra.npz', mz=mz, it=it, off=off, prec=m.precursor_mz.values.astype(np.float32), ce=ce,
             add_id=add_id, inst_id=inst, pol=(m.ionization_mode == 'negative').values.astype(np.int8), sidx=sidx,
             split=split)
    # hard negatives from the pool window (10 ppm), same formula first
    mass = P.mass.values
    negs = np.full((len(skeys), 64), -1, np.int64)
    form = P.formula.values
    for i, pi in enumerate(pool_idx):
        M = mass[pi]; tol = max(M * 1e-5, 0.002)
        lo, hi = np.searchsorted(mass, M - tol), np.searchsorted(mass, M + tol)
        c = np.arange(lo, hi); c = c[c != pi]
        if len(c) == 0:
            continue
        same = c[form[c] == form[pi]]; other = c[form[c] != form[pi]]
        c = np.concatenate([same[np.random.default_rng(i).permutation(len(same))],
                            other[np.random.default_rng(i + 1).permutation(len(other))]])[:64]
        negs[i, :len(c)] = c
    used = np.unique(negs[negs >= 0])
    remap = -np.ones(len(P), np.int64); remap[used] = np.arange(len(used))
    negs2 = np.where(negs >= 0, remap[np.maximum(negs, 0)], -1).astype(np.int32)
    # negatives need the same 2215-bit layout: Morgan from the pool + MACCS
    with Pool(4) as p:
        nmac = np.stack(p.map(maccs_bits, P.smiles.values[used].tolist(), chunksize=2000))
    nfp = np.packbits(np.concatenate([np.unpackbits(pfp[used], axis=1), nmac], 1), axis=1)
    np.save(f'{out}/negs.npy', negs2); np.save(f'{out}/negfp.npy', nfp)
    np.savez(f'{out}/structs.npz', fp=sfp, mass=mass[pool_idx], key=skeys.astype('U14'), pool_idx=pool_idx)
    print('spectra', len(sel), 'structures', len(skeys), 'neg pool', len(used),
          'split counts', np.bincount(split), 'peaks', off[-1])
