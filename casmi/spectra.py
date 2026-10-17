"""Spectrum utilities: cleaning, cosine / modified cosine (numba), compact store access."""
import numpy as np
import pandas as pd
from numba import njit

PROTON = 1.007276
ELECTRON = 0.000548579909

# neutral mass = (precursor_mz * |z| - adduct_shift) / n_mol ; shifts for singly charged ions
ADDUCTS = {
    '[M+H]+': (1, 1.007276), '[M+NH4]+': (1, 18.033823), '[M+Na]+': (1, 22.989218), '[M+K]+': (1, 38.963158),
    '[M-H2O+H]+': (1, 1.007276 - 18.010565), '[M-2H2O+H]+': (1, 1.007276 - 2 * 18.010565),
    '[M-H]-': (1, -1.007276), '[M-H2O-H]-': (1, -1.007276 - 18.010565), '[M+CH2O2-H]-': (1, 44.998201),
    '[M+Cl]-': (1, 34.969402), '[M]+': (1, -ELECTRON), '[2M+H]+': (2, 1.007276), '[2M+Na]+': (2, 22.989218),
    '[2M-H]-': (2, -1.007276), '[M+C2H4O2-H]-': (1, 59.013851),
}


def neutral_mass(prec_mz, adduct):
    n, s = ADDUCTS.get(adduct, (None, None))
    if n is None:
        return np.nan
    return (prec_mz - s) / n


def clean_peaks(mz, it, prec_mz, rel_floor=0.005, top_k=64, remove_prec=False):
    """Drop peaks above precursor+2 Da (and near the precursor if asked), below rel_floor, keep top_k; sqrt and L2."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    keep = mz <= prec_mz + 2.0
    if remove_prec:
        keep &= mz < prec_mz - 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return mz.astype(np.float32), it.astype(np.float32)
    keep = it >= rel_floor * it.max()
    mz, it = mz[keep], it[keep]
    if len(mz) > top_k:
        idx = np.argpartition(-it, top_k)[:top_k]
        mz, it = mz[idx], it[idx]
    o = np.argsort(mz)
    mz, it = mz[o], np.sqrt(it[o])
    n = np.sqrt((it ** 2).sum())
    return mz.astype(np.float32), (it / n if n > 0 else it).astype(np.float32)


@njit(cache=True)
def modcos(mza, ia, mzb, ib, shift, tol):
    """Greedy modified cosine: peaks match at equal m/z or at m/z + shift (b = a + shift). Inputs L2-normalised."""
    na, nb = len(mza), len(mzb)
    if na == 0 or nb == 0:
        return 0.0, 0
    # candidate pairs
    cap = na * 4 + 8
    sc = np.empty(cap, np.float64); pa = np.empty(cap, np.int64); pb = np.empty(cap, np.int64)
    k = 0
    for s_i in range(2):
        s = 0.0 if s_i == 0 else shift
        if s_i == 1 and abs(shift) < tol:
            break
        j0 = 0
        for i in range(na):
            t = mza[i] + s
            while j0 < nb and mzb[j0] < t - tol:
                j0 += 1
            j = j0
            while j < nb and mzb[j] <= t + tol:
                if k >= cap:
                    break
                sc[k] = ia[i] * ib[j]; pa[k] = i; pb[k] = j; k += 1
                j += 1
    if k == 0:
        return 0.0, 0
    order = np.argsort(-sc[:k])
    useda = np.zeros(na, np.bool_); usedb = np.zeros(nb, np.bool_)
    tot = 0.0; nm = 0
    for o in order:
        a = pa[o]; b = pb[o]
        if useda[a] or usedb[b]:
            continue
        useda[a] = True; usedb[b] = True
        tot += sc[o]; nm += 1
    return tot, nm


@njit(cache=True)
def modcos_shifted_flags(mza, ia, mzb, ib, shift, tol):
    """Like modcos but also returns per-peak-of-a match type: 0 none, 1 unshifted, 2 shifted (greedy)."""
    na, nb = len(mza), len(mzb)
    flag = np.zeros(na, np.int64)
    w = np.zeros(na, np.float64)
    cap = na * 4 + 8
    sc = np.empty(cap, np.float64); pa = np.empty(cap, np.int64); pb = np.empty(cap, np.int64); st = np.empty(cap, np.int64)
    k = 0
    for s_i in range(2):
        s = 0.0 if s_i == 0 else shift
        if s_i == 1 and abs(shift) < tol:
            break
        j0 = 0
        for i in range(na):
            t = mza[i] + s
            while j0 < nb and mzb[j0] < t - tol:
                j0 += 1
            j = j0
            while j < nb and mzb[j] <= t + tol:
                if k >= cap:
                    break
                sc[k] = ia[i] * ib[j]; pa[k] = i; pb[k] = j; st[k] = s_i + 1; k += 1
                j += 1
    order = np.argsort(-sc[:k])
    useda = np.zeros(na, np.bool_); usedb = np.zeros(nb, np.bool_)
    for o in order:
        a = pa[o]; b = pb[o]
        if useda[a] or usedb[b]:
            continue
        useda[a] = True; usedb[b] = True
        flag[a] = st[o]; w[a] = sc[o]
    return flag, w


class Store:
    """Memory-mapped compact store written by scripts/build_spec_store.py."""

    def __init__(self, path):
        self.mz = np.load(f'{path}/spec_mz.npy', mmap_mode='r')
        self.it = np.load(f'{path}/spec_int.npy', mmap_mode='r')
        self.off = np.load(f'{path}/spec_off.npy')
        self.meta = pd.read_parquet(f'{path}/spec_meta.parquet')

    def peaks(self, i):
        a, b = self.off[i], self.off[i + 1]
        return np.asarray(self.mz[a:b], np.float64), np.asarray(self.it[a:b], np.float64)
