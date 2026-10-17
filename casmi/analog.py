"""Analog search over the representative library (work/analog_lib.npz) with modified cosine."""
import numpy as np
from numba import njit, prange

from .spectra import modcos


@njit(parallel=True, cache=True)
def _scan(lmz, lit, loff, idx, lmass, qmz, qit, qmass, tol, min_match):
    out = np.zeros(len(idx), np.float64)
    nmt = np.zeros(len(idx), np.int64)
    for t in prange(len(idx)):
        j = idx[t]
        a, b = loff[j], loff[j + 1]
        s, nm = modcos(qmz, qit, lmz[a:b], lit[a:b], lmass[j] - qmass, tol)
        if nm >= min_match:
            out[t] = s
        nmt[t] = nm
    return out, nmt


class AnalogLib:
    def __init__(self, path='/home/user/work/analog_lib.npz'):
        z = np.load(path, allow_pickle=True)
        self.mz, self.it, self.off = z['mz'], z['it'].astype(np.float32), z['off']
        self.key, self.smiles, self.nmass, self.pol = z['key'], z['smiles'], z['nmass'], z['pol']
        self.adduct, self.prec, self.lib = z['adduct'], z['prec'], z['lib']
        # precursor-ion mass used for the shift: the shift applies between ion masses of the same adduct type;
        # across adducts we use neutral masses (fragments of different adducts rarely align anyway)
        self.order = {p: np.where(self.pol == p)[0] for p in (0, 1)}

    def search(self, qmz, qit, qmass, pol, window=(-350.0, 350.0), tol=0.01, min_match=3, exclude_keys=(),
               top=50):
        cand = self.order[pol]
        d = self.nmass[cand] - qmass
        cand = cand[(d >= window[0]) & (d <= window[1])]
        if len(cand) == 0:
            return []
        sc, nm = _scan(self.mz, self.it, self.off, cand.astype(np.int64), self.nmass.astype(np.float64),
                       qmz.astype(np.float64), qit.astype(np.float64), float(qmass), tol, min_match)
        o = np.argsort(-sc)
        res, seen = [], set()
        for t in o:
            if sc[t] <= 0:
                break
            j = cand[t]
            k = self.key[j]
            if k in exclude_keys or k in seen:
                continue
            seen.add(k)
            res.append((j, float(sc[t]), int(nm[t])))
            if len(res) >= top:
                break
        return res
