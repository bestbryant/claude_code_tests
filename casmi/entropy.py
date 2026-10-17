"""Spectral entropy similarity (Li et al., Nat. Methods 2021), weighted variant, numba."""
import numpy as np
from numba import njit


@njit(cache=True)
def _prep(mz, it):
    """Normalise to sum 1 and apply entropy weighting for low-entropy spectra."""
    s = it.sum()
    if s <= 0:
        return it
    p = it / s
    h = 0.0
    for v in p:
        if v > 0:
            h -= v * np.log(v)
    if h < 3.0:
        w = 0.25 + h * 0.25
        p = p ** w
        p = p / p.sum()
    return p


@njit(cache=True)
def entropy_sim(mza, ia, mzb, ib, tol):
    """Weighted entropy similarity of two centroided spectra (m/z sorted ascending)."""
    na, nb = len(mza), len(mzb)
    if na == 0 or nb == 0:
        return 0.0
    pa = _prep(mza, ia.astype(np.float64)); pb = _prep(mzb, ib.astype(np.float64))
    # greedy one-to-one matching by m/z proximity
    i = j = 0
    ha = 0.0; hb = 0.0; hab = 0.0
    for v in pa:
        if v > 0:
            ha -= v * np.log(v)
    for v in pb:
        if v > 0:
            hb -= v * np.log(v)
    # merged spectrum entropy: matched peaks summed, unmatched kept (each halved)
    used_b = np.zeros(nb, np.bool_)
    hmix = 0.0
    j0 = 0
    for i in range(na):
        while j0 < nb and mzb[j0] < mza[i] - tol:
            j0 += 1
        best = -1; bd = tol + 1.0
        j = j0
        while j < nb and mzb[j] <= mza[i] + tol:
            if not used_b[j]:
                d = abs(mzb[j] - mza[i])
                if d < bd:
                    bd = d; best = j
            j += 1
        if best >= 0:
            used_b[best] = True
            v = (pa[i] + pb[best]) / 2.0
        else:
            v = pa[i] / 2.0
        if v > 0:
            hmix -= v * np.log(v)
    for j in range(nb):
        if not used_b[j]:
            v = pb[j] / 2.0
            if v > 0:
                hmix -= v * np.log(v)
    return 1.0 - (2.0 * hmix - ha - hb) / np.log(4.0)
