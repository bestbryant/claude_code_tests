"""Fragment evidence v2 for the stage-2 ranker (stage-1 top 30): depth-mixed explained share and 1-cut coverage.

For a candidate, every connected fragment from cutting one or two bonds is enumerated (two-bond cuts only for
molecules with <= 100 bonds) with its neutral mass and minimum cut depth. Per query spectrum (peaks below
precursor - 1.5, linear intensity shares), a peak is explained by a fragment when peak - charge carrier - s * H
(s = -2..2; carriers by adduct: proton plus Na / K / NH4 / formate / Cl adduct ions) matches within 0.01 Da / 15 ppm:
  ps_mix   intensity share explained, 1-cut fragments weighted 1, fragments needing two cuts 0.6
  cov1     share of single bonds whose cleavage has an observable side that matches a peak
  covw1    the same weighted by the best matching peak's intensity share
Spectrum values are averaged (and maximised) over the query's spectra; frag2_feats adds within-query ranks / gaps and
the gap to the best same-formula candidate.
"""
import numpy as np
from numba import njit
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
H = 1.00782503207; E = 0.00054857990; PROTON = H - E
MONO = {'C': 12.0, 'H': H, 'N': 14.0030740048, 'O': 15.99491461956, 'S': 31.97207100, 'P': 30.97376163,
        'Cl': 34.96885268, 'F': 18.99840322, 'Br': 78.9183371, 'I': 126.904473, 'Si': 27.9769265, 'B': 11.0093054,
        'Se': 79.9165218, 'Na': 22.98976928, 'K': 38.9637064}
CARRIERS = {'[M+H]+': [PROTON], '[M-H2O+H]+': [PROTON], '[M-2H2O+H]+': [PROTON],
            '[M+NH4]+': [PROTON, MONO['N'] + 4 * H - E], '[M+Na]+': [PROTON, MONO['Na'] - E],
            '[M+K]+': [PROTON, MONO['K'] - E], '[M-H]-': [-PROTON], '[M-H2O-H]-': [-PROTON],
            '[M+CH2O2-H]-': [-PROTON, MONO['C'] + 2 * H + 2 * MONO['O'] - PROTON], '[M+Cl]-': [-PROTON, MONO['Cl'] + E]}
FRAG2 = ['ps_mix_mean', 'ps_mix_max', 'cov1_mean', 'covw1_mean']
FRAG2_FEATS = FRAG2 + ['ps_mix_mean_r', 'ps_mix_mean_gap', 'ps_mix_mean_fgap', 'covw1_mean_r', 'covw1_mean_fgap']


@njit(cache=True)
def _find(p, a):
    while p[a] != a:
        p[a] = p[p[a]]
        a = p[a]
    return a


@njit(cache=True)
def _cut(n, w, ea, eb, r1, r2, parent, acc, om, ok, cnt, k, nc0):
    for a in range(n):
        parent[a] = a; acc[a] = 0.0
    nc = n
    for e in range(ea.shape[0]):
        if e == r1 or e == r2:
            continue
        x = _find(parent, ea[e]); y = _find(parent, eb[e])
        if x != y:
            parent[x] = y; nc -= 1
    if nc <= nc0:                       # the cut does not disconnect anything (ring bond)
        return cnt
    for a in range(n):
        acc[_find(parent, a)] += w[a]
    for a in range(n):
        if parent[a] == a:
            om[cnt] = acc[a]; ok[cnt] = k; cnt += 1
    return cnt


@njit(cache=True)
def enum_frags(n, w, ea, eb, max2):
    """Masses (and cut depth 1 / 2) of the components after every 1-bond cut and, if n_bonds <= max2, every 2-bond
    cut. w: per-atom mass including its hydrogens."""
    nb = ea.shape[0]
    do2 = nb <= max2
    parent = np.empty(n, np.int64); acc = np.empty(n)
    for a in range(n):
        parent[a] = a
    nc0 = n
    for e in range(nb):
        x = _find(parent, ea[e]); y = _find(parent, eb[e])
        if x != y:
            parent[x] = y; nc0 -= 1
    cap = (nc0 + 1) * nb + ((nc0 + 2) * (nb * (nb - 1) // 2) if do2 else 0) + 4
    om = np.empty(cap); ok = np.empty(cap, np.int8)
    cnt = 0
    for i in range(nb):
        cnt = _cut(n, w, ea, eb, i, -1, parent, acc, om, ok, cnt, 1, nc0)
    if do2:
        for i in range(nb):
            for j in range(i + 1, nb):
                cnt = _cut(n, w, ea, eb, i, j, parent, acc, om, ok, cnt, 2, nc0)
    return om[:cnt], ok[:cnt]


@njit(cache=True)
def match_depth(ms, ks, mz, deltas, hlo, hhi, tol_da, tol_ppm):
    """Smallest cut depth of a fragment explaining each peak (0 = none); ms sorted."""
    out = np.zeros(mz.shape[0], np.int8)
    m = ms.shape[0]
    for p in range(mz.shape[0]):
        t = max(mz[p] * tol_ppm * 1e-6, tol_da)
        best = 99
        for d in range(deltas.shape[0]):
            for s in range(hlo, hhi + 1):
                x = mz[p] - deltas[d] - s * 1.00782503207
                j = np.searchsorted(ms, x - t)
                while j < m and ms[j] <= x + t:
                    if ks[j] < best:
                        best = ks[j]
                    j += 1
        out[p] = best if best < 99 else 0
    return out


def mol_record(smiles, max2=100):
    """Fragment table of one candidate (None if it does not parse, has no bonds or several components)."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None or m.GetNumBonds() == 0 or len(Chem.GetMolFrags(m)) > 1:
        return None
    w = np.array([MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H for a in m.GetAtoms()])
    b = np.array([(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in m.GetBonds()], np.int64).reshape(-1, 2)
    ea, eb = b[:, 0].copy(), b[:, 1].copy()
    om, ok = enum_frags(len(w), w, ea, eb, max2)
    r = np.round(om, 5)
    o = np.lexsort((ok, r)); r, ok2 = r[o], ok[o]
    first = np.r_[True, r[1:] != r[:-1]]
    ms, ks = om[o][first], ok2[first]
    o = np.argsort(ms)
    one, _ = enum_frags(len(w), w, ea, eb, -1)          # components of every disconnecting single cut, in pairs
    pairs = one.reshape(-1, 2) if len(one) % 2 == 0 else np.zeros((0, 2))
    return dict(ms=ms[o], ks=ks[o], pairs=pairs)


def spectrum_scores(rec, mz, it, prec, adduct):
    """(ps_mix, cov1, covw1) of one cleaned spectrum (mz, it as from clean_peaks) for one candidate record."""
    sel = mz < prec - 1.5
    if not sel.any():
        return 0.0, 0.0, 0.0
    wl = it.astype(np.float64) ** 2; wl = wl / wl.sum()
    mz_, wl_ = mz[sel].astype(np.float64), wl[sel]
    sg = 1.0 if adduct.endswith('+') else -1.0
    car = np.array(CARRIERS.get(adduct, [sg * PROTON]))
    k = match_depth(rec['ms'], rec['ks'], mz_, car, -2, 2, 0.01, 15.0)
    mix = float((wl_ * np.where(k == 1, 1.0, np.where(k == 2, 0.6, 0.0))).sum())
    lo, hi = max(50.0, float(mz.min()) - 0.5), prec - 2
    no = nm = 0; wm = 0.0
    for A, B in rec['pairs']:
        obs = [x for x in (A, B) if lo <= x + car[0] <= hi]
        if not obs:
            continue
        no += 1
        ms_ = np.sort(np.array(obs))
        kk = match_depth(ms_, np.ones(len(ms_), np.int8), mz_, car, -1, 1, 0.01, 15.0)
        if (kk > 0).any():
            nm += 1; wm += float(wl_[kk > 0].max())
    return mix, (nm / no if no else 0.0), (wm / no if no else 0.0)


def candidate_scores(smiles_list, spectra):
    """spectra: [(mz, it, prec, adduct)] cleaned query spectra -> dict of FRAG2 arrays over the candidates."""
    out = {c: np.zeros(len(smiles_list)) for c in FRAG2}
    if not spectra:
        return out
    for i, s in enumerate(smiles_list):
        rec = mol_record(s)
        if rec is None:
            continue
        v = np.array([spectrum_scores(rec, *sp) for sp in spectra])
        out['ps_mix_mean'][i], out['cov1_mean'][i], out['covw1_mean'][i] = v.mean(0)
        out['ps_mix_max'][i] = v[:, 0].max()
    return out


def frag2_feats(df, keys=('qid',)):
    """Within-query rank / gap and gap to the best same-formula candidate, in place (df has FRAG2 + formula)."""
    keys = [k for k in keys if k in df.columns]
    g = df.groupby(keys, sort=False) if keys else None
    gf = df.groupby(keys + ['formula'], sort=False)
    for c in ('ps_mix_mean', 'covw1_mean'):
        df[c + '_r'] = g[c].rank(ascending=False, method='min') if keys else df[c].rank(ascending=False, method='min')
        if c == 'ps_mix_mean':
            df[c + '_gap'] = df[c] - (g[c].transform('max') if keys else df[c].max())
        df[c + '_fgap'] = df[c] - gf[c].transform('max')
    return df
