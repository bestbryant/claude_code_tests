"""Library-hit profile of the test molecules: best weighted-entropy similarity to a training spectrum with the same
precursor (same polarity, |prec m/z diff| < tol_prec), per molecule. Used to estimate how many test molecules are class 1.

python scripts/test_libhits.py <test.parquet> <lib10.npz> <out.parquet>
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.entropy import entropy_sim
from casmi.spectra import clean_peaks, neutral_mass

test_p, lib_p, out = sys.argv[1:4]
z = np.load(lib_p, allow_pickle=True)
mz, it, off = z['mz'].astype(np.float64), z['it'].astype(np.float64), z['off']
key, nmass, pol, lib, prec = z['key'], z['nmass'], z['pol'], z['lib'], z['prec']
o = np.argsort(nmass); sm = nmass[o]
T = pd.read_parquet(test_p)
rows = []
for mid, g in T.groupby('molecule_id'):
    best = {}
    for _, r in g.iterrows():
        p = 1 if r.ionization_mode == 'negative' else 0
        try:
            qm = neutral_mass(r.precursor_mz, r.adduct)
        except Exception:
            continue
        qmz, qit = clean_peaks(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float), r.precursor_mz)
        if len(qmz) == 0:
            continue
        lo, hi = np.searchsorted(sm, qm - 0.006), np.searchsorted(sm, qm + 0.006)
        for j in o[lo:hi]:
            if pol[j] != p:
                continue
            a, b = off[j], off[j + 1]
            s = entropy_sim(qmz, qit, mz[a:b], it[a:b], 0.01)
            k = key[j]
            if s > best.get(k, (0, ''))[0]:
                best[k] = (s, lib[j])
    ks = sorted(best.items(), key=lambda kv: -kv[1][0])
    rows.append(dict(molecule_id=mid, n_iso=len(ks), best=ks[0][1][0] if ks else 0, best_key=ks[0][0] if ks else '',
                     best_lib=ks[0][1][1] if ks else '', second=ks[1][1][0] if len(ks) > 1 else 0))
D = pd.DataFrame(rows)
D.to_parquet(out)
print(D.describe())
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f'best >= {t}: {np.mean(D.best >= t):.3f}')
print(D.best_lib.value_counts().head())
