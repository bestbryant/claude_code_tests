"""Representative cleaned spectra per (structure, polarity) for analog search.

Up to N spectra per structure+polarity, preferring test adducts ([M+H]+/[M-H]- first), eV energies 20-60 and
timsTOF/QTOF sources. Writes work/analog_lib.npz (flat cleaned peaks + per-spectrum key/mass/polarity/adduct).
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.spectra import Store, clean_peaks, neutral_mass

N = 3
st = Store('/home/user/work/train_store')
m = st.meta
TEST_ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-',
            '[M-H2O-H]-', '[M-2H2O+H]+']
m = m[m.adduct.isin(TEST_ADD)].copy()
m['pol'] = (m.ionization_mode == 'negative').astype(np.int8)
ce = m.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
inst = m.instrument_type.fillna('').str.lower()
m['pri'] = ((~m.adduct.isin(['[M+H]+', '[M-H]-'])) * 4 + (~ce.between(15, 65)) * 2
            + (~(inst.str.contains('tims') | inst.str.contains('tof'))) * 1 - np.minimum(m.num_peaks, 200) / 1000.0)
m = m.sort_values(['inchikey14', 'pol', 'pri'])
sel = m.groupby(['inchikey14', 'pol']).head(N)
print('selected', len(sel), 'structures', sel.inchikey14.nunique())
mzs, its, lens, rows = [], [], [], []
for i, r in zip(sel.index.values, sel.itertuples()):
    mz, it = st.peaks(i)
    a, b = clean_peaks(mz, it, r.precursor_mz, rel_floor=0.005, top_k=48, remove_prec=False)
    if len(a) < 3:
        continue
    mzs.append(a); its.append(b); lens.append(len(a)); rows.append(i)
off = np.zeros(len(lens) + 1, np.int64); off[1:] = np.cumsum(lens)
S = m.loc[rows]
nm = np.array([neutral_mass(p, a) for p, a in zip(S.precursor_mz, S.adduct)])
np.savez('/home/user/work/analog_lib.npz', mz=np.concatenate(mzs), it=np.concatenate(its), off=off,
         row=np.array(rows), key=S.inchikey14.values.astype('U14'), smiles=S.normalized_smiles.values.astype(object),
         nmass=nm, pol=S.pol.values, adduct=S.adduct.values.astype('U16'), prec=S.precursor_mz.values,
         lib=S.ingest_lib.values.astype('U20'))
print('done', len(rows))
