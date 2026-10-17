"""Build the engine's resources directly from train.parquet (used in the Kaggle notebook and locally)."""
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .spectra import clean_peaks, neutral_mass

TEST_ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-',
            '[M-H2O-H]-', '[M-2H2O+H]+']
META = ['inchikey14', 'normalized_smiles', 'ingest_lib', 'adduct', 'ionization_mode', 'instrument_type',
        'precursor_mz', 'collision_energy_ev', 'num_peaks']


def build_analog_lib(train_path, out_path, per_struct=3, top_k=48):
    pf = pq.ParquetFile(train_path)
    m = pq.read_table(train_path, columns=META).to_pandas()
    m['row'] = np.arange(len(m))
    m = m[m.adduct.isin(TEST_ADD)].copy()
    m['pol'] = (m.ionization_mode == 'negative').astype(np.int8)
    ce = m.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    inst = m.instrument_type.fillna('').str.lower()
    m['pri'] = ((~m.adduct.isin(['[M+H]+', '[M-H]-'])) * 4 + (~ce.between(15, 65)) * 2
                + (~(inst.str.contains('tims') | inst.str.contains('tof'))) * 1 - np.minimum(m.num_peaks, 200) / 1000.0)
    m = m.sort_values(['inchikey14', 'pol', 'pri'])
    sel = m.groupby(['inchikey14', 'pol']).head(per_struct).sort_values('row')
    want = set(sel.row.values)
    peaks = {}
    base = 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz'])
        nrow = t.num_rows
        rows = [r for r in range(base, base + nrow) if r in want]
        if rows:
            mzl = t.column('ms2_mzs').combine_chunks(); itl = t.column('ms2_normalized_intensities').combine_chunks()
            off = mzl.offsets.to_numpy(); mz = mzl.values.to_numpy(); it = itl.values.to_numpy()
            prec = t.column('precursor_mz').to_numpy()
            for r in rows:
                i = r - base
                a, b = clean_peaks(mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]], prec[i], rel_floor=0.005, top_k=top_k)
                if len(a) >= 3:
                    peaks[r] = (a, b)
        base += nrow
    S = sel[sel.row.isin(peaks)]
    mzs = [peaks[r][0] for r in S.row]; its = [peaks[r][1] for r in S.row]
    off = np.zeros(len(mzs) + 1, np.int64); off[1:] = np.cumsum([len(x) for x in mzs])
    nm = np.array([neutral_mass(p, a) for p, a in zip(S.precursor_mz, S.adduct)])
    np.savez(out_path, mz=np.concatenate(mzs), it=np.concatenate(its), off=off, row=S.row.values,
             key=S.inchikey14.values.astype('U14'), smiles=S.normalized_smiles.values.astype(object), nmass=nm,
             pol=S.pol.values, adduct=S.adduct.values.astype('U16'), prec=S.precursor_mz.values,
             lib=S.ingest_lib.values.astype('U20'))
    return len(S)


def structures(train_path):
    """One row per inchikey14: smiles and exact mass."""
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    m = pq.read_table(train_path, columns=['inchikey14', 'normalized_smiles']).to_pandas()
    s = m.drop_duplicates('inchikey14').reset_index(drop=True)
    masses = []
    for smi in s.normalized_smiles:
        mol = Chem.MolFromSmiles(smi)
        masses.append(ExactMolWt(mol) if mol is not None else np.nan)
    s['mass'] = masses
    return s.rename(columns={'normalized_smiles': 'smiles'})


def hydrolyse_all(structs: pd.DataFrame, procs=4):
    from .hydrolysis import work
    with Pool(procs) as pool:
        res = pool.map(work, list(zip(structs.inchikey14, structs.smiles)), chunksize=500)
    rows = [r for rr in res for r in rr]
    return pd.DataFrame(rows, columns=['key', 'bond', 'acid', 'partner'])
