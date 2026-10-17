"""Compact spectrum store: top-K peaks per spectrum (peaks above precursor+2 Da dropped), flat arrays.

Writes <out>/spec_{mz,int,off}.npy and <out>/spec_meta.parquet (row order = parquet order).
Usage: python scripts/build_spec_store.py data/train.parquet work/train_store [K]
"""
import os, sys
import numpy as np, pyarrow.parquet as pq, pandas as pd

src, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 64
os.makedirs(out, exist_ok=True)
pf = pq.ParquetFile(src)
meta_cols = [c for c in ['molecule_id', 'spectrum_id', 'inchikey14', 'normalized_smiles', 'molecular_formula', 'ingest_lib',
                         'adduct', 'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_ev',
                         'base_peak_intensity', 'num_peaks'] if c in pf.schema_arrow.names]
mzs, ints, lens, metas = [], [], [], []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=meta_cols + ['ms2_mzs', 'ms2_normalized_intensities'])
    mzl = t.column('ms2_mzs').combine_chunks(); itl = t.column('ms2_normalized_intensities').combine_chunks()
    off = mzl.offsets.to_numpy(); mz = mzl.values.to_numpy(); it = itl.values.to_numpy()
    prec = t.column('precursor_mz').to_numpy()
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        m, v = mz[a:b], it[a:b]
        keep = m <= prec[i] + 2.0
        m, v = m[keep], v[keep]
        if len(m) > K:
            idx = np.argpartition(-v, K)[:K]
            m, v = m[idx], v[idx]
        o = np.argsort(m)
        mzs.append(m[o].astype(np.float32)); ints.append(v[o].astype(np.float32)); lens.append(len(m))
    metas.append(t.select(meta_cols).to_pandas())
    print('row group', rg, sum(lens), flush=True)
offs = np.zeros(len(lens) + 1, np.int64); offs[1:] = np.cumsum(lens)
np.save(f'{out}/spec_mz.npy', np.concatenate(mzs)); np.save(f'{out}/spec_int.npy', np.concatenate(ints).astype(np.float16))
np.save(f'{out}/spec_off.npy', offs)
pd.concat(metas, ignore_index=True).to_parquet(f'{out}/spec_meta.parquet')
print('done', len(lens), offs[-1])
