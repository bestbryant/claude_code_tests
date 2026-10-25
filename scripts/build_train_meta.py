"""train_meta.parquet: 10-column projection of the competition train.parquet, no filter, same row order
(docs/reproduce.md 1.1, G9). Read by build_pool.py, train_ret_ranker.py, make_split_keys.py and others.

python scripts/build_train_meta.py <train.parquet> <out.parquet>
Input: competition train.parquet (2,539,608 rows; local copy md5 b18b53c8). Output byte-identical to the Oct 17 file
(md5 82d7df60, pandas 3.0.5 / pyarrow 26.0.0); row count and md5 are checked before writing. Peak RSS ~1.2 GB.
"""
import hashlib, io, os, sys
import pandas as pd

COLS = ['inchikey14', 'normalized_smiles', 'molecular_formula', 'ingest_lib', 'adduct', 'instrument_type',
        'precursor_mz', 'collision_energy_ev', 'num_peaks', 'ionization_mode']
MD5 = '82d7df60fee38cecca498c3dff1e5b79'

src, dst = sys.argv[1:3]
M = pd.read_parquet(src, columns=COLS)
assert len(M) == 2539608, len(M)
buf = io.BytesIO()
M.to_parquet(buf)
b = buf.getvalue()
h = hashlib.md5(b).hexdigest()
assert h == MD5, f'train_meta md5 {h} != {MD5}'
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
open(dst, 'wb').write(b)
print('train_meta', len(M), h)
