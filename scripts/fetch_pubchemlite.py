"""PubChemLite for Exposomics 3.3.0 (2026-09-25; Zenodo record 22953038, DOI 10.5281/zenodo.22953038, CC BY 4.0) ->
<out_dir>/pcl.csv and <out_dir>/pcl.parquet, the PubChemLite input of build_pool.py (docs/reproduce.md 1.0 / 1.1, G10).

python scripts/fetch_pubchemlite.py <out_dir> [pool_v1.parquet]
pcl.csv: PubChemLite_exposomics_20260925.csv, 200,142,750 B, md5 3023e89b (the md5 Zenodo lists); the download is
skipped when <out_dir>/pcl.csv already has that size and md5.
pcl.parquet: read_csv(usecols=Identifier, FirstBlock, PubMed_Count, Patent_Count, SMILES, MonoisotopicMass,
AnnoTypeCount, BioPathway, FoodRelated), file order, 469,602 rows, md5 d9d51dd1. The Oct 18 file also had an in_pool
column (FirstBlock in pool v1 keys) that nothing reads; pass pool v1's pool.parquet ($W/pool, md5 707055a9) to add it
and get that file byte for byte (md5 ac08d8f5). md5s are checked before writing (pandas 3.0.5, pyarrow 26.0.0).
"""
import hashlib, io, os, subprocess, sys
import pandas as pd

URL = 'https://zenodo.org/api/records/22953038/files/PubChemLite_exposomics_20260925.csv/content'
SIZE, CSV_MD5 = 200142750, '3023e89b20e5b57bd3e2c70dc1366be1'
COLS = ['Identifier', 'FirstBlock', 'PubMed_Count', 'Patent_Count', 'SMILES', 'MonoisotopicMass', 'AnnoTypeCount',
        'BioPathway', 'FoodRelated']
MD5 = {False: 'd9d51dd1cf71131667324cf1d3bcbc97', True: 'ac08d8f5025fda5f45f2e6940064fc97'}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


out = sys.argv[1]; pool1 = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(out, exist_ok=True)
csv = f'{out}/pcl.csv'
if not (os.path.exists(csv) and os.path.getsize(csv) == SIZE and md5(csv) == CSV_MD5):
    subprocess.run(['curl', '-fsSL', '--retry', '3', '-o', csv + '.part', URL], check=True)
    n, h = os.path.getsize(csv + '.part'), md5(csv + '.part')
    assert n == SIZE and h == CSV_MD5, f'download: {n} B md5 {h}, expected {SIZE} B md5 {CSV_MD5}'
    os.replace(csv + '.part', csv)
print('pcl.csv', SIZE, CSV_MD5, flush=True)

D = pd.read_csv(csv, usecols=COLS)
assert len(D) == 469602, len(D)
if pool1:
    D['in_pool'] = D.FirstBlock.isin(set(pd.read_parquet(pool1, columns=['key']).key))
buf = io.BytesIO()
D.to_parquet(buf)
b = buf.getvalue()
h = hashlib.md5(b).hexdigest()
assert h == MD5[bool(pool1)], f'pcl.parquet md5 {h} != {MD5[bool(pool1)]}'
open(f'{out}/pcl.parquet', 'wb').write(b)
print('pcl.parquet', len(D), h)
