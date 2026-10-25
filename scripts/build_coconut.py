"""COCONUT 10-2026 tables: coconut_2026_10.parquet (pool rows, in_coco) and coconut_prior.parquet (8 prior features of
stage 1 / 2 and docw), docs/reproduce.md 1.1 and G8.

python scripts/build_coconut.py <lite_csv> <full_csv> <out_dir>
Either CSV may be given as the downloaded .zip (read directly, nothing unpacked). Release of 2026-10-01:
  lite https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv_lite-10-2026.zip   199,096,787 B
       zip sha256 a602502a4fa0950cc2e59eed690dccd082d1a051bf9f85ccda50a82dd22ca154
       csv sha256 fcdbf564b90d8a09e29b94b61250f3fe0d4332cb6f9f9ce45c3e829567b3b873 (576,925,391 B)
  full https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv-10-2026.zip        247,959,854 B
       zip sha256 95a0525b11a334389ceebb941d0a8ee9446892fccf4f8ff78d2184efc1c974ea
       csv sha256 ad4b174454c16dded7cbb40e96dd045b8d4203bc9e96473bb6d0c1c87ea4b46a (699,882,314 B)
coconut_2026_10.parquet: LITE CSV in file order (build_pool.py keeps the first row per ik14; the full-CSV order changes
  3,050 stereo-stripped SMILES): identifier, canonical_smiles, ik14 = standard_inchi_key[:14], exact_molecular_weight,
  np_likeness. 738,794 rows, md5 8e2fcf7e.
coconut_prior.parquet: FULL CSV (organisms, collections, dois, synonyms, cas are not in the lite one) grouped by ik14
  (sorted): n_org / n_coll / n_doi / n_syn = max number of '|'-separated entries (0 if empty), has_cas / has_name =
  max(notna), ann = max annotation_level, npl = max np_likeness, pathway = first non-null np_classifier_pathway,
  n_stereo = rows per ik14. 479,690 rows, md5 f8b83ce5.
Both outputs are byte-identical to the Oct 17 files (pandas 3.0.5, pyarrow 26.0.0); md5s are checked before writing.
"""
import hashlib, io, os, sys
import pandas as pd

MD5 = {'coconut_2026_10.parquet': '8e2fcf7e6234aa3b332f7a9bbedeaf22',
       'coconut_prior.parquet': 'f8b83ce5512deabd27431a9ae8b48c87'}
ROWS = {'coconut_2026_10.parquet': 738794, 'coconut_prior.parquet': 479690}

lite, full, out = sys.argv[1:4]


def save(df, name):
    assert len(df) == ROWS[name], f'{name}: {len(df)} rows != {ROWS[name]}'
    buf = io.BytesIO()
    df.to_parquet(buf)
    b = buf.getvalue()
    h = hashlib.md5(b).hexdigest()
    assert h == MD5[name], f'{name}: md5 {h} != {MD5[name]}'
    open(f'{out}/{name}', 'wb').write(b)
    print(name, len(df), h, flush=True)


os.makedirs(out, exist_ok=True)
L = pd.read_csv(lite, usecols=['identifier', 'canonical_smiles', 'standard_inchi_key', 'exact_molecular_weight',
                               'np_likeness'])
L['ik14'] = L.standard_inchi_key.str[:14]
save(L[['identifier', 'canonical_smiles', 'ik14', 'exact_molecular_weight', 'np_likeness']], 'coconut_2026_10.parquet')
del L

F = pd.read_csv(full, usecols=['standard_inchi_key', 'name', 'annotation_level', 'np_likeness', 'np_classifier_pathway',
                               'organisms', 'collections', 'dois', 'synonyms', 'cas'])
F['ik14'] = F.standard_inchi_key.str[:14]
for c, n in (('organisms', 'n_org'), ('collections', 'n_coll'), ('dois', 'n_doi'), ('synonyms', 'n_syn')):
    F[n] = F[c].str.count(r'\|').add(1).fillna(0).astype('int64')
F['has_cas'] = F.cas.notna().astype('int64')
F['has_name'] = F.name.notna().astype('int64')
P = F.groupby('ik14').agg(n_org=('n_org', 'max'), n_coll=('n_coll', 'max'), n_doi=('n_doi', 'max'),
                          n_syn=('n_syn', 'max'), has_cas=('has_cas', 'max'), has_name=('has_name', 'max'),
                          ann=('annotation_level', 'max'), npl=('np_likeness', 'max'),
                          pathway=('np_classifier_pathway', 'first'), n_stereo=('ik14', 'size')).reset_index()
save(P, 'coconut_prior.parquet')
