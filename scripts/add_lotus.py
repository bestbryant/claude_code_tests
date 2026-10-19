"""Pool v5 = pool v4 + LOTUS occurrence columns (LOTUS frozen dataset 230106, CC BY 4.0, Zenodo 7534071).

python scripts/add_lotus.py <pool_in_dir> <230106_frozen_metadata.csv.gz> <pool_out_dir>
Per connectivity key (InChIKey first block): distinct organisms / references, distinct land-plant (Streptophyta)
organisms, families and genera, fungal and bacterial organisms (counts stored as log1p), plant share of organisms.
The hidden test comes from plant extracts; these say whether, and how widely, a structure has been reported in plants.
pool_fp.npy and coconut_prior.parquet are hard-linked (identical row order).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.retrieval import LT_COLS

src, lotus, dst = sys.argv[1:4]
cols = ['structure_inchikey', 'organism_name', 'organism_taxonomy_02kingdom', 'organism_taxonomy_03phylum',
        'organism_taxonomy_06family', 'organism_taxonomy_08genus', 'reference_doi']
L = pd.read_csv(lotus, usecols=cols, dtype=str)
L['key'] = L.structure_inchikey.str[:14]
plant = L[L.organism_taxonomy_03phylum == 'Streptophyta']
fungi = L[L.organism_taxonomy_02kingdom == 'Fungi']
bact = L[L.organism_taxonomy_03phylum.isin(['Actinobacteria', 'Proteobacteria', 'Firmicutes', 'Cyanobacteria', 'Bacteroidetes'])]
g = L.groupby('key')
F = pd.DataFrame({'lt_n_org': g.organism_name.nunique(), 'lt_n_ref': g.reference_doi.nunique(),
                  'lt_n_plant': plant.groupby('key').organism_name.nunique(),
                  'lt_n_pfam': plant.groupby('key').organism_taxonomy_06family.nunique(),
                  'lt_n_pgen': plant.groupby('key').organism_taxonomy_08genus.nunique(),
                  'lt_n_fungi': fungi.groupby('key').organism_name.nunique(),
                  'lt_n_bact': bact.groupby('key').organism_name.nunique()}).fillna(0)
F['lt_plant_frac'] = F.lt_n_plant / F.lt_n_org.clip(lower=1)
for c in LT_COLS:
    if c != 'lt_plant_frac':
        F[c] = np.log1p(F[c])
os.makedirs(dst, exist_ok=True)
P = pd.read_parquet(f'{src}/pool.parquet')
J = F.reindex(P.key.values)
for c in LT_COLS:
    P[c] = J[c].fillna(0).values.astype(np.float32)
P.to_parquet(f'{dst}/pool.parquet')
for f in ('pool_fp.npy', 'coconut_prior.parquet'):
    if not os.path.exists(f'{dst}/{f}'):
        os.link(f'{src}/{f}', f'{dst}/{f}')
print('pool v5', len(P), 'with LOTUS occurrences', int((P.lt_n_org > 0).sum()), 'plant', int((P.lt_n_plant > 0).sum()))
