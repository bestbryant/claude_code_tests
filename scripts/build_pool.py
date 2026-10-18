"""Candidate pool = training structures U COCONUT (2026-10) U PubChemLite (2026-09-25), one row per standard InChIKey14.

Writes <out>/pool.parquet (key, smiles, formula, mass, ha, in_train, in_coco, in_pcl, pcl_pubmed, pcl_patent,
pcl_anno, pcl_bio, pcl_food) and <out>/pool_fp.npy (Morgan r2 2048 bits, packed uint8).
Usage: python scripts/build_pool.py <out_dir> [procs]
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')

out = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(out, exist_ok=True)
_G = None


def props(smi):
    global _G
    if _G is None:
        _G = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    fp = np.packbits(_G.GetFingerprintAsNumPy(m).astype(np.uint8))
    return (Chem.MolToSmiles(m), Chem.MolToInchiKey(m)[:14], rdMolDescriptors.CalcMolFormula(m), ExactMolWt(m),
            m.GetNumHeavyAtoms(), fp)


if __name__ == '__main__':
    tm = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['inchikey14', 'normalized_smiles'])
    tr = tm.drop_duplicates('inchikey14')
    co = pd.read_parquet('/home/user/ext/coconut/coconut_2026_10.parquet').drop_duplicates('ik14')
    co = co[co.exact_molecular_weight.between(80, 1600)]
    pcl = pd.read_parquet('/home/user/ext/pcl/pcl.parquet')
    pcl = pcl[pcl.MonoisotopicMass.between(80, 1600)]
    src = pd.concat([pd.DataFrame(dict(smiles=tr.normalized_smiles.values, train=1)),
                     pd.DataFrame(dict(smiles=co.canonical_smiles.values, train=0)),
                     pd.DataFrame(dict(smiles=pcl.SMILES.values, train=2, sidx=np.arange(len(pcl))))], ignore_index=True)
    with Pool(procs) as p:
        res = p.map(props, src.smiles.tolist(), chunksize=2000)
    ok = [r is not None for r in res]
    rows = [r for r in res if r is not None]
    df = pd.DataFrame([r[:5] for r in rows], columns=['smiles', 'key', 'formula', 'mass', 'ha'])
    df['train'] = src.train.values[ok]
    fps = np.stack([r[5] for r in rows])
    g = df.groupby('key')
    first = g.head(1).index.values
    flags = g.train.agg(['max', 'min'])
    P = df.loc[first].reset_index(drop=True)
    P['in_train'] = P.key.isin(set(df.loc[df.train == 1, 'key'])).astype(np.int8)
    P['in_coco'] = P.key.isin(set(df.loc[df.train == 0, 'key'])).astype(np.int8)
    P['in_pcl'] = P.key.isin(set(df.loc[df.train == 2, 'key'])).astype(np.int8)
    # PubChemLite annotation priors, keyed by the structure key recomputed from its SMILES
    pk = df.loc[df.train == 2, 'key'].values
    sidx = src.sidx.values[ok]
    pv = pcl.iloc[sidx[df.train.values == 2].astype(np.int64)]
    ann = pd.DataFrame(dict(key=pk, pcl_pubmed=pv.PubMed_Count.values, pcl_patent=pv.Patent_Count.values,
                            pcl_anno=pv.AnnoTypeCount.values, pcl_bio=pv.BioPathway.values,
                            pcl_food=pv.FoodRelated.values)).groupby('key').max()
    for c in ann.columns:
        P[c] = P.key.map(ann[c]).fillna(0).astype(np.float32)
    P = P.sort_values('mass').reset_index(drop=False).rename(columns={'index': 'src_row'})
    np.save(f'{out}/pool_fp.npy', fps[first][P.src_row.values])
    P.drop(columns=['src_row', 'train']).to_parquet(f'{out}/pool.parquet')
    print('pool', len(P), 'train', int(P.in_train.sum()), 'coconut', int(P.in_coco.sum()),
          'pcl', int(P.in_pcl.sum()), 'pcl only', int(((P.in_pcl == 1) & (P.in_coco == 0) & (P.in_train == 0)).sum()))
