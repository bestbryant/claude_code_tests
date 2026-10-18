"""Pool v4 = pool v2 + ChEBI / LIPID MAPS LMSD structures (both CC BY 4.0; standardised set of
sergiomalv/casmi-chebi-lmsd-structures, see /home/user/ext/chebi_lmsd/ATTRIBUTION.md).

Usage: python scripts/build_pool_bio.py <pool_in_dir> <bio_structures.parquet> <pool_out_dir> [procs]
Adds a membership flag in_bio (1 for every pool row whose key is in ChEBI/LMSD); new rows get in_train = in_coco =
in_pcl = 0 and zero PubChemLite columns. PubChem documentation counts (pc_*) are filled afterwards by
scripts/add_pubchem_pop.py, as for the other pools.
"""
import os, shutil, sys
from multiprocessing import Pool
import numpy as np, pandas as pd


def props(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    global _G
    try:
        _G
    except NameError:
        _G = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    m = Chem.MolFromSmiles(smi)
    if m is None or len(Chem.GetMolFrags(m)) != 1:
        return None
    mass = ExactMolWt(m)
    if not 100 <= mass <= 1300:
        return None
    Chem.RemoveStereochemistry(m)
    fp = np.packbits(_G.GetFingerprintAsNumPy(m).astype(np.uint8))
    return (Chem.MolToSmiles(m), Chem.MolToInchiKey(m)[:14], rdMolDescriptors.CalcMolFormula(m), mass,
            m.GetNumHeavyAtoms(), fp)


if __name__ == '__main__':
    src, bio_p, dst = sys.argv[1:4]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.makedirs(dst, exist_ok=True)
    P = pd.read_parquet(f'{src}/pool.parquet')
    B = pd.read_parquet(bio_p)
    bio_keys = set(B.inchikey14)
    new = B[~B.inchikey14.isin(set(P.key))].drop_duplicates('inchikey14')
    with Pool(procs) as pool:
        res = pool.map(props, new.smiles.tolist(), chunksize=500)
    rows = [r for r in res if r is not None]
    N = pd.DataFrame([r[:5] for r in rows], columns=['smiles', 'key', 'formula', 'mass', 'ha'])
    nfp = np.stack([r[5] for r in rows])
    keep = ~N.duplicated('key').values & ~N.key.isin(set(P.key)).values
    N, nfp = N[keep].reset_index(drop=True), nfp[keep]
    for c in P.columns:
        if c not in N.columns:
            N[c] = 0
    allp = pd.concat([P, N[P.columns]], ignore_index=True)
    allp['in_bio'] = allp.key.isin(bio_keys).astype(np.int8)
    allfp = np.concatenate([np.load(f'{src}/pool_fp.npy'), nfp])
    o = np.argsort(allp.mass.values, kind='stable')
    allp.iloc[o].reset_index(drop=True).to_parquet(f'{dst}/pool.parquet')
    np.save(f'{dst}/pool_fp.npy', allfp[o])
    shutil.copy(f'{src}/coconut_prior.parquet', f'{dst}/coconut_prior.parquet')
    print('pool v4', len(allp), 'added', len(N), 'in_bio', int(allp.in_bio.sum()))
