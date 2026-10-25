"""Pool v1 = scripts/build_pool.py as of commit 26b2f28 (verbatim; HEAD build_pool.py builds pool v2). Pool v1 is the
pool of casmi26-fp-data (prep_fp_data.py), casmi26-pf-data (prep_pf_data.py) and pf_negtrain.py (docs/reproduce.md G7).

Candidate pool = training structures U COCONUT (2026-10), one row per standard InChIKey14.

Writes <out>/pool.parquet (key, smiles, formula, mass, ha, in_train, in_coco) and <out>/pool_fp.npy (Morgan r2
2048 bits, packed uint8).
Usage: python scripts/build_pool_v1.py <out_dir> [procs] [train_meta.parquet] [coconut_2026_10.parquet]
(input defaults: /home/user/data/train_meta.parquet, /home/user/ext/coconut/coconut_2026_10.parquet)
Checks 726,198 rows, pool_fp.npy md5 6df2c180 and pool.parquet md5 707055a9 (reference environment of
docs/reproduce.md: row order needs AVX-512 numpy sort dispatch, G12; parquet bytes need pyarrow 26.0.0).
"""
import hashlib, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')

out = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TM = sys.argv[3] if len(sys.argv) > 3 else '/home/user/data/train_meta.parquet'
CO = sys.argv[4] if len(sys.argv) > 4 else '/home/user/ext/coconut/coconut_2026_10.parquet'
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
    tm = pd.read_parquet(TM, columns=['inchikey14', 'normalized_smiles'])
    tr = tm.drop_duplicates('inchikey14')
    co = pd.read_parquet(CO).drop_duplicates('ik14')
    co = co[co.exact_molecular_weight.between(80, 1600)]
    src = pd.concat([pd.DataFrame(dict(smiles=tr.normalized_smiles.values, train=1)),
                     pd.DataFrame(dict(smiles=co.canonical_smiles.values, train=0))], ignore_index=True)
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
    P['in_train'] = P.key.map(flags['max']).astype(np.int8)
    P['in_coco'] = P.key.map(lambda k: 1).astype(np.int8)
    # in_coco: key has a COCONUT row
    coco_keys = set(df.loc[df.train == 0, 'key'])
    P['in_coco'] = P.key.isin(coco_keys).astype(np.int8)
    P = P.sort_values('mass').reset_index(drop=False).rename(columns={'index': 'src_row'})
    np.save(f'{out}/pool_fp.npy', fps[first][P.src_row.values])
    P.drop(columns=['src_row', 'train']).to_parquet(f'{out}/pool.parquet')
    print('pool', len(P), 'train', int(P.in_train.sum()), 'coconut', int(P.in_coco.sum()),
          'both', int((P.in_train & P.in_coco).sum()))
    md5 = lambda f: hashlib.file_digest(open(f, 'rb'), 'md5').hexdigest()
    assert len(P) == 726198, len(P)
    assert md5(f'{out}/pool_fp.npy') == '6df2c1802f5908d82d8446191dcc1414', 'pool_fp.npy md5 (row order: G12)'
    assert md5(f'{out}/pool.parquet') == '707055a9731618d21fe14922d72c7c4c', 'pool.parquet md5 (pyarrow 26.0.0?)'
    print('pool v1 md5 ok')
