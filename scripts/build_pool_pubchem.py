"""Pool v3 = pool v2 + documented PubChem compounds (PubMed and/or patent records), streamed from the PubChem FTP.

Usage: python scripts/build_pool_pubchem.py <pool_in_dir> <pool_out_dir> <pubchem_dir> <min_pmid> <min_patent> [procs]
A CID is added when (pmid >= min_pmid or patent >= min_patent), its connectivity key is new to the pool, and its
exact mass is in 100-1300 Da. New rows get in_train = in_coco = in_pcl = 0 and in_pc = 1.
"""
import os, subprocess, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.keyset import pack

src, dst, pcd = sys.argv[1:4]
min_pmid, min_pat = int(sys.argv[4]), int(sys.argv[5])
procs = int(sys.argv[6]) if len(sys.argv) > 6 else 4
os.makedirs(dst, exist_ok=True)
FTP = 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras'


def props(item):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    global _G
    try:
        _G
    except NameError:
        _G = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cid, smi = item
    m = Chem.MolFromSmiles(smi)
    if m is None or len(Chem.GetMolFrags(m)) != 1:
        return None
    mass = ExactMolWt(m)
    if not 100 <= mass <= 1300:
        return None
    Chem.RemoveStereochemistry(m)
    fp = np.packbits(_G.GetFingerprintAsNumPy(m).astype(np.uint8))
    return (cid, Chem.MolToSmiles(m), Chem.MolToInchiKey(m)[:14], rdMolDescriptors.CalcMolFormula(m), mass,
            m.GetNumHeavyAtoms(), fp)


if __name__ == '__main__':
    t0 = time.time()
    pmid = np.load(f'{pcd}/cnt_pmid.npy', mmap_mode='r'); pat = np.load(f'{pcd}/cnt_patent.npy', mmap_mode='r')
    cid_ik = np.load(f'{pcd}/cid_ik.npy', mmap_mode='r')
    sel = (np.asarray(pmid) >= min_pmid) | (np.asarray(pat) >= min_pat)
    P = pd.read_parquet(f'{src}/pool.parquet')
    have = np.unique(pack(list(P.key)))
    ik = np.asarray(cid_ik)
    sel &= ik > 0
    sel &= ~np.isin(ik, have)
    print('selected CIDs', int(sel.sum()), f'{time.time()-t0:.0f}s', flush=True)
    p = subprocess.Popen(f'curl -s {FTP}/CID-SMILES.gz | gunzip -c', shell=True, stdout=subprocess.PIPE, bufsize=1 << 24)
    items = []
    for line in p.stdout:
        t = line.find(b'\t')
        if t <= 0:
            continue
        c = int(line[:t])
        if c < len(sel) and sel[c]:
            items.append((c, line[t + 1:].strip().decode()))
    print('smiles fetched', len(items), f'{time.time()-t0:.0f}s', flush=True)
    with Pool(procs) as pool:
        res = [r for r in pool.imap(props, items, chunksize=5000) if r is not None]
    print('parsed', len(res), f'{time.time()-t0:.0f}s', flush=True)
    N = pd.DataFrame([r[:6] for r in res], columns=['cid', 'smiles', 'key', 'formula', 'mass', 'ha'])
    nfp = np.stack([r[6] for r in res])
    keep = ~N.duplicated('key').values & ~N.key.isin(set(P.key)).values
    N, nfp = N[keep].reset_index(drop=True), nfp[keep]
    for c in P.columns:
        if c not in N.columns:
            N[c] = 0
    N['in_pc'] = 1; P['in_pc'] = 0
    allp = pd.concat([P, N[P.columns]], ignore_index=True)
    allfp = np.concatenate([np.load(f'{src}/pool_fp.npy'), nfp])
    o = np.argsort(allp.mass.values, kind='stable')
    allp.iloc[o].reset_index(drop=True).to_parquet(f'{dst}/pool.parquet')
    np.save(f'{dst}/pool_fp.npy', allfp[o])
    print('pool v3', len(allp), 'added', len(N), f'{time.time()-t0:.0f}s', flush=True)
