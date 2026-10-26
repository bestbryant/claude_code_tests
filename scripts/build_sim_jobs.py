"""Job files for GLACIER spectrum SIMULATION of pool structures (PF-syn: synthetic training spectra for the PeakFormer).

python scripts/build_sim_jobs.py <out_dir> <tag> [n_train] [n_coco] [n_files] [seed]
Structures (pool v4 SMILES): every candidate of the 620-query membership-neutral subset (/home/user/work/fpb_local/
sub.parquet: truths and their same-mass windows, as at test time where the whole pool is simulated) + n_train random
training structures + n_coco random COCONUT-not-train structures (both populations, so that "has a synthetic
spectrum" carries no source information). Each job file is a casmi.fwdrun input: items of up to 500 candidates sharing
three placeholder [M+H]+ query spectra at CE 20 / 40 / 60 eV (the Enveda timsTOF energies), so that fwdrun predicts
every candidate at the three energies (instrument token QTOF); the placeholder scores are ignored, the predictions are
kept with --dump-preds (scripts/make_sim_workers.py). Also writes <tag>_structs.parquet (pid, smiles, set).
Env POOL (pool.parquet; default pool v4, which rebuilds the pilot set byte-identically).
ALL=1 (production, PF-syn full pool): every pool row that the GLACIER runner can predict -- RDKit-parsable, elements in
ms-pred's VALID_ELEMENTS, <= 100 heavy atoms after RemoveHs (exactly casmi.fwdrun canon()), plus net formal charge 0
(SKIP_CHARGED, default 1; the runner predicts [M+H]+) and mass <= MASS_MAX (default 1300) -- with set = train /
coco (COCONUT-not-train) / other; n_train / n_coco are ignored. The structures are shuffled (seed) and assigned to
the n_files files greedily by cost (3 x heavy atoms), so every file mixes the sets and takes about the same time.
Writes <tag>_excluded.parquet (pid, reason) and <tag>_MANIFEST.json (counts per set, expected predictions per file).
"""
import json, os, sys
import numpy as np, pandas as pd

out, tag = sys.argv[1], sys.argv[2]
n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
n_coco = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
n_files = int(sys.argv[5]) if len(sys.argv) > 5 else 16
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rng = np.random.default_rng(seed)
POOL = os.environ.get('POOL', '/home/user/work/pool4/pool.parquet')
VALID = {'C', 'N', 'P', 'O', 'S', 'Si', 'I', 'H', 'Cl', 'F', 'Br', 'B', 'Se', 'Fe', 'Co', 'As', 'Na', 'K'}   # ms-pred chem_utils


def check(smi):
    """'' if the GLACIER runner predicts this structure as a neutral [M+H]+ candidate, else the reason."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 'parse'
    if any(a.GetSymbol() not in VALID for a in m.GetAtoms()):
        return 'element'
    if os.environ.get('SKIP_CHARGED', '1') == '1' and Chem.GetFormalCharge(m) != 0:
        return 'charged'
    return 'atoms' if Chem.RemoveHs(m).GetNumAtoms() > 100 else ''


if os.environ.get('ALL') == '1':
    import heapq, hashlib
    from multiprocessing import Pool
    P = pd.read_parquet(POOL, columns=['smiles', 'mass', 'in_train', 'in_coco'])
    with Pool(int(os.environ.get('PROCS', '4'))) as pool:
        why = np.array(pool.map(check, P.smiles.tolist(), chunksize=5000), dtype=object)
    why[(why == '') & (P.mass.values > float(os.environ.get('MASS_MAX', '1300')))] = 'mass'
    ok = np.where(why == '')[0]
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({'pid': np.where(why != '')[0], 'reason': why[why != '']}).to_parquet(f'{out}/{tag}_excluded.parquet')
    sset = np.where(P.in_train.values[ok] == 1, 'train', np.where(P.in_coco.values[ok] == 1, 'coco', 'other'))
    S = pd.DataFrame({'pid': ok, 'set': sset, 'smiles': P.smiles.values[ok]})
    S = S.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    from rdkit import Chem
    S['ha'] = [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in S.smiles]
    heap = [(0, f) for f in range(n_files)]; fid = np.zeros(len(S), np.int32)
    for i, c in enumerate(3 * S.ha.values):          # greedy: next structure to the cheapest file so far
        load, f = heapq.heappop(heap); fid[i] = f; heapq.heappush(heap, (load + int(c), f))
    S['file'] = fid
    S.to_parquet(f'{out}/{tag}_structs.parquet')
    spectra = [dict(mz=[100.0, 200.0], it=[1.0, 1.0], prec=2000.0, adduct='[M+H]+', mode='positive', instrument='timsTOF',
                    ce_ev=[float(ce)]) for ce in (20, 40, 60)]
    man = dict(pool=POOL, pool_md5=hashlib.md5(open(POOL, 'rb').read()).hexdigest(), seed=seed, n_files=n_files,
               excluded={k: int(v) for k, v in pd.Series(why[why != '']).value_counts().items()},
               sets={k: int(v) for k, v in S.set.value_counts().items()}, files={})
    for f in range(n_files):
        smi = S.smiles.values[fid == f].tolist()
        items = [dict(mid=f'{tag}_{f}_{i}', spectra=spectra, cands=smi[i:i + 500]) for i in range(0, len(smi), 500)]
        txt = json.dumps(items)
        open(f'{out}/{tag}.jobs_{f}.json', 'w').write(txt)
        man['files'][f'{tag}.jobs_{f}.json'] = dict(structures=len(smi), expected_predictions=3 * len(smi),
                                                     heavy_atoms=int(S.ha.values[fid == f].sum()), md5=hashlib.md5(txt.encode()).hexdigest())
    json.dump(man, open(f'{out}/{tag}_MANIFEST.json', 'w'), indent=1)
    print('structures', len(S), man['sets'], 'excluded', man['excluded'], 'files', n_files, 'predictions', 3 * len(S))
    sys.exit(0)
P = pd.read_parquet(POOL, columns=['smiles', 'in_train', 'in_coco'])
sub = np.unique(pd.read_parquet('/home/user/work/fpb_local/sub.parquet', columns=['pid']).pid.values)
rest = np.setdiff1d(np.arange(len(P)), sub)
tr = rest[P.in_train.values[rest] == 1]
co = rest[(P.in_coco.values[rest] == 1) & (P.in_train.values[rest] == 0)]
pick_tr = rng.choice(tr, min(n_train, len(tr)), replace=False)
pick_co = rng.choice(co, min(n_coco, len(co)), replace=False)
S = pd.DataFrame({'pid': np.r_[sub, pick_tr, pick_co],
                  'set': ['neutral'] * len(sub) + ['train'] * len(pick_tr) + ['coco'] * len(pick_co)})
S['smiles'] = P.smiles.values[S.pid.values]
S = S.sample(frac=1.0, random_state=seed).reset_index(drop=True)       # every file gets the same population mix
os.makedirs(out, exist_ok=True)
S.to_parquet(f'{out}/{tag}_structs.parquet')
spectra = [dict(mz=[100.0, 200.0], it=[1.0, 1.0], prec=2000.0, adduct='[M+H]+', mode='positive', instrument='timsTOF',
                ce_ev=[float(ce)]) for ce in (20, 40, 60)]
for f, part in enumerate(np.array_split(np.arange(len(S)), n_files)):
    smi = S.smiles.values[part].tolist()
    items = [dict(mid=f'{tag}_{f}_{i}', spectra=spectra, cands=smi[i:i + 500]) for i in range(0, len(smi), 500)]
    json.dump(items, open(f'{out}/{tag}.jobs_{f}.json', 'w'))
print('structures', len(S), S.set.value_counts().to_dict(), 'files', n_files, 'predictions', 3 * len(S))
