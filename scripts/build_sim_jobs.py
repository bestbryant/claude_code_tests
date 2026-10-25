"""Job files for GLACIER spectrum SIMULATION of pool structures (PF-syn: synthetic training spectra for the PeakFormer).

python scripts/build_sim_jobs.py <out_dir> <tag> [n_train] [n_coco] [n_files] [seed]
Structures (pool v4 SMILES): every candidate of the 620-query membership-neutral subset (/home/user/work/fpb_local/
sub.parquet: truths and their same-mass windows, as at test time where the whole pool is simulated) + n_train random
training structures + n_coco random COCONUT-not-train structures (both populations, so that "has a synthetic
spectrum" carries no source information). Each job file is a casmi.fwdrun input: items of up to 500 candidates sharing
three placeholder [M+H]+ query spectra at CE 20 / 40 / 60 eV (the Enveda timsTOF energies), so that fwdrun predicts
every candidate at the three energies (instrument token QTOF); the placeholder scores are ignored, the predictions are
kept with --dump-preds (scripts/make_sim_workers.py). Also writes <tag>_structs.parquet (pid, smiles, set).
"""
import json, os, sys
import numpy as np, pandas as pd

out, tag = sys.argv[1], sys.argv[2]
n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
n_coco = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
n_files = int(sys.argv[5]) if len(sys.argv) > 5 else 16
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rng = np.random.default_rng(seed)
P = pd.read_parquet('/home/user/work/pool4/pool.parquet', columns=['smiles', 'in_train', 'in_coco'])
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
