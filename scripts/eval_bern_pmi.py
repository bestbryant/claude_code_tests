"""PMI correction of the FP model's Bernoulli score: bern - lam * log p(bits | training-positive marginal).

python scripts/eval_bern_pmi.py <scored subset rows> [<more>...]  (scripts/fpb_local.sh outputs, fp_bern computed)
The bit head is trained by BCE on positives only, so its likelihood favours candidates that look like a typical training
structure (a population prior that window negatives cannot remove). The marginal is the spectrum-weighted bit frequency
of the training-split structures (casmi26-fp-data). Writes fp_bern_pmi05 / fp_bern_pmi1 columns and runs scripts/eval_pf_neutral.py.
"""
import os, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.fpmodel import N_BITS
from casmi.fpscore import FPScorer

DATA = '/home/user/ds/casmi26-fp-data'
T = np.load(f'{DATA}/structs.npz'); S = np.load(f'{DATA}/spectra.npz')
w = np.bincount(S['sidx'][S['split'] == 0], minlength=len(T['fp'])).astype(np.float64)
m = np.zeros(N_BITS)
for a in range(0, len(w), 20000):
    m += w[a:a + 20000] @ np.unpackbits(T['fp'][a:a + 20000], axis=1)[:, :N_BITS]
m = np.clip(m / w.sum(), 1e-4, 1 - 1e-4)
lm, l1m = np.log(m), np.log1p(-m)
P = pd.read_parquet('/home/user/work/pool4/pool.parquet', columns=['smiles'])
sc = FPScorer.__new__(FPScorer); sc._mac = {}; sc.pool_fp = np.load('/home/user/work/pool4/pool_fp.npy', mmap_mode='r')
prior = {}
for path in sys.argv[1:]:
    D = pd.read_parquet(path)
    new = np.setdiff1d(D.pid.unique(), list(prior))
    if len(new):
        bits = np.unpackbits(sc.mol_bits(new, P.smiles.values[new]), axis=1)[:, :N_BITS].astype(np.float64)
        prior.update(zip(new, bits @ lm + (1 - bits) @ l1m))
    pr = D.pid.map(prior).values
    for lam, nm in ((0.5, '05'), (1.0, '1')):
        D[f'fp_bern_pmi{nm}'] = D.fp_bern.values - lam * pr
    out = path.replace('.parquet', '_pmi.parquet'); D.to_parquet(out)
    print('==', os.path.basename(path), flush=True)
    subprocess.run([sys.executable, 'scripts/eval_pf_neutral.py', out, 'fp_bern,fp_bern_pmi05,fp_bern_pmi1'])
