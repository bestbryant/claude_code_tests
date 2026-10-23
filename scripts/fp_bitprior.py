"""Training-positive bit marginal of the binned FP model's 2,215 bits (Morgan 2048 + MACCS 167) for the PMI score
(casmi/fpscore.py FP_PMI): spectrum-weighted bit frequency of the training-split structures (split 0) of the FP data.

python scripts/fp_bitprior.py <fp_data_dir> <out.npy>     -> (2, N_BITS) float32: log m, log(1 - m)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.fpmodel import N_BITS

data, out = sys.argv[1:3]
T = np.load(f'{data}/structs.npz'); S = np.load(f'{data}/spectra.npz')
w = np.bincount(S['sidx'][S['split'] == 0], minlength=len(T['fp'])).astype(np.float64)
m = np.zeros(N_BITS)
for a in range(0, len(w), 20000):
    m += w[a:a + 20000] @ np.unpackbits(T['fp'][a:a + 20000], axis=1)[:, :N_BITS]
m = np.clip(m / w.sum(), 1e-4, 1 - 1e-4)
np.save(out, np.stack([np.log(m), np.log1p(-m)]).astype(np.float32))
print('bits', N_BITS, 'mean marginal', round(float(m.mean()), 4))
