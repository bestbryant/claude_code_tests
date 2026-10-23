"""Per-negative-row flag "is a training structure" for PeakFormer training (PF_NEG_TRAIN in scripts/train_pf.py).

python scripts/pf_negtrain.py <fp_data_dir> <pool_dir> <out.npy>
The negative rows of fp_data_dir/negfp.npy are the sorted unique pool rows of the window search in
scripts/prep_fp_data.py (repeated here as in scripts/prep_pf_data.py). COCONUT-not-train structures are never positives;
as window negatives they teach the model that such structures are never the answer (a population prior that hurts
hidden COCONUT answers: matched within-formula win rate vs COCONUT decoys exceeds that vs training decoys by +0.137).
A row is kept only if its structure is a training POSITIVE (has split-0 spectra in fp_data_dir): held-out validation
(split 1) and ranker (split 2) structures stay unseen, like the hidden answers (review wf_362c305b-918, Oct 23).
"""
import sys
import numpy as np, pandas as pd

src, pool_dir, out = sys.argv[1:4]
T = np.load(f'{src}/structs.npz', allow_pickle=True)
P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['mass', 'formula', 'in_train'])
pool_idx = T['pool_idx']; mass, form = P.mass.values, P.formula.to_numpy(dtype=object)
used = set()
for i, pi in enumerate(pool_idx):
    M = mass[pi]; tol = max(M * 1e-5, 0.002)
    lo, hi = np.searchsorted(mass, M - tol), np.searchsorted(mass, M + tol)
    c = np.arange(lo, hi); c = c[c != pi]
    if len(c) == 0:
        continue
    same = c[form[c] == form[pi]]; other = c[form[c] != form[pi]]
    c = np.concatenate([same[np.random.default_rng(i).permutation(len(same))],
                        other[np.random.default_rng(i + 1).permutation(len(other))]])[:64]
    used.update(c.tolist())
used = np.array(sorted(used))
assert len(used) == np.load(f'{src}/negfp.npy', mmap_mode='r').shape[0]
S = np.load(f'{src}/spectra.npz')
pos = pool_idx[np.bincount(S['sidx'][S['split'] == 0], minlength=len(pool_idx)) > 0]
tr = np.isin(used, pos)
np.save(out, tr)
print('negative rows', len(used), 'training structures', round(float(P.in_train.values[used].mean()), 3),
      'kept (split-0 positives)', round(float(tr.mean()), 3))
