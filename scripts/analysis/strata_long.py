"""Holdout MRR@25 of the long-schedule saner model (ahmedberatozer/casmi26-spectrum-model-long) vs the published model,
by stratum, from the two validation_predictions.csv files (same 1,000 holdout keys) + val1000_strata.parquet. Ours.
python scripts/analysis/strata_long.py <published_vp.csv> <long_vp.csv> <val1000_strata.parquet>"""
import sys
import numpy as np
import pandas as pd

p = pd.read_csv(sys.argv[1]).set_index('key')
a = pd.read_csv(sys.argv[2]).set_index('key')
st = pd.read_parquet(sys.argv[3])
d = p[['mrr25', 'top1']].join(a[['mrr25', 'top1']], rsuffix='_long', how='inner').join(st[['e180', 'coco']], how='left')


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return f'{x.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]'


for name, mask in [('all', d.index == d.index), ('e180', d.e180 == True), ('not e180', d.e180 == False),
                   ('coco', d.coco == True)]:
    x = d[mask]
    print(f'{name:9s} n={len(x):4d} pub {x.mrr25.mean():.4f} long {x.mrr25_long.mean():.4f} | long-pub '
          f'{boot(x.mrr25_long - x.mrr25)} | top1 pub {x.top1.mean():.3f} long {x.top1_long.mean():.3f}')
