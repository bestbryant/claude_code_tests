"""MRR@25 per class of one or more rankers on a rows file (e.g. the test-like validation rows).

python scripts/eval_tl.py <rows.parquet> <ranker.txt> [<ranker.txt> ...] [--cols lib,an_sum,...]
Class-2 queries are kept only when the truth is reachable without train.parquet (COCONUT or PubChemLite), as in
ranker training; --pmid also accepts a PubChem PMID record (pool v3).
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb

rows = sys.argv[1]
rankers = [a for a in sys.argv[2:] if a.endswith('.txt')]
cols = next((a.split('=', 1)[1].split(',') for a in sys.argv if a.startswith('--cols=')), [])
D = pd.read_parquet(rows)
t = D[D.label == 1].set_index('qid')
reach = (t.in_coco == 1) | (t.in_pcl == 1)
if '--pmid' in sys.argv:
    reach |= t.pc_pmid > 0
D = D[(D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)]


def mrr(D, col):
    o = D.sort_values(['qid', col], ascending=[True, False])
    o['r'] = o.groupby('qid').cumcount() + 1
    r = o[o.label == 1].set_index('qid').r.reindex(D.qid.unique())
    rr = (1 / r).where(r <= 25, 0).fillna(0)
    cls = D.drop_duplicates('qid').set_index('qid').cls.reindex(rr.index)
    return f'all {rr.mean():.4f}  cls1 {rr[cls == 1].mean():.4f} (n={int((cls == 1).sum())})  ' \
           f'cls2 {rr[cls == 2].mean():.4f} (n={int((cls == 2).sum())})'


for rk in rankers:
    f = open(rk.replace('.txt', '_feats.txt')).read().split()
    D['p'] = lgb.Booster(model_file=rk).predict(D[f].astype(np.float32))
    print(f'{rk.split("/")[-1]:28s}', mrr(D, 'p'))
for c in cols:
    print(f'{c:28s}', mrr(D, c))
