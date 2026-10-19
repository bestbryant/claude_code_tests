"""Stage-1 rankers on validation rows: MRR@25 overall, by class, and class 2 under documentation-shift weights.

python scripts/eval_s1.py <val_rows_fp.parquet> <doc_w.parquet>[,<doc_w2.parquet>] <ranker.txt> [<ranker.txt> ...]
Same query filter as training (reachable class 2, no artifact class 1). The weighted class-2 MRR approximates
class-2 queries whose truth is distributed like reachable structures WITHOUT public spectra (casmi/docw.py).
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb

rows, wps, models = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
D = pd.read_parquet(rows).sort_values(['qid', 'pid']).reset_index(drop=True)
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
t = D[D.label == 1].drop_duplicates('qid').set_index('qid')
reach = (t.in_coco == 1) | (t.in_pcl == 1) | (t.in_bio == 1)
D = D[((D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)) & ~D.qid.isin(t.index[(t.cls == 1) & (t.lib_n == 0)])]
t = D[D.label == 1].drop_duplicates('qid').set_index('qid')
c2 = t.index[t.cls == 2]
ws = [t.truth.map(pd.read_parquet(p).set_index('key').w).fillna(1.0).clip(0.02, 20)[c2] for p in wps]


def rr(col):
    o = D.sort_values(['qid', col], ascending=[True, False])
    o['r'] = o.groupby('qid').cumcount() + 1
    r = o[(o.label == 1) & (o.r <= 25)].groupby('qid').r.min().reindex(t.index)
    return (1 / r).fillna(0)


for mp in models:
    feats = open(mp.replace('.txt', '_feats.txt')).read().split()
    D['p'] = lgb.Booster(model_file=mp).predict(D[feats].astype(float).values)
    r = rr('p')
    w = ' '.join(f'{np.average(r[c2], weights=x):.4f}' for x in ws)
    print(f'{mp.split("/")[-1]:28s} all {r.mean():.4f} cls1 {r[t.cls == 1].mean():.4f} cls2 {r[c2].mean():.4f} cls2-weighted {w}', flush=True)
