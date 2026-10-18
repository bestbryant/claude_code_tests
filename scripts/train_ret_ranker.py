"""LambdaRank over retrieval candidates: train on ranker-set queries, report MRR@25 on held-out validation queries.

Class-2 queries are kept only when the truth is reachable through a non-training source (COCONUT or PubChemLite),
because a class-2 answer is by definition not a training structure.
Usage: python scripts/train_ret_ranker.py <train_rows> <val_rows> <model_out.txt> [--no-prior] [--no-fp] [pool_dir]
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.retrieval import FEATS, PRIOR_COLS, PCL_COLS, PC_COLS
from casmi.own import FP_FEATS

tr_p, va_p, out = sys.argv[1:4]
flags = set(a for a in sys.argv[4:] if a.startswith('--'))
pool_dir = next((a for a in sys.argv[4:] if not a.startswith('--')), '/home/user/work/pool2')
REACH_PMID = '--reach-pmid' in flags


def load(p):
    D = pd.read_parquet(p).sort_values(['qid', 'pid']).reset_index(drop=True)
    D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0           # also for rows dumped before the in-code fix
    t = D[D.label == 1].set_index('qid')
    # reachable for a class-2 test molecule: in COCONUT, in PubChemLite, or (pool v3) PubChem-documented (PMID record)
    reach = ((t.in_coco == 1) | (t.get('in_pcl', 0) == 1) | (t.get('pc_pmid', 0) > 0 if REACH_PMID else False))
    keep = (D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)
    return D[keep].reset_index(drop=True)


TR, VA = load(tr_p), load(va_p)
# class-1 queries twice: the simulation is 30% class 1 while the test mix is closer to 45:55 (better on validation)
c1 = TR[TR.cls == 1]
TR = pd.concat([TR, c1.assign(qid=c1.qid + 10_000_000)], ignore_index=True).sort_values(['qid', 'pid']).reset_index(drop=True)
feats = [f for f in FEATS + PCL_COLS + PC_COLS + FP_FEATS if f in TR.columns and f in VA.columns]
if '--no-prior' in flags:
    feats = [f for f in feats if f not in PRIOR_COLS + PCL_COLS[1:] + PC_COLS + ['in_coco']]
if '--no-membership' in flags:
    feats = [f for f in feats if f not in ('in_train', 'lib_n')]
if '--no-fp' in flags:
    feats = [f for f in feats if f not in FP_FEATS]
m = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['inchikey14', 'ingest_lib'])
npx = set(m.inchikey14[m.ingest_lib == 'enveda-np-examples'])
P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['key', 'in_coco']).set_index('key')
VA['subset'] = np.where(VA.truth.isin(npx), 'npx', np.where(VA.truth.map(P.in_coco) == 1, 'np', 'other'))


def mrr(df, col):
    r = df.sort_values(['qid', col], ascending=[True, False]).copy()
    r['rk'] = r.groupby('qid').cumcount() + 1
    hit = r[(r.label == 1) & (r.rk <= 25)].groupby('qid').rk.min()
    return float((1.0 / hit).reindex(df.qid.unique()).fillna(0).mean())


params = dict(objective='lambdarank', metric='map', eval_at=[25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=25, verbose=-1, num_threads=2)
g = TR.groupby('qid', sort=False).size().values
model = lgb.train(params, lgb.Dataset(TR[feats].astype(float).values, TR.label.values, group=g), num_boost_round=500)
VA['pred'] = model.predict(VA[feats].astype(float).values)
model.save_model(out)
open(out.replace('.txt', '_feats.txt'), 'w').write('\n'.join(feats))
print('features', len(feats), 'train queries', TR.qid.nunique(), 'val queries', VA.qid.nunique())
print(f'{"subset":8s} cls  n     ranker  lib    analog  frag' + ('   fp_cos' if 'fp_cos' in VA else ''))
for (sub, cls), d in VA.groupby(['subset', 'cls']):
    line = f'{sub:8s} {cls}  {d.qid.nunique():4d}  {mrr(d, "pred"):.4f}  {mrr(d, "lib"):.3f}  {mrr(d, "an_best"):.3f}  {mrr(d, "frag"):.3f}'
    if 'fp_cos' in d:
        line += f'   {mrr(d, "fp_cos"):.3f}'
    print(line)
for cls, d in VA.groupby('cls'):
    print('ALL cls', cls, 'ranker MRR', round(mrr(d, 'pred'), 4), 'truth in window', round(d.groupby('qid').label.max().mean(), 3))
imp = pd.Series(model.feature_importance('gain'), index=feats).sort_values(ascending=False)
print(imp.head(15).round(0).to_dict())
