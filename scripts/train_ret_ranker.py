"""LambdaRank over retrieval candidates: train on ranker-set queries, report MRR@25 on held-out validation queries.

Usage: python scripts/train_ret_ranker.py <train_rows.parquet> <val_rows.parquet> <model_out.txt> [extra feats csv]
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.retrieval import FEATS

tr_p, va_p, out = sys.argv[1:4]
extra = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else []
feats = FEATS + [f for f in extra]
TR = pd.read_parquet(tr_p).sort_values(['qid', 'pid']); VA = pd.read_parquet(va_p).sort_values(['qid', 'pid'])
val_sets = {}
m = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['inchikey14', 'ingest_lib'])
npx = set(m.inchikey14[m.ingest_lib == 'enveda-np-examples'])
P = pd.read_parquet('/home/user/work/pool/pool.parquet', columns=['key', 'in_coco']).set_index('key')
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
print(f'{"subset":8s} {"cls":3s} {"n":>5s}  ranker   lib-only  analog  frag' + ('  fp_cos  fp_bern' if 'fp_cos' in VA else ''))
for (sub, cls), d in VA.groupby(['subset', 'cls']):
    line = f'{sub:8s} {cls:3d} {d.qid.nunique():5d}  {mrr(d, "pred"):.4f}   {mrr(d, "lib"):.4f}   {mrr(d, "an_best"):.4f}  {mrr(d, "frag"):.4f}'
    if 'fp_cos' in d:
        line += f'  {mrr(d, "fp_cos"):.4f}  {mrr(d, "fp_bern"):.4f}'
    print(line)
for cls, d in VA.groupby('cls'):
    print('ALL cls', cls, 'ranker MRR', round(mrr(d, 'pred'), 4), 'truth in window', round(d.groupby('qid').label.max().mean(), 3))
imp = pd.Series(model.feature_importance('gain'), index=feats).sort_values(ascending=False)
print(imp.head(15).round(0).to_dict())
