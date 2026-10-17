"""Train a LightGBM LambdaRank model over NovelEngine candidates; report MRR@25 on held-out queries.

Evaluation is on held-out PubChem-absent queries (class-3 proxy), grouped by truth structure.
Usage: python scripts/train_novel_ranker.py <rows1.parquet,rows2.parquet,...> <model_out.txt>
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.novel import FEATS

paths, out = sys.argv[1].split(','), sys.argv[2]
parts = []
for i, p in enumerate(paths):
    d = pd.read_parquet(p)
    d = d[~d['empty'].astype(bool)] if 'empty' in d.columns else d
    d['set'] = p.split('rows_')[-1].split('.')[0]
    d['qid'] = d['qid'].astype(int) + 1_000_000 * i
    parts.append(d)
D = pd.concat(parts, ignore_index=True)
for f in FEATS:
    if f not in D.columns:
        D[f] = np.nan
D = D.sort_values(['qid', 'h'], ascending=[True, False])
D['hrank'] = D.groupby('qid').cumcount()
D = D[D.hrank < 400]                      # cap list length per query
X = D[FEATS].astype(float).values
y = D.label.values


def mrr(df, col):
    r = df.sort_values(['qid', col], ascending=[True, False])
    r['rk'] = r.groupby('qid').cumcount() + 1
    hit = r[(r.label == 1) & (r.rk <= 25)].groupby('qid').rk.min()
    return float((1.0 / hit).reindex(df.qid.unique()).fillna(0).mean())


# fold by truth structure; evaluate on the absent set
truth = D.truth.values
ut = np.unique(truth)
rng = np.random.default_rng(0)
fold_of = dict(zip(ut, rng.integers(0, 5, len(ut))))
fold = np.array([fold_of[t] for t in truth])
params = dict(objective='lambdarank', metric='map', eval_at=[25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=30,
              verbose=-1, num_threads=1)
D['pred'] = np.nan
for f in range(5):
    tr, te = fold != f, fold == f
    dtr = D[tr]
    gtr = dtr.groupby('qid', sort=False).size().values
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], group=gtr), num_boost_round=400)
    D.loc[te, 'pred'] = m.predict(X[te])
for s in sorted(D.set.unique()):
    sub = D[D.set == s]
    print(f'{s:8s} queries {sub.qid.nunique():5d}  heuristic MRR {mrr(sub, "h"):.4f}  LTR MRR {mrr(sub, "pred"):.4f}')
gall = D.groupby('qid', sort=False).size().values
m = lgb.train(params, lgb.Dataset(X, y, group=gall), num_boost_round=400)
m.save_model(out)
imp = pd.Series(m.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
print(imp.head(15).round(0).to_dict())
