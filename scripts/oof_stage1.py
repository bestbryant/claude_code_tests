"""Out-of-fold stage-1 scores for stage-2 training (stacking): 2-fold by query, same features and settings as the
stage-1 ranker, so stage 2 sees stage-1 scores of the quality it will meet on unseen molecules.

python scripts/oof_stage1.py <train_rows_fp.parquet> <stage1_feats.txt> <out.parquet> [--reach-pmid] [--drop-artifact]
--drop-artifact removes class-1 queries whose truth kept no library spectrum (all of them were the hidden query
spectra): they behave like class 2 with in_train = 1, which never happens at test time.
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb

rows, feats_p, out = sys.argv[1:4]
D = pd.read_parquet(rows).sort_values(['qid', 'pid']).reset_index(drop=True)
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
t = D[D.label == 1].set_index('qid')
reach = (t.in_coco == 1) | (t.get('in_pcl', 0) == 1)
if '--reach-pmid' in sys.argv:
    reach |= t.get('pc_pmid', 0) > 0
keep = (D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)
if '--drop-artifact' in sys.argv:
    keep &= ~D.qid.isin(t.index[(t.cls == 1) & (t.lib_n == 0)])
D = D[keep].reset_index(drop=True)
feats = open(feats_p).read().split()
params = dict(objective='lambdarank', metric='map', eval_at=[25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=25, verbose=-1, num_threads=4)
q = D.qid.unique(); fold = dict(zip(q, np.random.default_rng(0).integers(0, 2, len(q))))
D['fold'] = D.qid.map(fold)
D['s1_oof'] = np.nan
for k in (0, 1):
    tr = D[D.fold != k]
    c1 = tr[tr.cls == 1]
    tr = pd.concat([tr, c1.assign(qid=c1.qid + 10_000_000)]).sort_values(['qid', 'pid'])
    m = lgb.train(params, lgb.Dataset(tr[feats].astype(float).values, tr.label.values,
                                      group=tr.groupby('qid', sort=False).size().values), num_boost_round=500)
    te = D.fold == k
    D.loc[te, 's1_oof'] = m.predict(D.loc[te, feats].astype(float).values)
    print('fold', k, 'done', flush=True)


def mrr(df, col):
    r = df.sort_values(['qid', col], ascending=[True, False]).copy()
    r['rk'] = r.groupby('qid').cumcount() + 1
    hit = r[(r.label == 1) & (r.rk <= 25)].groupby('qid').rk.min()
    return round(float((1.0 / hit).reindex(df.qid.unique()).fillna(0).mean()), 4)


print('queries', D.qid.nunique(), 'OOF stage-1 MRR', mrr(D, 's1_oof'))
D.drop(columns=['fold']).to_parquet(out)
