"""List-confidence model: p(the main list does not contain the answer), for placing PubChem-channel proposals.

python scripts/train_pother.py <stage1 oof preds.parquet> <pred column> [out.txt]
Simulation: every query's stage-1 list as is (answer present, y = 0) and, for class-2 queries, the same list with the
truth removed (y = 1: an answer outside the pool, like a hidden "other" molecule). Features describe the list only
(score margins, the leading candidates' evidence and sources). 5-fold CV by query (both copies of a query in the
same fold), class-2 queries weighted by the documentation-shift weights. Prints AUC and the score distributions.
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = '/home/user/work'
preds, col = sys.argv[1:3]
out = sys.argv[3] if len(sys.argv) > 3 else None
COLS = ['qid', 'pid', 'label', 'cls', 'truth', 'lib', 'an_best', 'an_tan_best', 'frag', 'fp_cos', 'in_train', 'in_coco',
        'in_pcl', 'in_bio', 'pc_sid', 'pc_pmid', 'n_cand', 'n_form']
D = pd.concat([pd.read_parquet(f'{W}/ret10_train_fp.parquet', columns=COLS),
               pd.read_parquet(f'{W}/ret10_val_fp.parquet', columns=COLS).assign(qid=lambda d: d.qid + 100_000_000)])
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
P = pd.read_parquet(preds, columns=['qid', 'pid', col])
D = D.merge(P, on=['qid', 'pid'], how='inner').rename(columns={col: 'p'})
DW = pd.read_parquet(f'{W}/pool4/doc_w.parquet').set_index('key').w
from casmi.pother import FEATS, list_feats

rows = []
for q, g in D.groupby('qid', sort=False):
    c = int(g.cls.iloc[0]); tk = g[g.label == 1].truth
    w = float(np.clip(DW.get(tk.iloc[0], 1.0), 0.02, 20)) if (c == 2 and len(tk)) else 1.0
    rows.append(dict(qid=q, y=0, cls=c, w=w, **list_feats(g)))
    if c == 2 and (g.label == 1).any() and (g.label == 0).sum() >= 2:
        rows.append(dict(qid=q, y=1, cls=c, w=w, **list_feats(g[g.label == 0])))
F = pd.DataFrame(rows)
print('lists', len(F), 'answer removed', int(F.y.sum()), flush=True)
fold = pd.Series(np.random.default_rng(0).integers(0, 5, F.qid.nunique()), index=F.qid.unique())
F['fold'] = F.qid.map(fold); F['s'] = np.nan
prm = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4, seed=0)


def roc_auc_score(y, s, sample_weight=None):
    """Weighted Mann-Whitney AUC (ties count half)."""
    y = np.asarray(y); s = np.asarray(s, float); w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wn = np.where(y == 0, w, 0.0); wp = np.where(y == 1, w, 0.0)
    _, first = np.unique(s, return_index=True); last = np.r_[first[1:], len(s)]
    cum_neg = np.r_[0, np.cumsum(wn)]
    auc = 0.0
    for a, b in zip(first, last):
        auc += wp[a:b].sum() * (cum_neg[a] + 0.5 * wn[a:b].sum())
    return auc / (wp.sum() * wn.sum())


for k in range(5):
    tr, te = F[F.fold != k], F[F.fold == k]
    m = lgb.train(prm, lgb.Dataset(tr[FEATS].astype(float), tr.y, weight=tr.w), 300)
    F.loc[te.index, 's'] = m.predict(te[FEATS].astype(float))
c2 = F[F.cls == 2]
print('AUC all', round(roc_auc_score(F.y, F.s), 4), '| class-2 lists only', round(roc_auc_score(c2.y, c2.s), 4),
      '| doc-weighted class 2', round(roc_auc_score(c2.y, c2.s, sample_weight=c2.w), 4))
for y in (0, 1):
    print('y', y, 'score quantiles', np.round(np.quantile(F[F.y == y].s, [.1, .25, .5, .75, .9]), 3))
F[['qid', 'y', 'cls', 'w', 's'] + FEATS].to_parquet(f'{W}/pother_oof.parquet')
if out:
    m = lgb.train(prm, lgb.Dataset(F[FEATS].astype(float), F.y, weight=F.w), 300)
    m.save_model(out); open(out.replace('.txt', '_feats.txt'), 'w').write('\n'.join(FEATS))
    imp = pd.Series(m.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
    print('gain', (imp / imp.sum()).round(3).head(10).to_dict())
