"""Channel re-ranker: LambdaRank over the channel's best 100 proposals (scripts/channel_rows.py rows), evaluated
against the PeakFormer-only order (score = z(f.z) + z(cos)) on validation queries.

python scripts/train_channel_rr.py <train_rows.parquet> <val_rows.parquet> [model_out.txt]
MRR@25 / top-1 per query; a truth outside the best 100 is a miss either way. Reported for all Enamine-truth
validation queries and for the unreachable ones (noreach: not in COCONUT / PubChemLite / ChEBI-LMSD).
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.pcchannel import RR_FEATS

tr_p, va_p = sys.argv[1:3]
out = sys.argv[3] if len(sys.argv) > 3 else None
TR, VA = pd.read_parquet(tr_p), pd.read_parquet(va_p)
TR = TR[TR.groupby('qid').label.transform('max') == 1].sort_values(['qid', 'pc_score'], ascending=[True, False])
prm = dict(objective='lambdarank', metric='map', eval_at=[25], learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=25, verbose=-1,
           num_threads=4, seed=0)
FE = [f for f in os.environ.get('RR_FEATS', ','.join(RR_FEATS)).split(',') if f]


def evaluate(V, col):
    o = V.sort_values(['qid', col], ascending=[True, False]).copy()
    o['r'] = o.groupby('qid').cumcount() + 1
    r = o[o.label == 1].groupby('qid').r.min().reindex(V.qid.unique())
    rr = np.where(r <= 25, 1.0 / r, 0.0)
    return pd.DataFrame(dict(rr=rr, top1=(r == 1).values), index=r.index)


def report(V, name):
    q = V.drop_duplicates('qid').set_index('qid')
    for sub, m in (('all', slice(None)), ('noreach', q.noreach.values)):
        a, b = evaluate(V, 'pc_score').loc[q.index[m]], evaluate(V, 'rr').loc[q.index[m]]
        print(f'{name:10s} {sub:8s} n {len(a):4d}  PF-only MRR {a.rr.mean():.4f} top1 {a.top1.mean():.3f}  ->  '
              f're-ranked MRR {b.rr.mean():.4f} top1 {b.top1.mean():.3f}  (paired d {np.mean(b.rr - a.rr):+.4f} '
              f'se {np.std(b.rr - a.rr) / np.sqrt(len(a)):.4f})')


print('train queries', TR.qid.nunique(), 'rows', len(TR), '| features', FE)
for rounds in (100, 300):
    m = lgb.train(prm, lgb.Dataset(TR[FE].astype(float), TR.label, group=TR.groupby('qid', sort=False).size().values), rounds)
    VA['rr'] = m.predict(VA[FE].astype(float))
    report(VA, f'rounds {rounds}')
imp = pd.Series(m.feature_importance('gain'), index=FE).sort_values(ascending=False)
print('gain', (imp / imp.sum()).round(3).head(10).to_dict())
if out:
    m.save_model(out); open(out.replace('.txt', '_feats.txt'), 'w').write('\n'.join(FE))
