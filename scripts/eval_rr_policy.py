"""Slot policy with the channel re-ranker's confidence as an extra trigger (scripts/eval_enam_policy.py semantics).

python scripts/eval_rr_policy.py <sim_enam_rr.parquet> <cr_train rows> <cr_val rows> <cr_present rows>
A re-ranker trained on the train rows only scores the validation windows: "other" queries (cr_val, unreachable truths,
answer inside the window) and "present" queries (cr_present, answer in the pool, window without it). Triggers:
B slots (2, 4, .., 10) when rel > agg or rr_top > t_agg; rank-1 promotion when rel > promo or rr_top > t_promo
(rr_top = best re-ranker score of the window, rr_gap = best minus second).
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.pcchannel import RR_FEATS

sim_p, tr_p, va_p, pr_p = sys.argv[1:5]
S = pd.read_parquet(sim_p)
TR = pd.read_parquet(tr_p)
TR = TR[TR.groupby('qid').label.transform('max') == 1].sort_values(['qid', 'pc_score'], ascending=[True, False])
prm = dict(objective='lambdarank', metric='map', eval_at=[25], learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=25, verbose=-1,
           num_threads=4, seed=0)
m = lgb.train(prm, lgb.Dataset(TR[RR_FEATS].astype(float), TR.label, group=TR.groupby('qid', sort=False).size().values), 300)


def conf(path):
    V = pd.read_parquet(path)
    V['rr'] = m.predict(V[RR_FEATS].astype(float))
    V = V.sort_values(['qid', 'rr'], ascending=[True, False])
    V['k'] = V.groupby('qid').cumcount()
    top = V[V.k == 0].set_index('qid').rr
    sec = V[V.k == 1].set_index('qid').rr.reindex(top.index)
    return pd.DataFrame(dict(rr_top=top, rr_gap=top - sec.fillna(top - 5)))


co, cp = conf(va_p), conf(pr_p)
S['rr_top'] = np.nan; S['rr_gap'] = np.nan
o, p = S.kind == 'other', S.kind == 'present'
S.loc[o, 'rr_top'] = S.loc[o, 'qid'].map(co.rr_top).values; S.loc[o, 'rr_gap'] = S.loc[o, 'qid'].map(co.rr_gap).values
S.loc[p, 'rr_top'] = S.loc[p, 'qid'].map(cp.rr_top).values; S.loc[p, 'rr_gap'] = S.loc[p, 'qid'].map(cp.rr_gap).values
print('rr_top quantiles present', np.round(np.nanquantile(S[p].rr_top, [.5, .9, .95, .99]), 2),
      '| other', np.round(np.nanquantile(S[o].rr_top, [.1, .25, .5, .75]), 2))
print('rr_gap quantiles present', np.round(np.nanquantile(S[p].rr_gap, [.5, .9, .95, .99]), 2),
      '| other (truth #1)', np.round(np.nanquantile(S[o & (S.ch_rank == 1)].rr_gap, [.1, .25, .5]), 2))
pr, ot = S[p], S[o]
A, B = (4, 8, 12, 16, 20), (2, 4, 6, 8, 10)


def slots(d, agg, promo, t_agg, t_promo, col):
    res = []
    for lib, rel, c in zip(d.lib_max.values, d.rel.values, d[col].values):
        if lib >= 0.9 or not np.isfinite(rel):
            res.append(()); continue
        c = c if np.isfinite(c) else -1e9
        sl = B if ((agg is not None and rel > agg) or (t_agg is not None and c > t_agg)) else A
        if (promo is not None and rel > promo) or (t_promo is not None and c > t_promo):
            sl = (1,) + tuple(x for x in sl if x > 1)
        res.append(sl)
    return res


def mrr_present(sl):
    out = np.zeros(len(pr))
    for i, (r, s) in enumerate(zip(pr.main_rank.values, sl)):
        if r <= 0:
            continue
        for x in sorted(s):
            if x <= r:
                r += 1
        out[i] = 1.0 / r if r <= 25 else 0.0
    c1 = pr.cls.values == 1
    return 0.45 * out[c1].mean() + 0.55 * np.average(out[~c1], weights=pr.w.values[~c1])


def mrr_other(sl):
    return float(np.mean([1.0 / sorted(s)[c - 1] if 1 <= c <= len(s) else 0.0 for c, s in zip(ot.ch_rank.values, sl)]))


base = mrr_present([()] * len(pr))
qp = {c: np.nanquantile(pr[c], [.9, .95, .99]) for c in ('rr_top', 'rr_gap')}
cands = [('rel only (15, 60)', 15, 60, None, None, 'rr_top')]
for col in ('rr_top', 'rr_gap'):
    for qa, ta in zip(('p90', 'p95', 'p99'), qp[col]):
        for qb, tb in zip(('p95', 'p99'), qp[col][1:]):
            if tb < ta:
                continue
            cands.append((f'{col} agg>{qa} promo>{qb}', None, None, ta, tb, col))
            cands.append((f'rel + {col} agg>{qa} promo>{qb}', 15, 60, ta, tb, col))
print(f'{"policy":36s} d_present  other   ' + '  '.join(f'pi={x}' for x in (0.05, 0.1, 0.2, 0.3)))
for name, agg, promo, ta, tb, col in cands:
    a, b = mrr_present(slots(pr, agg, promo, ta, tb, col)), mrr_other(slots(ot, agg, promo, ta, tb, col))
    g = [(1 - x) * a + x * b - (1 - x) * base for x in (0.05, 0.1, 0.2, 0.3)]
    print(f'{name:36s} {a - base:+.4f}  {b:.4f}  ' + '  '.join(f'{v:+.4f}' for v in g))
