"""Evaluate same-formula forward-model re-ranking on exp_ice output (rows with ranker score s, rank r, ice score).

python scripts/eval_ice.py <ice_val.parquet | runner work dir> [lams...]
Uses fuse.rerank semantics: within same-formula groups (>= 2 scored members) inside the top-N, re-sort by
z(ranker) + lam * z(ice) and keep the group's slots.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/ext/ice_local')
from fuse import rerank

src = sys.argv[1]
lams = [float(x) for x in sys.argv[2:]] or [0.25, 0.5, 1.0, 2.0]
D = pd.read_parquet(src)
if 'ice' not in D or D.ice.isna().all():           # partial run: read the runner checkpoint
    d = json.load(open(src + '.work/ice_out.json'))
    D['ice'] = [d.get(str(q), {}).get(s) for q, s in zip(D.qid, D.smiles)]
done = D.groupby('qid').ice.apply(lambda v: v.notna().mean() > 0.5)
D = D[D.qid.isin(done.index[done])]
print('queries', D.qid.nunique(), 'cand scored', D.ice.notna().mean().round(3))


def mrr(D, lam):
    rr = []
    for q, g in D.groupby('qid'):
        g = g.sort_values('r')
        if lam is None:
            order = list(range(len(g)))
        else:
            order = rerank(list(g.smiles), list(g.key), list(g.s), list(g.formula), list(g.ice), lam=lam, top_n=len(g))
        lab = g.label.values[order]
        hit = np.where(lab == 1)[0]
        rr.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
    return np.mean(rr), np.mean(np.array(rr) == 1)


print('base', np.round(mrr(D, None), 4))
for lam in lams:
    print('lam', lam, np.round(mrr(D, lam), 4))
for c in sorted(D.cls.unique()):
    Dc = D[D.cls == c]
    print('cls', c, Dc.qid.nunique(), 'base', np.round(mrr(Dc, None), 4), 'lam0.5', np.round(mrr(Dc, 0.5), 4), 'lam1', np.round(mrr(Dc, 1.0), 4))
# ICE alone within the truth's formula group: how often is the truth the best-ICE isomer?
t = D[D.label == 1]
g = D[D.formula.isin(t.formula)].merge(t[['qid', 'formula']], on=['qid', 'formula'])
grp = g.groupby('qid')
n_iso = grp.size()
best_is_truth = grp.apply(lambda x: x.loc[x.ice.idxmax(), 'label'] == 1 if x.ice.notna().any() else np.nan)
print('truth formula groups with >=2 members', int((n_iso >= 2).sum()), 'ICE picks truth', best_is_truth[n_iso >= 2].mean().round(3),
      'ranker picks truth', grp.apply(lambda x: x.loc[x.s.idxmax(), 'label'] == 1)[n_iso >= 2].mean().round(3))
