"""LambdaRank over retrieval candidates: train on ranker-set queries, report MRR@25 on held-out validation queries.

Class-2 queries are kept only when the truth is reachable through a non-training source (COCONUT or PubChemLite),
because a class-2 answer is by definition not a training structure.
Usage: python scripts/train_ret_ranker.py <train_rows> <val_rows> <model_out.txt> [--no-prior] [--no-fp] [pool_dir]
--source-blind: no source membership or documentation (in_coco / in_bio / PubChemLite / PubChem / LOTUS / priors), the
second ranker whose top-1 is inserted at rank 2 (OWN_R2 in casmi/run_own.py); with --no-membership also no in_train / lib_n.
DOC_W=<weights.parquet> [DOC_ALPHA=a]: documentation-shift weights for class-2 queries (casmi/docw.py).
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.retrieval import FEATS, PRIOR_COLS, PCL_COLS, PC_COLS, LT_COLS, VEND_COLS
from casmi.own import FP_FEATS, FREL_FEATS, PF_FEATS, PF_FREL
from casmi.dreams_feats import DR_FEATS

tr_p, va_p, out = sys.argv[1:4]
flags = set(a for a in sys.argv[4:] if a.startswith('--'))
pool_dir = next((a for a in sys.argv[4:] if not a.startswith('--')), '/home/user/work/pool2')
REACH_PMID = '--reach-pmid' in flags
DROP_ARTIFACT = '--drop-artifact' in flags


def load(p):
    D = pd.read_parquet(p).sort_values(['qid', 'pid']).reset_index(drop=True)
    D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0           # also for rows dumped before the in-code fix
    t = D[D.label == 1].set_index('qid')
    # reachable for a class-2 test molecule: in COCONUT, in PubChemLite, or (pool v3) PubChem-documented (PMID record)
    reach = ((t.in_coco == 1) | (t.get('in_pcl', 0) == 1) | (t.get('in_bio', 0) == 1)
             | (t.get('pc_pmid', 0) > 0 if REACH_PMID else False))
    keep = (D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)
    if DROP_ARTIFACT:   # class-1 queries whose truth kept no library spectrum behave like class 2 with in_train = 1
        keep &= ~D.qid.isin(t.index[(t.cls == 1) & (t.lib_n == 0)])
    return D[keep].reset_index(drop=True)


TR, VA = load(tr_p), load(va_p)
if os.environ.get('DOC_TRANSPLANT'):
    # hidden-like class-2 truths: a simulated truth is a training structure, far better documented than a hidden
    # answer; its documentation / depositor columns are replaced by those of a random non-training candidate of the
    # same source (scripts/hlval.py uses the same transplant to evaluate). DOC_TRANSPLANT=<copies> adds that many
    # transplanted copies of every class-2 query (the original stays unless DOC_TRANSPLANT_KEEP=0).
    from casmi.retrieval import PRIOR_COLS as _P, PCL_COLS as _L, PC_COLS as _C, LT_COLS as _T, VEND_COLS as _V
    cols = [c for c in _P + _L + _C + _T + ['in_bio'] + _V if c in TR.columns]
    don = {k: TR[(TR.label == 0) & (TR.in_train == 0) & (TR.in_coco == k)][cols].values for k in (0, 1)}
    c2 = TR[TR.cls == 2]
    parts = [TR] if os.environ.get('DOC_TRANSPLANT_KEEP', '1') == '1' else [TR[TR.cls != 2]]
    for k in range(int(os.environ['DOC_TRANSPLANT'])):
        rng = np.random.default_rng(100 + k)
        X = c2.copy(); X['qid'] = X.qid + 20_000_000 * (k + 1)
        tr = np.flatnonzero(X.label.values == 1)
        coco = X.in_coco.values[tr] == 1
        vals = np.where(coco[:, None], don[1][rng.integers(0, len(don[1]), len(tr))], don[0][rng.integers(0, len(don[0]), len(tr))])
        X.iloc[tr, [X.columns.get_loc(c) for c in cols]] = vals
        parts.append(X)
    TR = pd.concat(parts, ignore_index=True).sort_values(['qid', 'pid']).reset_index(drop=True)
    print('documentation transplant: class-2 copies', os.environ['DOC_TRANSPLANT'], 'rows', len(TR), flush=True)
if os.environ.get('PRIOR_SHIFT'):
    # hidden-like spectral scores for class-2 truths (casmi/hlprior.py): a hidden COCONUT / other non-train answer carries
    # the population prior of the PeakFormer / FP models (never a training positive), a simulated truth does not
    from casmi.hlprior import penalties, apply_prior
    _pen = penalties(TR)
    apply_prior(TR, _pen, float(os.environ['PRIOR_SHIFT']))
    print('prior shift', os.environ['PRIOR_SHIFT'], {g: {c: round(v, 4) for c, v in d.items()} for g, d in _pen.items()}, flush=True)
# class-1 queries twice: the simulation is 30% class 1 while the test mix is closer to 45:55 (better on validation)
c1 = TR[TR.cls == 1]
TR = pd.concat([TR, c1.assign(qid=c1.qid + 10_000_000)], ignore_index=True).sort_values(['qid', 'pid']).reset_index(drop=True)
feats = [f for f in FEATS + PCL_COLS + PC_COLS + ['in_bio'] + LT_COLS + FP_FEATS + FREL_FEATS + DR_FEATS + PF_FEATS + PF_FREL + VEND_COLS
         if f in TR.columns and f in VA.columns]
if '--no-prior' in flags:
    feats = [f for f in feats if f not in PRIOR_COLS + PCL_COLS[1:] + PC_COLS + LT_COLS + ['in_coco']]
if '--source-blind' in flags:
    feats = [f for f in feats if f not in PRIOR_COLS + PCL_COLS + PC_COLS + LT_COLS + VEND_COLS + ['in_coco', 'in_bio']]
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
              lambdarank_truncation_level=25, verbose=-1, num_threads=2,
              deterministic=True, force_row_wise=True, seed=0)      # reproducible (docs/reproduce.md G3)
import json
params.update(json.loads(os.environ.get('RANKER_PARAMS', '{}')))      # e.g. '{"num_leaves": 63, "min_data_in_leaf": 50}'
g = TR.groupby('qid', sort=False).size().values
wts = None
if os.environ.get('DOC_W'):        # documentation-shift weights for class-2 queries (casmi/docw.py)
    from casmi.docw import query_weights
    wts = query_weights(TR, os.environ['DOC_W'], float(os.environ.get('DOC_ALPHA', '1')))
model = lgb.train(params, lgb.Dataset(TR[feats].astype(float).values, TR.label.values, weight=wts, group=g),
                  num_boost_round=500)
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
