"""Artifact test for documentation-transplant training: does a ranker detect transplanted rows (documentation drawn
independently of the structure) rather than learn hidden-like truths? python scripts/hl_transplant_artifact.py"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
import importlib.util
spec = importlib.util.spec_from_file_location('hl', '/home/user/claude_code_tests/scripts/hlval.py'); hl = importlib.util.module_from_spec(spec)
sys.argv = ['x']; spec.loader.exec_module(hl)
W = '/home/user/work'
D = hl.load(f'{W}/ret13_val_fp.parquet')
cols = [c for c in hl.DOC if c in D.columns]
t = D[D.label == 1].drop_duplicates('qid').set_index('qid')
cq = t.index[(t.cls == 2) & (t.in_coco == 1)]
D = D[D.qid.isin(cq)].reset_index(drop=True)
don = D[(D.label == 0) & (D.in_train == 0) & (D.in_coco == 1)][cols].values
pool_dec = np.flatnonzero(((D.label == 0) & (D.in_train == 0) & (D.in_coco == 1)).values)
for name in ('ret_ranker_v13a1', 'ret_ranker_v13tp'):
    bst = lgb.Booster(model_file=f'{W}/own_models/{name}.txt'); f = open(f'{W}/own_models/{name}_feats.txt').read().split()
    res = {}
    for k in range(5):
        rng = np.random.default_rng(k)
        # (c) one random COCONUT-not-train decoy per query transplanted, truth keeps its real documentation
        X = D.copy()
        g = pd.Series(pool_dec).groupby(X.qid.values[pool_dec]).apply(lambda s: rng.choice(s.values))
        idx = g.values.astype(int)
        X.iloc[idx, [X.columns.get_loc(c) for c in cols]] = don[rng.integers(0, len(don), len(idx))]
        X['p'] = bst.predict(X[f].astype(float).values)
        X['r'] = X.groupby('qid').p.rank(ascending=False, method='first')
        res.setdefault('transplanted decoy rank-1 share', []).append(float((X.r.values[idx] == 1).mean()))
        Y = D.copy(); Y['p'] = bst.predict(Y[f].astype(float).values); Y['r'] = Y.groupby('qid').p.rank(ascending=False, method='first')
        res.setdefault('same decoy rank-1 share untouched', []).append(float((Y.r.values[idx] == 1).mean()))
        # (b) every COCONUT-not-train candidate AND the truth transplanted: documentation carries no information
        Z = D.copy(); allc = np.r_[pool_dec, np.flatnonzero(Z.label.values == 1)]
        Z.iloc[allc, [Z.columns.get_loc(c) for c in cols]] = don[rng.integers(0, len(don), len(allc))]
        Z['p'] = bst.predict(Z[f].astype(float).values)
        o = Z.sort_values(['qid', 'p'], ascending=[True, False]); o['r'] = o.groupby('qid').cumcount() + 1
        rr = o[o.label == 1].groupby('qid').r.min()
        res.setdefault('MRR all transplanted', []).append(float(np.where(rr <= 25, 1 / rr, 0).mean()))
    print(name, {k: round(np.mean(v), 4) for k, v in res.items()}, '| chance rank-1 ~', round(float((1 / D.groupby('qid').size()).mean()), 4))
