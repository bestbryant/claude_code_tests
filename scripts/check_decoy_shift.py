"""Artifact probe for prior-shift-trained rankers (review wf_b897a7fd-fd5): shift ONE same-formula COCONUT-not-train decoy
per query down by lam x penalty and report the change of its ranker score / rank-1 share (a sound ranker scores it LOWER;
PRIOR_SHIFT rankers v13p05 / v13p1 scored it higher = they learned the shifted-truth pattern). Also truth-shift robustness."""
import sys, os, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
import importlib.util
spec = importlib.util.spec_from_file_location('hl', '/home/user/claude_code_tests/scripts/hlval.py'); hl = importlib.util.module_from_spec(spec)
sys.argv = ['x']; spec.loader.exec_module(hl)
from casmi.hlprior import penalties, _globals, SPEC
from casmi.own import formula_rel
W = '/home/user/work'; M = f'{W}/own_models'
t0 = time.time()
D = hl.load(f'{W}/ret13_val_fp.parquet')
G = _globals()
pen = penalties(D)
print('val penalty', {g: {c: round(v, 4) for c, v in d.items()} for g, d in pen.items()}, flush=True)
D['formula'] = G['FORMULA'][D.pid.values]
S = hl.strata(D)
cq = S.index[S == 'C']
D = D[D.qid.isin(cq)].reset_index(drop=True)
for c in SPEC: D[c] = D[c].astype('float64')
tf = D[D.label == 1].drop_duplicates('qid').set_index('qid').formula
cand = np.flatnonzero(((D.label == 0) & (D.in_train == 0) & (D.in_coco == 1) & (D.formula.values == tf.reindex(D.qid.values).values)).values)
print('C queries', D.qid.nunique(), 'with same-formula C decoy', len(np.unique(D.qid.values[cand])), 'rows', len(D), flush=True)
models = {n: (lgb.Booster(model_file=f'{M}/ret_ranker_{n}.txt'), open(f'{M}/ret_ranker_{n}_feats.txt').read().split())
          for n in ('v13', 'v13p05', 'v13p1')}


def recompute(X, cols):
    g = X.groupby('qid', sort=False)
    for c in cols:
        X[c + '_r'] = g[c].rank(ascending=False, method='min')
        X[c + '_gap'] = X[c] - g[c].transform('max')
    src = [c for c in ('pf_cos', 'pf_bern') if c in cols]
    if src:
        formula_rel(X, src)
    return X


def shift(X, idx, lam, cols, scale=None):
    X = X.copy()
    for c in cols:
        j = X.columns.get_loc(c)
        s = lam * pen['C'][c] * (1 if scale is None else scale)
        X.iloc[idx, j] = X.iloc[idx, j].values - s
    return recompute(X, cols)


def pred(X, n):
    b, f = models[n]
    return b.predict(X[f].astype(float).values, num_threads=2)


def rank_of(X, p, idx):
    r = pd.Series(p).groupby(X.qid.values).rank(ascending=False, method='first').values
    return r[idx]


def mrr_truth(X, p):
    o = X.assign(p=p).sort_values(['qid', 'p'], ascending=[True, False])
    o['r'] = o.groupby('qid').cumcount() + 1
    r = o[o.label == 1].groupby('qid').r.min()
    return float(np.where(r <= 25, 1 / r, 0).mean())


# ---- E1 / E2: decoy shift (truth untouched), all four columns and per-column
variants = {'all4': SPEC, 'pf_bern': ['pf_bern'], 'pf_cos': ['pf_cos'], 'fp_bern': ['fp_bern'], 'fp_cos': ['fp_cos'],
            'pf both': ['pf_bern', 'pf_cos'], 'fp both': ['fp_bern', 'fp_cos']}
res = {}
for k in range(3):
    rng = np.random.default_rng(k)
    idx = pd.Series(cand).groupby(D.qid.values[cand]).apply(lambda s: rng.choice(s.values)).values.astype(int)
    base = {n: pred(D, n) for n in models}
    for vname, cols in variants.items():
        for lam in ((1.0, 0.5) if vname == 'all4' else (1.0,)):
            X = shift(D, idx, lam, cols)
            for n in models:
                p = pred(X, n)
                key = (vname, lam, n)
                res.setdefault(key, []).append((float((p[idx] - base[n][idx]).mean()), float((rank_of(D, base[n], idx) == 1).mean()),
                                                float((rank_of(X, p, idx) == 1).mean()), float((p[idx] > base[n][idx]).mean())))
    print('draw', k, round(time.time() - t0), 's', flush=True)
print('\nE1/E2 decoy shift: mean d score | rank-1 share before -> after | share of decoys whose score rose')
for key, v in res.items():
    a = np.mean(v, axis=0)
    print(f'  {key[0]:8s} lam {key[1]:.1f} {key[2]:7s}  dscore {a[0]:+.3f}  r1 {a[1]:.4f} -> {a[2]:.4f}  rose {a[3]:.3f}', flush=True)

# ---- E3: truth shift functional form (doc transplant as in hlval, C stratum only, 3 draws)
tr = np.flatnonzero(D.label.values == 1)
forms = {'none': lambda r: None, 'const': lambda r: np.ones(len(tr)),
         'U(0,2)': lambda r: r.uniform(0, 2, len(tr)), 'Exp(1)': lambda r: r.exponential(1, len(tr)),
         'half0/half2': lambda r: 2.0 * (r.random(len(tr)) < 0.5)}
out = {}
for k in range(3):
    Dk = hl.transplant(D.drop(columns='formula'), np.random.default_rng(k)); Dk['formula'] = D.formula.values
    for c in SPEC: Dk[c] = Dk[c].astype('float64')
    for fn, f in forms.items():
        sc = f(np.random.default_rng(1000 + k))
        X = Dk if sc is None else shift(Dk, tr, 1.0, SPEC, sc)
        for n in models:
            out.setdefault((fn, n), []).append(mrr_truth(X, pred(X, n)))
    print('E3 draw', k, round(time.time() - t0), 's', flush=True)
print('\nE3 C-stratum MRR (doc transplant) by truth-shift form, lam mean 1')
for fn in forms:
    print(f'  {fn:12s} ' + '  '.join(f'{n} {np.mean(out[(fn, n)]):.4f}' for n in models), flush=True)
