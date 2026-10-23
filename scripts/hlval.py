"""Hidden-like validation (HL-val): stage-1 MRR on validation strata made to look like the hidden test, combined with
the LB-implied class mix into one HL-LB number.

python scripts/hlval.py <ranker.txt> [<ranker2.txt> ...]      env HL_ROWS_<cond>=<rows parquet> per scoring condition

The simulation is far easier than the hidden test (sim MRR ~0.9 vs LB 0.32) and several offline gains did not transfer.
Differences addressed here:
- instrument: every hidden spectrum is timsTOF, and the FP / PeakFormer models read the token as a chemistry prior.
  Conditions = feature sets computed under a token policy: 'kernel' = what the deployed kernel computes on hidden data
  (token 1 for every query: ret14 rows), 'tok3' = the OWN_PF_INST=3 override (ret13 rows), 'true' = the rows' own tokens.
- documentation: a simulated class-2 truth is a training structure, far better documented than a hidden answer. The
  truth's documentation / depositor columns are replaced by those of a random candidate of the same source that is not a
  training structure (COCONUT-not-train for COCONUT truths), 10 paired draws; HL_TRANSPLANT_P = share of truths
  transplanted (calibrated against the LB no-prior gap, scripts/hl_calib_doc.sh).
- mix: strata weighted by the hidden mix implied by the LB probes (Oct 23 design panel): class 1 0.18, COCONUT 0.29,
  other reachable 0.03; Enamine (0.056) and unreachable (~0.44) answers score 0 in the main lists.
Strata: T = class-1 queries (truth spectra partly in the library), C = class-2 COCONUT truths, O = class-2 PubChemLite /
ChEBI-LMSD truths outside COCONUT. Validation queries only (ret10pl_val_reach_c query set).
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.retrieval import PRIOR_COLS, PCL_COLS, PC_COLS, LT_COLS, VEND_COLS

W = '/home/user/work'
MIX = dict(T=0.18, C=0.29, O=0.03)
DOC = PRIOR_COLS + PCL_COLS + PC_COLS + LT_COLS + ['in_bio'] + VEND_COLS
CONDS = {c[len('HL_ROWS_'):]: p for c, p in os.environ.items() if c.startswith('HL_ROWS_')} or {
    'kernel': f'{W}/ret14_val_fp.parquet', 'tok3': f'{W}/ret13_val_fp.parquet', 'true': f'{W}/ret10pl_val_fp.parquet'}
NDRAW = int(os.environ.get('HL_DRAWS', '10'))
P_TRANS = float(os.environ.get('HL_TRANSPLANT_P', '1'))   # share of class-2 truths given transplanted documentation
qset = pd.read_parquet(f'{W}/ret10pl_val_reach_c.parquet', columns=['qid']).qid.unique()
V = pd.read_parquet(f'{W}/pool4/vendor.parquet') if os.path.exists(f'{W}/pool4/vendor.parquet') else None


def load(p):
    D = pd.read_parquet(p)
    D = D[D.qid.isin(qset)].sort_values(['qid', 'pid']).reset_index(drop=True)
    D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
    if V is not None:
        for c in VEND_COLS:
            if c not in D.columns:
                D[c] = V[c].values[D.pid.values].astype(float)
    return D


def strata(D):
    t = D[D.label == 1].drop_duplicates('qid').set_index('qid')
    s = pd.Series('x', index=t.index)
    s[t.cls == 1] = 'T'
    s[(t.cls == 2) & (t.in_coco == 1)] = 'C'
    s[(t.cls == 2) & (t.in_coco == 0)] = 'O'
    return s


def transplant(D, rng):
    """Truth documentation of class-2 queries <- a random non-training candidate of the same source."""
    D = D.copy()
    cols = [c for c in DOC if c in D.columns]
    don_c = D[(D.label == 0) & (D.in_train == 0) & (D.in_coco == 1)][cols].values
    don_o = D[(D.label == 0) & (D.in_train == 0) & (D.in_coco == 0)][cols].values
    tr = np.flatnonzero(((D.label == 1) & (D.cls == 2)).values)
    tr = tr[rng.random(len(tr)) < P_TRANS]
    coco = D.in_coco.values[tr] == 1
    vals = np.where(coco[:, None], don_c[rng.integers(0, len(don_c), len(tr))], don_o[rng.integers(0, len(don_o), len(tr))])
    D.iloc[tr, [D.columns.get_loc(c) for c in cols]] = vals
    return D


def rr(D, score):
    o = D.assign(p=score).sort_values(['qid', 'p'], ascending=[True, False])
    o['r'] = o.groupby('qid').cumcount() + 1
    r = o[o.label == 1].groupby('qid').r.min().reindex(D.qid.unique())
    return pd.Series(np.where(r <= 25, 1.0 / r, 0.0), index=r.index)


def evaluate(model_p):
    bst = lgb.Booster(model_file=model_p)
    feats = open(model_p.replace('.txt', '_feats.txt')).read().split()
    out = {}
    for cond, p in CONDS.items():
        D = load(p)
        if not all(f in D.columns for f in feats):
            out[cond] = None; continue
        plain = rr(D, bst.predict(D[feats].astype(float).values))
        S = strata(D).reindex(plain.index).fillna('x')
        draws = []
        for k in range(NDRAW):
            Dk = transplant(D, np.random.default_rng(k))
            draws.append(rr(Dk, bst.predict(Dk[feats].astype(float).values)))
        hl = pd.concat(draws, axis=1).mean(axis=1)
        m = {s: (float(plain[S == s].mean()), float(hl[S == s].mean()), int((S == s).sum())) for s in MIX}
        per_draw = [sum(MIX[s] * (d[S == s].mean() if s != 'T' else plain[S == s].mean()) for s in MIX) for d in draws]
        out[cond] = dict(strata=m, hl_lb=float(np.mean(per_draw)), hl_sd=float(np.std(per_draw)))
    return out


if __name__ == '__main__':
    for mp in sys.argv[1:]:
        res = evaluate(mp)
        print('==', os.path.basename(mp))
        for cond, r in res.items():
            if r is None:
                print(f'  {cond:7s} (features missing)'); continue
            st = '  '.join(f'{s}: sim {a:.3f} -> HL {b:.3f} (n {n})' for s, (a, b, n) in r['strata'].items())
            print(f'  {cond:7s} HL-LB {r["hl_lb"]:.4f} (sd over draws {r["hl_sd"]:.4f}) | {st}')
