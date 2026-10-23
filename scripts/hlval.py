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
- population prior (HL_PRIOR=1): the spectral models score structures that were never training positives lower
  (COCONUT-not-train only ever appeared as window negatives), while a simulated class-2 truth is a held-out training
  structure. Per scoring condition and column (pf_bern, pf_cos, fp_bern, fp_cos) the penalty = mean score of held-out
  training decoys (in_train, no split-0 spectra in casmi26-fp-data: never positives) minus that of COCONUT-not-train
  (resp. other non-train) decoys, matched within (query, Morgan-Tanimoto-to-truth bin) cells of same-formula decoys of
  class-2 queries; class-2 truths are shifted down by it inside the draws and rank / gap / within-formula features
  recomputed. Debiased models / scores (token policy, FP_PMI, training-structure negatives) shrink their own penalty.
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
PRIOR = os.environ.get('HL_PRIOR') == '1'
SPEC = ['pf_bern', 'pf_cos', 'fp_bern', 'fp_cos']
if PRIOR:
    from casmi.own import formula_rel, PF_FREL_SRC, FREL_SRC
    _P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['key', 'formula'])
    FORMULA = _P.formula.to_numpy(dtype=object)
    _T = np.load('/home/user/ds/casmi26-fp-data/structs.npz'); _S = np.load('/home/user/ds/casmi26-fp-data/spectra.npz')
    _seen = _T['key'][np.bincount(_S['sidx'][_S['split'] == 0], minlength=len(_T['key'])) > 0]
    SEEN = np.isin(_P.key.values, _seen)           # pool4 pid -> structure was a training positive of the FP / PF models
    POOLFP = np.load(f'{W}/pool4/pool_fp.npy', mmap_mode='r')
    POP8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)
    del _P, _T, _S, _seen


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


def penalties(D):
    """{'C': {col: pen}, 'O': {col: pen}}: score gap of held-out training decoys over non-train decoys (see docstring)."""
    cols = [c for c in SPEC if c in D.columns]
    t = D[(D.label == 1) & (D.cls == 2)][['qid', 'pid']].drop_duplicates('qid').set_index('qid').pid
    d = D[(D.label == 0) & D.qid.isin(t.index)]
    d = d[FORMULA[d.pid.values] == FORMULA[t.reindex(d.qid.values).values]]
    held = (d.in_train.values == 1) & ~SEEN[d.pid.values]
    a = np.asarray(POOLFP[t.reindex(d.qid.values).values]); b = np.asarray(POOLFP[d.pid.values])
    tan = POP8[a & b].sum(1) / np.maximum(POP8[a | b].sum(1), 1)
    d = d.assign(typ=np.where(held, 'held', np.where(d.in_train.values == 1, 'seen', np.where(d.in_coco.values == 1, 'C', 'O'))),
                 bin=np.digitize(tan, [0.2, 0.3, 0.4, 0.55, 0.7]))
    out = {}
    for grp in ('C', 'O'):
        x = d[d.typ.isin(['held', grp])].groupby(['qid', 'bin', 'typ'])[cols].mean().unstack('typ').dropna()
        out[grp] = {c: float((x[(c, 'held')] - x[(c, grp)]).mean()) for c in cols} if len(x) else {c: 0.0 for c in cols}
        out[grp]['cells'] = len(x)
    return out


def apply_prior(D, pen):
    """Shift class-2 truths' spectral scores down by the population penalty, then recompute derived features."""
    cols = [c for c in SPEC if c in D.columns]
    for grp, m in (('C', (D.label == 1) & (D.cls == 2) & (D.in_coco == 1)), ('O', (D.label == 1) & (D.cls == 2) & (D.in_coco == 0))):
        for c in cols:
            D.loc[m, c] = D.loc[m, c] - pen[grp][c]
    g = D.groupby('qid', sort=False)
    for c in cols:
        if c + '_r' in D.columns:
            D[c + '_r'] = g[c].rank(ascending=False, method='min')
        if c + '_gap' in D.columns:
            D[c + '_gap'] = D[c] - g[c].transform('max')
    src = [c for c in PF_FREL_SRC + FREL_SRC if c in cols and c + '_fz' in D.columns]
    if src:
        D['formula'] = FORMULA[D.pid.values]
        formula_rel(D, src)
        D.drop(columns='formula', inplace=True)
    return D


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
        pen = PEN.setdefault(p, penalties(D)) if PRIOR else None
        draws = []
        for k in range(NDRAW):
            Dk = transplant(D, np.random.default_rng(k))
            if PRIOR:
                Dk = apply_prior(Dk, pen)
            draws.append(rr(Dk, bst.predict(Dk[feats].astype(float).values)))
        hl = pd.concat(draws, axis=1).mean(axis=1)
        m = {s: (float(plain[S == s].mean()), float(hl[S == s].mean()), int((S == s).sum())) for s in MIX}
        per_draw = [sum(MIX[s] * (d[S == s].mean() if s != 'T' else plain[S == s].mean()) for s in MIX) for d in draws]
        out[cond] = dict(strata=m, hl_lb=float(np.mean(per_draw)), hl_sd=float(np.std(per_draw)), pen=pen)
    return out


PEN = {}


if __name__ == '__main__':
    for mp in sys.argv[1:]:
        res = evaluate(mp)
        print('==', os.path.basename(mp))
        for cond, r in res.items():
            if r is None:
                print(f'  {cond:7s} (features missing)'); continue
            st = '  '.join(f'{s}: sim {a:.3f} -> HL {b:.3f} (n {n})' for s, (a, b, n) in r['strata'].items())
            print(f'  {cond:7s} HL-LB {r["hl_lb"]:.4f} (sd over draws {r["hl_sd"]:.4f}) | {st}')
            if r['pen'] and mp == sys.argv[1]:
                print('          prior penalty ' + ' | '.join(f'{g}: ' + ' '.join(f'{c} {v:.3g}' for c, v in r['pen'][g].items()) for g in ('C', 'O')))
