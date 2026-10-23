"""Population-prior emulation for simulated class-2 truths (scripts/hlval.py HL_PRIOR, scripts/train_ret_ranker.py
PRIOR_SHIFT).

The spectral models (PeakFormer pf_*, binned FP fp_*) score structures that were never training positives lower: the
COCONUT-not-train structures only ever appeared as window negatives. A simulated class-2 truth is a held-out training
structure and lacks that penalty, a hidden COCONUT answer has it. Penalty per spectral column = mean score of held-out
training decoys (in_train, no split-0 spectra in casmi26-fp-data: never positives) minus that of COCONUT-not-train
(group C) resp. other non-train (group O) decoys, matched within (query, Morgan-Tanimoto-to-truth bin) cells of
same-formula decoys of class-2 queries. Decoys only: the truth itself is never used.
"""
import numpy as np, pandas as pd

from .own import formula_rel, PF_FREL_SRC, FREL_SRC

W = '/home/user/work'
SPEC = ['pf_bern', 'pf_cos', 'fp_bern', 'fp_cos']
_G = {}


def _globals():
    if not _G:
        P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['key', 'formula'])
        T = np.load('/home/user/ds/casmi26-fp-data/structs.npz'); S = np.load('/home/user/ds/casmi26-fp-data/spectra.npz')
        seen = T['key'][np.bincount(S['sidx'][S['split'] == 0], minlength=len(T['key'])) > 0]
        _G.update(FORMULA=P.formula.to_numpy(dtype=object), SEEN=P.key.isin(set(seen.tolist())).values,
                  POOLFP=np.load(f'{W}/pool4/pool_fp.npy', mmap_mode='r'),
                  POP8=np.array([bin(i).count('1') for i in range(256)], np.uint8))
    return _G


def penalties(D):
    """{'C': {col: pen, 'cells': n}, 'O': {...}}: score gap of held-out training decoys over non-train decoys."""
    g = _globals()
    cols = [c for c in SPEC if c in D.columns]
    t = D[(D.label == 1) & (D.cls == 2)][['qid', 'pid']].drop_duplicates('qid').set_index('qid').pid
    d = D[(D.label == 0) & D.qid.isin(t.index)]
    tp = t.reindex(d.qid.values).values
    d = d[g['FORMULA'][d.pid.values] == g['FORMULA'][tp]]
    tp = t.reindex(d.qid.values).values
    held = (d.in_train.values == 1) & ~g['SEEN'][d.pid.values]
    a = np.asarray(g['POOLFP'][tp]); b = np.asarray(g['POOLFP'][d.pid.values])
    tan = g['POP8'][a & b].sum(1) / np.maximum(g['POP8'][a | b].sum(1), 1)
    d = d.assign(typ=np.where(held, 'held', np.where(d.in_train.values == 1, 'seen', np.where(d.in_coco.values == 1, 'C', 'O'))),
                 bin=np.digitize(tan, [0.2, 0.3, 0.4, 0.55, 0.7]))
    out = {}
    for grp in ('C', 'O'):
        x = d[d.typ.isin(['held', grp])].groupby(['qid', 'bin', 'typ'])[cols].mean().unstack('typ').dropna()
        out[grp] = {c: float((x[(c, 'held')] - x[(c, grp)]).mean()) for c in cols} if len(x) else {c: 0.0 for c in cols}
        out[grp]['cells'] = len(x)
    return out


def apply_prior(D, pen, lam=1.0):
    """Shift class-2 truths' spectral scores down by lam * penalty (in place), then recompute rank / gap /
    within-formula features of the shifted columns."""
    cols = [c for c in SPEC if c in D.columns]
    for grp, m in (('C', (D.label == 1) & (D.cls == 2) & (D.in_coco == 1)), ('O', (D.label == 1) & (D.cls == 2) & (D.in_coco == 0))):
        for c in cols:
            D.loc[m, c] = D.loc[m, c] - lam * pen[grp][c]
    g = D.groupby('qid', sort=False)
    for c in cols:
        if c + '_r' in D.columns:
            D[c + '_r'] = g[c].rank(ascending=False, method='min')
        if c + '_gap' in D.columns:
            D[c + '_gap'] = D[c] - g[c].transform('max')
    src = [c for c in PF_FREL_SRC + FREL_SRC if c in cols and c + '_fz' in D.columns]
    if src:
        D['formula'] = _globals()['FORMULA'][D.pid.values]
        formula_rel(D, src)
        D.drop(columns='formula', inplace=True)
    return D
