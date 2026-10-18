"""Forward-model (GLACIER / ICEBERG, ms-pred with MassSpecGym weights) features for a two-stage ranker.

Stage 1 ranks every candidate in the mass window; its top K per query are scored by predicted-spectrum similarity
(the runner's `gl_ex_mean`: mean entropy similarity of the prediction to the query's positive [M+H]+ / [M+Na]+
spectra) and stage 2 re-ranks those K with the features below added to the stage-1 features.
"""
import numpy as np
import pandas as pd

FWD_FEATS = ['gl', 'gl_fz', 'gl_fr', 'gl_fgap', 'gl_z', 'gl_r', 'n_fgrp', 's1', 's1_r']


def top_k(D, score, k):
    """Top k rows per qid by `score` (a column name), with the stage-1 score and rank attached."""
    D = D.assign(s1=D[score].values).sort_values(['qid', 's1'], ascending=[True, False])
    D['s1_r'] = D.groupby('qid').cumcount() + 1
    return D[D.s1_r <= k].copy()


def _gz(g, col):
    """Within-group z-score over non-missing values (sample std; < 2 values or std 0 -> NaN / 0)."""
    m = g[col].transform('mean'); sd = g[col].transform('std'); n = g[col].transform('count')
    z = (g.obj[col] - m) / sd
    return z.where(n >= 2).where(~((sd == 0) & (n >= 2)), 0.0)


def add_features(T, scores, col='gl'):
    """T: top-k rows (qid, smiles, formula, s1, s1_r). scores: {str(qid): {smiles: score | None}}.
    Adds the FWD_FEATS columns (NaN where the query has no covered spectrum)."""
    T = T.copy()
    T[col] = pd.to_numeric(pd.Series([scores.get(str(q), {}).get(s) for q, s in zip(T.qid, T.smiles)],
                                     index=T.index, dtype=object), errors='coerce')
    gq = T.groupby('qid', sort=False)
    T['gl_z'] = _gz(gq, col)
    T['gl_r'] = gq[col].rank(ascending=False, method='min')
    gf = T.groupby(['qid', 'formula'], sort=False)
    T['n_fgrp'] = gf[col].transform('size').astype(float)
    T['gl_fz'] = _gz(gf, col).where(T.n_fgrp >= 2)
    T['gl_fr'] = gf[col].rank(ascending=False, method='min').where(T.n_fgrp >= 2)
    # gap to the best OTHER member of the formula group
    first = gf[col].transform('max')
    second = gf[col].transform(lambda v: v.nlargest(2).iloc[-1] if v.count() >= 2 else np.nan)
    is_top = T[col] == first
    other_best = np.where(is_top, second, first)
    T['gl_fgap'] = (T[col] - other_best).where((T.n_fgrp >= 2) & T[col].notna())
    return T
