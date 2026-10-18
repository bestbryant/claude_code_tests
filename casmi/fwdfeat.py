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


def _z(v):
    ok = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if ok.sum() >= 2:
        sd = v[ok].std(ddof=1)
        out[ok] = (v[ok] - v[ok].mean()) / sd if sd > 0 else 0.0
    return out


def add_features(T, scores, col='gl'):
    """T: top-k rows (qid, smiles, formula, s1, s1_r). scores: {str(qid): {smiles: score | None}}.
    Adds the FWD_FEATS columns (NaN where the query has no covered spectrum)."""
    T = T.copy()
    T[col] = [scores.get(str(q), {}).get(s) for q, s in zip(T.qid, T.smiles)]
    T[col] = pd.to_numeric(T[col], errors='coerce')
    for c in FWD_FEATS[1:7]:
        T[c] = np.nan
    parts = []
    for q, g in T.groupby('qid', sort=False):
        v = g[col].values.astype(float)
        g = g.copy()
        g['gl_z'] = _z(v)
        g['gl_r'] = pd.Series(-v).rank(method='min').values          # NaN stays NaN
        fz = np.full(len(g), np.nan); fr = np.full(len(g), np.nan); fgap = np.full(len(g), np.nan)
        n_f = np.zeros(len(g))
        for f, idx in g.groupby('formula').indices.items():
            n_f[idx] = len(idx)
            if len(idx) < 2:
                continue
            w = v[idx]
            fz[idx] = _z(w)
            fr[idx] = pd.Series(-w).rank(method='min').values
            for j, i in enumerate(idx):
                others = np.delete(w, j)
                if np.isfinite(w[j]) and np.isfinite(others).any():
                    fgap[i] = w[j] - np.nanmax(others)
        g['gl_fz'], g['gl_fr'], g['gl_fgap'], g['n_fgrp'] = fz, fr, fgap, n_f
        parts.append(g)
    return pd.concat(parts) if parts else T
