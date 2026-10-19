"""Own end-to-end predictor: retrieval features + fingerprint-model features -> LambdaRank -> top-25 SMILES.

Shared by offline validation (scripts/add_fp_feats.py) and the Kaggle inference runner (casmi/run_own.py), so the
features are computed identically in both places.
"""
import numpy as np
import pandas as pd

from .metric import key14
from .prep_meta import inst_family
from .spectra import clean_peaks, neutral_mass

FP_FEATS = ['fp_cos', 'fp_bern', 'fp_cos_r', 'fp_bern_r', 'fp_cos_gap', 'fp_bern_gap']


def fp_feats(df, cos, bern):
    """Add fingerprint-model columns to one query's candidate frame (in place)."""
    df['fp_cos'] = cos; df['fp_bern'] = bern
    for c in ('fp_cos', 'fp_bern'):
        df[c + '_r'] = df[c].rank(ascending=False, method='min')
        df[c + '_gap'] = df[c] - df[c].max()
    return df


FREL_SRC = ['fp_cos', 'fp_bern', 'an_tan_best', 'an_best', 'frag', 'lib']
FREL_FEATS = [f'{c}_{s}' for c in FREL_SRC for s in ('fz', 'fr', 'fgap')]


def formula_rel(df):
    """Within-formula relative evidence (isomers are the main confusion): z-score, rank and gap to the best of the
    same-formula candidates of each query ('qid' column if present, else one query), NaN / 0 for singletons."""
    keys = (['qid'] if 'qid' in df.columns else []) + ['formula']
    g = df.groupby(keys, sort=False)
    n = g[FREL_SRC[0]].transform('size')
    for c in FREL_SRC:
        m = g[c].transform('mean'); sd = g[c].transform('std')
        df[c + '_fz'] = ((df[c] - m) / sd).where(n >= 2).fillna(0).where(sd > 0, 0)
        df[c + '_fr'] = g[c].rank(ascending=False, method='min').where(n >= 2)
        df[c + '_fgap'] = (df[c] - g[c].transform('max')).where(n >= 2)
    return df


def query_from_rows(g):
    """Test rows of one molecule -> (cleaned query spectra for retrieval, raw spectra for the FP model, M)."""
    Q, raw, nms = [], [], []
    for r in g.itertuples():
        mz = np.asarray(r.ms2_mzs, np.float64); it = np.asarray(r.ms2_normalized_intensities, np.float64)
        prec = float(r.precursor_mz)
        pos = r.ionization_mode == 'positive'
        a, b = clean_peaks(mz, it, prec, rel_floor=0.005, top_k=64)
        if len(a) >= 3:
            Q.append(dict(mz=a, it=b, pol=0 if pos else 1))
        ce = r.collision_energy_ev
        ce = float(np.mean(ce)) if ce is not None and len(ce) else -1.0
        raw.append(dict(mz=mz, it=it, prec=prec, adduct=r.adduct, ce=ce, inst=inst_family(r.instrument_type)))
        nms.append(neutral_mass(prec, r.adduct))
    M = float(np.nanmedian(nms)) if len(nms) else np.nan
    return Q, raw, M


def stage1(R, fps, ranker, feats, g, k=30):
    """One molecule: stage-1 top-k candidate frame (all features, s1 = ranker score, s1_r = rank), one row per
    metric identity (key14), or None."""
    Q, raw, M = query_from_rows(g)
    if not Q or not np.isfinite(M):
        return None
    df = R.features(Q, M)
    if len(df) == 0:
        return None
    if fps is not None:
        cos, bern = fps.score(raw, df.pid.values, df.smiles.values)
        fp_feats(df, cos, bern)
    if FREL_FEATS[0] in feats:
        formula_rel(df)
    df['s1'] = ranker.predict(df[feats].astype(float).values)
    df = df.sort_values('s1', ascending=False)
    keep, seen = [], set()
    for i, s in zip(df.index, df.smiles.values):
        kk = key14(s)
        if kk is None or kk in seen:
            continue
        seen.add(kk); keep.append((i, kk))
        if len(keep) >= k:
            break
    out = df.loc[[i for i, _ in keep]].copy()
    out['key14'] = [kk for _, kk in keep]
    out['s1_r'] = np.arange(1, len(out) + 1)
    return out


def predict(R, fps, ranker, feats, g, n=25):
    """One molecule: ranked list of (smiles, score, key14)."""
    Q, raw, M = query_from_rows(g)
    if not Q or not np.isfinite(M):
        return []
    df = R.features(Q, M)
    if len(df) == 0:
        return []
    if fps is not None:
        cos, bern = fps.score(raw, df.pid.values, df.smiles.values)
        fp_feats(df, cos, bern)
    if FREL_FEATS[0] in feats:
        formula_rel(df)
    df['z'] = ranker.predict(df[feats].astype(float).values)
    df = df.sort_values('z', ascending=False)
    out, seen = [], set()
    for s, z in zip(df.smiles.values, df.z.values):
        k = key14(s)
        if k is None or k in seen:
            continue
        seen.add(k); out.append((s, float(z), k))
        if len(out) >= n:
            break
    return out
