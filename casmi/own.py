"""Own end-to-end predictor: retrieval features + fingerprint-model features -> LambdaRank -> top-25 SMILES.

Shared by offline validation (scripts/add_fp_feats.py) and the Kaggle inference runner (casmi/run_own.py), so the
features are computed identically in both places.
"""
import os
import numpy as np
import pandas as pd

from .metric import key14
from .prep_meta import inst_family
from .spectra import clean_peaks, neutral_mass

FP_FEATS = ['fp_cos', 'fp_bern', 'fp_cos_r', 'fp_bern_r', 'fp_cos_gap', 'fp_bern_gap']
PF_FEATS = [f.replace('fp_', 'pf_') for f in FP_FEATS]          # PeakFormer (casmi/peakformer.py) columns


def fp_feats(df, cos, bern, prefix='fp'):
    """Add fingerprint-model columns to one query's candidate frame (in place); prefix 'pf' for the PeakFormer."""
    df[prefix + '_cos'] = cos; df[prefix + '_bern'] = bern
    for c in (prefix + '_cos', prefix + '_bern'):
        df[c + '_r'] = df[c].rank(ascending=False, method='min')
        df[c + '_gap'] = df[c] - df[c].max()
    return df


FREL_SRC = ['fp_cos', 'fp_bern', 'an_tan_best', 'an_best', 'frag', 'lib']
FREL_FEATS = [f'{c}_{s}' for c in FREL_SRC for s in ('fz', 'fr', 'fgap')]
PF_FREL_SRC = ['pf_cos', 'pf_bern']
PF_FREL = [f'{c}_{s}' for c in PF_FREL_SRC for s in ('fz', 'fr', 'fgap')]


def formula_rel(df, src=FREL_SRC):
    """Within-formula relative evidence (isomers are the main confusion): z-score, rank and gap to the best of the
    same-formula candidates of each query ('qid' column if present, else one query), NaN / 0 for singletons."""
    keys = (['qid'] if 'qid' in df.columns else []) + ['formula']
    g = df.groupby(keys, sort=False)
    n = g[src[0]].transform('size')
    for c in src:
        m = g[c].transform('mean'); sd = g[c].transform('std')
        df[c + '_fz'] = ((df[c] - m) / sd).where(n >= 2).fillna(0).where(sd > 0, 0)
        df[c + '_fr'] = g[c].rank(ascending=False, method='min').where(n >= 2)
        df[c + '_fgap'] = (df[c] - g[c].transform('max')).where(n >= 2)
    return df


def add_dreams(df, R, dreams, g):
    """DreaMS analog-channel columns for one test molecule's candidate frame (in place); nothing is hidden."""
    from .dreams_feats import dreams_features
    from .retrieval import tanimoto_packed
    DL, emb = dreams
    idx = [i for i in g.index if i in emb]
    if not idx:
        for c in ('dr_best', 'dr_tan_best', 'dr_sum', 'dr_top1_tan', 'dr_self', 'dr_r', 'dr_gap'):
            df[c] = 0.0 if c != 'dr_r' else 1.0
        return df
    qe = np.stack([emb[i] for i in idx]).astype(np.float32)
    qp = np.array([0 if m == 'positive' else 1 for m in g.loc[idx, 'ionization_mode']], np.int8)
    key_fp = lambda k: np.asarray(R.fp[R.key_idx[k]]) if k in R.key_idx else None
    f = dreams_features(DL, qe, qp, df.key.values, np.asarray(R.fp[df.pid.values]), key_fp, tanimoto_packed)
    for c, v in f.items():
        df[c] = v
    return df


def query_from_rows(g, inst=None):
    """Test rows of one molecule -> (cleaned query spectra for retrieval, raw spectra for the FP model, M).
    inst: instrument token for the FP / PeakFormer models instead of the rows' own (OWN_PF_INST in stage1 / predict):
    both models learned "timsTOF => Enamine chemistry" (enveda-180 is 99.9% of timsTOF training spectra), so natural
    products measured on timsTOF score far better under the generic TOF / unknown token (validation NP queries:
    PeakFormer f.z MRR 0.314 timsTOF vs 0.482 TOF / 0.480 unknown; Enamine queries the reverse, 0.826 vs 0.635 / 0.688)."""
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
        raw.append(dict(mz=mz, it=it, prec=prec, adduct=r.adduct, ce=ce,
                        inst=inst_family(r.instrument_type) if inst is None else int(inst)))
        nms.append(neutral_mass(prec, r.adduct))
    M = float(np.nanmedian(nms)) if len(nms) else np.nan
    return Q, raw, M


def stage1(R, fps, ranker, feats, g, k=30, dreams=None, pfs=None, second=None):
    """One molecule: stage-1 top-k candidate frame (all features, s1 = ranker score, s1_r = rank), one row per
    metric identity (key14), or None. dreams = (DreamsLib, {test row index: embedding}) adds the DreaMS analog
    channel (casmi/dreams_feats.py); pfs = PeakFormer scorer (casmi/peakformer.py) adds the pf_* columns;
    second = (ranker, feats) of a second (source-blind) ranker: its top-1 is flagged r2_top = 1 (and appended when
    outside the top k) for the rank-2 insertion in casmi/run_own.py."""
    Q, raw, M = query_from_rows(g, os.environ.get('OWN_PF_INST'))
    if not Q or not np.isfinite(M):
        return None
    df = R.features(Q, M)
    if len(df) == 0:
        return None
    if fps is not None:
        cos, bern = fps.score(raw, df.pid.values, df.smiles.values)
        fp_feats(df, cos, bern)
    if pfs is not None:
        cos, fz = pfs.score(raw, df.pid.values, df.smiles.values)
        fp_feats(df, cos, fz, 'pf')
    if dreams is not None:
        add_dreams(df, R, dreams, g)
    if FREL_FEATS[0] in feats:
        formula_rel(df)
    if PF_FREL[0] in feats:
        formula_rel(df, PF_FREL_SRC)
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
    top2 = None
    if second is not None:
        s1b = second[0].predict(df[second[1]].astype(float).values)
        for i in df.index.values[np.argsort(-s1b, kind='stable')]:
            kk = key14(df.smiles[i])
            if kk is not None:
                top2 = (i, kk)
                break
        if top2 is not None and top2[1] not in seen:
            keep.append(top2)
    out = df.loc[[i for i, _ in keep]].copy()
    out['key14'] = [kk for _, kk in keep]
    out['s1_r'] = np.arange(1, len(out) + 1)
    if second is not None:
        out['r2_top'] = (out.key14 == (top2[1] if top2 else None)).astype(np.int8)
    return out


def predict(R, fps, ranker, feats, g, n=25, pfs=None):
    """One molecule: ranked list of (smiles, score, key14)."""
    Q, raw, M = query_from_rows(g, os.environ.get('OWN_PF_INST'))
    if not Q or not np.isfinite(M):
        return []
    df = R.features(Q, M)
    if len(df) == 0:
        return []
    if fps is not None:
        cos, bern = fps.score(raw, df.pid.values, df.smiles.values)
        fp_feats(df, cos, bern)
    if pfs is not None:
        cos, fz = pfs.score(raw, df.pid.values, df.smiles.values)
        fp_feats(df, cos, fz, 'pf')
    if FREL_FEATS[0] in feats:
        formula_rel(df)
    if PF_FREL[0] in feats:
        formula_rel(df, PF_FREL_SRC)
    spec = os.environ.get('OWN_SPEC_ONLY')      # LB probe: order by one spectral score (e.g. pf_bern), no ranker
    df['z'] = df[spec].astype(float).values if spec else ranker.predict(df[feats].astype(float).values)
    if os.environ.get('OWN_SPEC_SRC') == 'coco':  # keep COCONUT-not-train candidates only (pool flags)
        df = df[(df.in_coco == 1) & (df.in_train == 0)]
    elif os.environ.get('OWN_SPEC_SRC') == 'cotr':  # training + COCONUT-not-train candidates (the membership contest)
        df = df[(df.in_train == 1) | ((df.in_coco == 1) & (df.in_train == 0))]
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
