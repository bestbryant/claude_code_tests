"""Zenodo-extra library channel: the 3,717 Enveda-180 structures (Zenodo 21346580, CC BY 4.0) absent from
train.parquet, matched directly by spectrum (dataset bryantyu/casmi26-zen-extra, kernels/casmi26-zen-probe).

ZenLib(path).match(test_rows) -> [(similarity, smiles)] best first: structures within 10 ppm of the molecule's
neutral mass, scored by the best weighted entropy similarity between any test spectrum and any of the structure's
spectra of the same polarity. insert_zen() puts the best match at rank 1 when similarity >= t1, at rank 3 when >= t2
(defaults 0.7 / 0.5: on placeholder molecules that are not Zenodo extras the best similarity is 0.04 median, 0.35 at
the 90th and 0.65 at the 99th percentile).
"""
import numpy as np
from .entropy import entropy_sim
from .spectra import clean_peaks, neutral_mass


class ZenLib:
    def __init__(self, path):
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem.Descriptors import ExactMolWt
        Z = pd.read_parquet(path)
        S = Z.drop_duplicates('ik14')[['ik14', 'smiles']].reset_index(drop=True)
        mols = [Chem.MolFromSmiles(s) for s in S.smiles]
        S['mass'] = [ExactMolWt(m) if m is not None else np.nan for m in mols]
        self.S = S.dropna().sort_values('mass').reset_index(drop=True)
        pol = (~Z.ionmode.str.lower().str.contains('positive')).astype(int).values
        self.spec = {}
        for r, p in zip(Z.itertuples(), pol):
            try:
                pm = float(str(r.pepmass).split()[0])
            except ValueError:
                continue
            a, b = clean_peaks(np.asarray(r.mzs, float), np.asarray(r.intensities, float), pm, rel_floor=0.005, top_k=64)
            if len(a) >= 3:
                o = np.argsort(a); self.spec.setdefault(r.ik14, []).append((int(p), a[o], b[o]))

    def match(self, g):
        Ms = [neutral_mass(float(r.precursor_mz), r.adduct) for r in g.itertuples()]
        Ms = [x for x in Ms if np.isfinite(x)]
        if not Ms:
            return []
        M = float(np.median(Ms)); tol = max(M * 1e-5, 0.002)
        lo, hi = np.searchsorted(self.S.mass.values, M - tol), np.searchsorted(self.S.mass.values, M + tol)
        Q = []
        for r in g.itertuples():
            a, b = clean_peaks(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float),
                               float(r.precursor_mz), rel_floor=0.005, top_k=64)
            if len(a) >= 3:
                o = np.argsort(a); Q.append((0 if r.ionization_mode == 'positive' else 1, a[o], b[o]))
        out = []
        for j in range(lo, hi):
            best = 0.0
            for qp, qa, qb in Q:
                for zp, za, zb in self.spec.get(self.S.ik14[j], ()):
                    if zp == qp:
                        best = max(best, entropy_sim(qa, qb, za, zb, 0.01))
            out.append((best, self.S.smiles[j]))
        return sorted(out, reverse=True)


def insert_zen(final, matches, t1=0.7, t2=0.5, key14=None):
    """Best Zenodo match at rank 1 (similarity >= t1) or rank 3 (>= t2); final: [[smiles, score, key14], ...]."""
    if not matches or matches[0][0] < t2:
        return final
    sim, s = matches[0]
    kk = key14(s) if key14 else None
    out = [r for r in final if r[2] != kk]
    out.insert(0 if sim >= t1 else min(2, len(out)), [s, float(sim), kk])
    return out[:25]
