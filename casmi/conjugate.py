"""Conjugate engine: candidate = acid A + partner B (amine / alcohol) joined by an amide or ester bond.

Pairs are enumerated by exact mass (mA + mB - H2O = M) over a building-block library mined from training-set
hydrolysis, then ranked by diagnostic-ion evidence for each half plus a log-frequency prior.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')

H2O = 18.010565
PROTON = 1.007276
NH3 = 17.026549
HCOOH = 46.005479
CO2 = 43.989829
OH = 17.002740

# ion offsets relative to neutral piece mass, per role and polarity
ION_POS = {
    'acid': [PROTON, PROTON - H2O, -OH + 0.0, -OH - H2O, -OH - 2 * H2O, -OH - 28.0313],  # [A+H]+, -H2O, acylium series
    'partner': [PROTON, PROTON - H2O, PROTON - 2 * H2O, PROTON - NH3, PROTON - HCOOH, PROTON - H2O - NH3],
}
ION_NEG = {
    'acid': [-PROTON, -PROTON - H2O, -PROTON - CO2],
    'partner': [-PROTON, -PROTON - H2O, -PROTON - CO2, -PROTON - 2 * H2O],
}


class BlockLib:
    def __init__(self, pieces: pd.DataFrame, min_count=2, exclude_keys=()):
        P = pieces[~pieces.key.isin(set(exclude_keys))] if len(exclude_keys) else pieces
        self.acid_cnt = P.groupby('acid').key.nunique()
        self.part_cnt = P.groupby('partner').key.nunique()
        self.acid_cnt = self.acid_cnt[self.acid_cnt >= min_count]
        self.part_cnt = self.part_cnt[self.part_cnt >= min_count]
        self.acids = self._table(self.acid_cnt)
        self.parts = self._table(self.part_cnt)

    @staticmethod
    def _table(cnt):
        smi = cnt.index.values
        mass = np.array([_mass(s) for s in smi])
        ok = np.isfinite(mass)
        o = np.argsort(mass[ok])
        smi = smi[ok][o]
        return dict(smiles=smi, mass=mass[ok][o], cnt=cnt.values[ok][o].astype(np.float64),
                    index={x: i for i, x in enumerate(smi)})

    def loo(self, pieces_of_key: pd.DataFrame):
        """Counts with one structure's own pieces removed (leave-one-out); returns (acid_cnt, part_cnt) arrays."""
        ca, cp = self.acids['cnt'].copy(), self.parts['cnt'].copy()
        for x in set(pieces_of_key.acid):
            i = self.acids['index'].get(x)
            if i is not None:
                ca[i] -= 1
        for x in set(pieces_of_key.partner):
            i = self.parts['index'].get(x)
            if i is not None:
                cp[i] -= 1
        return ca, cp


_MC = {}


def _mass(s):
    if s not in _MC:
        m = Chem.MolFromSmiles(s)
        _MC[s] = ExactMolWt(m) if m is not None else np.nan
    return _MC[s]


def ion_evidence(masses, ions, qmz, qit, tol=0.01):
    """(n_pieces, n_ion_types) matrix: max query intensity matched by each ion type of each piece."""
    ev = np.zeros((len(masses), len(ions)))
    if len(qmz) == 0:
        return ev
    for c, off in enumerate(ions):
        t = masses + off
        lo = np.searchsorted(qmz, t - tol); hi = np.searchsorted(qmz, t + tol)
        for r in np.where(hi > lo)[0]:
            ev[r, c] = qit[lo[r]:hi[r]].max()
    return ev


def candidate_pairs(lib: BlockLib, M, qmz, qit, positive=True, tol_da=0.004, top=200, w_prior=0.15, counts=None):
    """Ranked list of dict(score, acid, partner, evA, evB, cntA, cntB, evA_vec, evB_vec) for neutral mass M."""
    A, B = dict(lib.acids), dict(lib.parts)
    if counts is not None:
        A['cnt'], B['cnt'] = counts
    ions = ION_POS if positive else ION_NEG
    EA = ion_evidence(A['mass'], ions['acid'], qmz, qit)
    EB = ion_evidence(B['mass'], ions['partner'], qmz, qit)
    evA, evB = EA.sum(1), EB.sum(1)
    out = []
    target = M + H2O
    lcA, lcB = np.log1p(np.maximum(A['cnt'], 0)), np.log1p(np.maximum(B['cnt'], 0))
    for i in range(len(A['mass'])):
        need = target - A['mass'][i]
        if need < 14:
            break
        if A['cnt'][i] <= 0:
            continue
        lo = np.searchsorted(B['mass'], need - tol_da); hi = np.searchsorted(B['mass'], need + tol_da)
        for j in range(lo, hi):
            if B['cnt'][j] <= 0:
                continue
            out.append((evA[i] + evB[j] + w_prior * (lcA[i] + lcB[j]), i, j))
    out.sort(key=lambda x: -x[0])
    res = []
    for s, i, j in out[:top]:
        res.append(dict(score=s, acid=A['smiles'][i], partner=B['smiles'][j], evA=evA[i], evB=evB[j],
                        cntA=A['cnt'][i], cntB=B['cnt'][j], evA_vec=EA[i], evB_vec=EB[j],
                        mA=A['mass'][i], mB=B['mass'][j], n_pairs=len(out)))
    return res


_RX = {
    'amide': AllChem.ReactionFromSmarts('[C:1](=[O:2])[OH].[N;!H0;!$(NC=O):3]>>[C:1](=[O:2])[N:3]'),
    'ester': AllChem.ReactionFromSmarts('[C:1](=[O:2])[OH].[O;H1;!$(OC=O):3][#6:4]>>[C:1](=[O:2])[O:3][#6:4]'),
    'glyco': AllChem.ReactionFromSmarts('[C;R:1]([O;R:5])[OH].[O;H1;!$(OC=O):3][#6:4]>>[C:1]([O:5])[O:3][#6:4]'),
}


def join(acid, partner, max_products=24):
    """Products of joining acid and partner: list of (smiles, linkage)."""
    a, b = Chem.MolFromSmiles(acid), Chem.MolFromSmiles(partner)
    if a is None or b is None:
        return []
    out, seen = [], set()
    for name, rx in _RX.items():
        try:
            prods = rx.RunReactants((a, b))
        except Exception:
            continue
        for p in prods:
            try:
                m = p[0]; Chem.SanitizeMol(m); s = Chem.MolToSmiles(m)
            except Exception:
                continue
            if s not in seen:
                seen.add(s); out.append((s, name))
            if len(out) >= max_products:
                return out
    return out
