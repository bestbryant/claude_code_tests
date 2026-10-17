"""Class-3 (novel structure) candidate engine.

Two generators, both restricted to structures absent from PubChem and from the training set:
  D  derivatives of spectral analogs (transform table), ranked by modification-site consistency + analog similarity
  C  amide / ester conjugates of building blocks mined from training-set hydrolysis, ranked by half-ion evidence
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

from .analog import AnalogLib
from .conjugate import BlockLib, candidate_pairs, join
from .gen import generate
from .keyset import KeySet
from .metric import key14
from .siteloc import ParentEvidence
from .spectra import clean_peaks, neutral_mass

RDLogger.DisableLog('rdApp.*')


def std_ik14(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


class NovelEngine:
    def __init__(self, analog_lib: AnalogLib, blocks: BlockLib, pubchem: KeySet, known_keys: KeySet, key_mass: dict,
                 n_parents=20, n_pairs=40):
        self.L, self.B, self.pc, self.known = analog_lib, blocks, pubchem, known_keys
        self.key_mass = key_mass
        self.n_parents, self.n_pairs = n_parents, n_pairs

    def _novel(self, smiles_list, exclude_keys=()):
        ks = [std_ik14(s) for s in smiles_list]
        ok = [k is not None for k in ks]
        known = self.pc.contains([k or 'A' * 14 for k in ks]) | self.known.contains([k or 'A' * 14 for k in ks])
        return [o and not kn and (k not in exclude_keys) for o, kn, k in zip(ok, known, ks)], ks

    def predict(self, spectra, M=None, exclude_keys=(), loo_counts=None, allow_keys=()):
        """spectra: list of dict(mz, it, prec, adduct, positive). Returns (deriv, conj): lists of (smiles, score, key14)."""
        Q = []
        for s in spectra:
            a, b = clean_peaks(s['mz'], s['it'], s['prec'], rel_floor=0.005, top_k=48)
            if len(a) >= 3:
                Q.append(dict(mz=a, it=b, pol=0 if s['positive'] else 1, prec=s['prec'], adduct=s['adduct']))
        if M is None:
            ms = [neutral_mass(s['prec'], s['adduct']) for s in spectra]
            M = float(np.nanmedian(ms))
        deriv = self._derive(Q, M, exclude_keys, allow_keys)
        conj = self._conjugate(Q, M, exclude_keys, loo_counts, allow_keys)
        return deriv, conj, M

    def _derive(self, Q, M, exclude_keys, allow_keys):
        L = self.L
        agg = {}
        for qi, q in enumerate(Q):
            for j, s, nm in L.search(q['mz'], q['it'], M, q['pol'], exclude_keys=set(exclude_keys), top=40):
                k = L.key[j]
                if k not in agg or s > agg[k][0]:
                    agg[k] = (s, j, qi)
        parents = sorted(agg.items(), key=lambda x: -x[1][0])[:self.n_parents]
        cands = {}
        for pr, (k, (s, j, qi)) in enumerate(parents):
            pm = self.key_mass.get(k, L.nmass[j])
            prods, pmol = generate(L.smiles[j], M - pm)
            if not prods:
                continue
            a, b = L.off[j], L.off[j + 1]
            q = Q[qi]
            ev = ParentEvidence(pmol, L.mz[a:b], L.it[a:b], q['mz'], q['it'], float(M - pm), positive=(q['pol'] == 0))
            for p in prods:
                site, _ = ev.score(p['sites'], p['removed'])
                sc = site + 0.5 * s
                if p['smiles'] not in cands or sc > cands[p['smiles']][0]:
                    cands[p['smiles']] = (sc, pr, s)
        return self._finish(cands, exclude_keys, allow_keys)

    def _conjugate(self, Q, M, exclude_keys, loo_counts, allow_keys):
        pos = [q for q in Q if q['pol'] == 0]
        use = pos if pos else Q
        if not use:
            return []
        peaks = {}
        for q in use:
            for x, y in zip(np.round(q['mz'], 3), q['it']):
                peaks[x] = max(peaks.get(x, 0), y)
        qmz = np.array(sorted(peaks)); qit = np.array([peaks[x] for x in qmz])
        pairs = candidate_pairs(self.B, M, qmz, qit, positive=bool(pos), top=self.n_pairs, counts=loo_counts)
        cands = {}
        for r, p in enumerate(pairs):
            for s in join(p[1], p[2]):
                if s not in cands:
                    cands[s] = (p[0], r, 0.0)
        return self._finish(cands, exclude_keys, allow_keys)

    def _finish(self, cands, exclude_keys, allow_keys):
        if not cands:
            return []
        smis = list(cands)
        ok, ks = self._novel(smis, exclude_keys)
        allow = set(allow_keys)
        out = []
        for s, o, k in zip(smis, ok, ks):
            if o or k in allow:
                out.append((s, cands[s][0], k))
        out.sort(key=lambda x: -x[1])
        res, seen = [], set()
        for s, sc, k in out:
            mk = key14(s)
            if mk is None or mk in seen:
                continue
            seen.add(mk)
            res.append((s, sc, mk))
            if len(res) >= 60:
                break
        return res


def interleave(deriv, conj, n=25, first='d'):
    out, seen = [], set()
    i = j = 0
    turn = first
    while len(out) < n and (i < len(deriv) or j < len(conj)):
        if (turn == 'd' and i < len(deriv)) or j >= len(conj):
            s = deriv[i]; i += 1
        else:
            s = conj[j]; j += 1
        turn = 'c' if turn == 'd' else 'd'
        if s[2] not in seen:
            seen.add(s[2]); out.append(s)
    return out


def exact_masses(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(ExactMolWt(m) if m is not None else np.nan)
    return np.array(out)
