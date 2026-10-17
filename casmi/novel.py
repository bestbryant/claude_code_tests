"""Class-3 (novel structure) candidate engine.

Two generators, both restricted to structures absent from PubChem and from the training set:
  d  derivatives of spectral analogs (transform table), with modification-site consistency evidence
  c  amide / ester conjugates of building blocks mined from training-set hydrolysis, with half-ion evidence
Every candidate carries a feature row; `rank` orders them with a learned model or the heuristic fallback.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .analog import AnalogLib
from .conjugate import BlockLib, candidate_pairs, join
from .derive import _TRANSFORMS
from .gen import generate
from .keyset import KeySet
from .metric import key14
from .siteloc import ParentEvidence
from .spectra import clean_peaks, neutral_mass

RDLogger.DisableLog('rdApp.*')

D_FEATS = ['psim', 'prank', 'pnm', 'site', 'site_tot', 'n_shift', 'n_unshift', 'steps', 'n_prod_parent', 'abs_delta',
           'delta_sign', 'parent_ha', 'n_parents', 'best_psim', 'psim_gap']
C_FEATS = ['cscore', 'evA', 'evB', 'cntA', 'cntB', 'pair_rank', 'n_pairs', 'is_amide', 'mA', 'mB', 'n_prod_pair'] + \
          [f'eA{i}' for i in range(6)] + [f'eB{i}' for i in range(6)]
Q_FEATS = ['M', 'n_spec', 'has_pos', 'is_d', 'ha', 'n_d', 'n_c']
T_FEATS = [f't_{n}' for n in _TRANSFORMS]
FEATS = Q_FEATS + D_FEATS + C_FEATS + T_FEATS


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
                 n_parents=20, n_pairs=40, model=None):
        self.L, self.B, self.pc, self.known = analog_lib, blocks, pubchem, known_keys
        self.key_mass = key_mass
        self.n_parents, self.n_pairs = n_parents, n_pairs
        self.model = model

    def candidates(self, spectra, M=None, exclude_keys=(), loo_counts=None, allow_keys=()):
        """spectra: list of dict(mz, it, prec, adduct, positive). Returns DataFrame of novel candidates with features."""
        Q = []
        for s in spectra:
            a, b = clean_peaks(s['mz'], s['it'], s['prec'], rel_floor=0.005, top_k=48)
            if len(a) >= 3:
                Q.append(dict(mz=a, it=b, pol=0 if s['positive'] else 1, prec=s['prec'], adduct=s['adduct']))
        if M is None:
            M = float(np.nanmedian([neutral_mass(s['prec'], s['adduct']) for s in spectra]))
        d = self._derive(Q, M, exclude_keys)
        c = self._conjugate(Q, M, loo_counts)
        rows = d + c
        if not rows:
            return pd.DataFrame(columns=['smiles', 'key', 'src'] + FEATS)
        df = pd.DataFrame(rows)
        df = self._filter(df, exclude_keys, allow_keys)
        if len(df) == 0:
            return df
        df['M'] = M; df['n_spec'] = len(Q); df['has_pos'] = float(any(q['pol'] == 0 for q in Q))
        df['is_d'] = (df.src == 'd').astype(float)
        df['n_d'] = float((df.src == 'd').sum()); df['n_c'] = float((df.src == 'c').sum())
        for f in FEATS:
            if f not in df:
                df[f] = np.nan
        df[T_FEATS] = df[T_FEATS].fillna(0)
        return df

    def _derive(self, Q, M, exclude_keys):
        L = self.L
        agg = {}
        for qi, q in enumerate(Q):
            for j, s, nm in L.search(q['mz'], q['it'], M, q['pol'], exclude_keys=set(exclude_keys), top=40):
                k = L.key[j]
                if k not in agg or s > agg[k][0]:
                    agg[k] = (s, j, qi, nm)
        parents = sorted(agg.items(), key=lambda x: -x[1][0])[:self.n_parents]
        best = parents[0][1][0] if parents else 0.0
        rows = {}
        for pr, (k, (s, j, qi, nm)) in enumerate(parents):
            pm = self.key_mass.get(k, L.nmass[j])
            prods, pmol = generate(L.smiles[j], M - pm)
            if not prods:
                continue
            a, b = L.off[j], L.off[j + 1]
            q = Q[qi]
            ev = ParentEvidence(pmol, L.mz[a:b], L.it[a:b], q['mz'], q['it'], float(M - pm), positive=(q['pol'] == 0),
                                cache_key=L.smiles[j])
            for p in prods:
                site, tot = ev.score(p['sites'], p['removed'])
                r = dict(smiles=p['smiles'], src='d', psim=s, prank=pr, pnm=nm, site=site, site_tot=tot,
                         n_shift=ev.n_shift, n_unshift=ev.n_unshift, steps=len(p['combo']), n_prod_parent=len(prods),
                         abs_delta=abs(M - pm), delta_sign=np.sign(M - pm), parent_ha=pmol.GetNumHeavyAtoms(),
                         n_parents=len(parents), best_psim=best, psim_gap=best - s, combo='+'.join(p['combo']))
                for t in p['combo']:
                    r[f't_{t}'] = r.get(f't_{t}', 0) + 1
                h = site + 0.5 * s
                if p['smiles'] not in rows or h > rows[p['smiles']]['h']:
                    r['h'] = h; rows[p['smiles']] = r
        return list(rows.values())

    def _conjugate(self, Q, M, loo_counts):
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
        rows = {}
        for r_, p in enumerate(pairs):
            prods = join(p['acid'], p['partner'])
            for smi, link in prods:
                if smi in rows:
                    continue
                r = dict(smiles=smi, src='c', cscore=p['score'], evA=p['evA'], evB=p['evB'], cntA=np.log1p(p['cntA']),
                         cntB=np.log1p(p['cntB']), pair_rank=r_, n_pairs=p['n_pairs'], is_amide=float(link == 'amide'),
                         mA=p['mA'], mB=p['mB'], n_prod_pair=len(prods), h=p['score'] - 0.3 - 0.02 * r_)
                for i, v in enumerate(p['evA_vec']):
                    r[f'eA{i}'] = v
                for i, v in enumerate(p['evB_vec']):
                    r[f'eB{i}'] = v
                rows[smi] = r
        return list(rows.values())

    def _filter(self, df, exclude_keys, allow_keys):
        ks = [std_ik14(s) for s in df.smiles]
        df = df.assign(std=ks)
        df = df[df['std'].notna()]
        known = self.pc.contains(list(df['std'])) | self.known.contains(list(df['std']))
        allow = np.array([k in set(allow_keys) for k in df['std']], bool)
        excl = np.array([k in set(exclude_keys) for k in df['std']], bool)
        df = df[(~known | allow) & (~excl | allow)].copy()
        df['ha'] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in df.smiles]
        df['key'] = [key14(s) for s in df.smiles]
        df = df[df.key.notna()].sort_values('h', ascending=False).drop_duplicates('key')
        return df

    def rank(self, df, n=25):
        if len(df) == 0:
            return df
        if self.model is not None:
            df = df.assign(z=self.model.predict(df[FEATS].astype(float).values))
        else:
            df = df.assign(z=df.h)
        return df.sort_values('z', ascending=False).head(n)


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
