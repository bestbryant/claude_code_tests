"""Own retrieval pipeline: candidate windows over the pool + per-candidate evidence features.

Channels
  lib     weighted entropy similarity of the query to the candidate's own library spectra (class 1)
  analog  spectral analogs (modified cosine) x structural Tanimoto to the candidate (class 2)
  frag    intensity explained by 1-2 bond-cleavage fragments of the candidate
  prior   pool membership and COCONUT documentation counts
Simulation controls: `hide_keys` removes structures from library + analog search (class 2), `drop_keys` also
removes them from the candidate pool (class 3), `hide_rows` removes specific library spectra (class 1).
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .analog import AnalogLib
from .entropy import entropy_sim
from .frag import fragments, annotate

RDLogger.DisableLog('rdApp.*')
_POP8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)

PRIOR_COLS = ['n_org', 'n_coll', 'n_doi', 'n_syn', 'has_cas', 'has_name', 'ann', 'npl']
FEATS = (['lib', 'lib_n', 'lib_r', 'an_best', 'an_tan_best', 'an_sum', 'an_top1_tan', 'an_r', 'frag', 'frag_n', 'frag_r',
          'in_train', 'in_coco', 'ha', 'ppm', 'n_cand', 'n_form', 'form_frac', 'lib_gap', 'an_gap', 'frag_gap']
         + PRIOR_COLS)
PCL_COLS = ['in_pcl', 'pcl_pubmed', 'pcl_patent', 'pcl_anno', 'pcl_bio', 'pcl_food']
PC_COLS = ['pc_sid', 'pc_pmid', 'pc_patent', 'pc_ncid']      # PubChem documentation counts (log1p)


def tanimoto_packed(a, B):
    """a: (nbytes,) packed fp; B: (n, nbytes) -> Tanimoto vector."""
    inter = _POP8[np.bitwise_and(B, a)].sum(1).astype(np.float64)
    union = _POP8[np.bitwise_or(B, a)].sum(1).astype(np.float64)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


class Retrieval:
    def __init__(self, pool_dir, analog_lib: AnalogLib, prior_path=None, ppm=10.0, min_da=0.002, lib_full=None):
        self.P = pd.read_parquet(f'{pool_dir}/pool.parquet')
        self.fp = np.load(f'{pool_dir}/pool_fp.npy', mmap_mode='r')
        self.mass = self.P.mass.values
        self.key_idx = {k: i for i, k in enumerate(self.P.key.values)}
        self.L = analog_lib
        self.LF = lib_full if lib_full is not None else analog_lib     # direct library matching (more spectra)
        self.lib_of = pd.Series(np.arange(len(self.LF.key))).groupby(self.LF.key).apply(list).to_dict()
        self.ppm, self.min_da = ppm, min_da
        pri = np.zeros((len(self.P), len(PRIOR_COLS)))
        if prior_path:
            pr = pd.read_parquet(prior_path).set_index('ik14')
            j = pr.reindex(self.P.key.values)[PRIOR_COLS]
            pri = np.log1p(j.fillna(0).clip(lower=0).values.astype(float))
            pri[:, PRIOR_COLS.index('npl')] = j['npl'].fillna(0).values
        self.prior = pri
        self._frag_cache = {}

    def window(self, M):
        tol = max(M * self.ppm * 1e-6, self.min_da)
        lo, hi = np.searchsorted(self.mass, M - tol), np.searchsorted(self.mass, M + tol)
        return np.arange(lo, hi)

    def _frags(self, i):
        f = self._frag_cache.get(i)
        if f is None:
            m = Chem.MolFromSmiles(self.P.smiles.values[i])
            f = fragments(m) if m is not None and m.GetNumHeavyAtoms() <= 80 else []
            if len(self._frag_cache) > 20000:
                self._frag_cache.clear()
            self._frag_cache[i] = f
        return f

    def features(self, Q, M, hide_keys=(), drop_keys=(), hide_rows=(), n_analog=20, an_pow=3.0, do_frag=True):
        """Q: list of cleaned query spectra dict(mz, it, pol). Returns DataFrame (one row per candidate)."""
        idx = self.window(M)
        if drop_keys:
            keep = ~np.isin(self.P.key.values[idx], list(drop_keys))
            idx = idx[keep]
        n = len(idx)
        if n == 0:
            return pd.DataFrame(columns=['pid', 'key', 'smiles'] + FEATS)
        keys = self.P.key.values[idx]
        L = self.L
        hide = set(hide_keys) | set(drop_keys)
        hrows = set(hide_rows)
        # --- lib channel
        lib = np.zeros(n); lib_n = np.zeros(n)
        for c, k in enumerate(keys):
            if k in hide:
                continue
            LF = self.LF
            for j in self.lib_of.get(k, ()):
                if int(LF.row[j]) in hrows:
                    continue
                a, b = LF.off[j], LF.off[j + 1]
                for q in Q:
                    if q['pol'] != LF.pol[j]:
                        continue
                    s = entropy_sim(q['mz'], q['it'], LF.mz[a:b], LF.it[a:b], 0.01)
                    if s > lib[c]:
                        lib[c] = s
                    lib_n[c] += 1
        # --- analog channel
        agg = {}
        for q in Q:
            for j, s, nm in L.search(q['mz'], q['it'], M, q['pol'], exclude_keys=hide, top=40):
                if int(L.row[j]) in hrows:
                    continue
                k = L.key[j]
                if k not in agg or s > agg[k]:
                    agg[k] = s
        an = sorted(agg.items(), key=lambda x: -x[1])[:n_analog]
        an_best = np.zeros(n); an_tan_best = np.zeros(n); an_sum = np.zeros(n); an_top1 = np.zeros(n)
        if an:
            C = np.asarray(self.fp[idx])
            for r, (k, s) in enumerate(an):
                pi = self.key_idx.get(k)
                if pi is None:
                    continue
                t = tanimoto_packed(np.asarray(self.fp[pi]), C)
                v = (s ** an_pow) * t
                an_best = np.maximum(an_best, v); an_tan_best = np.maximum(an_tan_best, t)
                an_sum += v
                if r == 0:
                    an_top1 = t
        # --- frag channel
        frag = np.zeros(n); frag_n = np.zeros(n)
        if do_frag:
            peaks = {}
            pos = [q for q in Q if q['pol'] == 0]
            use = pos if pos else Q
            for q in use:
                for x, y in zip(np.round(q['mz'], 3), q['it']):
                    peaks[x] = max(peaks.get(x, 0.0), y)
            qmz = np.array(sorted(peaks)); qit = np.array([peaks[x] for x in qmz]); tot = (qit ** 2).sum()
            keepq = qmz < M - 0.5            # ignore precursor region
            qmz, qit = qmz[keepq], qit[keepq]
            for c in range(n):
                f = self._frags(idx[c])
                if not f or len(qmz) == 0:
                    continue
                ann = annotate(f, qmz, positive=bool(pos), tol=0.01, hshift=(-2, -1, 0, 1))
                hit = np.array([len(a) > 0 for a in ann])
                frag[c] = (qit[hit] ** 2).sum() / max(tot, 1e-9); frag_n[c] = hit.mean()
        # --- assemble
        df = pd.DataFrame(dict(pid=idx, key=keys, smiles=self.P.smiles.values[idx], lib=lib, lib_n=lib_n,
                               an_best=an_best, an_tan_best=an_tan_best, an_sum=an_sum, an_top1_tan=an_top1,
                               frag=frag, frag_n=frag_n, in_train=self.P.in_train.values[idx],
                               in_coco=self.P.in_coco.values[idx], ha=self.P.ha.values[idx],
                               ppm=(self.mass[idx] - M) / M * 1e6, formula=self.P.formula.values[idx]))
        if hide:
            # a hidden structure stands in for a test molecule without public spectra: it must not look like a
            # training structure (class-2 answers are never in train.parquet)
            df.loc[df.key.isin(hide), 'in_train'] = 0
        for c, name in enumerate(PRIOR_COLS):
            df[name] = self.prior[idx, c]
        for c in PC_COLS:            # PubChem documentation counts per connectivity key (stored as log1p)
            if c in self.P.columns:
                df[c] = self.P[c].values[idx].astype(float)
        if 'in_bio' in self.P.columns:   # ChEBI / LIPID MAPS membership (pool v4)
            df['in_bio'] = self.P.in_bio.values[idx].astype(float)
        for c in PCL_COLS:           # PubChemLite membership and annotation priors (pool v2)
            if c in self.P.columns:
                v = self.P[c].values[idx].astype(float)
                df[c] = np.log1p(v) if c in ('pcl_pubmed', 'pcl_patent') else v
        df['n_cand'] = n
        fc = df.formula.map(df.formula.value_counts())
        df['n_form'] = fc; df['form_frac'] = fc / n
        for ch in ('lib', 'an_best', 'frag'):
            r = df[ch].rank(ascending=False, method='min')
            df[ch.split('_')[0] + '_r'] = r
            df[ch.split('_')[0] + '_gap'] = df[ch] - df[ch].max()
        return df
