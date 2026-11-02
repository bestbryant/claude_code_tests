"""Engine 2 clone: the FP channel adapter (docs/engine2_study.md option R step 2, change (b)). Our own code.

Engine 2's FPNet is replaced by our PeakFormer: the vendored rank_features takes model_raw = the per-candidate f.z of
pf_tpu_f (patches P1-P5, E16-E20), and compute_channels asks fp_raw(mid, cand, sub) for it. JoinedPF is that callable
for the replay:
  mode 'join'   every candidate present in our ret18cu rows takes its exact ret18cu pf_bern (looked up by (qid, pid));
                absent candidates are scored on the fly with the same PeakFormer;
  mode 'score'  every candidate is scored (the PF reproduction check, and PF-x later).
The query spectra for scoring are built exactly as scripts/add_fp_feats.py L35-41 builds them, over ALL q_rows of the
query: store m/z (float32 -> float64), store intensity (float16 -> float64), the RAW precursor m/z (no electron fix, as in
ret18cu), adduct, mean collision energy (-1 if absent) and the row's own instrument family (OWN_PF_INST is unset in
ret18cu). A single checkpoint gives the raw f.z of casmi/peakformer.py (PFEnsemble with one path == PFScorer);
rows and serving must use the same checkpoint set, because PFEnsemble z-scores per query for several checkpoints.
This module imports torch only inside the default scorer factory (lazily, in the worker that needs it).
"""
import numpy as np

from casmi.prep_meta import inst_family


def pf_factory(paths, threads=1):
    """Default scorer factory: PFEnsemble over the given checkpoint path(s), created lazily, torch on 1 thread."""
    paths = [paths] if isinstance(paths, str) else list(paths)

    def make():
        import torch
        torch.set_num_threads(threads)
        from casmi.peakformer import PFEnsemble
        return PFEnsemble(paths)
    return make


class QSpec:
    """The e2qspec_<split>.npz pack (distinct q_rows; store peaks + metadata), indexed by training row."""

    FIELDS = ('row', 'off', 'mz', 'it', 'prec', 'ik', 'npk', 'adduct', 'ingest_lib', 'instrument_type', 'ce')

    def __init__(self, z):
        self.z = {k: np.asarray(z[k]) for k in self.FIELDS}
        self.row = self.z['row']
        assert np.all(np.diff(self.row) > 0), 'qspec rows must be sorted and unique'

    def idx(self, rows):
        rows = np.asarray(rows, np.int64)
        j = np.searchsorted(self.row, rows)
        if np.any(j >= len(self.row)) or np.any(self.row[np.minimum(j, len(self.row) - 1)] != rows):
            raise KeyError(f'rows missing from qspec: {rows[(j >= len(self.row)) | (self.row[np.minimum(j, len(self.row) - 1)] != rows)][:5]}')
        return j

    def peaks(self, row):
        j = int(self.idx([row])[0])
        a, b = self.z['off'][j], self.z['off'][j + 1]
        return self.z['mz'][a:b], self.z['it'][a:b]

    def get(self, field, row):
        return self.z[field][int(self.idx([row])[0])]


def pf_spectra(qs, q_rows):
    """Query spectra for PeakFormer scoring (scripts/add_fp_feats.py L35-41), in q_rows order."""
    out = []
    for i in q_rows:
        j = int(qs.idx([i])[0])
        a, b = qs.z['off'][j], qs.z['off'][j + 1]
        out.append(dict(mz=np.asarray(qs.z['mz'][a:b], np.float64), it=np.asarray(qs.z['it'][a:b], np.float64),
                        prec=float(qs.z['prec'][j]), adduct=str(qs.z['adduct'][j]), ce=float(qs.z['ce'][j]),
                        inst=inst_family(str(qs.z['instrument_type'][j]) or None)))
    return out


class JoinedPF:
    """fp_raw(qid, cand, sub=None) -> float32[len(cand)] PeakFormer f.z (see the module docstring).

    pf_df: columns qid, pid, pf_bern sorted by (qid, pid) (e2pf_<split>.parquet); qspec: QSpec; qrows: {qid: [rows]};
    bank_smiles: SMILES per bank row (= pool row); scorer_factory() -> object with .score(spectra, pids, smiles) ->
    (cos, fz)."""

    def __init__(self, pf_df, qspec, bank_smiles, scorer_factory, mode='join', qrows=None):
        if mode not in ('join', 'score'):
            raise ValueError(mode)
        self.q = np.asarray(pf_df['qid'], np.int64); self.p = np.asarray(pf_df['pid'], np.int64)
        self.v = np.asarray(pf_df['pf_bern'], np.float32)
        key = self.q * (1 << 32) + self.p
        if len(key) > 1 and not np.all(np.diff(key) > 0):
            raise ValueError('pf_df must be sorted by (qid, pid) without duplicates')
        self.qs, self.smiles, self.factory, self.mode = qspec, bank_smiles, scorer_factory, mode
        self.qrows = qrows or {}
        self._scorer = None
        self.stats, self.src = {}, {}

    def scorer(self):
        if self._scorer is None:
            self._scorer = self.factory()
        return self._scorer

    def spectra(self, qid):
        return pf_spectra(self.qs, self.qrows[int(qid)])

    def lookup(self, qid, cand):
        """(values float32, found bool) of the ret18cu pf_bern for (qid, cand)."""
        cand = np.asarray(cand, np.int64)
        a, b = np.searchsorted(self.q, qid, 'left'), np.searchsorted(self.q, qid, 'right')
        pp, vv = self.p[a:b], self.v[a:b]
        j = np.searchsorted(pp, cand)
        jc = np.minimum(j, max(len(pp) - 1, 0))
        found = (j < len(pp)) & (pp[jc] == cand) if len(pp) else np.zeros(len(cand), bool)
        val = np.zeros(len(cand), np.float32)
        val[found] = vv[jc[found]]
        return val, found

    def __call__(self, qid, cand, sub=None):
        qid = int(qid); cand = np.asarray(cand, np.int64)
        if self.mode == 'join':
            val, found = self.lookup(qid, cand)
        else:
            val, found = np.zeros(len(cand), np.float32), np.zeros(len(cand), bool)
        miss = np.flatnonzero(~found)
        if len(miss):
            pids = cand[miss]
            _, fz = self.scorer().score(self.spectra(qid), pids, np.asarray(self.smiles, dtype=object)[pids])
            fz = np.asarray(fz, np.float32)
            assert fz.shape == (len(miss),), (fz.shape, len(miss))
            val[miss] = fz
        self.stats[qid] = (int(found.sum()), int(len(miss)))
        self.src[qid] = (~found).astype(np.int8)
        return val
