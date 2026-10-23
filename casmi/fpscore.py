"""Score candidates with a trained SparseDualEncoder: embedding cosine and fingerprint Bernoulli log-likelihood.

FP_PMI=<npy (2, N_BITS): log m, log(1 - m)> (scripts/fp_bitprior.py): the Bernoulli score becomes a PMI,
log p(bits | spectrum) - FP_PMI_LAM * log p(bits | training-positive marginal). The bit head is trained by BCE on positives
only, so its raw likelihood favours candidates that look like a typical training structure (docs/plan.md, Oct 23)."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys

from .fpmodel import SparseDualEncoder, sparse_spec, sparse_bits, N_BITS, NB_FRAG, NB_NL

RDLogger.DisableLog('rdApp.*')
ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-', '[M-H2O-H]-',
       '[M-2H2O+H]+', '[2M+H]+', '[2M+Na]+', '[2M-H]-', '[M]+', '[M+C2H4O2-H]-']


def top_peaks(mz, it, prec, k=64):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    keep = mz <= prec + 2.0
    mz, it = mz[keep], it[keep]
    if len(mz) > k:
        o = np.argpartition(-it, k)[:k]
        mz, it = mz[o], it[o]
    o = np.argsort(mz)
    return mz[o].astype(np.float32), it[o].astype(np.float32)


class FPScorer:
    def __init__(self, weights, pool_fp=None, device='cpu'):
        sd = torch.load(weights, map_location=device)
        self.fine = sd['peaks.weight'].shape[0] > NB_FRAG + NB_NL          # 0.1 Da bins present
        self.m = SparseDualEncoder(d=sd['peaks.weight'].shape[1], emb=sd['spec_emb.weight'].shape[0], fine=self.fine)
        self.m.load_state_dict(sd)
        self.m.eval()
        self.pool_fp = pool_fp          # packed Morgan 2048 per pool row (MACCS added on the fly)
        self._mac = {}
        pmi = os.environ.get('FP_PMI')
        self.prior = np.load(pmi) * float(os.environ.get('FP_PMI_LAM', '1')) if pmi else None

    def _maccs(self, pid, smiles):
        v = self._mac.get(pid)
        if v is None:
            m = Chem.MolFromSmiles(smiles)
            v = np.array(list(MACCSkeys.GenMACCSKeys(m)), np.uint8) if m is not None else np.zeros(167, np.uint8)
            if len(self._mac) > 300000:
                self._mac.clear()
            self._mac[pid] = v
        return v

    def mol_bits(self, pids, smiles):
        morgan = np.unpackbits(np.asarray(self.pool_fp[pids]), axis=1)
        mac = np.stack([self._maccs(p, s) for p, s in zip(pids, smiles)])
        return np.packbits(np.concatenate([morgan, mac], 1), axis=1)

    def spec_embed(self, spectra):
        """spectra: list of dict(mz, it, prec, adduct, ce, inst) raw peaks -> (mean emb (1, d), mean bit probs)."""
        mzs, its, offs, prec, ce, add, inst, pol = [], [], [0], [], [], [], [], []
        for s in spectra:
            a, b = top_peaks(s['mz'], s['it'], s['prec'])
            mzs.append(a); its.append(b); offs.append(offs[-1] + len(a))
            prec.append(s['prec']); ce.append(s.get('ce', -1.0)); add.append(ADD.index(s['adduct']) if s['adduct'] in ADD else 0)
            inst.append(s.get('inst', 1)); pol.append(0 if s['adduct'].endswith('+') else 1)
        idx = np.arange(len(spectra))
        ids, bo, wts, meta = sparse_spec(np.concatenate(mzs), np.concatenate(its), np.array(offs), idx,
                                         np.array(prec, np.float32), np.array(ce, np.float32), np.array(add, np.int8),
                                         np.array(inst, np.int8), np.array(pol, np.int8), fine=self.fine)
        with torch.no_grad():
            e, logits = self.m.encode_spec(torch.from_numpy(ids), torch.from_numpy(bo), torch.from_numpy(wts),
                                           torch.from_numpy(meta))
            e = F.normalize(e.mean(0, keepdim=True), dim=-1)
            lp = F.logsigmoid(logits).mean(0).numpy(); lq = F.logsigmoid(-logits).mean(0).numpy()
        return e, lp, lq

    def score(self, spectra, pids, smiles):
        e, lp, lq = self.spec_embed(spectra)
        packed = self.mol_bits(pids, smiles)
        ids, bo = sparse_bits(packed)
        with torch.no_grad():
            m = self.m.encode_mol(torch.from_numpy(ids), torch.from_numpy(bo))
            cos = (e @ m.T).squeeze(0).numpy()
        bits = np.unpackbits(packed, axis=1)[:, :N_BITS].astype(np.float32)
        if self.prior is not None:
            lp, lq = lp - self.prior[0], lq - self.prior[1]
        bern = bits @ lp + (1 - bits) @ lq
        return cos, bern
