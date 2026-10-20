"""Peak-token transformer spectrum -> structure scorer ("PeakFormer").

Each of the (up to 64) peaks is a token built from sinusoidal embeddings of its exact m/z and of its neutral loss
(precursor m/z - m/z), wavelengths 0.01 .. 1000 Da (phases reduced in float64, so 0.01 Da structure survives), plus
its intensity. A global token carries the precursor m/z (same sinusoids) and the metadata (adduct, polarity,
collision energy, instrument family, number of merged spectra). Transformer encoder -> global token ->
  z    logits over the multi-block structure fingerprint (casmi/richfp.py, kept bits); a candidate with on-bits f
       scores f . z (its Bernoulli log-likelihood up to a candidate-independent constant)
  emb  a spectrum embedding; candidates are embedded from the same bits (EmbeddingBag MLP), score = cosine.
Both scores are trained against same-mass (10 ppm window) pool negatives and in-batch negatives
(scripts/train_pf.py), plus a fingerprint BCE on z.

Views at inference: every spectrum on its own plus, per adduct group with >= 2 spectra, merged spectra of up to 4
(union of the max-normalised peak lists, top 64) - the merge augmentation used in training. z and the embedding of
the query are the means over its views.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 64
N_ADD, N_INST = 15, 6
N_META = N_ADD + 2 + 2 + N_INST + 2          # adduct, polarity, (ce, ce known), instrument, (prec / 1000, merged)
ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-', '[M-H2O-H]-',
       '[M-2H2O+H]+', '[2M+H]+', '[2M+Na]+', '[2M-H]-', '[M]+', '[M+C2H4O2-H]-']


def meta_matrix(add_id, pol, ce, inst_id, prec, n_merged):
    """(n, N_META) float32 metadata rows (numpy inputs of length n)."""
    n = len(add_id)
    X = np.zeros((n, N_META), np.float32)
    r = np.arange(n)
    X[r, np.clip(add_id.astype(np.int64), 0, N_ADD - 1)] = 1.0
    o = N_ADD
    X[r, o + pol.astype(np.int64)] = 1.0
    o += 2
    X[:, o] = np.where(ce >= 0, np.minimum(ce, 150) / 100.0, 0.0)
    X[:, o + 1] = (ce >= 0).astype(np.float32)
    o += 2
    X[r, o + np.clip(inst_id.astype(np.int64), 0, N_INST - 1)] = 1.0
    o += N_INST
    X[:, o] = prec / 1000.0
    X[:, o + 1] = (np.asarray(n_merged, np.float32) - 1) / 3.0
    return X


def pad_peaks(mz_list, it_list, prec):
    """Lists of raw peak arrays -> (n, K) float64 m/z, float32 relative intensity (max 1), bool mask; peaks above
    precursor + 2 dropped, top K by intensity kept."""
    n = len(mz_list)
    M = np.zeros((n, K), np.float64); I = np.zeros((n, K), np.float32); Z = np.zeros((n, K), bool)
    for j, (mz, it) in enumerate(zip(mz_list, it_list)):
        mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
        keep = (mz <= prec[j] + 2.0) & (it > 0) & (mz > 0)
        mz, it = mz[keep], it[keep]
        if len(mz) > K:
            o = np.argpartition(-it, K)[:K]
            mz, it = mz[o], it[o]
        if len(mz) == 0:
            continue
        M[j, :len(mz)] = mz; I[j, :len(mz)] = it / it.max(); Z[j, :len(mz)] = True
    return M, I, Z


def merge_peaks(mz_list, it_list):
    """Merged view of several spectra: their max-normalised peak lists concatenated (pad_peaks then keeps the top K
    by intensity), as the merge augmentation in scripts/train_pf.py does."""
    mz = np.concatenate([np.asarray(m, np.float64) for m in mz_list])
    it = np.concatenate([np.asarray(i, np.float64) / max(float(np.max(i)) if len(i) else 1.0, 1e-12) for i in it_list])
    return mz, it


class PeakFormer(nn.Module):
    def __init__(self, n_bits, d=256, layers=6, heads=8, emb=256, d_mol=512, n_freq=64, drop=0.1):
        super().__init__()
        lam = np.logspace(-2, 3, n_freq)
        self.register_buffer('freq', torch.tensor(1.0 / lam, dtype=torch.float64), persistent=False)
        self.mz_proj = nn.Linear(2 * n_freq, d)
        self.nl_proj = nn.Linear(2 * n_freq, d)
        self.int_proj = nn.Linear(2, d)
        self.prec_proj = nn.Linear(2 * n_freq, d)
        self.meta_proj = nn.Linear(N_META, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.spec_emb = nn.Linear(d, emb)
        self.fp_head = nn.Linear(d, n_bits)
        self.bits = nn.EmbeddingBag(n_bits, d_mol, mode='sum')
        self.mol = nn.Sequential(nn.GELU(), nn.Linear(d_mol, d_mol), nn.GELU(), nn.Linear(d_mol, emb))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07), dtype=torch.float32))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.1), dtype=torch.float32))   # scale of f . z

    def _sin(self, x):
        """x float64 (...,) masses -> (..., 2 n_freq) float32 sinusoids (phase reduced in float64)."""
        cyc = x.unsqueeze(-1) * self.freq
        ph = ((2 * math.pi) * (cyc - torch.floor(cyc))).float()
        return torch.cat([torch.sin(ph), torch.cos(ph)], -1)

    def encode_spec(self, mz, it, mask, prec, meta):
        """mz (B, K) float64, it (B, K) relative intensity, mask (B, K) bool valid, prec (B,) float64, meta
        (B, N_META) -> (normalised embedding (B, emb), bit logits z (B, n_bits)), both float32."""
        nl = prec.unsqueeze(1) - mz
        ints = torch.stack([it.sqrt(), torch.log(it.clamp_min(1e-4)) / 4 + 1], -1)
        tok = self.mz_proj(self._sin(mz)) + self.nl_proj(self._sin(nl)) + self.int_proj(ints)
        g = self.meta_proj(meta) + self.prec_proj(self._sin(prec))
        x = torch.cat([g.unsqueeze(1), tok], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        h = self.norm(self.enc(x, src_key_padding_mask=pad)[:, 0])
        return F.normalize(self.spec_emb(h).float(), dim=-1), self.fp_head(h).float()

    def encode_mol(self, idx, off):
        """Kept on-bit ids (flat int64) + bag offsets -> normalised molecule embeddings."""
        return F.normalize(self.mol(self.bits(idx, off)).float(), dim=-1)


def csr_rows(ptr, idx, rows):
    """CSR rows -> (flat int64 bit ids, int64 bag offsets, int64 row id per bit) for the selected rows."""
    starts, ends = ptr[rows], ptr[rows + 1]
    lens = ends - starts
    rid = np.repeat(np.arange(len(rows)), lens)
    pos = np.repeat(starts - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
    off = np.r_[0, np.cumsum(lens)[:-1]]
    return idx[pos].astype(np.int64), off.astype(np.int64), rid.astype(np.int64)


def fz_scores(z, bit_ids, row_of_bit, z_row, n_rows):
    """f . z per candidate: z (Q, n_bits); candidate c belongs to query z_row[c]; bit_ids / row_of_bit are the flat
    on-bit ids and their candidate index. -> (n_rows,)."""
    g = z.reshape(-1).gather(0, z_row[row_of_bit] * z.shape[1] + bit_ids)
    return torch.zeros(n_rows, dtype=z.dtype, device=z.device).index_add_(0, row_of_bit, g)


def views(spectra, merged=True):
    """Per-spectrum views plus (merged=True) merged views of up to 4 spectra per adduct group with >= 2 spectra."""
    mzl, itl, prec, add, pol, ce, inst, nm = [], [], [], [], [], [], [], []
    for s in spectra:
        mzl.append(s['mz']); itl.append(s['it']); prec.append(s['prec'])
        add.append(ADD.index(s['adduct']) if s['adduct'] in ADD else 0)
        pol.append(0 if str(s['adduct']).endswith('+') else 1)
        ce.append(s.get('ce', -1.0)); inst.append(s.get('inst', 1)); nm.append(1)
    groups = {}
    for j, a in enumerate(add):
        groups.setdefault(a, []).append(j)
    for a, jj in groups.items():
        if len(jj) < 2 or not merged:
            continue
        for c in range(0, len(jj), 4):
            part = jj[c:c + 4]
            if len(part) < 2:
                continue
            m, i = merge_peaks([mzl[j] for j in part], [itl[j] for j in part])
            mzl.append(m); itl.append(i); prec.append(float(np.median([prec[j] for j in part]))); add.append(a)
            pol.append(pol[part[0]])
            ces = [ce[j] for j in part if ce[j] >= 0]
            ce.append(float(np.mean(ces)) if ces else -1.0); inst.append(inst[part[0]]); nm.append(len(part))
    prec = np.array(prec, np.float64)
    M, I, Z = pad_peaks(mzl, itl, prec)
    meta = meta_matrix(np.array(add), np.array(pol), np.array(ce, np.float32), np.array(inst), prec, nm)
    return M, I, Z, prec, meta


def embed_query(model, spectra, dev='cpu', merged=True):
    """Spectra of one query (dicts mz, it, prec, adduct, ce, inst) -> (normalised mean view embedding (1, emb),
    mean bit logits (n_bits,))."""
    M, I, Z, prec, meta = views(spectra, merged)
    ok = Z.any(1)
    if not ok.any():
        ok[:] = True
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a[ok])).to(dev)
    with torch.no_grad():
        e, z = model.encode_spec(t(M), t(I), t(Z), t(prec), t(meta))
    return F.normalize(e.mean(0, keepdim=True), dim=-1), z.mean(0)


class PFScorer:
    """Like casmi.fpscore.FPScorer: score(spectra, pids, smiles) -> (cosine, f . z). Candidate fingerprints are
    computed from SMILES (casmi/richfp.py) and cached by pool id."""

    def __init__(self, weights, pool_fp=None, device='cpu'):
        from .richfp import N_RAW
        sd = torch.load(weights, map_location=device, weights_only=False)
        cfg = sd.pop('_cfg'); keep = np.asarray(sd.pop('_keep'))
        self.m = PeakFormer(**cfg)
        self.m.load_state_dict(sd)
        self.m.eval()
        self.dev = device
        self.km = -np.ones(N_RAW, np.int64); self.km[keep] = np.arange(len(keep))
        self._fp = {}

    def onbits(self, pid, smiles):
        v = self._fp.get(pid)
        if v is None:
            from .richfp import raw_onbits
            a = self.km[raw_onbits(smiles)]
            v = a[a >= 0]
            if len(self._fp) > 400000:
                self._fp.clear()
            self._fp[pid] = v
        return v

    def score(self, spectra, pids, smiles):
        e, z = embed_query(self.m, spectra, self.dev)
        lists = [self.onbits(p, s) for p, s in zip(pids, smiles)]
        lens = np.array([len(a) for a in lists], np.int64)
        ids = torch.from_numpy(np.concatenate(lists).astype(np.int64) if lens.sum() else np.zeros(0, np.int64))
        off = torch.from_numpy(np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64))
        rid = torch.from_numpy(np.repeat(np.arange(len(lists)), lens).astype(np.int64))
        with torch.no_grad():
            m = self.m.encode_mol(ids, off)
            cos = (e @ m.T).squeeze(0).cpu().numpy()
            fz = fz_scores(z.unsqueeze(0), ids, rid, torch.zeros(len(lists), dtype=torch.long), len(lists)).cpu().numpy()
        return cos, fz
