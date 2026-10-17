"""Spectrum <-> structure dual encoder (binned-spectrum MLP + fingerprint MLP), contrastive + fingerprint BCE.

Spectrum features: sqrt intensities in 1 Da fragment bins (0-1500) and neutral-loss bins (0-500), plus adduct,
polarity, collision energy, instrument family and precursor mass. Molecule features: 2215 bits (Morgan r2 2048 +
MACCS 167). Score(spectrum, candidate) = cosine(emb_s, emb_m) / tau.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_FRAG, NB_NL, N_ADD, N_INST = 1500, 500, 15, 6
N_META = N_ADD + 2 + 1 + N_INST + 1
D_IN = NB_FRAG + NB_NL + N_META
N_BITS = 2215


def featurize(mz_flat, it_flat, off, idx, prec, ce, add_id, inst_id, pol):
    """Dense float32 (len(idx), D_IN) inputs for spectra `idx` from flat peak arrays."""
    n = len(idx)
    X = np.zeros((n, D_IN), np.float32)
    starts, ends = off[idx], off[idx + 1]
    lens = ends - starts
    rows = np.repeat(np.arange(n), lens)
    # flat peak indices without a python loop: start of each row repeated + position within the row
    csum = np.cumsum(lens) - lens
    pk = np.repeat(starts, lens) + (np.arange(lens.sum()) - np.repeat(csum, lens))
    mz = mz_flat[pk].astype(np.float32); it = np.sqrt(it_flat[pk].astype(np.float32))
    p = prec[idx][rows]
    fb = np.clip(np.round(mz).astype(np.int64), 0, NB_FRAG - 1)
    np.add.at(X, (rows, fb), it)
    nl = p - mz
    ok = (nl > 0.5) & (nl < NB_NL - 0.5)
    np.add.at(X, (rows[ok], NB_FRAG + np.round(nl[ok]).astype(np.int64)), it[ok])
    # L2 normalise the peak part
    nrm = np.sqrt((X[:, :NB_FRAG + NB_NL] ** 2).sum(1, keepdims=True)) + 1e-6
    X[:, :NB_FRAG + NB_NL] /= nrm
    o = NB_FRAG + NB_NL
    a = np.clip(add_id[idx].astype(np.int64), 0, N_ADD - 1)
    X[np.arange(n), o + a] = 1.0
    o += N_ADD
    X[np.arange(n), o + pol[idx].astype(np.int64)] = 1.0
    o += 2
    c = ce[idx]
    X[:, o] = np.where(c >= 0, np.minimum(c, 150) / 100.0, -0.5)
    o += 1
    X[np.arange(n), o + np.clip(inst_id[idx].astype(np.int64), 0, N_INST - 1)] = 1.0
    o += N_INST
    X[:, o] = prec[idx] / 1000.0
    return X


class DualEncoder(nn.Module):
    def __init__(self, d=1024, emb=512, drop=0.1):
        super().__init__()
        self.spec = nn.Sequential(nn.Linear(D_IN, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(drop))
        self.spec_emb = nn.Linear(d, emb)
        self.fp_head = nn.Linear(d, N_BITS)
        self.mol = nn.Sequential(nn.Linear(N_BITS, d), nn.GELU(), nn.Linear(d, emb))
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float32))

    def encode_spec(self, x):
        h = self.spec(x)
        return F.normalize(self.spec_emb(h), dim=-1), self.fp_head(h)

    def encode_mol(self, bits):
        return F.normalize(self.mol(bits), dim=-1)


def unpack_bits(packed):
    return np.unpackbits(packed, axis=-1)[..., :N_BITS].astype(np.float32)


def sparse_spec(mz_flat, it_flat, off, idx, prec, ce, add_id, inst_id, pol):
    """EmbeddingBag inputs for spectra `idx`: (bin ids, bag offsets, weights) + dense metadata (n, N_META)."""
    n = len(idx)
    starts, ends = off[idx], off[idx + 1]
    lens = ends - starts
    rows = np.repeat(np.arange(n), lens)
    csum = np.cumsum(lens) - lens
    pk = np.repeat(starts, lens) + (np.arange(lens.sum()) - np.repeat(csum, lens))
    mz = mz_flat[pk].astype(np.float32); it = np.sqrt(it_flat[pk].astype(np.float32))
    nl = prec[idx][rows] - mz
    fb = np.clip(np.round(mz).astype(np.int64), 0, NB_FRAG - 1)
    ok = (nl > 0.5) & (nl < NB_NL - 0.5)
    ids = np.concatenate([fb, NB_FRAG + np.round(nl[ok]).astype(np.int64)])
    wts = np.concatenate([it, it[ok]])
    rr = np.concatenate([rows, rows[ok]])
    o = np.argsort(rr, kind='stable')
    ids, wts, rr = ids[o], wts[o], rr[o]
    nrm = np.sqrt(np.bincount(rr, weights=wts ** 2, minlength=n)) + 1e-6
    wts = wts / nrm[rr]
    bag_off = np.concatenate([[0], np.cumsum(np.bincount(rr, minlength=n))[:-1]])
    meta = featurize_meta(idx, prec, ce, add_id, inst_id, pol)
    return ids, bag_off, wts.astype(np.float32), meta


def featurize_meta(idx, prec, ce, add_id, inst_id, pol):
    n = len(idx)
    X = np.zeros((n, N_META), np.float32)
    X[np.arange(n), np.clip(add_id[idx].astype(np.int64), 0, N_ADD - 1)] = 1.0
    o = N_ADD
    X[np.arange(n), o + pol[idx].astype(np.int64)] = 1.0
    o += 2
    c = ce[idx]
    X[:, o] = np.where(c >= 0, np.minimum(c, 150) / 100.0, -0.5)
    o += 1
    X[np.arange(n), o + np.clip(inst_id[idx].astype(np.int64), 0, N_INST - 1)] = 1.0
    o += N_INST
    X[:, o] = prec[idx] / 1000.0
    return X


def sparse_bits(packed):
    """Packed fingerprints -> (on-bit ids, bag offsets) for EmbeddingBag."""
    b = unpack_bits(packed)
    r, c = np.nonzero(b)
    off = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=len(b)))[:-1]])
    return c.astype(np.int64), off.astype(np.int64)


class SparseDualEncoder(nn.Module):
    def __init__(self, d=768, emb=256, drop=0.1):
        super().__init__()
        self.peaks = nn.EmbeddingBag(NB_FRAG + NB_NL, d, mode='sum')
        self.meta = nn.Linear(N_META, d)
        self.spec = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(d, d), nn.GELU())
        self.spec_emb = nn.Linear(d, emb)
        self.fp_head = nn.Linear(d, N_BITS)
        self.bits = nn.EmbeddingBag(N_BITS, d, mode='sum')
        self.mol = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, emb))
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float32))

    def encode_spec(self, ids, off, wts, meta):
        h = self.spec(self.peaks(ids, off, per_sample_weights=wts) + self.meta(meta))
        return F.normalize(self.spec_emb(h), dim=-1), self.fp_head(h)

    def encode_mol(self, ids, off):
        return F.normalize(self.mol(self.bits(ids, off)), dim=-1)
