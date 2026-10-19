"""DreaMS analog channel: library neighbours by DreaMS embedding cosine instead of modified cosine.

Library = one representative spectrum per (structure, polarity) of the analog library (casmi/dreams_emb.py shards).
For a query (its spectra embedded with casmi.dreams_emb.DreamsEmbedder), the top-n_analog library structures by
cosine (same polarity, hidden structures / spectra excluded exactly as in Retrieval.features) are propagated to the
candidates by Morgan Tanimoto, like the modified-cosine analog channel:
  dr_best      max_j cos_j^pow * tan(cand, j)          dr_tan_best  max_j tan(cand, j)
  dr_sum       sum_j cos_j^pow * tan(cand, j)          dr_top1_tan  tan(cand, best neighbour)
  dr_self      best cosine to the candidate's own library representative (direct, instrument-robust match)
  dr_r / dr_gap  within-query rank / gap of dr_best
"""
import numpy as np

DR_FEATS = ['dr_best', 'dr_tan_best', 'dr_sum', 'dr_top1_tan', 'dr_self', 'dr_r', 'dr_gap']


class DreamsLib:
    def __init__(self, shard_paths, lib_key, lib_pol):
        """shard_paths: npz files (row, emb) from the embedding jobs; lib_key / lib_pol: per train.parquet row
        lookups (dict row -> key, dict row -> pol) for the representatives."""
        rows, embs = [], []
        for p in shard_paths:
            z = np.load(p)
            rows.append(z['row']); embs.append(z['emb'])
        self.row = np.concatenate(rows)
        self.E = np.concatenate(embs).astype(np.float32)
        self.key = np.array([lib_key[r] for r in self.row])
        self.pol = np.array([lib_pol[r] for r in self.row], np.int8)
        self.by_pol = {p: np.where(self.pol == p)[0] for p in (0, 1)}
        self.self_idx = {}
        for i, k in enumerate(self.key):
            self.self_idx.setdefault(k, []).append(i)

    def neighbours(self, e, pol, hide_keys=(), hide_rows=(), top=60):
        """Library indices and cosines of the best `top` representatives of one embedded query spectrum."""
        cand = self.by_pol[pol]
        s = self.E[cand] @ e
        o = np.argpartition(-s, min(top + 50, len(s) - 1))[:top + 50]
        o = o[np.argsort(-s[o])]
        out = []
        for t in o:
            j = cand[t]
            if self.key[j] in hide_keys or int(self.row[j]) in hide_rows:
                continue
            out.append((j, float(s[t])))
            if len(out) >= top:
                break
        return out


def dreams_features(DL, q_embs, q_pols, cand_keys, cand_fp, key_fp, tanimoto, hide_keys=(), hide_rows=(),
                    n_analog=20, pw=3.0):
    """q_embs: (n_spec, d) L2-normalised query embeddings with polarities q_pols; cand_keys / cand_fp: candidate
    keys and packed fingerprints; key_fp(key) -> packed fingerprint or None; tanimoto(fp, C) -> (n,) similarities.
    Returns dict of DR_FEATS arrays (n candidates)."""
    n = len(cand_keys)
    agg, self_best = {}, np.zeros(n)
    hide_keys, hide_rows = set(hide_keys), set(hide_rows)
    pos = {k: i for i, k in enumerate(cand_keys)}
    for e, p in zip(q_embs, q_pols):
        for j, s in DL.neighbours(e, int(p), hide_keys, hide_rows):
            k = DL.key[j]
            if k not in agg or s > agg[k]:
                agg[k] = s
            c = pos.get(k)
            if c is not None and s > self_best[c]:
                self_best[c] = s
    an = sorted(agg.items(), key=lambda x: -x[1])[:n_analog]
    best = np.zeros(n); tan_best = np.zeros(n); tot = np.zeros(n); top1 = np.zeros(n)
    for r, (k, s) in enumerate(an):
        f = key_fp(k)
        if f is None:
            continue
        t = tanimoto(f, cand_fp)
        v = (max(s, 0.0) ** pw) * t
        best = np.maximum(best, v); tan_best = np.maximum(tan_best, t); tot += v
        if r == 0:
            top1 = t
    import pandas as pd
    rank = pd.Series(best).rank(ascending=False, method='min').values      # as an_r / lib_r
    return dict(dr_best=best, dr_tan_best=tan_best, dr_sum=tot, dr_top1_tan=top1, dr_self=self_best,
                dr_r=rank, dr_gap=best - (best.max() if n else 0.0))
