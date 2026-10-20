"""PubChem proposal channel: documented PubChem compounds outside the pool (scripts/build_pubchem_tier.py), proposed
for a test molecule from its 10 ppm neutral-mass window and scored by the PeakFormer (casmi/peakformer.py).

  propose(M, spectra)  -> frame of the window's top-n_pop compounds by documentation (pop), with the PeakFormer
                          cosine / f.z, their within-window z-scores and score = z(f.z) + z(cos) + lam * pop
  insert(final, props, slots, k)  -> the final ranked list with the best k proposals at the given 1-based slots
                          (proposals whose metric key is already listed are skipped)
"""
import numpy as np
import pandas as pd


class PubChemTier:
    def __init__(self, path):
        z = np.load(path, mmap_mode='r')
        self.mass = np.asarray(z['mass']); self.pop = np.asarray(z['pop'])
        self.cid = z['cid']; self.off = np.asarray(z['smi_off']); self.smi = z['smi']

    def smiles(self, i):
        return bytes(self.smi[self.off[i]:self.off[i + 1]]).decode()

    def window(self, M, ppm=10.0, n_pop=2000):
        """Tier rows within ppm of neutral mass M, the n_pop most documented."""
        tol = max(M * ppm * 1e-6, 0.002)
        lo, hi = np.searchsorted(self.mass, M - tol), np.searchsorted(self.mass, M + tol)
        idx = np.arange(lo, hi)
        if len(idx) > n_pop:
            idx = idx[np.argsort(-self.pop[idx], kind='stable')[:n_pop]]
        return idx


def propose(tier, pfs, M, spectra, n_pop=2000, lam=0.25):
    """Scored proposals for one molecule (pfs: PFEnsemble / PFScorer; spectra: raw dicts as for the FP model)."""
    idx = tier.window(M, n_pop=n_pop)
    if len(idx) == 0 or pfs is None:
        return pd.DataFrame(columns=['smiles', 'cid', 'pop', 'pc_cos', 'pc_fz', 'pc_score'])
    smi = [tier.smiles(i) for i in idx]
    cos, fz = pfs.score(spectra, [-1 - int(i) for i in idx], smi)        # negative ids: never collide with pool ids
    zs = lambda v: (v - v.mean()) / (v.std() or 1.0)
    f = pd.DataFrame(dict(smiles=smi, cid=np.asarray(tier.cid[idx]).astype(np.int64), pop=tier.pop[idx], pc_cos=cos,
                          pc_fz=fz))
    f['pc_score'] = zs(f.pc_fz.values) + zs(f.pc_cos.values) + lam * f['pop'].values
    return f.sort_values('pc_score', ascending=False).reset_index(drop=True)


def insert(final, props, slots=(4, 8, 12, 16, 20), k=5, key14=None):
    """final: [[smiles, score, key14], ...] ranked; props: propose() frame. Returns the merged top-25 list."""
    out = list(final)
    have = {r[2] for r in out}
    added = 0
    for s in props.smiles.values:
        if added >= min(k, len(slots)):
            break
        kk = key14(s) if key14 else None
        if kk is None or kk in have:
            continue
        pos = min(slots[added] - 1, len(out))
        out.insert(pos, [s, float('nan'), kk])
        have.add(kk); added += 1
    return out[:25]
