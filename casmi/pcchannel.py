"""PubChem proposal channel: documented PubChem compounds outside the pool (scripts/build_pubchem_tier.py), proposed
for a test molecule from its 10 ppm neutral-mass window and scored by the PeakFormer (casmi/peakformer.py).

  propose(M, spectra)  -> frame of the window's top-n_pop compounds by documentation (pop), with the PeakFormer
                          cosine / f.z, their within-window z-scores and score = z(f.z) + z(cos) + lam * pop
  insert(final, props, slots, k)  -> the final ranked list with the best k proposals at the given 1-based slots
                          (proposals whose metric key is already listed are skipped); a list shorter than 25 is
                          then filled up with the remaining proposals
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
    for s in props.smiles.values:             # vacant places at the end of a short list cost nothing to fill
        if len(out) >= 25:
            break
        kk = key14(s) if key14 else None
        if kk is not None and kk not in have:
            out.append([s, float('nan'), kk]); have.add(kk)
    return out[:25]


class FullTier:
    """Full PubChem tier (kernels/casmi26-pc-full-build): directory with mass.npy (float32, sorted), cid.npy,
    smi_off.npy, smi.npy; no documentation counts."""

    def __init__(self, d, pop_path=None):
        self.mass = np.load(f'{d}/mass.npy', mmap_mode='r'); self.cid = np.load(f'{d}/cid.npy', mmap_mode='r')
        self.off = np.load(f'{d}/smi_off.npy', mmap_mode='r'); self.smi = np.load(f'{d}/smi.npy', mmap_mode='r')
        # documentation prior aligned to the rows (dataset bryantyu/casmi26-pc-full-pop), or None
        self.pop = np.load(pop_path, mmap_mode='r') if pop_path else None

    def smiles(self, i):
        return bytes(self.smi[self.off[i]:self.off[i + 1]]).decode()

    def window(self, M, ppm=10.0, cap=150000):
        """Tier rows within ppm of M; a window larger than cap is thinned by a uniform stride (never truncated at
        one mass edge)."""
        tol = max(M * ppm * 1e-6, 0.002)
        lo, hi = np.searchsorted(self.mass, M - tol), np.searchsorted(self.mass, M + tol)
        if hi - lo <= cap:
            return np.arange(lo, hi)
        return np.linspace(lo, hi - 1, cap).astype(np.int64)


def finalize(f, exclude=None, n=25):
    """Best n proposals of a scored frame with distinct metric identity (tautomer-canonical InChIKey14, so stereo /
    tautomer duplicates do not waste places), skipping structures whose standard InChIKey14 is in `exclude`."""
    from rdkit import Chem
    from .metric import key14
    keep, seen = [], set()
    for i, s in enumerate(f.smiles.values):
        k = key14(s)
        if k is None or k in seen:
            continue
        if exclude:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            Chem.RemoveStereochemistry(m)
            if Chem.MolToInchiKey(m)[:14] in exclude:
                continue
        seen.add(k); keep.append(i)
        if len(keep) >= n:
            break
    return f.iloc[keep].reset_index(drop=True)


def propose_full(tier, pfs, M, spectra, n1=2000, exclude=None, lam=0.0, pop_union=0):
    """Two-pass proposals from the full tier: pass 1 scores every window compound by f.z over its Morgan r2 block
    (casmi/richfp.py block 0, cheap); the best n1 (plus the pop_union best-documented rows) get the full PeakFormer
    score; score = z(f.z) + z(cos) + lam * pop; then finalize() (distinct metric identities, `exclude` keys dropped)."""
    from rdkit import Chem
    from .peakformer import embed_query
    idx = tier.window(M)
    cols = ['smiles', 'cid', 'pop', 'pc_cos', 'pc_fz', 'pc_score']
    if len(idx) == 0 or pfs is None:
        return pd.DataFrame(columns=cols)
    m0 = pfs.ms[0] if hasattr(pfs, 'ms') else pfs
    _, z = embed_query(m0.m, spectra, m0.dev)
    zb = np.zeros(4096, np.float32)
    kept = m0.km[:4096]
    zb[kept >= 0] = z.numpy()[kept[kept >= 0]]
    from .richfp import _gens
    gen = _gens()[0]
    smi = [tier.smiles(i) for i in idx]
    s1 = np.full(len(idx), -np.inf, np.float32)
    for j, s in enumerate(smi):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            s1[j] = zb[list(gen.GetFingerprint(mol).GetOnBits())].sum()
    ok = np.isfinite(s1)
    if not ok.any():
        return pd.DataFrame(columns=cols)
    top = np.argsort(-s1)[:min(n1, int(ok.sum()))]
    pw = np.asarray(tier.pop[idx], np.float32) if tier.pop is not None else np.zeros(len(idx), np.float32)
    if pop_union and tier.pop is not None:
        pu = np.argsort(-np.where(ok, pw, -np.inf))[:min(pop_union, int(ok.sum()))]
        top = np.union1d(top, pu)
    ssel = [smi[j] for j in top]
    cos, fz = pfs.score(spectra, [-1 - int(idx[j]) for j in top], ssel)
    zs = lambda v: (v - v.mean()) / (v.std() or 1.0)
    f = pd.DataFrame(dict(smiles=ssel, cid=np.asarray(tier.cid[idx[top]]).astype(np.int64), pop=pw[top], pc_cos=cos,
                          pc_fz=fz))
    f['pc_score'] = zs(f.pc_fz.values) + zs(f.pc_cos.values) + lam * f['pop'].values
    return finalize(f.sort_values('pc_score', ascending=False).reset_index(drop=True), exclude)
