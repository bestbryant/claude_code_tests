"""In-silico fragments with atom sets (1-2 bond cleavages) for peak annotation and site localisation."""
import numpy as np
from rdkit import Chem

H = 1.00782503207
PROTON = 1.007276
MONO = {'C': 12.0, 'H': H, 'N': 14.0030740048, 'O': 15.99491461956, 'S': 31.97207100, 'P': 30.97376163,
        'Cl': 34.96885268, 'F': 18.99840322, 'Br': 78.9183371, 'I': 126.904473, 'Si': 27.9769265, 'B': 11.0093054,
        'Se': 79.9165218, 'Na': 22.98976928, 'K': 38.9637064}


def _atom_mass(a):
    return MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H


def fragments(mol, max_cuts=2, max_frags=4000):
    """Return list of (atom_bitmask:int, neutral_mass:float, n_cuts:int) for connected fragments
    produced by breaking up to `max_cuts` bonds (pairs restricted to ring bonds of the same ring or any two chain bonds)."""
    n = mol.GetNumAtoms()
    am = np.array([_atom_mass(a) for a in mol.GetAtoms()])
    adj = [[] for _ in range(n)]
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append((j, b.GetIdx())); adj[j].append((i, b.GetIdx()))
        bonds.append((i, j, b.GetIdx(), b.IsInRing()))
    ri = mol.GetRingInfo()
    ring_of = {}
    for r, br in enumerate(ri.BondRings()):
        for bi in br:
            ring_of.setdefault(bi, set()).add(r)

    def comps(cut):
        seen = [False] * n
        out = []
        for s in range(n):
            if seen[s]:
                continue
            stack = [s]; seen[s] = True; mask = 0
            while stack:
                u = stack.pop(); mask |= (1 << u)
                for v, bi in adj[u]:
                    if bi in cut or seen[v]:
                        continue
                    seen[v] = True; stack.append(v)
            out.append(mask)
        return out

    res = {}
    chain = [b for b in bonds if not b[3]]
    for b in chain:
        cs = comps({b[2]})
        if len(cs) == 2:
            for c in cs:
                res.setdefault(c, 1)
    if max_cuts >= 2 and len(chain) <= 45:
        pairs = []
        for x in range(len(chain)):
            for y in range(x + 1, len(chain)):
                pairs.append((chain[x][2], chain[y][2]))
        ringb = [b for b in bonds if b[3]]
        for x in range(len(ringb)):
            for y in range(x + 1, len(ringb)):
                if ring_of.get(ringb[x][2], set()) & ring_of.get(ringb[y][2], set()):
                    pairs.append((ringb[x][2], ringb[y][2]))
        for p in pairs[:max_frags]:
            for c in comps(set(p)):
                if c not in res:
                    res[c] = 2
    out = []
    for mask, k in res.items():
        idx = [i for i in range(n) if (mask >> i) & 1]
        out.append((mask, float(am[idx].sum()), k))
    return out


def annotate(frags, mz, positive=True, tol=0.01, hshift=(-2, -1, 0, 1)):
    """For each peak, list of (mask, n_cuts) of fragments whose ion m/z matches."""
    if not frags:
        return [[] for _ in mz]
    fm = np.array([f[1] for f in frags])
    sign = 1.0 if positive else -1.0
    ann = [[] for _ in mz]
    for k in hshift:
        ion = fm + sign * PROTON + k * H
        o = np.argsort(ion); s = ion[o]
        for p, m in enumerate(mz):
            lo, hi = np.searchsorted(s, m - tol), np.searchsorted(s, m + tol)
            for t in range(lo, hi):
                f = frags[o[t]]
                ann[p].append((f[0], f[2]))
    return ann
