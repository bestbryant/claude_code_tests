"""Derivative generation with modification-site tracking.

Each product carries `sites` (parent atoms bonded to newly added atoms or whose H count / bonds changed)
and `removed` (parent atoms absent from the product), so spectra can localise the modification.
"""
import itertools
from typing import Dict, List

from rdkit import Chem
from rdkit.Chem import AllChem

from .derive import _TRANSFORMS, MASS

_RXN = {}


def _rxn(name):
    if name not in _RXN:
        _RXN[name] = AllChem.ReactionFromSmarts(_TRANSFORMS[name][0])
    return _RXN[name]


def _dm(d):
    return sum(MASS[e] * v for e, v in d.items())


_COMBOS = None


def combos(table=None):
    global _COMBOS
    if _COMBOS is None:
        names = list(_TRANSFORMS)
        out = [((n,), _dm(_TRANSFORMS[n][1])) for n in names]
        for a, b in itertools.combinations_with_replacement(names, 2):
            d = dict(_TRANSFORMS[a][1])
            for e, v in _TRANSFORMS[b][1].items():
                d[e] = d.get(e, 0) + v
            out.append(((a, b), _dm(d)))
        _COMBOS = out
    return _COMBOS


def _tag(mol):
    for a in mol.GetAtoms():
        if not a.HasProp('pidx'):
            continue
    return mol


def _apply(mol, name, max_products):
    rxn = _rxn(name)
    out = []
    try:
        prods = rxn.RunReactants((mol,))
    except Exception:
        return out
    seen = set()
    for p in prods:
        m = p[0]
        for a in m.GetAtoms():
            if not a.HasProp('pidx') and a.HasProp('react_atom_idx'):
                ra = mol.GetAtomWithIdx(a.GetIntProp('react_atom_idx'))
                if ra.HasProp('pidx'):
                    a.SetIntProp('pidx', ra.GetIntProp('pidx'))
        try:
            Chem.SanitizeMol(m)
            s = Chem.MolToSmiles(m)
        except Exception:
            continue
        if s in seen:
            continue
        seen.add(s); out.append(m)
        if len(out) >= max_products:
            break
    return out


def site_info(parent, prod):
    """sites: parent atoms adjacent to new atoms or with changed H count; removed: parent atoms not in product."""
    pmap = {}
    new = []
    for a in prod.GetAtoms():
        if a.HasProp('pidx'):
            pmap[a.GetIdx()] = a.GetIntProp('pidx')
        else:
            new.append(a.GetIdx())
    present = set(pmap.values())
    removed = [i for i in range(parent.GetNumAtoms()) if i not in present]
    sites = set()
    for ni in new:
        for nb in prod.GetAtomWithIdx(ni).GetNeighbors():
            if nb.GetIdx() in pmap:
                sites.add(pmap[nb.GetIdx()])
    for pi_, i in pmap.items():
        if prod.GetAtomWithIdx(pi_).GetTotalNumHs() != parent.GetAtomWithIdx(i).GetTotalNumHs():
            sites.add(i)
    for i in removed:
        for nb in parent.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() in present:
                sites.add(nb.GetIdx())
    return sorted(sites), removed


def generate(parent_smiles, delta, tol=0.003, max_per_parent=300, budget=800):
    """Products of `parent_smiles` whose mass differs by ~delta: list of dict(smiles, combo, sites, removed)."""
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return [], None
    for a in mol.GetAtoms():
        a.SetIntProp('pidx', a.GetIdx())
    todo = [c for c, m in combos() if abs(m - delta) <= tol]
    out = {}
    for combo in todo:
        mols = [mol]
        for depth, name in enumerate(combo):
            nxt = []
            for m in mols[:24]:
                nxt.extend(_apply(m, name, 32 if depth == 0 else 16))
                budget -= 1
                if budget <= 0:
                    break
            mols = nxt
            if not mols or budget <= 0:
                break
        for m in mols:
            s = Chem.MolToSmiles(m)
            if s == parent_smiles or s in out:
                continue
            sites, removed = site_info(mol, m)
            out[s] = dict(smiles=s, combo=combo, sites=sites, removed=removed)
        if len(out) >= max_per_parent or budget <= 0:
            break
    return list(out.values())[:max_per_parent], mol
