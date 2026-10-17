"""Hydrolysis of amide / ester / glycosidic bonds into (acid, partner) building-block pairs."""
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

RX = {
    'amide': '[C;!R:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]',
    'ester': '[C:1](=[O:2])[O;!R:3][#6:4]>>[C:1](=[O:2])O.[O:3][#6:4]',
    'glyco': '[C;R:1]([O;R:5])[O;!R:3][#6:4]>>[C:1]([O:5])O.[O:3][#6:4]',
}
_R = None


def canon(m):
    m = Chem.Mol(m); Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def work(args):
    """(key, smiles) -> list of (key, bond, acid_smiles, partner_smiles)."""
    global _R
    if _R is None:
        _R = {k: AllChem.ReactionFromSmarts(v) for k, v in RX.items()}
    key, smi = args
    m = Chem.MolFromSmiles(smi)
    out = []
    if m is None:
        return out
    for name, rx in _R.items():
        try:
            prods = rx.RunReactants((m,))
        except Exception:
            continue
        seen = set()
        for ps in prods:
            if len(ps) != 2:
                continue
            try:
                for p in ps:
                    Chem.SanitizeMol(p)
                a, b = canon(ps[0]), canon(ps[1])
            except Exception:
                continue
            if '.' in a or '.' in b or (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((key, name, a, b))
    return out
