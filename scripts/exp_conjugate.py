"""How many PubChem-absent structures are conjugates of two PubChem-known pieces?

Hydrolyse each amide / ester / thioester / glycosidic C-O bond (one at a time) and look the two pieces up in the
sorted PubChem InChIKey14 set. Usage: python scripts/exp_conjugate.py <keyfile> <out.parquet>
"""
import sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

keys = open(sys.argv[1]).read().split()
ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet').set_index('inchikey14')
pc = np.loadtxt('/home/user/ext/pc_ik14.txt', dtype='S14')  # sorted (LC_ALL=C)
print('pubchem keys', len(pc), flush=True)


def in_pc(k):
    b = k.encode()
    i = np.searchsorted(pc, b)
    return i < len(pc) and pc[i] == b


HYD = {
    'amide': AllChem.ReactionFromSmarts('[C:1](=[O:2])[N;!$(N-[C,S]=O)&!R:3]>>[C:1](=[O:2])O.[N:3]'),
    'amide_r': AllChem.ReactionFromSmarts('[C;!R:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]'),
    'ester': AllChem.ReactionFromSmarts('[C:1](=[O:2])[O;!R:3][#6:4]>>[C:1](=[O:2])O.[O:3][#6:4]'),
    'glyco': AllChem.ReactionFromSmarts('[C;R:1]([O;R:5])[O;!R:3][#6:4]>>[C:1]([O:5])O.[O:3][#6:4]'),
    'thioester': AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3][#6:4]>>[C:1](=[O:2])O.[S:3][#6:4]'),
}

rows = []
for n, k in enumerate(keys):
    smi = ts.loc[k, 'smiles'] if k in ts.index else None
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        continue
    best = None
    for name, rxn in HYD.items():
        try:
            prods = rxn.RunReactants((m,))
        except Exception:
            continue
        for ps in prods:
            pieces = []
            ok = True
            for p in ps:
                try:
                    Chem.SanitizeMol(p)
                except Exception:
                    ok = False; break
                frs = Chem.GetMolFrags(p, asMols=True)
                pieces.extend(frs)
            if not ok or len(pieces) != 2:
                continue
            pk = [Chem.MolToInchiKey(x)[:14] for x in pieces]
            known = [in_pc(x) for x in pk]
            ha = sorted(x.GetNumHeavyAtoms() for x in pieces)
            cand = (sum(known), ha[0], name, Chem.MolToSmiles(pieces[0]), Chem.MolToSmiles(pieces[1]))
            if best is None or cand[:2] > best[:2]:
                best = cand
    rows.append(dict(key=k, smiles=smi, n_known=best[0] if best else -1, small_ha=best[1] if best else 0,
                     bond=best[2] if best else None, a=best[3] if best else None, b=best[4] if best else None))
    if n % 2000 == 0:
        df = pd.DataFrame(rows); print(n, df.n_known.value_counts().to_dict(), flush=True)
df = pd.DataFrame(rows); df.to_parquet(sys.argv[2])
print('FINAL', len(df), df.n_known.value_counts().to_dict())
print(df[df.n_known == 2].bond.value_counts().to_dict())
