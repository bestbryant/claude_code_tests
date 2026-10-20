"""Multi-block structure fingerprint for the PeakFormer bit head (score = fingerprint . predicted bit logits).

Raw layout (RDKit generators, bits): Morgan r2 4096 | Morgan r3 4096 | feature-Morgan r2 2048 | RDKit paths <= 6
2048 | atom pairs 2048 | topological torsions 2048 | MACCS 167 = 16,551 raw bits. Training keeps the informative
bits (set in >= 1 / 2000 training structures and not in nearly all of them); `keep` maps raw -> kept index.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as G

RDLogger.DisableLog('rdApp.*')
BLOCKS = [('morgan2', 4096), ('morgan3', 4096), ('fmorgan2', 2048), ('rdk6', 2048), ('ap', 2048), ('tt', 2048),
          ('maccs', 167)]
N_RAW = sum(n for _, n in BLOCKS)
_GENS = None


def _gens():
    global _GENS
    if _GENS is None:
        _GENS = [G.GetMorganGenerator(radius=2, fpSize=4096), G.GetMorganGenerator(radius=3, fpSize=4096),
                 G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()),
                 G.GetRDKitFPGenerator(maxPath=6, fpSize=2048), G.GetAtomPairGenerator(fpSize=2048),
                 G.GetTopologicalTorsionGenerator(fpSize=2048)]
    return _GENS


def raw_onbits(smiles):
    """Sorted raw on-bit indices (int32) of one SMILES; empty if it does not parse."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return np.zeros(0, np.int32)
    out, base = [], 0
    for g, (_, n) in zip(_gens(), BLOCKS):
        out.append(np.asarray(list(g.GetFingerprint(m).GetOnBits()), np.int32) + base)
        base += n
    out.append(np.asarray(list(MACCSkeys.GenMACCSKeys(m).GetOnBits()), np.int32) + base)
    return np.concatenate(out)


def to_csr(lists, keep_map=None):
    """List of raw on-bit arrays -> (ptr int64, idx uint16) in the kept layout (keep_map: raw -> kept or -1)."""
    if keep_map is not None:
        lists = [keep_map[a][keep_map[a] >= 0] for a in lists]
    ptr = np.zeros(len(lists) + 1, np.int64)
    ptr[1:] = np.cumsum([len(a) for a in lists])
    idx = np.concatenate(lists).astype(np.uint16) if len(lists) else np.zeros(0, np.uint16)
    return ptr, idx
