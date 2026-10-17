"""Competition metric: MRR@25 on the tautomer-canonical InChIKey14 (RDKit 2026.03.3).

Mirrors metric/casmi-mean-reciprocal-rank: stereo is removed before tautomer canonicalisation.
"""
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
_TAUT = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=500_000)
def key14(smiles):
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(_TAUT.Canonicalize(mol))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def reciprocal_rank(guesses, truth_key, k=25):
    for i, s in enumerate(guesses[:k]):
        if key14(s) == truth_key:
            return 1.0 / (i + 1)
    return 0.0


def rank_of(keys, truth_key, k=25):
    """1-based rank of truth_key in an already-keyed list, or 0 if absent from the top k."""
    for i, kk in enumerate(keys[:k]):
        if kk == truth_key:
            return i + 1
    return 0
