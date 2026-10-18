"""Attach PubChem documentation counts (log1p SID / PMID / patent counts, number of CIDs) to a pool and/or row files.

Usage: python scripts/add_pubchem_pop.py <key_pop.parquet> <pool_dir> [rows.parquet ...]
Rows are updated in place (joined on the candidate key), so already-dumped simulations need no re-run.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.keyset import pack

_pop = pd.read_parquet(sys.argv[1])          # sorted by key (groupby output); searched with numpy, not a pandas index
POP_KEY = _pop.key.values.astype(np.uint64)
POP_VAL = np.log1p(_pop[['sid', 'pmid', 'patent', 'n_cids']].fillna(0).values.astype(np.float32))
del _pop
assert np.all(POP_KEY[:-1] <= POP_KEY[1:])
pool_dir = sys.argv[2]


def lookup(keys):
    ok = np.array([isinstance(k, str) and len(k) == 14 and k.isalpha() and k.isupper() for k in keys])
    out = pd.DataFrame(0.0, index=range(len(keys)), columns=['pc_sid', 'pc_pmid', 'pc_patent', 'pc_ncid'])
    if ok.any():
        v = pack([k for k, o in zip(keys, ok) if o])
        i = np.minimum(np.searchsorted(POP_KEY, v), len(POP_KEY) - 1)
        hit = POP_KEY[i] == v
        vals = np.where(hit[:, None], POP_VAL[i], 0.0)
        out.loc[np.where(ok)[0], :] = vals
    return out


P = pd.read_parquet(f'{pool_dir}/pool.parquet')
L = lookup(list(P.key))
for c in L.columns:
    P[c] = L[c].values.astype(np.float32)
P.to_parquet(f'{pool_dir}/pool.parquet')
print('pool', len(P), 'with PubChem record', float((P.pc_ncid > 0).mean()))
for rp in sys.argv[3:]:
    D = pd.read_parquet(rp)
    L = lookup(list(D.key))
    for c in L.columns:
        D[c] = L[c].values
    D.to_parquet(rp)
    print(rp, 'truth with PubChem record', float((D[D.label == 1].pc_ncid > 0).mean()))
