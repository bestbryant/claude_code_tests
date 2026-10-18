"""Attach PubChem documentation counts (log1p SID / PMID / patent counts, number of CIDs) to a pool and/or row files.

Usage: python scripts/add_pubchem_pop.py <key_pop.parquet> <pool_dir> [rows.parquet ...]
Rows are updated in place (joined on the candidate key), so already-dumped simulations need no re-run.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.keyset import pack

pop = pd.read_parquet(sys.argv[1]).set_index('key')
pool_dir = sys.argv[2]


def lookup(keys):
    ok = np.array([isinstance(k, str) and len(k) == 14 and k.isalpha() and k.isupper() for k in keys])
    out = pd.DataFrame(0.0, index=range(len(keys)), columns=['pc_sid', 'pc_pmid', 'pc_patent', 'pc_ncid'])
    if ok.any():
        v = pack([k for k, o in zip(keys, ok) if o])
        r = pop.reindex(v)
        vals = np.log1p(r[['sid', 'pmid', 'patent', 'n_cids']].fillna(0).values.astype(float))
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
