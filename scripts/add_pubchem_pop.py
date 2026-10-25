"""Attach PubChem documentation counts (log1p SID / PMID / patent counts, number of CIDs) to a pool and/or row files.

Usage: python scripts/add_pubchem_pop.py <key_pop.parquet> <pool_dir> [rows.parquet ...]
Rows are updated in place (joined on the candidate key), so already-dumped simulations need no re-run.
key_pop.parquet (105.9M rows sorted by key, row groups of 1M rows) is streamed one row group at a time against the
sorted query keys, so memory is one row group (~100 MB) plus the pool; loading it whole needed ~12 GB (G11). Output is
identical to the full load.
"""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.keyset import pack

POP = sys.argv[1]                            # sorted by key, one row per key (groupby output)
COLS = ['sid', 'pmid', 'patent', 'n_cids']
pool_dir = sys.argv[2]


def pop_vals(v):
    """packed keys v -> float32 [n, 4] log1p counts (0 where the key has no PubChem record)."""
    o = np.argsort(v, kind='stable'); vs = v[o]
    vals = np.zeros((len(v), 4), np.float32)
    f, prev = pq.ParquetFile(POP), None
    for g in range(f.num_row_groups):         # row group = batch (iter_batches buffers ~1 GB of the file)
        b = f.read_row_group(g, columns=['key'] + COLS)
        k = b.column('key').to_numpy()
        if not len(k):
            continue
        assert np.all(k[:-1] < k[1:]) and (prev is None or prev < k[0]), 'key_pop not sorted / unique'
        prev = k[-1]
        lo, hi = np.searchsorted(vs, k[0]), np.searchsorted(vs, k[-1], side='right')
        if lo == hi:
            continue
        i = np.searchsorted(k, vs[lo:hi])     # k[0] <= query <= k[-1], so i < len(k)
        h = k[i] == vs[lo:hi]
        X = np.column_stack([b.column(c).fill_null(0).to_numpy(zero_copy_only=False) for c in COLS])
        vals[o[lo:hi][h]] = np.log1p(X[i[h]].astype(np.float32))
    return vals


def lookup(keys):
    ok = np.array([isinstance(k, str) and len(k) == 14 and k.isalpha() and k.isupper() for k in keys])
    out = pd.DataFrame(0.0, index=range(len(keys)), columns=['pc_sid', 'pc_pmid', 'pc_patent', 'pc_ncid'])
    if ok.any():
        out.loc[np.where(ok)[0], :] = pop_vals(pack([k for k, o in zip(keys, ok) if o]))
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
