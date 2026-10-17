"""Dump NovelEngine candidate rows (features + label) for held-out structures, leave-one-out.

Each query hides the structure's spectra from analog search, removes its pieces from the building-block counts,
and allows the truth through the PubChem / known filters (simulating "this structure were novel").
Usage: python scripts/dump_novel_rows.py <set: absent|np|any> <n> <out.parquet> [seed] [procs]
"""
import os, sys, time
os.environ['NUMBA_NUM_THREADS'] = '1'
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.analog import AnalogLib
from casmi.conjugate import BlockLib
from casmi.keyset import KeySet, pack
from casmi.metric import key14
from casmi.novel import NovelEngine
from casmi.spectra import Store

which, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet')
tsi = ts.set_index('inchikey14')
L = AnalogLib()
P = pd.read_parquet('/home/user/work/blocks_pieces.parquet'); PK = {k: g for k, g in P.groupby('key')}
B = BlockLib(P, min_count=1)
E = NovelEngine(L, B, KeySet.load('/home/user/ext/pc_keys_u64.npy'), KeySet(np.unique(pack(list(ts.inchikey14)))),
                dict(zip(ts.inchikey14, ts.mass)))
st = Store('/home/user/work/train_store'); meta = st.meta
TEST_ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-']
grp = meta[meta.adduct.isin(TEST_ADD)].groupby('inchikey14').groups
absent = set(open('/home/user/work/train_not_pc.txt').read().split())
pool = pd.read_parquet('/home/user/pub/pool/pool_meta.parquet')
if which == 'absent':
    keys = [k for k in absent]
elif which == 'np':
    np_keys = set(pool.loc[pool.src == 3, 'key'])
    keys = [k for k, mk in zip(ts.inchikey14, ts.key) if mk in np_keys and k not in absent]
else:
    keys = [k for k in ts.inchikey14 if k not in absent]
keys = sorted(k for k in keys if k in grp and 150 < tsi.loc[k, 'mass'] < 1000)
rng = np.random.default_rng(seed)
keys = list(rng.choice(keys, size=min(n, len(keys)), replace=False))


def one(args):
    qi, k = args
    r = np.random.default_rng(seed * 100003 + qi)
    idx = list(grp[k]); r.shuffle(idx); idx = idx[:int(r.integers(1, 6))]
    spectra = []
    for i in idx:
        mz, it = st.peaks(i); row = meta.loc[i]
        spectra.append(dict(mz=mz, it=it, prec=row.precursor_mz, adduct=row.adduct, positive=row.adduct.endswith('+')))
    counts = B.loo(PK[k]) if k in PK else None
    try:
        df = E.candidates(spectra, M=tsi.loc[k, 'mass'], exclude_keys={k}, loo_counts=counts, allow_keys={k})
    except Exception as e:
        print('fail', k, repr(e), flush=True)
        return None
    tk = key14(tsi.loc[k, 'smiles'])
    if len(df) == 0:
        return pd.DataFrame(dict(qid=[qi], truth=[k], label=[0], empty=[True]))
    df['qid'] = qi; df['truth'] = k; df['label'] = (df.key == tk).astype(int); df['empty'] = False
    return df.drop(columns=['std'])


if __name__ == '__main__':
    t0 = time.time(); res = []
    with Pool(procs) as pool_:
        for i, df in enumerate(pool_.imap_unordered(one, list(enumerate(keys)), chunksize=4)):
            if df is not None:
                res.append(df)
            if i % 200 == 0:
                print(i, f'{time.time()-t0:.0f}s', flush=True)
    D = pd.concat(res, ignore_index=True)
    D.to_parquet(out)
    q = D.groupby('qid').label.max()
    print('FINAL queries', D.qid.nunique(), 'rows', len(D), 'truth present', q.mean().round(3), f'{time.time()-t0:.0f}s')
