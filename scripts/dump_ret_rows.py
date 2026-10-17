"""Simulated retrieval queries -> candidate feature rows for ranker training / validation.

Usage: python scripts/dump_ret_rows.py <set: np|npx|any|keys:<file>> <n> <out.parquet> [seed] [procs] [p_class1]
Each held-out structure: 1-5 spectra (test adducts, timsTOF preferred) as the query. Class 1 hides only those
spectra; class 2 hides every spectrum of the structure (it stays in the candidate pool).
"""
import os, sys, time
os.environ['NUMBA_NUM_THREADS'] = '1'
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.analog import AnalogLib
from casmi.retrieval import Retrieval
from casmi.spectra import Store, clean_peaks

which, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
p1 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
R = Retrieval('/home/user/work/pool', AnalogLib(), '/home/user/ext/coconut/coconut_prior.parquet')
st = Store('/home/user/work/train_store'); meta = st.meta
TEST_ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-',
            '[M-H2O-H]-', '[M-2H2O+H]+']
mm = meta[meta.adduct.isin(TEST_ADD)]
grp = mm.groupby('inchikey14').groups
tims = set(mm.index[mm.instrument_type.fillna('').str.contains('tims', case=False)])
P = R.P.set_index('key')
if which.startswith('keys:'):
    keys = sorted(set(open(which[5:]).read().split()))
elif which == 'npx':
    keys = sorted(set(meta.inchikey14[meta.ingest_lib == 'enveda-np-examples']))
elif which == 'np':
    keys = sorted(set(P.index[(P.in_coco == 1) & (P.in_train == 1)]))
else:
    keys = sorted(set(P.index[P.in_train == 1]))
excl = set(open('/home/user/work/val_keys.txt').read().split()) if os.path.exists('/home/user/work/val_keys.txt') and 'train' in out else set()
keys = [k for k in keys if k in grp and k in P.index and 120 < P.loc[k, 'mass'] < 1300 and k not in excl]
rng = np.random.default_rng(seed)
keys = list(rng.choice(keys, size=min(n, len(keys)), replace=False))


def one(args):
    qi, k = args
    r = np.random.default_rng(seed * 7919 + qi)
    idx = list(grp[k])
    t_idx = [i for i in idx if i in tims]
    pool_idx = t_idx if t_idx and r.random() < 0.8 else idx
    r.shuffle(pool_idx)
    q_rows = pool_idx[:int(r.integers(1, 6))]
    Q = []
    for i in q_rows:
        mz, it = st.peaks(i); row = meta.loc[i]
        a, b = clean_peaks(mz, it, row.precursor_mz, rel_floor=0.005, top_k=64)
        if len(a) >= 3:
            Q.append(dict(mz=a, it=b, pol=0 if row.adduct.endswith('+') else 1))
    if not Q:
        return None
    cls = 1 if r.random() < p1 else 2
    M = P.loc[k, 'mass']
    try:
        if cls == 1:
            df = R.features(Q, M, hide_rows=set(int(i) for i in q_rows))
        else:
            df = R.features(Q, M, hide_keys={k})
    except Exception as e:
        print('fail', k, repr(e), flush=True)
        return None
    df['qid'] = qi; df['truth'] = k; df['cls'] = cls; df['label'] = (df.key == k).astype(np.int8)
    df['n_q'] = len(Q)
    return df.drop(columns=['smiles', 'formula'])


if __name__ == '__main__':
    t0 = time.time(); res = []
    with Pool(procs) as pool_:
        for i, df in enumerate(pool_.imap_unordered(one, list(enumerate(keys)), chunksize=2)):
            if df is not None:
                res.append(df)
            if i % 100 == 0:
                print(i, f'{time.time()-t0:.0f}s', flush=True)
    D = pd.concat(res, ignore_index=True)
    D.to_parquet(out)
    q = D.groupby('qid').label.max()
    print('FINAL queries', D.qid.nunique(), 'rows', len(D), 'truth in window', q.mean().round(3), f'{time.time()-t0:.0f}s')
