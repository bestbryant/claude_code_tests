"""Part 3a: exact re-dump of the lib / analog channels for the SAME v18c validation class-1 queries (same query
spectra, same mass window -> PeakFormer / FP / frag / prior features and GLACIER scores stay valid) under
hidden-like hiding rules. Replicates scripts/dump_ret_rows.py one() (query build) + Retrieval.features (do_frag=False).
Variants (rows of the truth hidden from the lib10 library AND the analog library):
  orig : the query rows only (as the v18c dump; sanity check against ret18c_val_reach_c)
  tl   : + every timsTOF (= Enveda) spectrum of the truth (dump_ret_rows tl: rule)
  xinst: + every truth spectrum of the same instrument class as any query spectrum (tims / orbi / qtof / other)
  xlab : + every truth spectrum from the same ingest_lib as any query spectrum
python redump_same.py <out.parquet> [procs] [max_queries]
"""
import os, sys, time
os.environ['NUMBA_NUM_THREADS'] = '1'; os.environ['VENDOR_PATH'] = '/nonexistent'
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/home/user/claude_code_tests')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casmi.analog import AnalogLib
from casmi.retrieval import Retrieval
from casmi.spectra import Store, clean_peaks, neutral_mass
ELECTRON = 0.00054858
out = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
from common import inst_class
R = Retrieval('/home/user/work/pool_v18c', AnalogLib(), '/home/user/ext/coconut/coconut_prior.parquet',
              lib_full=AnalogLib('/home/user/work/lib10.npz'))
st = Store('/home/user/work/train_store'); meta = st.meta
ic = inst_class(meta)
D = pd.read_parquet('/home/user/work/ret18c_val_reach_c.parquet', columns=['qid', 'truth', 'cls', 'q_rows'])
Qs = D[D.cls == 1].drop_duplicates('qid')[['qid', 'truth', 'q_rows']].values.tolist()[:nmax]
rows_of = meta.groupby('inchikey14').indices
VARS = ('orig', 'tl', 'xinst', 'xlab')
KEEP = ['pid', 'lib', 'lib_n', 'lib_r', 'lib_gap', 'an_best', 'an_tan_best', 'an_sum', 'an_top1_tan', 'an_r', 'an_gap']

_search = R.L.search
_memo = {}
def search_memo(qmz, qit, qmass, pol, **kw):
    k = (qmz.tobytes(), qit.tobytes(), float(qmass), int(pol), tuple(sorted(kw.get('exclude_keys', ()))))
    if k not in _memo:
        _memo[k] = _search(qmz, qit, qmass, pol, **kw)
    return _memo[k]
R.L.search = search_memo


def one(args):
    qid, k, qr = args
    _memo.clear()
    q_rows = [int(x) for x in qr.split(',')]
    Q, nms = [], []
    for i in q_rows:
        mz, it = st.peaks(i); row = meta.loc[i]
        prec = row.precursor_mz
        if row.ingest_lib == 'enveda-180' and row.adduct.endswith('+'):
            prec -= ELECTRON
        a, b = clean_peaks(mz, it, prec, rel_floor=0.005, top_k=64)
        if len(a) >= 3:
            Q.append(dict(mz=a, it=b, pol=0 if row.adduct.endswith('+') else 1))
            nms.append(neutral_mass(prec, row.adduct))
    M = float(np.nanmedian(nms))
    tr = np.asarray(rows_of.get(k, []))
    qic = set(ic.loc[q_rows]); qlib = set(meta.ingest_lib.loc[q_rows])
    hide = {'orig': set(q_rows),
            'tl': set(q_rows) | set(int(i) for i in tr if ic.iat[i] == 'tims'),
            'xinst': set(q_rows) | set(int(i) for i in tr if ic.iat[i] in qic),
            'xlab': set(q_rows) | set(int(i) for i in tr if meta.ingest_lib.iat[i] in qlib)}
    res = []
    for v in VARS:
        df = R.features(Q, M, hide_rows=hide[v], do_frag=False)[KEEP]
        df['qid'] = qid; df['var'] = v; df['n_hidden_truth'] = len(hide[v]); res.append(df)
    return pd.concat(res, ignore_index=True)


if __name__ == '__main__':
    t0 = time.time(); res = []
    with Pool(procs) as pool:
        for j, df in enumerate(pool.imap_unordered(one, Qs, chunksize=2)):
            res.append(df)
            if (j + 1) % 25 == 0:
                print(j + 1, f'{time.time()-t0:.0f}s', flush=True)
    pd.concat(res, ignore_index=True).to_parquet(out)
    print('done', len(res), f'{time.time()-t0:.0f}s')
