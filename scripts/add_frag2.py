"""Add fragment evidence v2 (casmi/fragscore.py) to stage-2 top-k rows.

python scripts/add_frag2.py <in.top.parquet> <out.top.parquet> [procs]
Query spectra = the simulated query's training spectra (q_rows), cleaned as at inference (clean_peaks, rel floor
0.005, top 64); enveda-180 positive precursors are corrected by the electron mass as in the DreaMS query embedding.
"""
import os, sys, time
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.fragscore import FRAG2, candidate_scores, frag2_feats
from casmi.spectra import Store, clean_peaks

ELEC = 0.000548579909
st = meta = None


def qspec(q_rows):
    out = []
    for i in [int(x) for x in str(q_rows).split(',')]:
        mz, it = st.peaks(i); r = meta.loc[i]
        prec = float(r.precursor_mz) - (ELEC if r.ingest_lib == 'enveda-180' and r.adduct.endswith('+') else 0.0)
        a, b = clean_peaks(mz, it, prec, rel_floor=0.005, top_k=64)
        if len(a) >= 3:
            out.append((a.astype(np.float64), b.astype(np.float64), prec, r.adduct))
    return out


def job(args):
    qid, q_rows, smiles = args
    return qid, smiles, candidate_scores(smiles, qspec(q_rows))


if __name__ == '__main__':
    src, out = sys.argv[1:3]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    st = Store('/home/user/work/train_store'); meta = st.meta
    T = pd.read_parquet(src)
    jobs = [(q, g.q_rows.iloc[0], list(g.smiles)) for q, g in T.groupby('qid', sort=False)]
    t0 = time.time(); res = {}
    with Pool(procs) as p:
        for n, (q, smi, sc) in enumerate(p.imap_unordered(job, jobs, chunksize=4)):
            for j, s in enumerate(smi):
                res[(q, s)] = [sc[c][j] for c in FRAG2]
            if n % 500 == 0:
                print(n, 'of', len(jobs), f'{time.time() - t0:.0f}s', flush=True)
    V = np.array([res[(q, s)] for q, s in zip(T.qid, T.smiles)])
    for j, c in enumerate(FRAG2):
        T[c] = V[:, j]
    frag2_feats(T)
    T.to_parquet(out)
    print('done', len(T), f'{time.time() - t0:.0f}s')
