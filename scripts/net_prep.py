"""Pseudo-test for test-test networking (docs/research_oct25.md item 5): the validation queries of a pipeline build as
one "test set", with their stage-2 top-k lists and every query-query spectral similarity.

python scripts/net_prep.py <fwd val prefix> <stage2.txt> <rows_val.parquet> <pool_dir> <out.npz>
  fwd val prefix: <prefix>.top.parquet + <prefix>.jobs_*.gl.json (scripts/build_fwd_jobs.py, GLACIER workers)
  rows_val.parquet: the full validation rows (q_rows of every query, also those outside the reach-filtered top set)
Writes (per query of the top set, in qid order): qid, cls, truth pid, M (measured neutral mass), pol mask (1 = +, 2 = -);
the top-k table (qid index, pid, s2, label) and pairwise query similarity: modified cosine (max over the two queries'
spectra of one polarity, shift = precursor difference, tol 0.01 Da) and plain cosine, both with matched-peak counts.
Query spectra are rebuilt exactly as scripts/dump_ret_rows.py built them (clean_peaks 0.005 / top 64, enveda-180
positive-mode electron fix).
"""
import os, sys
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.spectra import Store, clean_peaks, neutral_mass, modcos

ELECTRON = 0.00054858
TOL = 0.01


@njit(cache=True)
def pair_sims(mz, it, off, prec, pol, q_of, nq):
    """Max modified cosine / cosine (and matched peaks) over spectrum pairs of the same polarity, per query pair."""
    n = len(prec)
    MC = np.zeros((nq, nq), np.float32); CO = np.zeros((nq, nq), np.float32)
    MN = np.zeros((nq, nq), np.int16); CN = np.zeros((nq, nq), np.int16)
    for a in range(n):
        for b in range(a + 1, n):
            qa, qb = q_of[a], q_of[b]
            if qa == qb or pol[a] != pol[b]:
                continue
            A = slice(off[a], off[a + 1]); B = slice(off[b], off[b + 1])
            s, k = modcos(mz[A], it[A], mz[B], it[B], prec[b] - prec[a], TOL)
            if s > MC[qa, qb]:
                MC[qa, qb] = s; MC[qb, qa] = s; MN[qa, qb] = k; MN[qb, qa] = k
            s, k = modcos(mz[A], it[A], mz[B], it[B], 0.0, TOL)
            if s > CO[qa, qb]:
                CO[qa, qb] = s; CO[qb, qa] = s; CN[qa, qb] = k; CN[qb, qa] = k
    return MC, MN, CO, CN


if __name__ == '__main__':
    prefix, s2_path, rows_p, pool_dir, out = sys.argv[1:6]
    import glob, json, lightgbm as lgb
    from casmi.fwdfeat import add_features
    T = pd.read_parquet(prefix + '.top.parquet')
    sc = {}
    for f in sorted(glob.glob(prefix + '.jobs_*.gl.json')):
        for q, d in json.load(open(f)).items():
            sc.setdefault(q, {}).update(d)
    T = add_features(T, sc).sort_values(['qid', 's1_r']).reset_index(drop=True)
    m = lgb.Booster(model_file=s2_path)
    feats = open(s2_path.replace('.txt', '_feats.txt')).read().split()
    T['s2'] = m.predict(T[feats].astype(np.float32))
    qids = np.sort(T.qid.unique()); qi = {q: i for i, q in enumerate(qids)}
    R = pd.read_parquet(rows_p, columns=['qid', 'truth', 'cls', 'q_rows']).drop_duplicates('qid').set_index('qid').loc[qids]
    P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['key'])
    pid_of = dict(zip(P.key, np.arange(len(P))))
    st = Store('/home/user/work/train_store'); meta = st.meta
    mzs, its, precs, pols, qof, Ms = [], [], [], [], [], []
    for q in qids:
        nms = []
        for i in map(int, R.loc[q, 'q_rows'].split(',')):
            mz, it = st.peaks(i); row = meta.loc[i]
            prec = row.precursor_mz
            if row.ingest_lib == 'enveda-180' and row.adduct.endswith('+'):
                prec -= ELECTRON
            a, b = clean_peaks(mz, it, prec, rel_floor=0.005, top_k=64)
            if len(a) >= 3:
                mzs.append(a.astype(np.float64)); its.append(b.astype(np.float64)); precs.append(prec)
                pols.append(0 if row.adduct.endswith('+') else 1); qof.append(qi[q])
                nms.append(neutral_mass(prec, row.adduct))
        Ms.append(float(np.nanmedian(nms)) if nms else np.nan)
    off = np.r_[0, np.cumsum([len(a) for a in mzs])].astype(np.int64)
    pol = np.array(pols, np.int64); q_of = np.array(qof, np.int64)
    MC, MN, CO, CN = pair_sims(np.concatenate(mzs), np.concatenate(its), off, np.array(precs), pol, q_of, len(qids))
    polmask = np.zeros(len(qids), np.int8)
    for q, p in zip(q_of, pol):
        polmask[q] |= 1 << p
    np.savez(out, qid=qids, cls=R.cls.values.astype(np.int8), truth=np.array([pid_of.get(k, -1) for k in R.truth]),
             M=np.array(Ms), polmask=polmask, n_spec=np.bincount(q_of, minlength=len(qids)),
             t_q=T.qid.map(qi).values.astype(np.int32), t_pid=T.pid.values.astype(np.int64), t_s1=T.s1.values,
             t_s2=T.s2.values, t_label=T.label.values.astype(np.int8), MC=MC, MN=MN, CO=CO, CN=CN)
    print('queries', len(qids), 'spectra', len(precs), 'top rows', len(T),
          'pairs with modcos >= 0.6:', int((MC >= 0.6).sum() // 2), '>= 0.7:', int((MC >= 0.7).sum() // 2))
