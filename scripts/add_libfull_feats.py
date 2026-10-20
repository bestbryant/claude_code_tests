"""Library-match feature against ALL training spectra of a candidate (not the 10-per-structure lib10 library).

python scripts/add_libfull_feats.py <rows.parquet> <out.parquet> [lib_all.npz]
Adds lib_all (best weighted-entropy similarity over every same-polarity training spectrum of the candidate, test adducts
only), lib_all_n (number of compared pairs) and lib_all_r / lib_all_gap. Hiding exactly as scripts/dump_ret_rows.py
(standard mode): class 2 hides the truth structure, class 1 hides the query spectra. Query spectra are cleaned as in
the dump (top 64 peaks, 0.5% floor, electron-mass corrected precursor for positive enveda-180 rows).
"""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.analog import AnalogLib
from casmi.build import build_analog_lib
from casmi.entropy import entropy_sim
from casmi.spectra import clean_peaks

ELECTRON = 0.00054858
rows_p, out = sys.argv[1:3]
lib_p = sys.argv[3] if len(sys.argv) > 3 else '/home/user/work/lib_all.npz'
if not os.path.exists(lib_p):
    t0 = time.time(); n = build_analog_lib('/home/user/data/train.parquet', lib_p, per_struct=100000)
    print('built', lib_p, n, f'{time.time() - t0:.0f}s', flush=True)
L = AnalogLib(lib_p)
lib_of = pd.Series(np.arange(len(L.key))).groupby(L.key).apply(list).to_dict()
D = pd.read_parquet(rows_p).reset_index(drop=True)
meta = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['ingest_lib', 'adduct'])
Q = D.drop_duplicates('qid')[['qid', 'cls', 'truth', 'q_rows']].copy()
Q['rows'] = Q.q_rows.map(lambda s: [int(x) for x in s.split(',')])
need = sorted({r for rr in Q.rows for r in rr})
pf = pq.ParquetFile('/home/user/data/train.parquet')
want = np.zeros(pf.metadata.num_rows, bool); want[need] = True
spec, base = {}, 0
for g in range(pf.metadata.num_row_groups):
    n = pf.metadata.row_group(g).num_rows
    loc = np.where(want[base:base + n])[0].tolist()
    if loc:
        t = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz']).take(loc).to_pandas()
        for j, (mz, it, prec) in zip(loc, zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz)):
            r = base + j
            add = meta.adduct.values[r] or ''
            if meta.ingest_lib.values[r] == 'enveda-180' and add.endswith('+'):
                prec -= ELECTRON
            a, b = clean_peaks(np.asarray(mz, np.float64), np.asarray(it, np.float64), prec, rel_floor=0.005, top_k=64)
            if len(a) >= 3:
                spec[r] = (a, b, 0 if add.endswith('+') else 1)
    base += n
print('query spectra', len(spec), 'of', len(need), flush=True)
lib_all = np.zeros(len(D)); lib_all_n = np.zeros(len(D))
qinfo = Q.set_index('qid')
t0 = time.time()
for n_done, (qid, g) in enumerate(D.groupby('qid', sort=False)):
    q = qinfo.loc[qid]
    qs = [spec[r] for r in q.rows if r in spec]
    hr = set(q.rows) if q.cls == 1 else set()
    for i, k in zip(g.index.values, g.key.values):
        if q.cls == 2 and k == q.truth:
            continue
        best, cnt = 0.0, 0
        for j in lib_of.get(k, ()):
            if hr and int(L.row[j]) in hr:
                continue
            a, b = L.off[j], L.off[j + 1]
            for qm, qi, qp in qs:
                if qp != L.pol[j]:
                    continue
                s = entropy_sim(qm, qi, L.mz[a:b], L.it[a:b], 0.01)
                best = max(best, s); cnt += 1
        lib_all[i], lib_all_n[i] = best, cnt
    if n_done % 500 == 0:
        print('queries', n_done, f'{time.time() - t0:.0f}s', flush=True)
D['lib_all'] = lib_all; D['lib_all_n'] = np.log1p(lib_all_n)
gq = D.groupby('qid').lib_all
D['lib_all_r'] = gq.rank(ascending=False, method='min'); D['lib_all_gap'] = D.lib_all - gq.transform('max')
D.to_parquet(out)
print('done', out, D.shape)
