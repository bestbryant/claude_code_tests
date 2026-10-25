"""GLACIER similarity column 'gl' for retrieval rows from the PF-syn simulations (scripts/prep_syn_data.py syn.npz),
with the stage-2 definition (casmi/fwdrun.py): weighted entropy similarity (0.01 Da) between each positive [M+H]+
query spectrum and the candidate's predicted spectrum at the simulated CE nearest the query's, averaged over the
query's spectra; NaN when the query has no positive [M+H]+ spectrum or the candidate was not simulated.

python scripts/add_gl_feats.py <rows.parquet> <syn.npz> <out.parquet>   (rows need qid, pid, q_rows)
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.entropy import entropy_sim
from casmi.fwdrun import query_peaks
from casmi.spectra import Store

rows, synp, out = sys.argv[1:4]
D = pd.read_parquet(rows)
Z = np.load(synp)
pid = Z['pid']; sidx = Z['sidx']; ce = Z['ce']; PM, PI = Z['PM'], Z['PI']
pred = {}            # pid -> {ce: (mz sorted, it)}
for j in range(len(sidx)):
    m, i = PM[j].astype(np.float64), PI[j].astype(np.float64)
    ok = i > 0
    o = np.argsort(m[ok])
    pred.setdefault(int(pid[sidx[j]]), {})[float(ce[j])] = (m[ok][o], i[ok][o] / max(i[ok].max(), 1e-12) if ok.any() else i[ok])
st = Store('/home/user/work/train_store'); meta = st.meta
gl = np.full(len(D), np.nan)
pos = {k: v for k, v in zip(D.index, range(len(D)))}
cov = 0
for qid, g in D.groupby('qid', sort=False):
    qs = []
    for i in map(int, g.q_rows.iloc[0].split(',')):
        r = meta.loc[i]
        if r.adduct != '[M+H]+' or str(r.ionization_mode).lower() != 'positive':
            continue
        mz, it = st.peaks(i)
        a, b = query_peaks(np.asarray(mz, float), np.asarray(it, float), float(r.precursor_mz))
        c = r.collision_energy_ev
        qce = float(np.mean(c)) if c is not None and len(c) else 30.0
        if len(a) >= 2:
            qs.append((a, b, qce))
    if not qs:
        continue
    cov += 1
    for ix, p in zip(g.index, g.pid.values):
        P = pred.get(int(p))
        if not P:
            continue
        v = []
        for a, b, qce in qs:
            m, i = P[min(P, key=lambda e: abs(e - qce))]
            if len(m):
                v.append(float(entropy_sim(a, b, m, i, 0.01)))
        if v:
            gl[pos[ix]] = np.mean(v)
D['gl'] = gl
D.to_parquet(out)
print('queries', D.qid.nunique(), 'with [M+H]+ spectra', cov, '| rows with gl', int(np.isfinite(gl).sum()), 'of', len(D))
