"""PMI correction of the PeakFormer bit score: pf_bern (= f.z, sum of bit logits over the candidate's on-bits) minus
lam * sum of logit(m) over the same bits, m = training-positive bit marginal (casmi26-pf-data fpx.npz, train split;
half spectrum-weighted, half structure-uniform like the structure-balanced batches of model d).

python scripts/eval_pf_pmi.py <rows with pf_bern> [<more>...]   -> pf_bern_pmi05 / pf_bern_pmi1 + eval_pf_neutral.py
Writes the per-bit prior to /home/user/work/pf_bitprior.npy (logit(m) over the model's kept bits) for reuse.
"""
import os, subprocess, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = '/home/user/work'
pri_p = f'{W}/pf_bitprior.npy'
keep = np.asarray(torch.load(f'{W}/pf_local/pf_model_d_final.pt', map_location='cpu', weights_only=False)['_keep'])
if not os.path.exists(pri_p):
    S = np.load('/home/user/ds/casmi26-fp-data/spectra.npz')
    X = np.load('/home/user/ds/casmi26-pf-data/fpx.npz')
    assert np.array_equal(X['keep'], keep)
    ptr = X['s_ptr']; n = len(ptr) - 1
    cnt = np.bincount(S['sidx'][S['split'] == 0], minlength=n).astype(np.float64)
    w = 0.5 * cnt / cnt.sum() + 0.5 * (cnt > 0) / (cnt > 0).sum()
    idx = X['s_idx']; m = np.zeros(len(keep))
    for a in range(0, n, 20000):
        b = min(n, a + 20000); lens = np.diff(ptr[a:b + 1])
        m += np.bincount(idx[ptr[a]:ptr[b]].astype(np.int64), weights=np.repeat(w[a:b], lens), minlength=len(keep))
    m = np.clip(m, 1e-5, 1 - 1e-5)
    np.save(pri_p, np.log(m) - np.log1p(-m))
lg = np.load(pri_p)
from casmi.richfp import N_RAW
km = -np.ones(N_RAW, np.int64); km[keep] = np.arange(len(keep))
C = np.load(f'{W}/richfp_ret10.npz'); cpid, cptr, cidx = C['pid'], C['ptr'], C['idx']
pos = dict(zip(cpid.tolist(), range(len(cpid))))
for path in sys.argv[1:]:
    D = pd.read_parquet(path)
    pr = {}
    for p in D.pid.unique():
        j = pos.get(int(p))
        if j is None:
            pr[p] = np.nan; continue
        a = km[cidx[cptr[j]:cptr[j + 1]].astype(np.int64)]
        pr[p] = lg[a[a >= 0]].sum()
    v = D.pid.map(pr).values
    print(os.path.basename(path), 'pids without cached bits', int(np.isnan(v).sum()), flush=True)
    for lam, nm in ((0.5, '05'), (1.0, '1')):
        D[f'pf_bern_pmi{nm}'] = D.pf_bern.values - lam * np.nan_to_num(v)
    out = path.replace('.parquet', '_pfpmi.parquet'); D.to_parquet(out)
    subprocess.run([sys.executable, 'scripts/eval_pf_neutral.py', out, 'pf_bern,pf_bern_pmi05,pf_bern_pmi1'])
