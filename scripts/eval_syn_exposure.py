"""Exposure-bias diagnostic for a PF-syn pair on the neutral rows: per query z-score each arm's pf column, then compare
the syn-minus-ctl shift of decoys whose structure had synthetic spectra in training (simulated, not dropped) with
decoys that had none. A positive gap = the model favours structures it saw as synthetic positives (membership bias).

python scripts/eval_syn_exposure.py <ctl rows.parquet> <syn rows.parquet> <sim structs.parquet> <dropped pids.txt>
(pilot: /home/user/ds/casmi26-sim-jobs/sim1_structs.parquet, syn_drop_B.txt)."""
import sys, numpy as np, pandas as pd
ctl, syn, structs, drop = sys.argv[1:5]
S = pd.read_parquet(structs, columns=['pid']).pid.values
sim = np.setdiff1d(S, np.loadtxt(drop, dtype=np.int64))
cols = ['qid', 'pid', 'label', 'in_train', 'in_coco', 'pf_cos', 'pf_bern']
A = pd.read_parquet(ctl, columns=cols); B = pd.read_parquet(syn, columns=cols)
for c in ('pf_bern', 'pf_cos'):
    z = lambda D: D.groupby('qid')[c].transform(lambda s: (s - s.mean()) / (s.std() or 1.0))
    d = (z(B) - z(A)).values
    m = (A.label == 0).values; s = np.isin(A.pid.values, sim)
    for name, pop in (('coco', (A.in_coco == 1) & (A.in_train == 0)), ('train', A.in_train == 1), ('other', (A.in_coco == 0) & (A.in_train == 0))):
        pop = pop.values & m
        a, b = d[pop & s], d[pop & ~s]
        print(f'{c} {name:5s} sim n={len(a):6d} dz={a.mean() if len(a) else np.nan:+.4f} | unsim n={len(b):6d} dz={b.mean() if len(b) else np.nan:+.4f} | gap {(a.mean() if len(a) else np.nan) - (b.mean() if len(b) else np.nan):+.4f}')
