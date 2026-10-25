"""A/B split of the 620-query membership-neutral evaluation set for the PF-syn pilot (review wf_4f2803ec-ac0).

python scripts/syn_ab_split.py <sub.parquet> <out_dir> [seed]
Half A of the queries keeps its windows in the synthetic training set (the deployment regime: the whole pool is
simulated, test answers included); half B is held out: every pool id that occurs only in B windows (in no A window)
is written to <out_dir>/syn_drop_B.txt (PF_SYN_DROP for scripts/train_pf_xla.py) and the qids of each half to
<out_dir>/neutral_A.txt / neutral_B.txt. B measures whether the prior removal generalises to never-simulated
structures; A (at full-pool density, PF_SYN_NEUTRAL_W) is the decision half.
"""
import os, sys
import numpy as np, pandas as pd

sub, out = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
D = pd.read_parquet(sub, columns=['qid', 'pid'])
q = np.random.default_rng(seed).permutation(np.unique(D.qid.values))
A, B = np.sort(q[:len(q) // 2]), np.sort(q[len(q) // 2:])
pa = set(D.pid[D.qid.isin(A)]); pb = set(D.pid[D.qid.isin(B)])
drop = np.array(sorted(pb - pa), np.int64)
os.makedirs(out, exist_ok=True)
np.savetxt(f'{out}/syn_drop_B.txt', drop, fmt='%d')
np.savetxt(f'{out}/neutral_A.txt', A, fmt='%d'); np.savetxt(f'{out}/neutral_B.txt', B, fmt='%d')
print('queries A', len(A), 'B', len(B), '| pids in A windows', len(pa), 'in B windows', len(pb), 'shared', len(pa & pb),
      '| dropped (B only)', len(drop))
