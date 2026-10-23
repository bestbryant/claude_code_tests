"""Membership-neutral evaluation of spectral scores (PeakFormer / FP columns) on class-2 COCONUT validation truths.

python scripts/eval_pf_neutral.py <rows_fp.parquet> [score columns, default pf_bern,pf_cos,fp_bern,fp_cos]
Within the truth's formula group, the truth's score is compared with every same-formula decoy; decoys are split into
training structures (in_train) and COCONUT-not-train structures and compared within (query, Morgan-Tanimoto-to-truth bin)
cells holding both types. Reported per score: matched win rate vs training
decoys (the prior-free measure of isomer discrimination: training decoys are positives elsewhere, so no membership prior
helps or hurts them), matched win rate vs COCONUT decoys, and the excess (COCONUT - training) = the population prior
"COCONUT-not-train is never the answer" (raw within-formula MRR rewards that prior; hidden COCONUT answers do not get it).
"""
import sys
import numpy as np, pandas as pd

rows = sys.argv[1]
cols = (sys.argv[2] if len(sys.argv) > 2 else 'pf_bern,pf_cos,fp_bern,fp_cos').split(',')
W = '/home/user/work'
D = pd.read_parquet(rows, columns=['qid', 'pid', 'label', 'cls', 'in_train', 'in_coco'] + cols)
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['formula'])
D['formula'] = P.formula.to_numpy(dtype=object)[D.pid.values]
fp = np.load(f'{W}/pool4/pool_fp.npy', mmap_mode='r')
POP8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)
t = D[(D.label == 1) & (D.cls == 2) & (D.in_coco == 1)]
bins = np.array([0, 0.2, 0.3, 0.4, 0.55, 0.7, 1.01])
rec = []
for r in t.itertuples():
    g = D[(D.qid == r.qid) & (D.label == 0) & (D.formula == r.formula)]
    g = g[(g.in_train == 1) | ((g.in_coco == 1) & (g.in_train == 0))]
    if len(g) == 0:
        continue
    a = np.asarray(fp[r.pid]); B = np.asarray(fp[g.pid.values])
    inter = POP8[np.bitwise_and(B, a)].sum(1); union = POP8[np.bitwise_or(B, a)].sum(1)
    tan = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    b = np.digitize(tan, bins[1:-1])
    for c in cols:
        win = (getattr(r, c) > g[c].values).astype(float) + 0.5 * (getattr(r, c) == g[c].values)
        for typ, m in (('train', g.in_train.values == 1), ('coco', g.in_train.values == 0)):
            for k in np.unique(b[m]):
                sel = m & (b == k)
                rec.append((c, r.qid, int(k), typ, float(win[sel].mean()), float(tan[sel].mean())))
R = pd.DataFrame(rec, columns=['col', 'qid', 'bin', 'typ', 'win', 'tan'])
print('class-2 COCONUT truths with same-formula decoys:', t.qid.nunique())
# (query, Tanimoto bin) cells holding both decoy types: the prior is concentrated in near analogs (Tanimoto > 0.4),
# so pair-weighted averages (dominated by distant decoys) hide it
for c in cols:
    q = R[R.col == c].pivot_table(index=['qid', 'bin'], columns='typ', values='win').dropna()
    hi = q[q.index.get_level_values('bin') >= 3]
    print(f'{c:8s} vs training decoys {q.train.mean():.3f} | vs COCONUT decoys {q.coco.mean():.3f} | excess (prior) '
          f'{(q.coco - q.train).mean():+.3f} (cells {len(q)}) | near analogs (Tanimoto > 0.4): train {hi.train.mean():.3f} '
          f'coco {hi.coco.mean():.3f} excess {(hi.coco - hi.train).mean():+.3f} (cells {len(hi)})')
