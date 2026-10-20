"""Component metrics of fingerprint-model scores on retrieval rows (plan gate for PeakFormer).

python scripts/eval_pf.py <rows.parquet> [score columns ...]   (default fp_cos fp_bern pf_cos pf_bern; pool v4 rows)
Class-2 queries whose truth is in COCONUT and not a training structure: within-formula MRR of each score (rank of the
truth among same-formula candidates) and pairwise win rates of the truth against same-formula train / COCONUT decoys;
also the within-query MRR over all candidates. Gate (docs/coconut_gap_plan.md): within-formula MRR >= .46 and wins
vs COCONUT decoys >= .93 (fp_cos baseline .408 / .893).
"""
import sys
import numpy as np, pandas as pd

rows = sys.argv[1]
cols = sys.argv[2:] or ['fp_cos', 'fp_bern', 'pf_cos', 'pf_bern']
D = pd.read_parquet(rows)
if 'formula' not in D:
    D['formula'] = pd.read_parquet('/home/user/work/pool4/pool.parquet',
                                   columns=['formula']).formula.to_numpy(dtype=object)[D.pid.values]
cols = [c for c in cols if c in D.columns]
D = D[D.cls == 2]
t = D[D.label == 1].drop_duplicates('qid').set_index('qid')
keepq = t.index[(t.in_coco == 1) & (t.in_train == 0)] if 'in_train' in t else t.index[t.in_coco == 1]
D = D[D.qid.isin(keepq)]
if 'pf_cos' in D and 'pf_bern' in D:
    g = D.groupby('qid')
    for c in ('pf_cos', 'pf_bern'):
        D[c + '_zq'] = (D[c] - g[c].transform('mean')) / g[c].transform('std').replace(0, 1)
    D['pf_sum'] = D.pf_cos_zq + D.pf_bern_zq; cols.append('pf_sum')
tf = D[D.label == 1].drop_duplicates('qid').set_index('qid').formula
D['same_f'] = D.formula.values == D.qid.map(tf).values
print('queries', D.qid.nunique(), 'rows', len(D), 'mean same-formula group', round(D[D.same_f].groupby('qid').size().mean(), 1))
for c in cols:
    tv = D[D.label == 1].drop_duplicates('qid').set_index('qid')[c]
    d = D.assign(tv=D.qid.map(tv))
    neg = d[d.label == 0]
    sf = neg[neg.same_f]
    r_f = 1 + sf.groupby('qid').apply(lambda x: (x[c] > x.tv).sum(), include_groups=False).reindex(tv.index).fillna(0)
    r_a = 1 + neg.groupby('qid').apply(lambda x: (x[c] > x.tv).sum(), include_groups=False).reindex(tv.index).fillna(0)
    w_tr = (sf[sf.in_train == 1].tv > sf[sf.in_train == 1][c]).mean() if 'in_train' in sf else np.nan
    cc = sf[(sf.in_coco == 1) & (sf.get('in_train', 0) == 0)]
    w_co = (cc.tv > cc[c]).mean()
    print(f'{c:10s} within-formula MRR {np.mean(1 / r_f):.4f} top1 {np.mean(r_f == 1):.4f} | all-candidates MRR '
          f'{np.mean(1 / r_a):.4f} | wins vs train decoys {w_tr:.4f} vs COCONUT decoys {w_co:.4f}')
