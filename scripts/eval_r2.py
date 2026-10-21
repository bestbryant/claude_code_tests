"""Offline gate for the second-ranker insertion (OWN_R2 in casmi/run_own.py): the main stage-1 ranker's list with the
second ranker's top-1 moved to a fixed slot (when it ranks below that slot), MRR@25 by class / subset, class 2 also
doc-shift weighted, and a test-mix estimate (45 % class 1, 55 % weighted class 2).

python scripts/eval_r2.py <rows.parquet> <main_ranker.txt> <second_ranker.txt> [slots, e.g. 2,3,4]
"""
import os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.docw import query_weights

rows, rk1, rk2 = sys.argv[1:4]
SLOTS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '2,3,4').split(',')]
W = '/home/user/work'
D = pd.read_parquet(rows)
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
for c, p in (('s1', rk1), ('s2', rk2)):
    f = open(p.replace('.txt', '_feats.txt')).read().split('\n')
    D[c] = lgb.Booster(model_file=p).predict(D[f].astype(float).values)
D['w'] = query_weights(D, f'{W}/pool4/doc_w.parquet', 1.0)
m = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['inchikey14', 'ingest_lib'])
npx = set(m.inchikey14[m.ingest_lib == 'enveda-np-examples'])
P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['key', 'in_coco']).set_index('key')

res = []
for q, g in D.groupby('qid', sort=False):
    o = g.sort_values('s1', ascending=False)
    lab = o.label.values; pid = o.pid.values
    top2 = g.pid.values[np.argmax(g.s2.values)]
    t = g[g.label == 1]
    tk = t.truth.iloc[0] if len(t) else None
    rec = dict(qid=q, cls=int(g.cls.iloc[0]), w=float(g.w.iloc[0]),
               sub='npx' if tk in npx else ('np' if P.in_coco.get(tk, 0) == 1 else 'other'),
               r2_top1=float(g.label.values[np.argmax(g.s2.values)] == 1))

    def rr(order):
        h = np.flatnonzero(order[:25] == 1)
        return 1.0 / (h[0] + 1) if len(h) else 0.0
    rec['main'] = rr(lab)
    rec['second'] = rr(g.sort_values('s2', ascending=False).label.values)
    j = int(np.flatnonzero(pid == top2)[0])
    for s in SLOTS:
        if j < s - 1:
            rec[f'ins{s}'] = rec['main']
        else:
            l2 = np.r_[lab[:s - 1], lab[j], np.delete(lab, j)[s - 1:]]
            rec[f'ins{s}'] = rr(l2)
    res.append(rec)
R = pd.DataFrame(res)
cols = ['main', 'second'] + [f'ins{s}' for s in SLOTS]
print('queries', len(R), '| second-ranker top-1 correct', round(R.r2_top1.mean(), 3))
for (c, sub), d in R.groupby(['cls', 'sub']):
    print(f'cls {c} {sub:6s} n {len(d):5d}', ' '.join(f'{k} {d[k].mean():.4f}' for k in cols))
c1, c2 = R[R.cls == 1], R[R.cls == 2]
for k in cols:
    a, b = c1[k].mean(), np.average(c2[k], weights=c2.w)
    print(f'{k:7s} cls1 {a:.4f} cls2 {c2[k].mean():.4f} cls2w {b:.4f} mix {0.45 * a + 0.55 * b:.4f}')
