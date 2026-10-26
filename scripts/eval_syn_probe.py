"""Offline analog of the LB spectral-only probes (kernels/casmi26-specpf*-coco, OWN_SPEC_ONLY=pf_bern) on the 620-query
membership-neutral rows (/home/user/work/fpb_local/sub.parquet scored by scripts/add_fp_feats.py): per class-2 query,
reciprocal rank (cap 25, ties half) of the truth among the candidates of one source set ordered by a pf column; paired
bootstrap (2,000 draws) of the difference between two scorings, overall and per half.

python scripts/eval_syn_probe.py <rows A.parquet> <rows B.parquet> [coco|cotr] [qids_A.txt qids_B.txt]
coco: COCONUT-not-train candidates (as OWN_SPEC_SRC=coco); cotr: training + COCONUT-not-train (OWN_SPEC_SRC=cotr).
Anchor (Oct 26): PF-d (fpb_local/sub.parquet) -> PF-b (v17b/sub_pf.parquet) pf_bern coco +0.0636 [+0.041, +0.086]
while the LB probe pair specpfd-coco -> specpfb-coco moved +0.031 (ratio 0.49).
"""
import sys
import numpy as np, pandas as pd

COLS = ['qid', 'pid', 'label', 'cls', 'in_train', 'in_coco', 'pf_cos', 'pf_bern']


def rr(p, c, src):
    D = pd.read_parquet(p, columns=COLS)
    D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0          # hidden truths count as not-train (simulation)
    keep = (D.in_coco == 1) & (D.in_train == 0)
    if src == 'cotr':
        keep |= D.in_train == 1
    D = D[keep]
    t = D[(D.label == 1) & (D.cls == 2)][['qid', c]].rename(columns={c: 't'})
    D = D.merge(t, on='qid')
    r = D[D.label == 0].assign(b=lambda x: (x[c] > x.t) + 0.5 * (x[c] == x.t)).groupby('qid').b.sum()
    q = D.qid.unique(); r = r.reindex(q).fillna(0) + 1
    return pd.Series(np.where(r <= 25, 1 / r, 0.0), index=q)


if __name__ == '__main__':
    a, b = sys.argv[1], sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else 'coco'
    halves = {'A': np.loadtxt(sys.argv[4], dtype=np.int64), 'B': np.loadtxt(sys.argv[5], dtype=np.int64)} if len(sys.argv) > 5 else {}
    for c in ('pf_bern', 'pf_cos'):
        x, y = rr(a, c, src), rr(b, c, src)
        y = y.reindex(x.index)
        for h, hq in [('all', None)] + list(halves.items()):
            m = np.ones(len(x), bool) if hq is None else np.isin(x.index.values, hq)
            d = (y - x).values[m]
            rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print(f'{src} {c:7s} [{h:3s}] n={len(d):3d} A={x.values[m].mean():.4f} B={y.values[m].mean():.4f} '
                  f'delta={d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]')
