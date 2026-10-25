"""Paired syn-vs-ctl comparison on the membership-neutral set (PF-syn pilot gate, review wf_4f2803ec-ac0).

python scripts/eval_pf_pair.py <ctl rows_fp.parquet> <syn rows_fp.parquet> [gl rows.parquet] [qids_A.txt qids_B.txt]
Cells as scripts/eval_pf_neutral.py: within the truth's formula group, (query, Morgan-Tanimoto-to-truth bin) cells
holding both training and COCONUT-not-train decoys; per arm and column the matched win rate vs training decoys, vs
COCONUT decoys and the excess (COCONUT - training). Paired bootstrap over queries (2,000 draws, the same draws for both
arms): delta point estimate and 95% interval. Columns: pf_cos (primary), pf_bern, and with a gl file (scripts/
add_gl_feats.py) the combinations z(pf) + z(gl) (pf_cos_gl, pf_bern_gl) within each (query, formula) group -- PF-syn must add beyond GLACIER,
which stage 2 already has -- on GLACIER-covered queries, plus pf alone on uncovered ones. With A/B qid files the halves
are reported separately (A: windows simulated, the decision half; B: held out).
Gate (pre-registered Oct 25): on half A, pf_cos delta win vs training decoys >= +0.06 with the 95% interval above 0,
delta win vs COCONUT decoys >= -0.01, excess_syn <= excess_ctl and >= -0.02; and the pf_cos_gl delta on covered A
queries > 0. Half B is reported (generalisation), not gated.
"""
import sys
import numpy as np, pandas as pd

W = '/home/user/work'
ctl_p, syn_p = sys.argv[1], sys.argv[2]
gl_p = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
halves = {'A': np.loadtxt(sys.argv[4], dtype=np.int64), 'B': np.loadtxt(sys.argv[5], dtype=np.int64)} if len(sys.argv) > 5 else {}
P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['formula'])
FORM = P.formula.to_numpy(dtype=object)
fp = np.load(f'{W}/pool4/pool_fp.npy', mmap_mode='r')
POP8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)
bins = np.array([0, 0.2, 0.3, 0.4, 0.55, 0.7, 1.01])


def load(p, extra=None):
    D = pd.read_parquet(p, columns=['qid', 'pid', 'label', 'cls', 'in_train', 'in_coco', 'pf_cos', 'pf_bern'])
    D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
    D['formula'] = FORM[D.pid.values]
    if extra is not None:
        D = D.merge(extra, on=['qid', 'pid'], how='left')
        z = lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
        grp = D.groupby(['qid', 'formula'])
        for c in ('pf_cos', 'pf_bern'):
            D[c + '_gl'] = grp[c].transform(z) + grp['gl'].transform(z)
    return D


def cells(D, cols):
    """(col, qid) -> (sum of train wins, sum of coco wins, number of cells) over cells holding both decoy types."""
    t = D[(D.label == 1) & (D.cls == 2) & (D.in_coco == 1)]
    out = {}
    for r in t.itertuples():
        g = D[(D.qid == r.qid) & (D.label == 0) & (D.formula == r.formula)]
        g = g[(g.in_train == 1) | ((g.in_coco == 1) & (g.in_train == 0))]
        if len(g) == 0:
            continue
        a = np.asarray(fp[r.pid]); B = np.asarray(fp[g.pid.values])
        tan = POP8[np.bitwise_and(B, a)].sum(1) / np.maximum(POP8[np.bitwise_or(B, a)].sum(1), 1)
        b = np.digitize(tan, bins[1:-1]); tr = g.in_train.values == 1
        for c in cols:
            tv = getattr(r, c)
            if not np.isfinite(tv):
                continue
            gv = g[c].values
            win = (tv > gv).astype(float) + 0.5 * (tv == gv)
            st, sc, n = 0.0, 0.0, 0
            for k in np.unique(b):
                m1, m2 = tr & (b == k), ~tr & (b == k)
                if m1.any() and m2.any():
                    st += win[m1].mean(); sc += win[m2].mean(); n += 1
            if n:
                out[(c, r.qid)] = (st, sc, n)
    return out


gl = pd.read_parquet(gl_p, columns=['qid', 'pid', 'gl']) if gl_p else None
cols = ['pf_cos', 'pf_bern'] + (['pf_cos_gl', 'pf_bern_gl'] if gl is not None else [])
arms = {name: cells(load(p, gl), cols) for name, p in (('ctl', ctl_p), ('syn', syn_p))}
covered = set(gl.qid[np.isfinite(gl.gl)]) if gl is not None else set()
rng = np.random.default_rng(0)
for hname, hq in [('all', None)] + list(halves.items()):
    for c in cols:
        for scope in (['covered', 'uncovered'] if c.endswith('_gl') else ['all']):
            cc = c[:-3] if scope == 'uncovered' else c      # uncovered queries: pf alone
            q = sorted({k[1] for k in arms['ctl'] if k[0] == cc} & {k[1] for k in arms['syn'] if k[0] == cc})
            if hq is not None:
                q = [x for x in q if x in set(hq.tolist())]
            if scope == 'covered':
                q = [x for x in q if x in covered]
            elif scope == 'uncovered':
                q = [x for x in q if x not in covered]
            if not q:
                continue
            M = {a: np.array([arms[a][(cc, x)] for x in q]) for a in arms}

            def stats(idx):
                r = {}
                for a in M:
                    s = M[a][idx].sum(0)
                    r[a] = (s[0] / s[2], s[1] / s[2])
                return (r['syn'][0] - r['ctl'][0], r['syn'][1] - r['ctl'][1],
                        (r['syn'][1] - r['syn'][0]) - (r['ctl'][1] - r['ctl'][0]), r)
            d_tr, d_co, d_ex, r = stats(np.arange(len(q)))
            bs = np.array([stats(rng.integers(0, len(q), len(q)))[:3] for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            print(f'[{hname:3s}] {cc:12s} {scope:9s} n={len(q):3d} | ctl train {r["ctl"][0]:.3f} coco {r["ctl"][1]:.3f} '
                  f'excess {r["ctl"][1] - r["ctl"][0]:+.3f} | syn train {r["syn"][0]:.3f} coco {r["syn"][1]:.3f} excess '
                  f'{r["syn"][1] - r["syn"][0]:+.3f} | delta train {d_tr:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}] coco {d_co:+.3f} '
                  f'[{lo[1]:+.3f},{hi[1]:+.3f}] excess {d_ex:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}]')
