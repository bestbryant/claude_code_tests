"""Offline evaluation of test-test networking on a scripts/net_prep.py pseudo-test (Oct 25 result: NOT adopted).

python scripts/net_eval.py <net_prep.npz> <pool_dir> [nap|confident]
nap: NAP-style consensus -- for each query, the modified-cosine-weighted mean over its spectral neighbours (modcos >= thr,
     >= mn matched peaks) of sum_d softmax(s2_r / tau)(d) * Tanimoto(c, d), z-scored within the query and added to the
     stage-2 score with weight lam. Guard: class-2 truths of OTHER queries are dropped from neighbour lists (in the
     simulation they are training structures with library spectra from the neighbour's side; in a real test they are not).
confident: only for queries whose stage-2 top-1 gap is below qgap, max Tanimoto of each candidate to the top-1 of a
     confident neighbour (gap >= nbgap; gap >= 2 -> top-1 correct 93% on v18c val), optionally only neighbours whose
     neutral-mass difference is a common transform (0, CH2, O, H2, hexose, deoxyhexose, pentose, glucuronide, CH2O, acetyl,
     C2H4, H2O, malonyl, sulfate/phosphate-like 79.957, C2H4O, CH4).
v18c val (979 queries): nap -0.004 to -0.16 overall at every setting; confident at most +0.0013 overall (one or two
queries), larger gains only on post-hoc subsets of 8-60 queries. Spectral neighbours are weak structural evidence here
(modcos >= 0.7 with >= 6 matched peaks: truths Tanimoto >= 0.5 in 26% of pairs).
"""
import sys, itertools
import numpy as np
z = np.load(sys.argv[1])
fp = np.load(sys.argv[2] + '/pool_fp.npy', mmap_mode='r')
n = len(z['qid']); cls = z['cls']; truth = z['truth']
tq, tpid, ts2, tlab = z['t_q'], z['t_pid'], z['t_s2'], z['t_label']
MC, MN = z['MC'], z['MN']
ups = np.unique(tpid); ui = {p: i for i, p in enumerate(ups)}
F = np.unpackbits(np.asarray(fp[ups]), axis=1).astype(np.float32); cnt = F.sum(1)
rows = [np.where(tq == q)[0] for q in range(n)]
hidden = set(truth[cls == 2].tolist())        # class-2 truths: not training structures in a real test
def tani(a, b):
    A, B = F[[ui[p] for p in a]], F[[ui[p] for p in b]]
    i = A @ B.T
    return i / (A.sum(1)[:, None] + B.sum(1)[None, :] - i)
def mrr(score):
    rr = np.zeros(n)
    for q in range(n):
        r = rows[q]; l = tlab[r]
        if l.any():
            o = np.argsort(-score[r], kind='stable'); rank = np.where(l[o] == 1)[0][0] + 1; rr[q] = 1 / rank
    return rr
base = mrr(ts2)
print('base stage-2 MRR', base.mean().round(4), {c: base[cls == c].mean().round(4) for c in (1, 2)})
def net_scores(thr, mn, tau, guard, k=30):
    net = np.zeros(len(tq)); has = np.zeros(n, bool)
    P = {}
    for r in range(n):
        idx = rows[r]; s = ts2[idx]
        keep = np.array([(not guard) or (p not in hidden) or p == truth[r] for p in tpid[idx]])
        idx, s = idx[keep], s[keep]
        o = np.argsort(-s)[:k]; idx, s = idx[o], s[o]
        e = np.exp((s - s.max()) / tau); P[r] = (tpid[idx], e / e.sum())
    for q in range(n):
        nb = np.where((MC[q] >= thr) & (MN[q] >= mn))[0]
        if len(nb) == 0: continue
        has[q] = True
        cq = tpid[rows[q]]; tot = np.zeros(len(cq)); wsum = 0
        for r in nb:
            w = MC[q, r]; d, p = P[r]
            tot += w * (tani(cq, d) @ p); wsum += w
        net[rows[q]] = tot / wsum
    return net, has
MODE = sys.argv[3] if len(sys.argv) > 3 else 'confident'
if MODE == 'nap':
    for thr, mn, tau, guard in itertools.product((0.6, 0.7, 0.8), (4, 6), (0.5, 1.0), (True, False)):
        net, has = net_scores(thr, mn, tau, guard)
        res = []
        for lam in (0.5, 1, 2, 4):
            # within-query z of net added to s2 (s2 scale ~ lambdarank)
            adj = ts2.copy()
            for q in np.where(has)[0]:
                r = rows[q]; v = net[r]; sd = v.std()
                if sd > 0: adj[r] = ts2[r] + lam * (v - v.mean()) / sd * ts2[r].std()
            m = mrr(adj)
            res.append(f'lam{lam}: all {m.mean() - base.mean():+.4f} netq {m[has].mean() - base[has].mean():+.4f}')
        print(f'thr {thr} mn {mn} tau {tau} guard {guard} netq {has.sum()} ({has.mean():.2f}) base_netq {base[has].mean():.3f} | ' + ' | '.join(res), flush=True)
else:
    M = z['M']
    DELTAS = np.array([0, 14.01565, 15.99491, 2.01565, 162.05282, 146.05791, 132.04226, 176.03209, 30.01056, 42.01056,
                       28.0313, 18.01056, 86.00039, 79.95682, 44.0262, 16.0313])
    dM = np.abs(M[:, None] - M[None, :])
    tol = np.maximum(0.003, 5e-6 * np.maximum(M[:, None], M[None, :]))
    TR = np.zeros((n, n), bool)
    for d in DELTAS:
        TR |= np.abs(dM - d) <= tol
    np.fill_diagonal(TR, False)
    # per-query confidence: stage-2 top1 - top2 gap, top-1 pid
    top1 = np.zeros(n, np.int64); gap = np.zeros(n)
    for q in range(n):
        s = ts2[rows[q]]; o = np.argsort(-s); top1[q] = tpid[rows[q]][o[0]]; gap[q] = s[o[0]] - s[o[1]] if len(o) > 1 else 9
    acc1 = np.array([tlab[rows[q]][np.argmax(ts2[rows[q]])] == 1 for q in range(n)])
    for g in (0.5, 1, 2, 3):
        print(f'gap >= {g}: share {np.mean(gap >= g):.2f}, top-1 correct {acc1[gap >= g].mean():.2f}; below: {acc1[gap < g].mean():.2f}')
    for thr, mn, need_tr, gconf, gq in itertools.product((0.5, 0.6, 0.7), (4, 6), (True, False), (1, 2, 3), (0.5, 1, 99)):
        A = (MC >= thr) & (MN >= mn)
        if need_tr: A &= TR
        A &= (gap >= gconf)[None, :]          # neighbour must be confident
        tgt = np.where(A.any(1) & (gap < gq))[0]
        if len(tgt) < 5: continue
        res = []
        for lam in (0.1, 0.25, 0.5, 1.0):
            adj = ts2.copy()
            for q in tgt:
                r = rows[q]; cq = tpid[r]
                nb = np.where(A[q])[0]
                t = tani(cq, top1[nb]).max(1)          # max Tanimoto to a confident neighbour's top-1
                sd = ts2[r].std()
                adj[r] = ts2[r] + lam * t * sd * 2
            m = mrr(adj)
            res.append(f'{lam}: {m.mean() - base.mean():+.4f} / {m[tgt].mean() - base[tgt].mean():+.3f}')
        print(f'mc {thr} mn {mn} tr {need_tr} nbgap {gconf} qgap< {gq}: tgt {len(tgt)} base {base[tgt].mean():.3f} | ' + ' | '.join(res), flush=True)
