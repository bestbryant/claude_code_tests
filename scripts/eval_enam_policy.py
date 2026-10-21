"""Evaluate insertion policies for a tier proposal channel on scripts/sim_enam_policy.py output.

python scripts/eval_enam_policy.py <sim.parquet>
present queries: the truth keeps its main-list rank, pushed down by every inserted proposal at or above it;
other queries: the truth is reachable only through the channel, at slot[c - 1] when its channel rank c <= #slots.
Expected MRR for a share pi of "other" molecules: (1 - pi) * present + pi * other, present = 0.45 class 1 +
0.55 doc-weighted class 2. Reported as the gain over no channel.
"""
import sys
import numpy as np, pandas as pd

R = pd.read_parquet(sys.argv[1])
pr, ot = R[R.kind == 'present'].copy(), R[R.kind == 'other'].copy()
print('present', len(pr), 'other', len(ot), '| other: channel top1', round((ot.ch_rank == 1).mean(), 3),
      'MRR', round(float(np.where((ot.ch_rank >= 1) & (ot.ch_rank <= 25), 1 / ot.ch_rank.clip(lower=1), 0).mean()), 4))
print('rel quantiles present', np.round(np.nanquantile(pr.rel, [.5, .75, .9, .95, .99]), 1),
      '| other', np.round(np.nanquantile(ot.rel, [.05, .1, .25, .5, .75]), 1))
print('lib_max >= 0.9: present', round((pr.lib_max >= 0.9).mean(), 3), 'other', round((ot.lib_max >= 0.9).mean(), 3))


def mrr_present(slots_of):
    out = np.zeros(len(pr))
    for i, (r, sl) in enumerate(zip(pr.main_rank.values, slots_of(pr))):
        if r <= 0:
            continue
        for s in sorted(sl):
            if s <= r:
                r += 1
        out[i] = 1.0 / r if r <= 25 else 0.0
    c1 = pr.cls.values == 1
    return 0.45 * out[c1].mean() + 0.55 * np.average(out[~c1], weights=pr.w.values[~c1])


def mrr_other(slots_of):
    out = np.zeros(len(ot))
    for i, (c, sl) in enumerate(zip(ot.ch_rank.values, slots_of(ot))):
        if 1 <= c <= len(sl):
            out[i] = 1.0 / sorted(sl)[c - 1]
    return out.mean()


A, B, C = (4, 8, 12, 16, 20), (2, 4, 6, 8, 10), (1, 3, 5, 7, 9)


def gated(base, gate=0.9, hi=None, tau=None, hi_s=None, s_tau=None):
    def f(d):
        res = []
        for lib, rel, st in zip(d.lib_max.values, d.rel.values, d.s_top.values):
            if lib >= gate or not np.isfinite(rel):
                res.append(())
            elif hi is not None and rel > tau:
                res.append(hi)
            elif hi_s is not None and st > s_tau:
                res.append(hi_s)
            else:
                res.append(base)
        return res
    return f


base_p = mrr_present(lambda d: [()] * len(d))
pols = {'A fixed 4..20': gated(A), 'B fixed 2..10': gated(B), 'C fixed 1..9': gated(C),
        'A no gate': gated(A, gate=9)}
qs = np.nanquantile(pr.rel, [.75, .9, .95, .99])
for q, tau in zip(['p75', 'p90', 'p95', 'p99'], qs):
    pols[f'A, B if rel>{q} ({tau:.0f})'] = gated(A, hi=B, tau=tau)
    pols[f'A, C if rel>{q} ({tau:.0f})'] = gated(A, hi=C, tau=tau)
    pols[f'B, C if rel>{q} ({tau:.0f})'] = gated(B, hi=C, tau=tau)
def knobs(agg, promo, gate=0.9):
    """casmi/run_own.pc_channel semantics: slots A; B when rel > PC_REL_AGG; rank-1 promotion when rel > PC_PROMO."""
    def f(d):
        res = []
        for lib, rel in zip(d.lib_max.values, d.rel.values):
            if lib >= gate or not np.isfinite(rel):
                res.append(()); continue
            sl = B if (agg is not None and rel > agg) else A
            if promo is not None and rel > promo:
                sl = (1,) + tuple(x for x in sl if x > 1)
            res.append(sl)
        return res
    return f


for agg in (None, 5, 15, 25):
    for promo in (None, 25, 43, 60, 81):
        if agg is None and promo is None:
            continue
        pols[f'knobs agg {agg} promo {promo}'] = knobs(agg, promo)
print(f'no channel: present {base_p:.4f}')
print(f'{"policy":32s} d_present  other   ' + '  '.join(f'pi={p}' for p in (0.05, 0.1, 0.2, 0.3)))
for name, f in pols.items():
    p, o = mrr_present(f), mrr_other(f)
    gains = [(1 - pi) * p + pi * o - (1 - pi) * base_p for pi in (0.05, 0.1, 0.2, 0.3)]
    print(f'{name:32s} {p - base_p:+.4f}  {o:.4f}  ' + '  '.join(f'{g:+.4f}' for g in gains))
