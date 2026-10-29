"""Part 3a: v18c lists on the exact same-query re-dump (redump_same.py) of the 341 val class-1 queries.
Each variant replaces lib / analog columns of the class-1 queries' full-window rows, then stage 1 -> top 30 ->
GLACIER features (existing scores; a candidate new to the top 30 has none) -> stage2_v18c."""
import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/c1trust/d3_hiddenlike')
from evalpipe import *
OUT = '/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/c1trust/d3_hiddenlike'
X = pd.read_parquet(OUT + '/redump_same_val.parquet')
I = pd.read_parquet(OUT + '/c1_inventory.parquet'); I = I[I.set == 'val'].set_index('qid')
D = load_rows(); cls_of = D.drop_duplicates('qid').set_index('qid').cls; allq = D.qid.unique()
COLS = ['lib', 'lib_n', 'lib_r', 'lib_gap', 'an_best', 'an_tan_best', 'an_sum', 'an_top1_tan', 'an_r', 'an_gap']
sc = gl_scores()
res, Ts, tl = {}, {}, {}
for v in ('orig', 'tl', 'xinst', 'xlab'):
    x = X[X['var'] == v].set_index(['qid', 'pid'])[COLS]
    Dv = D.set_index(['qid', 'pid']); idx = Dv.index.intersection(x.index)
    assert len(idx) == len(x), (v, len(idx), len(x))
    Dv.loc[idx, COLS] = x.loc[idx, COLS].values
    Dv = Dv.reset_index()
    T = run(Dv); Ts[v] = T
    R = decompose(T, all_qids=allq, cls_of=cls_of); res[v] = R
    t = Dv[(Dv.label == 1) & (Dv.cls == 1)].set_index('qid'); tl[v] = t[['lib', 'lib_n', 'an_best']]
    newc = T[T.cls == 1].apply(lambda r: r.qid if (str(r.qid) in sc and r.smiles not in sc[str(r.qid)]) else None, axis=1).notna().sum()
    print(f'{v}: class-1 top-30 rows without a GLACIER score (new entrants in GLACIER-covered queries) {newc}')
base = res['orig']; b1 = base[base.cls == 1].set_index('qid')
assert np.allclose(b1.rr.values, decompose(run(D), all_qids=allq, cls_of=cls_of).set_index('qid').reindex(b1.index).rr.values)
pd.set_option('display.width', 220)

def block(R, qs):
    r = R[R.qid.isin(qs)]
    return dict(n=len(r), full=r.rr.mean(), restricted=r.rr_res.mean(), loss=r.loss.mean(), intop=r.intop.mean(),
                top1=(r.rr == 1).mean(), co_above=r.n_co.mean(), ot_above=r.n_ot.mean())

q1 = b1.index
pops = {'all class 1': set(q1), 'tims queries': set(I.index[I.q_ic == 'tims']),
        'non-tims queries': set(I.index[I.q_ic != 'tims'])}
rows = []
for v, R in res.items():
    keep = set(tl[v].index[tl[v].lib_n > 0])          # truth still has >= 1 library comparison under this rule
    changed = set(tl[v].index[(tl[v].lib_n != tl['orig'].lib_n.reindex(tl[v].index))])
    for pn, qs in list(pops.items()) + [('retained-lib (all)', keep & set(q1)), ('retained-lib tims', keep & pops['tims queries']),
                                         ('changed & retained', changed & keep), ('lost all lib', set(q1) - keep)]:
        rows.append(dict(variant=v, pop=pn, **block(R, qs)))
S = pd.DataFrame(rows)
print(S.round(4).to_string(index=False))
print('\ntruth lib under each rule (class-1 val queries, full window):')
for v in tl:
    t = tl[v]
    print(f'  {v:6s} lib_n>0 {(t.lib_n>0).mean():.3f}  lib median (lib_n>0) {t.lib[t.lib_n>0].median():.3f}  mean {t.lib[t.lib_n>0].mean():.3f}  an_best median {t.an_best.median():.3f}')
print('\npaired deltas vs orig (class 1, bootstrap SE):')
for v in ('tl', 'xinst', 'xlab'):
    r1 = res[v][res[v].cls == 1].set_index('qid').reindex(q1)
    for pn, qs in (('all', q1), ('tims', [q for q in q1 if q in pops['tims queries']])):
        a = b1.loc[qs]; b = r1.loc[qs]
        d1, s1 = boot_delta(a.rr.values, b.rr.values); d2, s2 = boot_delta(a.loss.values, b.loss.values)
        print(f'  {v:6s} {pn:5s} dMRR {d1:+.4f} (se {s1:.4f})  dloss {d2:+.4f} (se {s2:.4f})')
pd.concat(res, names=['variant']).reset_index(level=0).to_parquet(OUT + '/part3a_perquery.parquet')
S.to_csv(OUT + '/part3a_summary.csv', index=False)
