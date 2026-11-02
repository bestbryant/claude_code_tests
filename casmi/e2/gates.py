"""Engine 2 clone: MRR, channel tables and the step-2 gates of the replay (docs/engine2_study.md option R step 2; spec
section 5). Our own code, pandas / numpy only; thresholds live in casmi/e2/config.py GATES.

Population: the headline numbers use ALL queries of the run (no reach filter, no artifact drop, as section 3.2 /
npx_channels.py); queries without rows (no target, empty window, error) count as 0. The ranker-comparable view
(cls == 1 | reach, minus art_ret; scripts/train_ret_ranker.py L24-34 with --drop-artifact) is reported alongside.
Windows: 'full' = every Engine 2 candidate; 'tc' = rows with in_train_pool | in_coco_pool | label (the train U COCONUT
window of npx_channels.py, where megayak measured); features always come from the full window, tc only re-ranks.
MRR@25 = mrr() on rows sorted by (qid, pid); ties keep that order.
"""
import numpy as np
import pandas as pd

from . import config as C

SUBSETS = ('npx', 'np', 'other')


def mrr(df, col):
    """Verbatim copy of scripts/train_ret_ranker.py L86-90 (input pre-sorted by (qid, pid), as its load() does)."""
    r = df.sort_values(['qid', col], ascending=[True, False]).copy()
    r['rk'] = r.groupby('qid').cumcount() + 1
    hit = r[(r.label == 1) & (r.rk <= 25)].groupby('qid').rk.min()
    return float((1.0 / hit).reindex(df.qid.unique()).fillna(0).mean())


def mrr_over(df, col, qids):
    """mrr() over an explicit query list: queries of qids without rows (or without the truth in the top 25) count 0."""
    qids = np.unique(np.asarray(qids, np.int64))
    if len(qids) == 0:
        return float('nan')
    d = df[df.qid.isin(qids)]
    if len(d) == 0:
        return 0.0
    r = d.sort_values(['qid', col], ascending=[True, False]).copy()
    r['rk'] = r.groupby('qid').cumcount() + 1
    hit = r[(r.label == 1) & (r.rk <= 25)].groupby('qid').rk.min()
    return float((1.0 / hit).reindex(qids).fillna(0).mean())


def tc_window(df):
    return df[(df.in_train_pool == 1) | (df.in_coco_pool == 1) | (df.label == 1)]


def channel_table(rows, queries, cols, windows=('full', 'tc'), view=None):
    """MRR@25 per subset x cls x window for each column (over all queries of the cell from `queries`)."""
    Q = queries if view is None else queries[view(queries)]
    cols = [c for c in cols if c in rows.columns]
    out = []
    for w in windows:
        R = rows if w == 'full' else tc_window(rows)
        for (sub, cls), q in Q.groupby(['subset', 'cls']):
            qids = q.qid.values
            d = R[R.qid.isin(qids)]
            e = dict(subset=sub, cls=int(cls), window=w, n=int(len(qids)),
                     truth_in=round(float(d.groupby('qid').label.max().reindex(qids).fillna(0).mean()), 4) if len(qids) else None)
            for c in cols:
                e[c] = round(mrr_over(d, c, qids), 4)
            out.append(e)
    return out


def ranker_view(q):
    return ((q.cls == 1) | (q.reach.astype(bool))) & (q.art_ret == 0)


def _cell(table, sub, cls, window, col):
    for e in table:
        if e['subset'] == sub and e['cls'] == cls and e['window'] == window:
            return e.get(col), e['n']
    return None, 0


def _col(q, prefix):
    """The single column of q named prefix + <N> (e.g. truth_in_ana200), or None."""
    c = [k for k in q.columns if k.startswith(prefix) and k[len(prefix):].isdigit()]
    return q[c[0]] if c else None


def leakage(rows, queries, ctl):
    """G1: class-2 truths carry no library / analog evidence of their own structure, class-1 hidden rows never match,
    re-picks are consistent, and the hiding-free control finds the truth (lv >= 0.9 in >= 95% of control queries)."""
    G = C.GATES
    q = queries[queries.status == 'ok']
    c2all = q[q.cls == 2]
    c2 = c2all[c2all.truth_in_e2 == 1]
    t = rows[rows.label == 1].set_index('qid')
    t2 = t.reindex(c2.qid.values)
    ana, anapol = _col(c2all, 'truth_in_ana'), _col(c2all, 'truth_in_anapol')
    v = dict(
        c2_queries=int(len(c2all)), c2_truth_in_window=int(len(c2)),
        c2_truth_b00_nonzero=int((t2.b00.fillna(0) != 0).sum()),
        c2_truth_lx0_nonzero=int((t2.lx0.fillna(0) != 0).sum()),
        c2_truth_lx3_nonzero=int((t2.lx3.fillna(0) != 0).sum()),
        c2_truth_lx6_nonzero=int((t2.lx6.fillna(0) != 0).sum()),
        c2_truth_libcnt_nonzero=int((c2.truth_libcnt.fillna(0) != 0).sum()),
        c2_missing_truth_row=int(t2.b00.isna().sum()),
        c2_truth_in_ana=int((ana == 1).sum()) if ana is not None else -1,
        c2_truth_in_anapol=int((anapol == 1).sum()) if anapol is not None else -1,
        c2_lib_leak=int(c2all.lib_leak.sum()),
        c2_ana_removed_total=int(c2all[['ana_removed_all', 'ana_removed_pos', 'ana_removed_neg']].sum().sum()),
    )
    c1 = q[q.cls == 1]
    v.update(c1_queries=int(len(c1)), c1_lib_leak=int(c1.lib_leak.sum()),
             c1_repick_in_hidden=int(c1.repick_in_hidden.sum()), c1_rep_inconsistent=int((c1.rep_consistent != 1).sum()),
             c1_repicks={n: int((c1[f'rep_{n}'] == 'repick').sum()) for n in ('all', 'pos', 'neg')},
             c1_drops={n: int((c1[f'rep_{n}'] == 'drop').sum()) for n in ('all', 'pos', 'neg')},
             c1_lib_removed_mean=float(c1.lib_removed.mean()) if len(c1) else 0.0,
             c1_art_e2=int(c1.art_e2.sum()) if 'art_e2' in c1 else None)
    ctl = list(ctl or [])
    have = [c for c in ctl if c.get('lv') is not None]
    n_ok = sum(1 for c in have if c['lv'] >= G['G1_control_lv'])
    v.update(control_n=len(ctl), control_with_truth=len(have), control_lv_ok=n_ok,
             control_frac=(n_ok / len(ctl)) if ctl else None)
    ok_c2 = all(v[k] == 0 for k in ('c2_truth_b00_nonzero', 'c2_truth_lx0_nonzero', 'c2_truth_lx3_nonzero',
                                    'c2_truth_lx6_nonzero', 'c2_truth_libcnt_nonzero', 'c2_missing_truth_row',
                                    'c2_truth_in_ana', 'c2_truth_in_anapol', 'c2_lib_leak'))
    ok_c1 = v['c1_lib_leak'] == 0 and v['c1_repick_in_hidden'] == 0 and v['c1_rep_inconsistent'] == 0
    ok_ctl = bool(ctl) and v['control_frac'] >= G['G1_control_min_frac']
    return dict(values=v, thresholds=dict(control_min_frac=G['G1_control_min_frac'], control_lv=G['G1_control_lv'],
                                          leaks=0), ok=bool(ok_c2 and ok_c1 and ok_ctl),
                parts=dict(class2=ok_c2, class1=ok_c1, control=ok_ctl))


def window_gate(queries, rows):
    """G2a bank alignment at the ret18cu target, G2b Engine 2 target with every difference explained, G2c truth in window
    and joined PeakFormer share."""
    G = C.GATES
    q = queries[queries.status.isin(['ok', 'empty_window'])]
    n = len(q)
    same_mret = float(q.win_same_mret.mean()) if n else 0.0
    cats = q.win_cat.value_counts().to_dict()
    ident = float((q.win_cat == 'same').mean()) if n else 0.0
    other = int((q.win_cat == 'other').sum())
    t_e2, t_ret = int(q.truth_in_e2.sum()), int(q.truth_in_ret.sum())
    nrows = len(rows)
    joined = float((rows.pf_src == 0).mean()) if nrows else 0.0
    g2a = same_mret >= G['G2a_identity_mret']
    g2b = ident >= G['G2b_identity_e2'] and other == 0
    g2c = t_e2 >= t_ret - G['G2c_truth_slack'] and joined >= G['G2c_joined_share']
    return dict(values=dict(n=n, identity_at_mret=round(same_mret, 5), identity_at_e2_target=round(ident, 5),
                            categories={k: int(v) for k, v in cats.items()}, unexplained=other,
                            truth_in_e2=t_e2, truth_in_ret=t_ret, pf_joined_share=round(joined, 6),
                            pf_scored_rows=int((rows.pf_src == 1).sum()) if nrows else 0,
                            fallback30=int(q.fallback30.sum())),
                thresholds=dict(G2a=G['G2a_identity_mret'], G2b=G['G2b_identity_e2'], G2b_edge_da=G['G2b_edge_da'],
                                G2c_truth_slack=G['G2c_truth_slack'], G2c_joined=G['G2c_joined_share']),
                ok=bool(g2a and g2b and g2c), parts=dict(G2a=g2a, G2b=g2b, G2c=g2c))


def parity(table, table_alt=None):
    """G3 on npx (spec section 5): library b00 class 1 full; analog b05 class 2 full or tc; MetFrag b21 class 2 tc."""
    G = C.GATES
    lib, n1 = _cell(table, 'npx', 1, 'full', 'b00')
    ana_f, n2 = _cell(table, 'npx', 2, 'full', 'b05')
    ana_t, _ = _cell(table, 'npx', 2, 'tc', 'b05')
    frag_t, _ = _cell(table, 'npx', 2, 'tc', 'b21')
    frag_f, _ = _cell(table, 'npx', 2, 'full', 'b21')
    v = dict(npx_c1_n=n1, npx_c2_n=n2, lib_b00_c1_full=lib, ana_b05_c2_full=ana_f, ana_b05_c2_tc=ana_t,
             frag_b21_c2_tc=frag_t, frag_b21_c2_full=frag_f,
             ref_ours=dict(lib_c1=0.861, an_best_c2_full=0.439, an_best_c2_tc=0.464, frag_c2_tc=0.261,
                           frag_c2_full=0.207, pf_bern_c2_full=0.501, pf_bern_c2_tc=0.529))
    if table_alt is not None:
        v['ana_b05_a200_c2_full'] = _cell(table_alt, 'npx', 2, 'full', 'b05_a200')[0]
        v['ana_b05_a200_c2_tc'] = _cell(table_alt, 'npx', 2, 'tc', 'b05_a200')[0]
    ok_lib = lib is not None and lib >= G['G3_lib_c1']
    ok_ana = (ana_f is not None and ana_f >= G['G3_ana_c2_full']) or (ana_t is not None and ana_t >= G['G3_ana_c2_tc'])
    ok_frag = frag_t is not None and frag_t >= G['G3_frag_c2_tc']
    return dict(values=v, thresholds=dict(lib_c1_full=G['G3_lib_c1'], ana_c2_full=G['G3_ana_c2_full'],
                                          ana_c2_tc=G['G3_ana_c2_tc'], frag_c2_tc=G['G3_frag_c2_tc']),
                ok=bool(ok_lib and ok_ana and ok_frag), parts=dict(library=ok_lib, analog=ok_ana, metfrag=ok_frag))


def throughput(queries, timing=None):
    """G4: mean channel-phase CPU seconds per query (worker process CPU, one numba thread, PeakFormer fallback included,
    MetFrag excluded) <= 3.0."""
    G = C.GATES
    q = queries[queries.status.isin(['ok', 'empty_window'])]
    ch = float(q.cpu_channels_s.mean()) if len(q) else float('nan')
    v = dict(n=int(len(q)), cpu_channels_s_mean=round(ch, 4),
             cpu_channels_s_p50=round(float(q.cpu_channels_s.median()), 4) if len(q) else None,
             cpu_channels_s_p95=round(float(q.cpu_channels_s.quantile(0.95)), 4) if len(q) else None,
             cpu_channels_s_max=round(float(q.cpu_channels_s.max()), 4) if len(q) else None,
             cpu_explain_s_mean=round(float(q.cpu_explain_s.mean()), 4) if len(q) else None,
             cpu_feat_s_mean=round(float(q.cpu_feat_s.mean()), 4) if len(q) else None)
    if timing:
        v.update(timing)
    return dict(values=v, thresholds=dict(cpu_s_per_query=G['G4_cpu_s_per_query']),
                ok=bool(len(q) and ch <= G['G4_cpu_s_per_query']))


def pf_check(pfc):
    """G5: PeakFormer rescoring of sampled queries reproduces the joined ret18cu pf_bern:
    max|fz - pf_bern| <= 1e-3 * max(1, std_q(pf_bern)) + 1e-4 for every sampled query."""
    G = C.GATES
    pfc = list(pfc or [])
    bad = [p for p in pfc if not (p.get('error') is None and p['max_abs'] <= G['G5_rel'] * max(1.0, p['std']) + G['G5_abs'])]
    v = dict(n=len(pfc), n_bad=len(bad), max_abs=max((p['max_abs'] for p in pfc if p.get('error') is None), default=None),
             bad=[dict(qid=p['qid'], max_abs=p.get('max_abs'), std=p.get('std'), error=p.get('error')) for p in bad[:20]])
    return dict(values=v, thresholds=dict(rel=G['G5_rel'], abs=G['G5_abs']), ok=bool(pfc) and not bad)


def evaluate(rows, queries, ctl, pfc, setup, ref=None, split='val', tag='', complete=True, limited=False, meta=None):
    """All gates -> report dict (spec section 5 layout). rows: e2r rows of the finished queries; queries: one row per
    SELECTED query (status 'pending' for unfinished ones); ctl / pfc: control and PF-check results; setup: G0 dict;
    ref: e2pf rows of the selected queries with label / subset / cls / in_train_pool / in_coco_pool (ret18cu window)."""
    rows = rows.sort_values(['qid', 'pid']).reset_index(drop=True)
    alt_cols = [c for c in rows.columns if c.endswith('_a200')]
    cols = list(C.CHANNEL_COLS)
    table = channel_table(rows, queries, cols)
    table_alt = channel_table(rows, queries, alt_cols) if alt_cols else None
    gates = {'G0': setup, 'G1': leakage(rows, queries, ctl), 'G2': window_gate(queries, rows)}
    gates['G3'] = parity(table, table_alt)
    gates['G4'] = throughput(queries, (meta or {}).get('timing_extra'))
    gates['G5'] = pf_check(pfc)
    n_err = int(queries.status.astype(str).str.startswith('error').sum())
    n_pending = int((queries.status == 'pending').sum())
    if limited:                                        # smoke / subset runs: G0, G1, G5 (+ errors); the rest is info
        hard = ['G0', 'G1', 'G5']
    elif split == 'val':
        hard = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5']
    else:                                              # train: G3 informational (no npx queries in the train split)
        hard = ['G0', 'G1', 'G2', 'G4', 'G5']
    for k, g in gates.items():
        g['hard'] = k in hard
    failed = [k for k in hard if not gates[k].get('ok')]
    if n_err:
        failed.append(f'errors:{n_err}')
    status = 'FAIL' if failed else ('PARTIAL' if (n_pending or not complete) else 'PASS')
    tables = dict(channels=table, channels_a200=table_alt,
                  ranker_view=channel_table(rows, queries, cols, view=ranker_view))
    if ref is not None and len(ref):
        tables['ret18cu_ref'] = channel_table(ref, queries, list(C.REF_COLS))
        tables['ret18cu_ref_ranker_view'] = channel_table(ref, queries, list(C.REF_COLS), view=ranker_view)
    cnt = queries.groupby(['subset', 'cls']).size()
    return dict(split=split, tag=tag, status=status, failed=failed, hard_gates=hard, n_queries=int(len(queries)),
                n_done=int(len(queries) - n_pending), n_errors=n_err, n_rows=int(len(rows)),
                counts={f'{a}_c{b}': int(v) for (a, b), v in cnt.items()},
                statuses=queries.status.astype(str).str.slice(0, 40).value_counts().to_dict(),
                gates=gates, tables=tables, **(meta or {}))


def gate_lines(rep):
    """Human-readable 'E2 GATE <name>: PASS|FAIL|INFO ...' lines."""
    out = []
    for k, g in rep['gates'].items():
        word = ('PASS' if g.get('ok') else 'FAIL') if g.get('hard') else ('INFO-ok' if g.get('ok') else 'INFO-fail')
        parts = g.get('parts')
        out.append(f"E2 GATE {k}: {word}" + (f" parts={parts}" if parts else '') +
                   f" values={_short(g.get('values'))}")
    out.append(f"E2 GATE STATUS: {rep['status']} failed={rep['failed']} queries={rep['n_done']}/{rep['n_queries']} "
               f"errors={rep['n_errors']}")
    return out


def _short(v, n=600):
    s = str(v)
    return s if len(s) <= n else s[:n] + '...'
