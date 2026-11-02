"""Engine 2 clone: leakage-hiding decisions for the replay of our simulated queries (docs/engine2_study.md option R
step 2, change (c)). Our own code, pure numpy; it never imports or calls the vendored Engine 2 (casmi/e2/pv.py,
casmi/e2/engine.py). The vendored engine calls lib_keep / analog_drop inside molecule_channels (patches E8-E13).

Semantics (one query = one hide spec; L, I and POOL are never mutated):
  class 1  hidden rows = ALL q_rows of the query (scripts/dump_ret_rows.py L88-89 hides all of them, not only the
           spectra that survive cleaning). They are removed from the +-10 ppm library match. In each analog store
           ('all' / 'pos' / 'neg') the truth's representative is re-picked under the store's own rule when the
           original representative is hidden (the engine then computes the re-picked row's similarity); if no row of
           the truth remains, the truth is absent from that store. The truth's other library spectra stay (lv > 0 is
           legitimate).
  class 2  every library row of the truth structure (row_p == truth_p) is removed from the library match, and every
           analog entry of the truth is REMOVED from the results of all three stores (not set to similarity 0:
           best_tan / top_tan, base columns 10-11, ignore the weights, so a zero-similarity truth inside the top N would
           still give the truth Tanimoto 1).
Representative rule = E2:casmi_engine.py L207-210 (`_reps`): order the rows of a structure by (tims desc, npk desc,
row asc), where npk = the raw peak count np.diff(L['off']) and only 'ok' rows (finite nm, row_p >= 0) take part.
"""
import numpy as np

STORES = ('all', 'pos', 'neg')
_POL = {'all': None, 'pos': 1, 'neg': -1}


def rule_csr(row_p, ok, tims, npk):
    """All ok rows ordered by (row_p asc, tims desc, npk desc, row asc) -> dict(p, start, rows): the rows of structure
    p[k] in representative-rule order are rows[start[k]:start[k + 1]]."""
    row_p = np.asarray(row_p, np.int64); ok = np.asarray(ok, bool)
    tims = np.asarray(tims, bool); npk = np.asarray(npk, np.int64)
    rows = np.flatnonzero(ok).astype(np.int64)
    o = np.lexsort((rows, -npk[rows], ~tims[rows], row_p[rows]))
    srows = rows[o]
    p, start = np.unique(row_p[srows], return_index=True)
    return dict(p=p.astype(np.int64), start=np.r_[start, len(srows)].astype(np.int64), rows=srows)


def rule_rows(csr, p):
    """Rows of structure p in rule order (int64, empty when p has no ok row)."""
    k = int(np.searchsorted(csr['p'], p))
    if k >= len(csr['p']) or csr['p'][k] != p:
        return np.zeros(0, np.int64)
    return csr['rows'][csr['start'][k]:csr['start'][k + 1]]


def make_hide(cls, truth_p, q_rows, csr=None, pol=None):
    """Hide spec of one query. q_rows = all query rows (class 1 hides all of them). Class 1 needs csr (rule_csr over the
    library) and pol (L['pol'], +1 / -1 per row) to pre-compute rep[name] = (orig, new) per store: orig = the store's
    representative of the truth, new = the first rule row that is not hidden (-1 if absent)."""
    cls = int(cls)
    if cls not in (1, 2):
        raise ValueError(f'class {cls}')
    spec = dict(cls=cls, truth_p=int(truth_p), rows=np.unique(np.asarray(q_rows, np.int64)))
    if cls == 1:
        if csr is None or pol is None:
            raise ValueError('class 1 needs csr and pol')
        rr = rule_rows(csr, int(truth_p))
        hid = np.isin(rr, spec['rows'])
        rep = {}
        for name in STORES:
            s = _POL[name]
            m = np.ones(len(rr), bool) if s is None else (np.asarray(pol)[rr] == s)
            cand = rr[m]
            orig = int(cand[0]) if len(cand) else -1
            left = cand[~hid[m]]
            rep[name] = (orig, int(left[0]) if len(left) else -1)
        spec['rep'] = rep
    return spec


def new_diag(hide):
    """Per-query hiding diagnostics, filled by lib_keep / analog_drop and the engine's _hide_analog."""
    return dict(lib_removed=0, lib_leak=0, rep={n: 'none' for n in STORES}, rep_row={n: -1 for n in STORES},
                rep_sim={}, ana_removed={n: 0 for n in STORES}, hidden=hide is not None)


def lib_keep(crow, row_p_crow, hide, diag):
    """Boolean mask over the +-10 ppm library rows crow: False for hidden rows. lib_leak is recomputed independently
    over the kept rows and must stay 0."""
    crow = np.asarray(crow, np.int64); rp = np.asarray(row_p_crow, np.int64)
    if hide['cls'] == 1:
        keep = ~np.isin(crow, hide['rows'])
        leak = int(np.isin(crow[keep], hide['rows']).sum())
    else:
        keep = rp != hide['truth_p']
        leak = int((rp[keep] == hide['truth_p']).sum())
    diag['lib_removed'] += int((~keep).sum())
    diag['lib_leak'] += leak
    return keep


def analog_drop(p, s, hide, name, diag):
    """Analog results (p structure ids, s similarities) of store `name` -> (p2, s2, new_row). Inputs are not mutated,
    dtypes are preserved. new_row >= 0 asks the caller to append the re-picked representative's similarity."""
    p = np.asarray(p); s = np.asarray(s)
    t = hide['truth_p']
    if hide['cls'] == 2:
        m = p == t
        diag['ana_removed'][name] += int(m.sum())
        return p[~m], s[~m], -1
    orig, new = hide['rep'][name]
    if orig < 0 or not np.isin(orig, hide['rows']):
        return p, s, -1
    m = p == t
    diag['ana_removed'][name] += int(m.sum())
    diag['rep'][name] = 'repick' if new >= 0 else 'drop'
    diag['rep_row'][name] = int(new)
    return p[~m], s[~m], int(new)
