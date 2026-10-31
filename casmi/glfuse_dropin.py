"""Notebook-side forward-model helpers, drop-in for the fork's `gl_fuse.py` (and the `fuse.py` functions the fork
uses) (fix F5b, clean-room).

Written from the behavioural specification scratchpad f5b/glue_spec.md (sections 3, 9.2, 11) and our own casmi code.
No code of the unlicensed runner glue was used. `run_runner` follows the subprocess wrapper `n1_run` of the fork's
frozen notebook (FZ c41:1007-1040, Apache-2.0 notebook code). Pure Python (+ the standard library); safe to import in
the notebook.

- build_ice_input(test_df, cands_by_mid, covered_only=True): runner input items from test rows (spec 3.1).
- ice_candidates(smis, formulas, top_n=40): the members of same-formula groups among a ranked list's top n (3.2).
- run_gl(...) / run_ice(...): run a runner program in its own process; never raise; return {mid: {smiles: score}}.
- rerank(...): within-formula z-fusion with one re-scorer (11.2); rerank_multi(...): with several (11.3).
- placeholder_scores(items): all-null per-molecule dicts for the "ICEBERG dropped" option (12).
"""
import json
import math
import numbers
import os
import subprocess
import sys
import time

COVERED = ('[M+H]+', '[M+Na]+')
GRACE_S = 300


# ------------------------------------------------------------------------------------------------ input ----------
def _missing(v):
    """None, NaN, pandas NA / NaT."""
    if v is None or type(v).__name__ in ('NAType', 'NaTType'):
        return True
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isnan(float(v))


def _text(v):
    return None if _missing(v) else str(v)


def _floats(v):
    """A float list from an array-like; [] when missing or not iterable."""
    if v is None or isinstance(v, (str, bytes)):
        return []
    try:
        return [float(x) for x in v]
    except TypeError:
        return []


def _ce_list(v):
    """A float list from an array-like; None when missing or not iterable (a bare number, as in spec 3.1, is None:
    the runner then treats the CE as unknown)."""
    if v is None or isinstance(v, (str, bytes, numbers.Number)):
        return None
    try:
        return [float(x) for x in v]
    except TypeError:
        return None


def build_ice_input(test_df, cands_by_mid, covered_only=True):
    """Runner input items [{mid, spectra, cands}] in the order of cands_by_mid (= runner priority). Spectra are the
    test rows of the molecule: mz / it float lists, prec, adduct, mode, instrument, ce_ev (float list | None),
    ce_orig, ce_units (strings | None). A row that cannot be converted is skipped. covered_only keeps positive
    [M+H]+ / [M+Na]+ spectra only. Molecules without spectra are kept (spectra = [])."""
    by = {}
    for r in test_df.to_dict('records'):
        try:
            sp = dict(mz=_floats(r.get('ms2_mzs')), it=_floats(r.get('ms2_normalized_intensities')),
                      prec=float(r.get('precursor_mz')), adduct=_text(r.get('adduct')),
                      mode=_text(r.get('ionization_mode')), instrument=_text(r.get('instrument_type')),
                      ce_ev=_ce_list(r.get('collision_energy_ev')), ce_orig=_text(r.get('collision_energy_orig')),
                      ce_units=_text(r.get('collision_energy_orig_units')))
        except Exception:
            continue
        if covered_only and not (sp['mode'] == 'positive' and sp['adduct'] in COVERED):
            continue
        by.setdefault(str(r.get('molecule_id')), []).append(sp)
    return [dict(mid=str(m), spectra=by.get(str(m), []), cands=[str(c) for c in cands])
            for m, cands in cands_by_mid.items()]


def ice_candidates(smis, formulas, top_n=40):
    """Entries of the ranked list's first min(top_n, n) positions whose formula occurs >= 2 times there, in rank
    order, de-duplicated by SMILES string (only these can be moved by a within-formula re-rank)."""
    head = list(zip(smis, formulas))[:max(0, min(int(top_n), len(smis)))]
    count = {}
    for _, f in head:
        count[f] = count.get(f, 0) + 1
    return list(dict.fromkeys(s for s, f in head if count[f] >= 2))


# ------------------------------------------------------------------------------------------------ runners --------
def _here(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def run_runner(script, pkg, items, workdir, stem, device='cuda', budget_s=0, site=None, wheels=None, extra_args=(),
               grace_s=GRACE_S, tag='runner'):
    """Run `python <script> <pkg> <stem>_in.json <stem>_out.json ...` in workdir; never raises. Stale outputs are
    deleted first; the subprocess gets budget + grace seconds (no limit for budget 0); after a timeout the runner's
    last checkpointed out.json is still read. Returns the out.json dict, {} when missing / unreadable."""
    try:
        os.makedirs(workdir, exist_ok=True)
        fin, fout = os.path.join(workdir, f'{stem}_in.json'), os.path.join(workdir, f'{stem}_out.json')
        for f in (fout, fout + '.meta.json'):
            if os.path.exists(f):
                os.remove(f)
        with open(fin, 'w') as f:
            json.dump(items, f)
        cmd = [sys.executable, script, pkg, fin, fout, '--device', str(device), '--budget', str(float(budget_s or 0))]
        if site:
            cmd += ['--site', str(site)]
        if wheels:
            cmd += ['--wheels', str(wheels)]
        cmd += [str(x) for x in (extra_args or [])]
        timeout = float(budget_s) + float(grace_s) if budget_s and float(budget_s) > 0 else None
        t0 = time.time()
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace',
                               timeout=timeout)
            tail = f'rc {r.returncode}\n' + (r.stdout or '')[-3000:]
        except subprocess.TimeoutExpired as e:
            so = e.stdout.decode(errors='replace') if isinstance(e.stdout, bytes) else str(e.stdout or '')
            tail = f'TIMEOUT after {time.time() - t0:.0f}s\n' + so[-2900:]
        print(f'[{tag}] {tail}', flush=True)
        try:
            with open(fout + '.meta.json') as f:
                m = json.load(f)
            print(f'[{tag}] meta: status {m.get("status")} device {m.get("device")} rdkit {m.get("rdkit")} '
                  f'molecules {m.get("n_mols_scored")}/{m.get("n_mols")} scores {m.get("n_scored")}/{m.get("n_cands")} '
                  f'predictions {m.get("n_pred")} failed {m.get("n_pred_fail")} {m.get("seconds")}s', flush=True)
        except Exception as e:
            print(f'[{tag}] meta unreadable: {e!r}', flush=True)
        if not os.path.exists(fout):
            return {}
        with open(fout) as f:
            out = json.load(f)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        print(f'[{tag}] wrapper failed: {e!r}', flush=True)
        return {}


def run_gl(pkg, items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=0, site=None, wheels=None,
           extra_args=(), grace_s=GRACE_S, runner=None):
    """GLACIER scores {mid: {smiles: score | None}} via <runner> (default <pkg>/gl_runner.py, else glrun_dropin.py
    next to this file) on gl_in.json / gl_out.json in workdir. Never raises ({} on any failure)."""
    script = runner or (os.path.join(pkg, 'gl_runner.py') if os.path.exists(os.path.join(pkg, 'gl_runner.py'))
                        else _here('glrun_dropin.py'))
    return run_runner(script, pkg, items, workdir, 'gl', device, budget_s, site, wheels, extra_args, grace_s, 'gl')


def run_ice(pkg, items, workdir='/kaggle/working/ice_work', device='cuda', budget_s=0, site=None, extra_args=(),
            grace_s=GRACE_S, runner=None, wheels=None):
    """ICEBERG scores via <runner> (default <pkg>/ice_runner.py) on ice_in.json / ice_out.json; never raises.
    (We have no clean ICEBERG runner yet: see scratchpad f5b/iceberg_plan.md.)"""
    script = runner or os.path.join(pkg, 'ice_runner.py')
    return run_runner(script, pkg, items, workdir, 'ice', device, budget_s, site, wheels, extra_args, grace_s, 'ice')


def placeholder_scores(items):
    """{mid: {smiles: None}} for every runner input molecule: what the ICEBERG runner writes before scoring. Used
    when ICEBERG is not run, so the pipeline's truthy per-molecule ICE-dict gates (which also gate every GLACIER use,
    spec 9.3) stay open while no ICEBERG value is ever finite."""
    out = {}
    for it in items:
        out.setdefault(str(it['mid']), {}).update({str(c): None for c in it.get('cands') or []})
    return out


# ------------------------------------------------------------------------------------------------ fusion ---------
def _usable(x):
    """A z-score input: a Python int / float (numpy float64 is one, float32 is not) that is finite."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def _z(vals):
    """Within-group z-scores: (x - mean) / sample SD (ddof 1) over usable values; 0 for unusable values, and for
    all values when fewer than 2 are usable or the SD is not > 0.
    The arithmetic is the textbook two-pass definition (mean = sum / n; SD = sqrt(sum of squared deviations /
    (n - 1)), Python's sum()) on purpose: exact ties between a ranker z and a re-scorer z (e.g. +-1/sqrt(2) in a
    2-member group at lam 1) must cancel the same way as the reference's pandas-semantics z."""
    xs = [float(v) for v in vals if _usable(v)]
    if len(xs) >= 2:
        centre = sum(xs) / len(xs)
        spread = math.sqrt(sum((x - centre) ** 2 for x in xs) / (len(xs) - 1))
        if spread > 0:
            return [(float(v) - centre) / spread if _usable(v) else 0.0 for v in vals]
    return [0.0] * len(vals)


def _lookup(scores, smis, keys):
    """Re-scorer values aligned with smis: a list is taken as aligned; a dict is looked up by exact SMILES string,
    then (when that gives nothing) by the candidate's key."""
    n = len(smis)
    if not isinstance(scores, dict):                  # list / tuple / array: aligned with smis
        v = list(scores)[:n]
        return v + [None] * (n - len(v))
    out = []
    for i, s in enumerate(smis):
        v = scores.get(s)
        if v is None and keys is not None and i < len(keys):
            try:
                v = scores.get(keys[i])
            except TypeError:
                v = None
        out.append(v)
    return out


def _clean(v):
    """Finite float or None (non-numeric -> None)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _groups(formulas, m):
    g = {}
    for p in range(m):
        f = formulas[p] if p < len(formulas) else None
        if f is not None:
            g.setdefault(f, []).append(p)
    return [pos for pos in g.values() if len(pos) >= 2]


def _write_back(order, pos, fused):
    """Members sorted by (-fused, current position), written into the group's own ascending slots."""
    ranked = sorted(range(len(pos)), key=lambda i: (-fused[i], pos[i]))
    for slot, i in zip(pos, ranked):
        order[slot] = pos[i]


def rerank(smis, keys, scores, formulas, ice_scores, lam=0.5, top_n=60, min_covered=2):
    """Permutation of indices: inside each same-formula group among the first min(top_n, n) positions that has
    >= max(2, min_covered) members with a usable re-scorer value, members are re-ordered by z(scores) + lam *
    z(re-scorer) (z within the group) and written back into the group's slots; everything else stays put.
    Identity for an empty list or a falsy ice_scores."""
    n = len(smis)
    order = list(range(n))
    if n == 0 or not ice_scores:
        return order
    vals = [_clean(v) for v in _lookup(ice_scores, smis, keys)]     # re-scorer values -> finite float | None
    need = max(2, int(min_covered))
    for pos in _groups(formulas, min(int(top_n), n)):
        if sum(1 for p in pos if vals[p] is not None) < need:
            continue
        zs = _z([scores[p] for p in pos])
        zr = _z([vals[p] for p in pos])
        _write_back(order, pos, [a + lam * b for a, b in zip(zs, zr)])
    return order


def rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9):
    """rerank with several re-scorers: fused = z(scores) + sum_k lam_k z(score_k) over the re-scorers that cover >= 2
    group members; a group is processed when at least one re-scorer covers >= max(2, min_covered) members. Empty /
    None dicts are absent (identity when none is present). When >= 2 re-scorers contribute, fused values are put on a
    tie_eps grid (round(x / tie_eps)) so float-noise near-ties keep the current order."""
    if len(lams) != len(score_dicts):
        raise ValueError(f'{len(score_dicts)} score dicts but {len(lams)} weights')
    n = len(smis)
    order = list(range(n))
    present = [(d, float(l)) for d, l in zip(score_dicts, lams) if d]
    if n == 0 or not present:
        return order
    cols = [([_clean(v) for v in _lookup(d, smis, keys)], l) for d, l in present]
    need = max(2, int(min_covered))
    for pos in _groups(formulas, min(int(top_n), n)):
        cov = [sum(1 for p in pos if c[p] is not None) for c, _ in cols]
        if not any(k >= need for k in cov):
            continue
        fused = _z([scores[p] for p in pos])
        used = 0
        for (c, l), k in zip(cols, cov):
            if k >= 2:
                zk = _z([c[p] for p in pos])
                fused = [a + l * b for a, b in zip(fused, zk)]
                used += 1
        if used >= 2 and tie_eps > 0:             # tie_eps = 0 disables the grid
            fused = [round(x / tie_eps) for x in fused]
        _write_back(order, pos, fused)
    return order
