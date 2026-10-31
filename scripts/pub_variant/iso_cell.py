# ── ISO: isomer reranker on the final answer lists (ours; runs right after the final layer) ──────────────────────
# heliosli-style (team LB 0.486) with our PF-syn f.z as the model score. Per molecule, over its submitted list:
# score = z(-log rank) + ISO_ALPHA * (z(pf_fz) + ISO_COS_W * z(pf_cos))   (z over the list, population std; std 0 ->
# zeros; pf_cos = the PF-syn cosine of pf_scores.parquet, not read at all when ISO_COS_W is 0).
# ISO_SCOPE 'formula': only the candidates with the rank-1 candidate's formula are re-sorted by score, inside the positions
# they already hold; 'all': every candidate. Protected candidates never move: training structures (ISO_KEEP_TRAIN: RDKit
# InChIKey14 in the inchikey14 column of train.parquet) and SMILES RDKit cannot read. ISO_FREEZE_LEADER: rank 1 is held
# in place as well (ranks 2-25 only; scope 'formula' still re-sorts the leader's formula). Skipped molecules: best library
# match >= ISO_LIB_SKIP, a listed candidate without a finite PF-syn f.z (neural_candidate_features, then
# outsider_neural_features, then pf_scores.parquet) or, when ISO_COS_W > 0, without a finite pf_cos (pf_scores.parquet,
# pf_ok rows; counted as skipped pf), fewer than 2 candidates. ISO_ALPHA 0: submission.csv is not touched.
# Any failure restores the pre-ISO submission.csv byte for byte (WARNING). Report: iso_report.json, one 'ISO <status> | ...'
# line ('ISO applied' only when ISO ran; a fallback reports zero changes, the aborted attempt as attempted_*).
# Commit run (not IS_RERUN, ISO_ALPHA > 0): an 'ISO dryrun' line too, the same rerank with the library skip off, written to
# a scratch file that is read back and deleted (the visible molecules all have library match 1.0, so the real path skips
# them). PF_MID_W > 0: a 'PF_MID trace' line, how many molecules the final layer's hook re-ranked (pf_mid_trace.json).
import io, os, json, math, time, shutil, hashlib, traceback
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors
except Exception as _iso_import_error:                          # ISO then falls back (submission.csv untouched)
    Chem = rdBase = rdMolDescriptors = None
    print('WARNING ISO: RDKit import failed:', repr(_iso_import_error)[:200], flush=True)


def iso_z(v):
    """z-scores over one list (population std); all zeros when the std is 0."""
    v = np.asarray(v, float)
    sd = float(v.std())
    return (v - float(v.mean())) / sd if sd > 1e-12 else np.zeros(len(v))


@lru_cache(maxsize=200000)
def iso_formula(s):
    """RDKit molecular formula of a SMILES; None when RDKit cannot read it."""
    try:
        m = Chem.MolFromSmiles(str(s))
        return rdMolDescriptors.CalcMolFormula(m) if m is not None else None
    except Exception:
        return None


@lru_cache(maxsize=200000)
def iso_key14(s):
    """First block of the RDKit InChIKey (the inchikey14 of train.parquet); None when RDKit cannot make one."""
    try:
        m = Chem.MolFromSmiles(str(s))
        k = Chem.MolToInchiKey(m) if m is not None else ''
        return k[:14] if len(k) >= 14 else None
    except Exception:
        return None


def iso_order(cands, pf, alpha, scope='formula', train_keys=frozenset(), keep_train=True,
              formula_of=iso_formula, key_of=iso_key14, cos=None, cos_w=0.0, freeze_leader=False):
    """New order of one answer list (pure). cands: SMILES in submitted order (>= 2); pf: finite PF-syn f.z per candidate;
    cos: finite PF-syn cosine per candidate, read only when cos_w > 0 (score = z(-log rank) + alpha * (z(pf) +
    cos_w * z(cos))). freeze_leader: position 1 is held in place like a protected candidate (scope 'formula' still uses
    its formula). Returns (new list, protected, moved): protected = in-scope candidates held in place (training structure
    or unreadable; a frozen leader is not counted), moved = size of the re-sorted group (0 when fewer than 2 candidates
    are movable)."""
    n = len(cands)
    assert n >= 2 and len(pf) == n and scope in ('formula', 'all'), (n, len(pf), scope)
    assert float(cos_w) >= 0 and isinstance(freeze_leader, bool), (cos_w, freeze_leader)
    model = iso_z(pf)
    if cos_w:                                                           # ISO_COS_W 0: cos is not looked at
        assert cos is not None and len(cos) == n, 'pf_cos per candidate'
        model = model + float(cos_w) * iso_z(cos)
    score = iso_z(-np.log(np.arange(1, n + 1, dtype=float))) + float(alpha) * model
    lead = formula_of(cands[0])
    group, protected = [], 0
    for i, s in enumerate(cands):
        f = formula_of(s)
        if scope == 'formula' and (lead is None or f != lead):
            continue
        held = f is None                                                # unreadable SMILES never move
        if not held and keep_train:
            k = key_of(s)
            held = k is None or k in train_keys
        if held:
            protected += 1
            continue
        if i == 0 and freeze_leader:                                    # ISO_FREEZE_LEADER: rank 1 stays
            continue
        group.append(i)
    new = list(cands)
    if len(group) < 2:
        return new, protected, 0
    for slot, i in zip(group, sorted(group, key=lambda j: -score[j])):      # stable: ties keep the submitted order
        new[slot] = cands[i]
    return new, protected, len(group)


def iso_rerank(lists, pf_of, lib_max, alpha, lib_skip=0.9, scope='formula', train_keys=frozenset(), keep_train=True,
               considered=None, formula_of=iso_formula, key_of=iso_key14, cos_of=None, cos_w=0.0, freeze_leader=False):
    """All answer lists (pure). lists: [(molecule_id, [SMILES, ...]), ...]; pf_of: {(molecule_id, SMILES): f.z};
    lib_max: {molecule_id: best library match}; considered: molecule ids to rerank (None = every row); cos_of:
    {(molecule_id, SMILES): pf_cos}, read only when cos_w > 0 (a listed candidate without a finite pf_cos then skips the
    molecule: skipped_pf, of which skipped_cos had every f.z). Returns (new lists in the same order, stats)."""
    assert not cos_w or cos_of is not None, 'cos_w > 0 needs cos_of'
    st = dict(considered=0, skipped_lib=0, skipped_pf=0, skipped_cos=0, skipped_short=0, reranked=0, changed_rows=0,
              top1_changed=0, protected=0, moved_candidates=0)
    finite = lambda v: all(x is not None and math.isfinite(float(x)) for x in v)
    out = []
    for mid, cands in lists:
        mid, cands, new = str(mid), list(cands), list(cands)
        if considered is None or mid in considered:
            st['considered'] += 1
            lm = lib_max.get(mid)
            pf = [pf_of.get((mid, s)) for s in cands]
            cos = [cos_of.get((mid, s)) for s in cands] if cos_w else None
            if lm is not None and math.isfinite(float(lm)) and float(lm) >= lib_skip:
                st['skipped_lib'] += 1
            elif not finite(pf) or (cos_w and not finite(cos)):
                st['skipped_pf'] += 1
                st['skipped_cos'] += int(finite(pf))
            elif len(cands) < 2:
                st['skipped_short'] += 1
            else:
                new, prot, moved = iso_order(cands, pf, alpha, scope, train_keys, keep_train, formula_of, key_of,
                                             cos, cos_w, freeze_leader)
                st['reranked'] += 1
                st['protected'] += prot
                st['moved_candidates'] += sum(a != b for a, b in zip(new, cands))
                st['changed_rows'] += int(new != cands)
                st['top1_changed'] += int(new[0] != cands[0])
        out.append((mid, new))
    return out, st


def iso_pf_lookup(frames, col='pf_fz'):
    """{(molecule_id, SMILES): PF-syn score col} (pf_fz = f.z, pf_cos = cosine) from frames with molecule_id, smiles, col
    (and pf_ok, if present); the first finite score in frame order wins (frames that are None, empty or without col are
    passed over)."""
    out = {}
    for f in frames:
        if f is None or not len(f) or col not in f.columns:
            continue
        v = pd.to_numeric(f[col], errors='coerce').to_numpy(float)
        ok = np.isfinite(v)
        if 'pf_ok' in f.columns:
            ok &= f.pf_ok.eq(True).to_numpy()
        for m, s, x in zip(f.molecule_id.astype(str).to_numpy()[ok], f.smiles.astype(str).to_numpy()[ok], v[ok]):
            out.setdefault((m, s), float(x))
    return out


def iso_train_keys(train_path):
    """InChIKey14 of every training structure (only the inchikey14 column of train.parquet is read)."""
    return frozenset(pd.read_parquet(train_path, columns=['inchikey14']).inchikey14.dropna().astype(str).unique())


def iso_check(before, after, sample_ids):
    """after is a valid answer file: sample_submission rows in order, 1-25 unique non-empty SMILES per row, and per row
    the candidate set of before."""
    assert list(after.columns) == ['molecule_id', 'smiles'], 'columns'
    assert after.molecule_id.tolist() == list(sample_ids) == before.molecule_id.tolist(), 'rows'
    a, b = after.smiles.str.split(';'), before.smiles.str.split(';')
    assert a.map(len).between(1, 25).all(), 'list length'
    assert a.map(lambda v: all(x.strip() for x in v) and len(set(v)) == len(v)).all(), 'SMILES'
    assert all(sorted(x) == sorted(y) for x, y in zip(a, b)), 'candidate set changed'


def iso_summary(rep):
    return (f"ISO {rep.get('status')} | alpha {rep.get('alpha')} | scope {rep.get('scope')} | cos w {rep.get('cos_w')} | "
            f"freeze leader {rep.get('freeze_leader')} | molecules considered "
            f"{rep.get('considered', 0)} | skipped lib {rep.get('skipped_lib', 0)} | skipped pf {rep.get('skipped_pf', 0)} | "
            f"changed rows {rep.get('changed_rows', 0)} | top1 changed {rep.get('top1_changed', 0)} | protected "
            f"{rep.get('protected', 0)} | {rep.get('seconds', 0)} s")


ISO_OUTCOME = ('reranked', 'changed_rows', 'top1_changed', 'protected', 'moved_candidates')


def iso_frame(lists):
    """Answer frame (molecule_id, smiles) of [(molecule_id, [SMILES, ...]), ...]."""
    return pd.DataFrame({'molecule_id': [m for m, _ in lists], 'smiles': [';'.join(v) for _, v in lists]})


def iso_write_checked(frame, tmp):
    """Write frame as CSV to tmp and check that it reads back equal."""
    frame.to_csv(tmp, index=False)
    assert pd.read_csv(tmp, dtype=str, keep_default_na=False).equals(frame), 'CSV roundtrip'


def iso_dryrun(sub, lists, sub_path, sample_ids, pf_of, lib_max, alpha, scope, train_keys, keep_train, considered,
               formula_of=iso_formula, key_of=iso_key14, cos_of=None, cos_w=0.0, freeze_leader=False):
    """Check of the rerank path the commit run cannot reach (its visible molecules all have library match 1.0): the same
    rerank with the library skip off, validated, written to a scratch CSV that is read back, moved with os.replace and
    deleted. submission.csv is not touched; never raises. Returns its counts (status 'ok' or 'failed')."""
    tmp, dst = Path(sub_path).with_name('iso_dryrun.csv.tmp'), Path(sub_path).with_name('iso_dryrun.csv')
    try:
        new_lists, st = iso_rerank(lists, pf_of, lib_max, alpha, math.inf, scope, train_keys, keep_train, considered,
                                   formula_of, key_of, cos_of, cos_w, freeze_leader)
        new = iso_frame(new_lists)
        iso_check(sub, new, sample_ids)
        iso_write_checked(new, tmp)
        data = tmp.read_bytes()
        os.replace(tmp, dst)
        assert dst.read_bytes() == data, 'dry-run file changed in the move'
        out = dict(status='ok', **{k: st[k] for k in ('considered', 'skipped_lib', 'skipped_pf', 'skipped_cos',
                                                      'skipped_short') + ISO_OUTCOME},
                   changed_ids=[m for (m, a), (_, b) in zip(lists, new_lists) if a != b][:50])
    except Exception as exc:
        traceback.print_exc()
        out = dict(status='failed', error=f'{type(exc).__name__}: {str(exc)[:400]}')
    finally:
        for p in (tmp, dst):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    if out['status'] == 'ok':
        print(f"ISO dryrun ok | library skip off | molecules considered {out['considered']} | skipped pf {out['skipped_pf']} | "
              f"reranked {out['reranked']} | changed rows {out['changed_rows']} | top1 changed {out['top1_changed']} | "
              f"protected {out['protected']} | moved candidates {out['moved_candidates']}", flush=True)
    else:
        print(f"WARNING ISO dryrun failed (submission.csv not touched): {out['error']}", flush=True)
    return out


def iso_pf_mid_trace(nf, score_fn):
    """Trace of the final layer's PF_MID_W hook (diagnostic only): per molecule of neural_candidate_features, whether
    score_fn (the hook) gave a score other than neural_score (PF used) and a different stable neural order."""
    st = dict(molecules=0, pf_used=0, order_changed=0)
    for _, g in nf.groupby('molecule_id', sort=False):
        base = g['neural_score'].to_numpy(float)
        sc = np.asarray(score_fn(g), float)
        assert sc.shape == base.shape, (sc.shape, base.shape)
        st['molecules'] += 1
        st['pf_used'] += int(not np.array_equal(sc, base, equal_nan=True))
        st['order_changed'] += int(not np.array_equal(np.argsort(-base, kind='stable'), np.argsort(-sc, kind='stable')))
    return st


def iso_apply(sub_path, nf_path, out_path, pf_path, train_path, sample_ids, alpha, lib_skip=0.9, keep_train=True,
              scope='formula', report_path=None, skip=None, processed_only=True, formula_of=iso_formula,
              key_of=iso_key14, dryrun=False, cos_w=0.0, freeze_leader=False):
    """Rerank the answer file sub_path in place (all inputs are arguments; no notebook globals). skip: a status string
    that leaves the file untouched (e.g. 'skipped_fast_commit'). processed_only: rerank only the molecules the pipeline
    processed (neural_candidate_features.processed; the 12 visible molecules in the smoke commit, all in the rerun).
    dryrun: also run iso_dryrun (library skip off, scratch file only; report key 'dryrun'). cos_w (ISO_COS_W >= 0): weight
    of z(pf_cos) next to z(pf_fz), pf_cos from pf_path (pf_ok rows) only when cos_w > 0. freeze_leader
    (ISO_FREEZE_LEADER, bool): rank 1 never moves.
    Never fatal: on any exception the input bytes are written back (WARNING) and the outcome counters are reported as 0
    (the aborted attempt's as attempted_*). Returns the report (also report_path)."""
    t0 = time.time()
    sub_path = Path(sub_path)
    rep = dict(status='off', alpha=alpha, scope=scope, cos_w=cos_w, freeze_leader=freeze_leader, lib_skip=lib_skip,
               keep_train=keep_train, considered=0, skipped_lib=0, skipped_pf=0, skipped_cos=0, skipped_short=0,
               reranked=0, changed_rows=0, top1_changed=0, protected=0)
    before = sub_path.read_bytes()
    rep['input_sha256'] = hashlib.sha256(before).hexdigest()
    block = None
    try:
        if skip:
            rep['status'] = str(skip)
        elif float(alpha) != 0.0:
            alpha = float(alpha)
            block = rdBase.BlockLogs()                          # no RDKit parse warnings in the log
            assert math.isfinite(alpha) and alpha > 0, f'ISO_ALPHA {alpha!r}'
            assert scope in ('formula', 'all'), f'ISO_SCOPE {scope!r}'
            assert not isinstance(cos_w, bool) and math.isfinite(float(cos_w)) and float(cos_w) >= 0, f'ISO_COS_W {cos_w!r}'
            assert isinstance(freeze_leader, bool), f'ISO_FREEZE_LEADER {freeze_leader!r}'
            cos_w = float(cos_w)
            sub = pd.read_csv(io.BytesIO(before), dtype=str, keep_default_na=False)
            iso_check(sub, sub, sample_ids)
            nf = pd.read_parquet(nf_path)
            mids = nf.molecule_id.astype(str)
            lib = pd.to_numeric(nf.library_max, errors='coerce').groupby(mids.to_numpy()).max()
            lib_max = {str(m): float(v) for m, v in lib.items()}
            considered = set(mids[nf.processed.eq(True).to_numpy()]) if processed_only else None
            frames = [nf] + [pd.read_parquet(p) if p is not None and Path(p).is_file() else None for p in (out_path, pf_path)]
            pf_of = iso_pf_lookup(frames)
            cos_of = iso_pf_lookup(frames[2:], 'pf_cos') if cos_w else None     # pf_scores.parquet only
            train = iso_train_keys(train_path) if keep_train else frozenset()
            lists = [(m, s.split(';')) for m, s in zip(sub.molecule_id, sub.smiles)]
            new_lists, st = iso_rerank(lists, pf_of, lib_max, alpha, lib_skip, scope, train, keep_train, considered,
                                       formula_of, key_of, cos_of, cos_w, freeze_leader)
            new = iso_frame(new_lists)
            iso_check(sub, new, sample_ids)
            rep.update(st, status='applied', train_keys=len(train), pf_keys=len(pf_of))
            if cos_w:
                rep['cos_keys'] = len(cos_of)
            if dryrun:
                rep['dryrun'] = iso_dryrun(sub, lists, sub_path, sample_ids, pf_of, lib_max, alpha, scope, train, keep_train,
                                           considered, formula_of, key_of, cos_of, cos_w, freeze_leader)
            if st['changed_rows']:
                shutil.copyfile(sub_path, sub_path.with_name(sub_path.stem + '_pre_iso.csv'))
                tmp = sub_path.with_name(sub_path.name + '.iso.tmp')
                iso_write_checked(new, tmp)
                os.replace(tmp, sub_path)
    except Exception as exc:
        traceback.print_exc()
        rep.update({'attempted_' + k: rep.get(k, 0) for k in ISO_OUTCOME})   # nothing was applied: outcome counters 0
        rep.update(dict.fromkeys(ISO_OUTCOME, 0), status='fallback', error=f'{type(exc).__name__}: {str(exc)[:400]}')
        try:
            if sub_path.read_bytes() != before:
                tmp = sub_path.with_name(sub_path.name + '.iso.tmp')
                tmp.write_bytes(before)
                os.replace(tmp, sub_path)
            rep['restored'] = sub_path.read_bytes() == before
        except Exception as exc2:
            rep['restore_error'] = repr(exc2)[:300]
        print(f'WARNING ISO failed, pre-ISO submission.csv kept: {rep["error"]}', flush=True)
    finally:
        block = None                                            # RDKit logging back on
    rep['output_sha256'] = hashlib.sha256(sub_path.read_bytes()).hexdigest()
    rep['seconds'] = round(time.time() - t0, 1)
    if report_path is not None:
        try:
            Path(report_path).write_text(json.dumps(rep, indent=2, default=str))
        except Exception:
            traceback.print_exc()
    print(iso_summary(rep), flush=True)
    return rep


# ---- run (everything above is definitions) ----
_iso_sub = ROOT / 'submission.csv'
_iso_before = _iso_sub.read_bytes()
_iso_fast = True                                                # until the guards below are read
try:
    _iso_fast = not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast'
    if _iso_fast:
        _iso_skip = 'skipped_fast_commit'
    elif float(ISO_ALPHA) != 0.0 and time.time() - NB_T0 > 8.7 * 3600:
        _iso_skip = 'skipped_time_guard'
    else:
        _iso_skip = None
    ISO_REPORT = iso_apply(_iso_sub, ROOT / 'neural_candidate_features.parquet', ROOT / 'outsider_neural_features.parquet',
                           ROOT / 'pf_scores.parquet', COMP / 'train.parquet', SAMPLE_IDS, ISO_ALPHA, ISO_LIB_SKIP,
                           ISO_KEEP_TRAIN, ISO_SCOPE, ROOT / 'iso_report.json', skip=_iso_skip, dryrun=not IS_RERUN,
                           cos_w=ISO_COS_W, freeze_leader=ISO_FREEZE_LEADER)
except Exception as _iso_exc:                                   # iso_apply is fail-safe itself; this guards the call
    traceback.print_exc()
    ISO_REPORT = dict(status='fallback', alpha=globals().get('ISO_ALPHA'), scope=globals().get('ISO_SCOPE'),
                      cos_w=globals().get('ISO_COS_W'), freeze_leader=globals().get('ISO_FREEZE_LEADER'),
                      error=f'wrapper {type(_iso_exc).__name__}: {str(_iso_exc)[:300]}')
    if _iso_sub.read_bytes() != _iso_before:
        _iso_sub.with_name('submission.csv.iso.tmp').write_bytes(_iso_before)
        os.replace(_iso_sub.with_name('submission.csv.iso.tmp'), _iso_sub)
    ISO_REPORT['restored'] = _iso_sub.read_bytes() == _iso_before
    print(f'WARNING ISO wrapper error, pre-ISO submission.csv kept: {ISO_REPORT["error"]}', flush=True)
    print(iso_summary(ISO_REPORT), flush=True)
    try:
        (ROOT / 'iso_report.json').write_text(json.dumps(ISO_REPORT, indent=2, default=str))
    except Exception:
        traceback.print_exc()

try:
    _iso_mid_on = float(globals().get('PF_MID_W', 0.0) or 0.0) > 0 and not _iso_fast
except Exception:
    _iso_mid_on = False
if _iso_mid_on:                                                 # diagnostic only; reads, never writes answers
    try:
        PF_MID_TRACE = dict(w=PF_MID_W, **iso_pf_mid_trace(pd.read_parquet(ROOT / 'neural_candidate_features.parquet'),
                                                           globals()['_mid_neural_score']))
        print(f"PF_MID trace | w {PF_MID_W} | molecules {PF_MID_TRACE['molecules']} | PF used "
              f"{PF_MID_TRACE['pf_used']} | neural order changed {PF_MID_TRACE['order_changed']}", flush=True)
    except Exception as _iso_exc:
        PF_MID_TRACE = dict(w=PF_MID_W, error=f'{type(_iso_exc).__name__}: {str(_iso_exc)[:300]}')
        print(f"WARNING PF_MID trace failed: {PF_MID_TRACE['error']}", flush=True)
    try:
        (ROOT / 'pf_mid_trace.json').write_text(json.dumps(PF_MID_TRACE, indent=2, default=str))
    except Exception:
        traceback.print_exc()

final = pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
assert list(final.columns) == ['molecule_id', 'smiles']
assert final.molecule_id.tolist() == SAMPLE_IDS and final.molecule_id.is_unique
assert final.smiles.str.split(';').map(len).between(1, 25).all()
assert final.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
print('SUBMISSION READY:', ROOT / 'submission.csv', 'molecules', len(final), 'ISO', json.dumps(
    {k: ISO_REPORT.get(k) for k in ('status', 'alpha', 'scope', 'cos_w', 'freeze_leader', 'considered', 'reranked',
                                    'changed_rows', 'top1_changed', 'protected', 'output_sha256', 'error')
     if k in ISO_REPORT}, default=str))
