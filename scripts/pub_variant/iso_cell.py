# ── ISO: isomer reranker on the final answer lists (ours; runs right after the final layer) ──────────────────────
# heliosli-style (team LB 0.486) with our PF-syn f.z as the model score. Per molecule, over its submitted list:
# score = z(-log rank) + ISO_ALPHA * z(pf_fz)   (z over the list, population std; std 0 -> zeros).
# ISO_SCOPE 'formula': only the candidates with the rank-1 candidate's formula are re-sorted by score, inside the positions
# they already hold; 'all': every candidate. Protected candidates never move: training structures (ISO_KEEP_TRAIN: RDKit
# InChIKey14 in the inchikey14 column of train.parquet) and SMILES RDKit cannot read. Skipped molecules: best library
# match >= ISO_LIB_SKIP, a listed candidate without a finite PF-syn f.z (neural_candidate_features, then
# outsider_neural_features, then pf_scores.parquet), fewer than 2 candidates. ISO_ALPHA 0: submission.csv is not touched.
# Any failure restores the pre-ISO submission.csv byte for byte (WARNING). Report: iso_report.json, one 'ISO ...' line.
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
              formula_of=iso_formula, key_of=iso_key14):
    """New order of one answer list (pure). cands: SMILES in submitted order (>= 2); pf: finite PF-syn f.z per candidate.
    Returns (new list, protected, moved): protected = in-scope candidates held in place (training structure or
    unreadable), moved = size of the re-sorted group (0 when fewer than 2 candidates are movable)."""
    n = len(cands)
    assert n >= 2 and len(pf) == n and scope in ('formula', 'all'), (n, len(pf), scope)
    score = iso_z(-np.log(np.arange(1, n + 1, dtype=float))) + float(alpha) * iso_z(pf)
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
        group.append(i)
    new = list(cands)
    if len(group) < 2:
        return new, protected, 0
    for slot, i in zip(group, sorted(group, key=lambda j: -score[j])):      # stable: ties keep the submitted order
        new[slot] = cands[i]
    return new, protected, len(group)


def iso_rerank(lists, pf_of, lib_max, alpha, lib_skip=0.9, scope='formula', train_keys=frozenset(), keep_train=True,
               considered=None, formula_of=iso_formula, key_of=iso_key14):
    """All answer lists (pure). lists: [(molecule_id, [SMILES, ...]), ...]; pf_of: {(molecule_id, SMILES): f.z};
    lib_max: {molecule_id: best library match}; considered: molecule ids to rerank (None = every row).
    Returns (new lists in the same order, stats)."""
    st = dict(considered=0, skipped_lib=0, skipped_pf=0, skipped_short=0, reranked=0, changed_rows=0, top1_changed=0,
              protected=0, moved_candidates=0)
    out = []
    for mid, cands in lists:
        mid, cands, new = str(mid), list(cands), list(cands)
        if considered is None or mid in considered:
            st['considered'] += 1
            lm = lib_max.get(mid)
            pf = [pf_of.get((mid, s)) for s in cands]
            if lm is not None and math.isfinite(float(lm)) and float(lm) >= lib_skip:
                st['skipped_lib'] += 1
            elif not all(v is not None and math.isfinite(float(v)) for v in pf):
                st['skipped_pf'] += 1
            elif len(cands) < 2:
                st['skipped_short'] += 1
            else:
                new, prot, moved = iso_order(cands, pf, alpha, scope, train_keys, keep_train, formula_of, key_of)
                st['reranked'] += 1
                st['protected'] += prot
                st['moved_candidates'] += sum(a != b for a, b in zip(new, cands))
                st['changed_rows'] += int(new != cands)
                st['top1_changed'] += int(new[0] != cands[0])
        out.append((mid, new))
    return out, st


def iso_pf_lookup(frames):
    """{(molecule_id, SMILES): PF-syn f.z} from frames with molecule_id, smiles, pf_fz (and pf_ok, if present); the
    first finite score in frame order wins (frames that are None, empty or without pf_fz are passed over)."""
    out = {}
    for f in frames:
        if f is None or not len(f) or 'pf_fz' not in f.columns:
            continue
        v = pd.to_numeric(f.pf_fz, errors='coerce').to_numpy(float)
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
    return (f"ISO {rep.get('status')} | alpha {rep.get('alpha')} | scope {rep.get('scope')} | molecules considered "
            f"{rep.get('considered', 0)} | skipped lib {rep.get('skipped_lib', 0)} | skipped pf {rep.get('skipped_pf', 0)} | "
            f"changed rows {rep.get('changed_rows', 0)} | top1 changed {rep.get('top1_changed', 0)} | protected "
            f"{rep.get('protected', 0)} | {rep.get('seconds', 0)} s")


def iso_apply(sub_path, nf_path, out_path, pf_path, train_path, sample_ids, alpha, lib_skip=0.9, keep_train=True,
              scope='formula', report_path=None, skip=None, processed_only=True, formula_of=iso_formula,
              key_of=iso_key14):
    """Rerank the answer file sub_path in place (all inputs are arguments; no notebook globals). skip: a status string
    that leaves the file untouched (e.g. 'skipped_fast_commit'). processed_only: rerank only the molecules the pipeline
    processed (neural_candidate_features.processed; the 12 visible molecules in the smoke commit, all in the rerun).
    Never fatal: on any exception the input bytes are written back (WARNING). Returns the report (also report_path)."""
    t0 = time.time()
    sub_path = Path(sub_path)
    rep = dict(status='off', alpha=alpha, scope=scope, lib_skip=lib_skip, keep_train=keep_train, considered=0,
               skipped_lib=0, skipped_pf=0, skipped_short=0, reranked=0, changed_rows=0, top1_changed=0, protected=0)
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
            sub = pd.read_csv(io.BytesIO(before), dtype=str, keep_default_na=False)
            iso_check(sub, sub, sample_ids)
            nf = pd.read_parquet(nf_path)
            mids = nf.molecule_id.astype(str)
            lib = pd.to_numeric(nf.library_max, errors='coerce').groupby(mids.to_numpy()).max()
            lib_max = {str(m): float(v) for m, v in lib.items()}
            considered = set(mids[nf.processed.eq(True).to_numpy()]) if processed_only else None
            frames = [nf] + [pd.read_parquet(p) if p is not None and Path(p).is_file() else None for p in (out_path, pf_path)]
            pf_of = iso_pf_lookup(frames)
            train = iso_train_keys(train_path) if keep_train else frozenset()
            new_lists, st = iso_rerank([(m, s.split(';')) for m, s in zip(sub.molecule_id, sub.smiles)], pf_of, lib_max,
                                       alpha, lib_skip, scope, train, keep_train, considered, formula_of, key_of)
            new = pd.DataFrame({'molecule_id': [m for m, _ in new_lists], 'smiles': [';'.join(v) for _, v in new_lists]})
            iso_check(sub, new, sample_ids)
            rep.update(st, status='applied', train_keys=len(train), pf_keys=len(pf_of))
            if st['changed_rows']:
                shutil.copyfile(sub_path, sub_path.with_name(sub_path.stem + '_pre_iso.csv'))
                tmp = sub_path.with_name(sub_path.name + '.iso.tmp')
                new.to_csv(tmp, index=False)
                assert pd.read_csv(tmp, dtype=str, keep_default_na=False).equals(new), 'CSV roundtrip'
                os.replace(tmp, sub_path)
    except Exception as exc:
        traceback.print_exc()
        rep.update(status='fallback', error=f'{type(exc).__name__}: {str(exc)[:400]}')
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
try:
    if not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast':
        _iso_skip = 'skipped_fast_commit'
    elif float(ISO_ALPHA) != 0.0 and time.time() - NB_T0 > 8.7 * 3600:
        _iso_skip = 'skipped_time_guard'
    else:
        _iso_skip = None
    ISO_REPORT = iso_apply(_iso_sub, ROOT / 'neural_candidate_features.parquet', ROOT / 'outsider_neural_features.parquet',
                           ROOT / 'pf_scores.parquet', COMP / 'train.parquet', SAMPLE_IDS, ISO_ALPHA, ISO_LIB_SKIP,
                           ISO_KEEP_TRAIN, ISO_SCOPE, ROOT / 'iso_report.json', skip=_iso_skip)
except Exception as _iso_exc:                                   # iso_apply is fail-safe itself; this guards the call
    traceback.print_exc()
    ISO_REPORT = dict(status='fallback', alpha=globals().get('ISO_ALPHA'), scope=globals().get('ISO_SCOPE'),
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

final = pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
assert list(final.columns) == ['molecule_id', 'smiles']
assert final.molecule_id.tolist() == SAMPLE_IDS and final.molecule_id.is_unique
assert final.smiles.str.split(';').map(len).between(1, 25).all()
assert final.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
print('SUBMISSION READY:', ROOT / 'submission.csv', 'molecules', len(final), 'ISO', json.dumps(
    {k: ISO_REPORT.get(k) for k in ('status', 'alpha', 'scope', 'considered', 'reranked', 'changed_rows', 'top1_changed',
                                    'protected', 'output_sha256', 'error') if k in ISO_REPORT}, default=str))
