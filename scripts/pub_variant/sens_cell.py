# ── SENS: saner seed ensemble in the neural term (ours; scripts/make_pub_variant.py SENS=K) ─────────────────────────
# Runs where the PF-syn cell above runs: the cell below calls it after score_final_lists / score_outside_candidates and
# before build_consensus. The published saner model (BUNDLE final_model.pt, seed 20261023) has scored the final lists
# (neural_candidate_features.parquet) and the outside candidates (outsider_neural_features.parquet). Here the pinned
# neural_scores.py of cell 8 scores the SAME inputs (frozen_candidate_evidence.parquet, outsider_evidence.parquet,
# effective_query_spectra.parquet; the outside candidates whenever their features file exists, as every reader of it
# then sees one scale) once more per extra checkpoint of the same recipe: final_model_seed<s>.pt of the kernel
# output bryantyu/casmi26-saner-seeds for the first SENS - 1 seeds of SENS_SEEDS (file sha256 checked against its
# final_models_sha256.json), each from a temporary bundle folder that holds the published model.py / features.py /
# config.json (checked against MODEL_PINS) and that checkpoint as final_model.pt, with cell 9's command line, working
# directory and environment (plus a time limit). Then, with the SENS members = the published model + those seeds:
#   SENS_MODE 'raw'  neural_score := mean of the members' neural_score
#   SENS_MODE 'z'    neural_score := mean of z_m(neural_score_m), z_m(x) = (x - mean) / std (population std; 0 -> 1) of
#                    member m's scores over the molecule's listed candidates; outside candidates use the same maps
#   both modes       neural_cos / neural_fp := means of the members' values (the fragment checks gate on their absolute
#                    differences, so they keep their units; in 'raw' mode the mean score stays cos / T + w fp)
# A molecule of the final lists changes only when every one of its rows is processed, neural_ok and finite (score, cos,
# fp) for every member. An outside candidate of a changed molecule takes the ensemble when it is neural_ok and finite for
# every member, else the published member alone on the ensemble's scale (raw: its own values; z: z_published). Every
# other row keeps the published values. Audit columns: neural_score_published / neural_cos_published /
# neural_fp_published, neural_score_seed<s> per extra seed, sens_members (members averaged; 1 = the published member
# alone, 0 = row unchanged). Atomic writes (tmp + os.replace; the published files kept in sens_backup/); any failure
# keeps or restores the published files (WARNING). Report sens_report.json; lines 'SENS setup', 'SENS seeds' and
# 'SENS <status> | K | mode | molecules | rows | changed top1 (neural) | rank corr | seconds'.
import glob, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd

SENS_PUBLISHED_SEED = 20261023         # the seed of the published final_model.pt (its config.json)
SENS_SOURCE = 'casmi26-saner-seeds'    # kernel output bryantyu/casmi26-saner-seeds (kernel_sources)
SENS_COLS = ('neural_score', 'neural_cos', 'neural_fp')
SENS_ALIGN = ('molecule_id', 'smiles', 'baseline_rank')


def sens_z_map(v):
    """z-map of one member's scores over a molecule's listed candidates: x -> (x - mean) / std (population std; std 0
    -> 1, as the holdout evaluation of casmi26-saner-seeds)."""
    v = np.asarray(v, np.float64)
    mu, sd = float(v.mean()), float(v.std())
    sd = sd if sd > 0 else 1.0
    return lambda x: (np.asarray(x, np.float64) - mu) / sd


def _sens_spearman(a, b):
    """Spearman correlation of two score vectors; None when either is constant."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r = pd.Series(a).rank().corr(pd.Series(b).rank())
    return float(r) if np.isfinite(r) else None


def _sens_stack(base, members, what):
    """{column: (members x rows) float64} for score / cos / fp (member 0 = base) and the usable mask (neural_ok and
    all three finite); every member must hold base's rows in base's order."""
    frames = [base] + list(members)
    for k, f in enumerate(frames):
        miss = [c for c in SENS_ALIGN + SENS_COLS + ('neural_ok',) if c not in f.columns]
        assert not miss, f'{what}: member {k} lacks {miss}'
    for k, f in enumerate(members, 1):
        assert len(f) == len(base), f'{what}: member {k} has {len(f)} rows, the published file {len(base)}'
        for c in SENS_ALIGN:
            assert f[c].astype(str).tolist() == base[c].astype(str).tolist(), f'{what}: member {k} differs in {c}'
    vals = {c: np.stack([pd.to_numeric(f[c], errors='coerce').to_numpy(np.float64) for f in frames]).reshape(
        len(frames), len(base)) for c in SENS_COLS}
    ok = np.stack([f.neural_ok.fillna(False).astype(bool).to_numpy() for f in frames]).reshape(len(frames), len(base))
    for c in SENS_COLS:
        ok &= np.isfinite(vals[c])
    return vals, ok


def _sens_frame(frame, vals, new, members, seeds):
    """frame with the ensemble in SENS_COLS and the audit columns (published values, each extra seed's score)."""
    out = frame.copy()
    for c in SENS_COLS:
        assert f'{c}_published' not in out.columns, f'{c}_published exists: SENS already ran on this file'
        out[f'{c}_published'] = vals[c][0]
    for k, s in enumerate(seeds, 1):
        out[f'neural_score_seed{s}'] = vals['neural_score'][k]
    for c in SENS_COLS:
        out[c] = new[c]
    out['sens_members'] = members
    return out


def sens_ensemble_frames(main, outs, members_main, members_outs=None, mode='raw', seeds=None):
    """Seed ensemble of the neural term (pure; the rule is in the cell header).

    main: the published neural_candidate_features; outs: the published outsider_neural_features or None;
    members_main / members_outs: the extra seeds' neural_scores.py outputs on the same evidence files (lists of frames
    with the published file's rows in its order; members_outs one per extra seed when outs is given); seeds: one label
    per extra member (audit column neural_score_seed<label>). Returns (new_main, new_outs, stats)."""
    assert mode in ('raw', 'z'), f'mode {mode!r}'
    members_main = list(members_main)
    seeds = list(seeds) if seeds is not None else list(range(1, len(members_main) + 1))
    assert members_main and len(seeds) == len(members_main) and len(set(map(str, seeds))) == len(seeds), seeds
    K = 1 + len(members_main)
    vals, ok = _sens_stack(main, members_main, 'final lists')
    proc = main.processed.fillna(False).astype(bool).to_numpy()
    new = {c: vals[c][0].copy() for c in SENS_COLS}
    nmem = np.zeros(len(main), np.int64)
    stats = dict(k=K, mode=mode, molecules_total=0, molecules=0, inactive=0, kept_not_ok=0, rows_main=0, outsiders=0,
                 outsiders_published_only=0, outsiders_kept=0, top1_changed=0, rho=[],
                 per_seed={str(s): dict(rho=[], top1_agree=[]) for s in seeds})
    maps = {}
    for mid, idx in main.groupby('molecule_id', sort=False).indices.items():
        stats['molecules_total'] += 1
        if not (proc[idx].all() and ok[0, idx].all()):
            stats['inactive'] += 1                  # the published scorer did not use this molecule: unchanged
            continue
        if not ok[:, idx].all():
            stats['kept_not_ok'] += 1               # an extra member is not neural_ok / not finite here: published
            continue
        s = vals['neural_score'][:, idx]
        if mode == 'raw':
            e = s.mean(0)
            maps[str(mid)] = None
        else:
            zm = [sens_z_map(s[k]) for k in range(K)]
            e = np.mean([zm[k](s[k]) for k in range(K)], 0)
            maps[str(mid)] = zm
        new['neural_score'][idx] = e
        for c in ('neural_cos', 'neural_fp'):
            new[c][idx] = vals[c][:, idx].mean(0)
        nmem[idx] = K
        stats['molecules'] += 1
        stats['rows_main'] += len(idx)
        stats['top1_changed'] += int(np.argmax(e) != np.argmax(s[0]))     # argmax = stable_rank's first place
        if len(idx) > 2:
            r = _sens_spearman(s[0], e)
            if r is not None:
                stats['rho'].append(r)
        for k, sd in enumerate(seeds, 1):
            d = stats['per_seed'][str(sd)]
            d['top1_agree'].append(int(np.argmax(s[k]) == np.argmax(s[0])))
            r = _sens_spearman(s[0], s[k]) if len(idx) > 2 else None
            if r is not None:
                d['rho'].append(r)
    new_main = _sens_frame(main, vals, new, nmem, seeds)
    new_outs = None
    if outs is not None:
        members_outs = list(members_outs or [])
        assert len(members_outs) == len(members_main), 'one outside-candidate frame per extra member'
        vo, oko = _sens_stack(outs, members_outs, 'outside candidates')
        newo = {c: vo[c][0].copy() for c in SENS_COLS}
        nmo = np.zeros(len(outs), np.int64)
        for mid, idx in outs.groupby('molecule_id', sort=False).indices.items():
            if str(mid) not in maps:                # molecule unchanged in the final lists: published
                stats['outsiders_kept'] += len(idx)
                continue
            zm = maps[str(mid)]
            full = oko[:, idx].all(0)
            alone = ~full & oko[0, idx]
            s = vo['neural_score'][:, idx]
            with np.errstate(invalid='ignore'):
                if zm is None:
                    e, e1 = s.mean(0), s[0]
                else:
                    e, e1 = np.mean([zm[k](s[k]) for k in range(K)], 0), zm[0](s[0])
                cf = {c: vo[c][:, idx].mean(0) for c in ('neural_cos', 'neural_fp')}
            newo['neural_score'][idx[full]] = e[full]
            newo['neural_score'][idx[alone]] = e1[alone]
            for c in ('neural_cos', 'neural_fp'):
                newo[c][idx[full]] = cf[c][full]
            nmo[idx[full]] = K
            nmo[idx[alone]] = 1
            stats['outsiders'] += int(full.sum())
            stats['outsiders_published_only'] += int(alone.sum())
            stats['outsiders_kept'] += int((~full & ~alone).sum())
        new_outs = _sens_frame(outs, vo, newo, nmo, seeds)
    stats['rows'] = stats['rows_main'] + stats['outsiders']
    stats['rho'] = round(float(np.mean(stats['rho'])), 4) if stats['rho'] else None
    for d in stats['per_seed'].values():
        d['molecules'] = len(d['top1_agree'])
        d['rho'] = round(float(np.mean(d['rho'])), 4) if d['rho'] else None
        d['top1_agree'] = round(float(np.mean(d['top1_agree'])), 4) if d['top1_agree'] else None
    return new_main, new_outs, stats


def _sens_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sens_source_dir():
    """Folder of the bryantyu/casmi26-saner-seeds output: the one holding final_models_sha256.json (paths containing
    casmi26-saner-seeds first, then the shortest)."""
    hits = sorted(glob.glob(str(INPUT / '**' / 'final_models_sha256.json'), recursive=True),
                  key=lambda p: (SENS_SOURCE not in p, len(p), p))
    if not hits:
        raise FileNotFoundError('final_models_sha256.json not under /kaggle/input: attach the output of '
                                'bryantyu/casmi26-saner-seeds (kernel_sources)')
    return Path(hits[0]).parent


def sens_checkpoints(src, seeds):
    """{seed: record} of final_model_seed<s>.pt in src, its sha256 against final_models_sha256.json (and the .sha256
    file next to it); the table itself."""
    table = json.loads((Path(src) / 'final_models_sha256.json').read_text())
    out = {}
    for s in seeds:
        name = f'final_model_seed{s}.pt'
        p = Path(src) / name
        if not p.is_file():
            raise FileNotFoundError(f'{name} is not in {src} (its refit missing?): set SENS / SENS_SEEDS in cell 1')
        entry = table.get(name) or {}
        side = p.with_name(name + '.sha256')
        got = _sens_sha(p)
        out[s] = dict(file=str(p), sha256=got, json_sha256=entry.get('file_sha256'),
                      sidecar_sha256=(side.read_text().split() or [None])[0] if side.is_file() else None,
                      ok=got == entry.get('file_sha256'), epochs=entry.get('epochs'), bytes=p.stat().st_size)
    return out, table


def sens_decision(src):
    """The SEEDS SUMMARY decision of casmi26-saner-seeds (seeds_report.json), for the log only."""
    try:
        d = json.loads((Path(src) / 'seeds_report.json').read_text()).get('decision') or {}
        return {k: d.get(k) for k in ('verdict', 'k', 'mode', 'members', 'mrr25', 'pub_mrr25')}
    except Exception as exc:
        return dict(verdict=None, error=f'{type(exc).__name__}: {str(exc)[:120]}')


def _sens_bundle(dst, ckpt, sha):
    """Temporary bundle folder: the published model.py / features.py / config.json + the checkpoint as final_model.pt."""
    dst.mkdir(parents=True)
    for name in ('model.py', 'features.py', 'config.json'):
        shutil.copyfile(Path(BUNDLE) / name, dst / name)
        assert _sens_sha(dst / name) == MODEL_PINS[name], f'{name} differs from the published pin'
    shutil.copyfile(ckpt, dst / 'final_model.pt')
    assert _sens_sha(dst / 'final_model.pt') == sha, f'{ckpt} changed while copied'


def _sens_run_scorer(bundle, evidence, out, log, timeout):
    """The pinned neural_scores.py as cell 9 runs it (score_final_lists / score_outside_candidates: the same arguments,
    working directory and environment), with a time limit; returns (seconds, input failures)."""
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONUNBUFFERED': '1'}
    env.pop('INC_PANEL', None)
    args = [sys.executable, CODE / 'neural_scores.py', bundle, 'final_model.pt', evidence,
            ROOT / 'effective_query_spectra.parquet', out, 'cuda']
    t0 = time.time()
    with open(log, 'w') as handle:
        proc = subprocess.run([str(a) for a in args], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'neural_scores.py exit {proc.returncode} ({Path(log).name}): '
                           f'{Path(log).read_text()[-600:]}')
    return time.time() - t0, len(json.loads(Path(str(out) + '.errors.json').read_text()))


def sens_line(rep):
    return (f"SENS {rep.get('status')} | K {rep.get('k')} | mode {rep.get('mode')} | molecules {rep.get('molecules', 0)} | "
            f"rows {rep.get('rows', 0)} | changed top1 (neural) {rep.get('top1_changed', 0)} | "
            f"rank corr {rep.get('rho')} | {rep.get('seconds')} s"
            + (f" | {rep['error'][:240]}" if rep.get('error') else ''))


def ours_seed_ensemble():
    """Saner seed ensemble in neural_candidate_features / outsider_neural_features (in place); never fatal: any failure
    keeps (restores) the published scores."""
    t0 = time.time()
    K, mode = int(globals().get('SENS', 0) or 0), globals().get('SENS_MODE', 'raw')
    rep = dict(status='off', k=K, mode=mode, seeds=[], molecules=0, rows=0, top1_changed=0, rho=None, failures=0)
    if K <= 0:
        print('SENS off (SENS = 0): the published saner scores', flush=True)
        return rep
    main_p, out_p = ROOT / 'neural_candidate_features.parquet', ROOT / 'outsider_neural_features.parquet'
    jobs = [(main_p, ROOT / 'frozen_candidate_evidence.parquet')]
    if out_p.is_file():           # every reader of the outside candidates (consensus, fragment check, final-layer fill)
        jobs.append((out_p, ROOT / 'outsider_evidence.parquet'))      # gets the same scale as the final lists
    work, bak = ROOT / 'sens_work', ROOT / 'sens_backup'
    replaced = []
    try:
        assert mode in ('raw', 'z'), f'SENS_MODE {mode!r}'
        seeds = [int(s) for s in SENS_SEEDS][:max(K - 1, 0)]
        assert K >= 2 and len(seeds) == K - 1 and len(set(seeds)) == len(seeds) and SENS_PUBLISHED_SEED not in seeds, \
            f'SENS {K} needs {K - 1} distinct extra seeds in SENS_SEEDS {SENS_SEEDS!r}'
        rep['seeds'] = seeds
        assert NEURAL_OK and main_p.is_file(), 'no published neural scores'
        assert all(ev.is_file() for _, ev in jobs), 'an evidence file of the published scoring is missing'
        rep['outsiders_file'] = len(jobs) > 1
        budget = min(float(SENS_MAX_SECONDS), 8.0 * 3600 - (time.time() - NB_T0))
        if budget < 120:
            raise TimeoutError(f'only {budget:.0f} s left (SENS_MAX_SECONDS / the 8.0 h guard)')
        deadline = t0 + budget
        assert _sens_sha(CODE / 'neural_scores.py') == MODULE_SHA256['neural_scores.py'], 'neural_scores.py pin'
        src = sens_source_dir()
        ck, table = sens_checkpoints(src, seeds)
        pub = (table.get(f'final_model_seed{SENS_PUBLISHED_SEED}.pt') or {}).get('file_sha256')
        dec = sens_decision(src)
        rep.update(source=str(src), checkpoints={str(s): v for s, v in ck.items()}, seeds_decision=dec,
                   published_copy_matches_pin=None if pub is None else pub == MODEL_PINS['final_model.pt'])
        n_ok = sum(v['ok'] for v in ck.values())
        n_side = sum(v['sidecar_sha256'] == v['sha256'] for v in ck.values())
        dm = dec.get('members')                     # the decision's member seeds (published first), when listed
        same = not isinstance(dm, list) or sorted(map(str, dm)) == sorted(map(str, [SENS_PUBLISHED_SEED] + seeds))
        note = '' if dec.get('k') in (None, K) and dec.get('mode') in (None, mode) and same else \
            (f" | NOTE: build K {K} mode {mode}, seeds decision k {dec.get('k')} mode {dec.get('mode')}"
             + ('' if same else f" members {dm}"))
        print(f"SENS setup | K {K} (published + {','.join(map(str, seeds))}) | mode {mode} | source {src} | sha256 vs "
              f"final_models_sha256.json {n_ok}/{len(ck)} ok (.sha256 files {n_side}/{len(ck)}) | published copy = pin "
              f"{rep['published_copy_matches_pin']} | seeds decision {dec.get('verdict')} k {dec.get('k')} mode "
              f"{dec.get('mode')}{note}", flush=True)
        bad = [s for s, v in ck.items() if not v['ok']]
        if bad:
            raise ValueError(f'sha256 differs from final_models_sha256.json: {bad}')
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        members, runs = {p: [] for p, _ in jobs}, []
        for s in seeds:
            b = work / f'bundle_{s}'
            _sens_bundle(b, Path(ck[s]['file']), ck[s]['sha256'])
            for path, evidence in jobs:
                left = deadline - time.time()
                if left < 30:
                    raise TimeoutError(f'SENS budget spent before seed {s} on {path.name}')
                out = work / f'{path.stem}_seed{s}.parquet'
                sec, nerr = _sens_run_scorer(b, evidence, out, work / f'{path.stem}_seed{s}.log', left)
                members[path].append(pd.read_parquet(out))
                runs.append(dict(seed=s, file=path.name, seconds=round(sec, 1), input_failures=nerr))
                rep['failures'] += nerr
            shutil.rmtree(b)
        rep['runs'] = runs
        main = pd.read_parquet(main_p)
        outs = pd.read_parquet(out_p) if len(jobs) > 1 else None
        new_main, new_outs, st = sens_ensemble_frames(main, outs, members[main_p],
                                                      members[out_p] if outs is not None else None, mode, seeds)
        for old, new in ((main, new_main), (outs, new_outs)):
            if old is None:
                continue
            assert len(new) == len(old) and all(new[c].equals(old[c]) for c in old.columns if c not in SENS_COLS), \
                'a column other than the neural scores changed'
            assert np.isfinite(new.neural_score.to_numpy(np.float64)).all(), 'non-finite neural score'
        if st['molecules'] == 0:
            raise RuntimeError(f'no molecule ensembled: {st}')
        writes = [(main_p, new_main)] + ([(out_p, new_outs)] if outs is not None else [])
        bak.mkdir(exist_ok=True)
        for path, frame in writes:
            shutil.copyfile(path, bak / path.name)
            tmp = path.with_name(path.name + '.sens.tmp')
            frame.to_parquet(tmp, index=False)
            back = pd.read_parquet(tmp, columns=list(SENS_COLS) + ['sens_members'])
            assert len(back) == len(frame) and all(np.array_equal(back[c].to_numpy(np.float64),
                                                                  frame[c].to_numpy(np.float64)) for c in SENS_COLS), \
                f'{path.name}: read-back differs'
        for path, _ in writes:
            os.replace(path.with_name(path.name + '.sens.tmp'), path)
            replaced.append(path)
        rep.update(st, status='applied')
    except Exception as exc:
        for path in replaced:                       # all-or-nothing: put the published files back
            try:
                shutil.copyfile(bak / path.name, path)
            except Exception:
                traceback.print_exc()
        rep.update(status='fallback_published', error=f'{type(exc).__name__}: {str(exc)[:400]}')
        print(f"WARNING SENS failed, published saner scores kept: {rep['error']}", flush=True)
    finally:
        try:
            for b in (sorted(work.glob('bundle_*')) if work.is_dir() else []):
                shutil.rmtree(b, ignore_errors=True)    # checkpoint copies: not left in the output
            for path, _ in jobs:
                path.with_name(path.name + '.sens.tmp').unlink(missing_ok=True)
        except Exception:
            traceback.print_exc()
    rep['seconds'] = round(time.time() - t0, 1)
    try:
        (ROOT / 'sens_report.json').write_text(json.dumps(rep, indent=2, default=str))
    except Exception:
        pass
    if rep['status'] == 'applied':
        print('SENS seeds | vs published on the ensembled lists | ' + ' | '.join(
            f"{s} rho {d['rho']} top1 {d['top1_agree']}" for s, d in rep['per_seed'].items()), flush=True)
    print(sens_line(rep), flush=True)
    return rep
