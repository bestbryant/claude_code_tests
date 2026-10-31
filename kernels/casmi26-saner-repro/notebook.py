"""casmi26-saner-repro (F7 in docs/fork_eligibility.md, asset A37): retrain the saner neural scorer from the code published
in sanpier/casmi-spectrum-model-v1 (Apache-2.0) on a Kaggle T4, then compare the retrained checkpoints with the published
ones on the published validation candidate sets. Nothing here feeds a submission.

Steps; each prints 'F7 CHECK <name>: OK | FAIL | INFO {...}', and the whole record is /kaggle/working/repro_report.json.
 1. Published bundle: sha256 of every file against the bundle's own sha256.json. The sources (config.json, features.py,
    model.py, prepare.py, train.py, inference.py, runtime_test.py) are copied to /kaggle/working/saner_repro/src.
 2. Environment: versions against the published environment.json. RDKit is pinned to 2026.3.3 (offline wheel, as in the
    pubm lineage), because the tautomer-canonical InChIKey14, the split and the candidate sets all depend on it.
 3. The one documented patch: 'normalized_smiles' is added to the SMILES alias list of prepare.py. The published
    prepare.py only accepts 'smiles' and 'canonical_smiles', and train.parquet has neither. The dataset card names this
    fix. The unified diff is printed and saved as saner_repro/prepare_normalized_smiles.diff.
 4. Published runtime_test.self_test(cfg), then the launcher saner_repro/launch.py (printed below): prepare.prepare(cfg)
    -> train.run(cfg) with the published config.json: seed 20261023, 16 epochs x 500 steps, batch 96, patience 4, a
    holdout stage, then a full refit for best_epoch epochs with seed + 1, AMP on CUDA. The launcher adds one speed-only memo
    (features.structure for the unique SMILES, evaluated in a process pool). The pool returns the same tuples, which a
    spot check recomputes in-process.
 5. Seed reference (time-guarded): the published holdout stage once more with seed + 1000 on the same prepared data. It
    measures the run-to-run spread that a not-bit-exact retrain has to be judged against.
 6. Evaluation on the published validation_candidates.npz: 1,000 held-out molecules, 256 candidates each, gold first.
    The candidates are mapped by InChIKey14 through the published molecules.parquet. The metric is the published
    train.validate (diagnostic MRR@25, pessimistic ties). Checkpoints compared:
      pub_best  = published best_validation_model.pt (holdout model; training_result.json: 0.48441 at epoch 16)
      rep_best  = our holdout model (clean on these molecules) -- the primary reproduction number
      ref_best  = seed-reference holdout model
      pub_final = published final_model.pt (the scorer used in pubm), rep_final = our refit
    The two final models were trained on the validation molecules, so their MRR on this set is contaminated. It is
    reported only to compare the two, together with the score correlation of the pair.
 7. sha256 (file and state-dict content) of every new checkpoint; the large training cache is deleted at the end.
"""
import difflib, hashlib, importlib.metadata, json, os, random, shutil, subprocess, sys, time, traceback
from pathlib import Path

T0 = time.time()
INPUT = Path(os.environ.get('F7_INPUT', '/kaggle/input'))
WORKING = Path(os.environ.get('F7_WORKING', '/kaggle/working'))
RW = WORKING / 'saner_repro'
SRC = RW / 'src'
SEEDREF = RW / 'seedref'
REPORT_PATH = WORKING / 'repro_report.json'
SEED_REF = True               # step 5 on/off
SEED_REF_OFFSET = 1000
SEED_REF_START_BY_MIN = 80    # skip step 5 if the main training has not finished by then (total budget < 2 h)
SPOT_CHECK = 2000             # memo entries recomputed with the original features.structure after prepare()
HOLDOUT_TOL = 0.02            # |rep_best - pub_best| diagnostic MRR@25 accepted as reproduced
PUB_MRR_TOL = 0.002           # pub_best recomputed here vs training_result.json (evaluation-harness parity)
PUBLISHED_HOLDOUT_MRR = 0.48440741342338733
SOURCES = ['config.json', 'features.py', 'model.py', 'prepare.py', 'train.py', 'inference.py', 'runtime_test.py']
ALIAS_OLD = "'smiles':['smiles','canonical_smiles']"
ALIAS_NEW = "'smiles':['smiles','canonical_smiles','normalized_smiles']"
REPORT = dict(kernel='casmi26-saner-repro', task='F7 saner retrain demonstration (docs/fork_eligibility.md, A37)',
              bundle='sanpier/casmi-spectrum-model-v1', checks={}, errors=[])
RW.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def minutes():
    return round((time.time() - T0) / 60, 2)


def _js(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def save_report():
    REPORT['minutes'] = minutes()
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=_js))


def check(name, ok, **detail):
    REPORT['checks'][name] = dict(ok=ok, **detail)
    tag = 'INFO' if ok is None else ('OK' if ok else 'FAIL')
    log(f'F7 CHECK {name}: {tag} ' + json.dumps(detail, default=_js)[:1500])
    save_report()


def guarded(name, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        REPORT['errors'].append(dict(step=name, error=repr(e), trace=traceback.format_exc()[-3000:]))
        log(f'F7 ERROR in {name}: {e!r}\n{traceback.format_exc()}')
        save_report()
        return None


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def run_logged(args, logfile, **kw):
    """Run a child process, stream its output to this log and to `logfile`; return (exit code, seconds)."""
    t = time.time()
    with open(logfile, 'w') as fh:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
        for line in p.stdout:
            sys.stdout.write(line); fh.write(line)
        rc = p.wait()
    sys.stdout.flush()
    return rc, round(time.time() - t, 1)


# ── 1. Published bundle ──────────────────────────────────────────────────────────────────────────────────────
def find_bundle():
    hits = [p.parent for p in INPUT.rglob('sha256.json')
            if all((p.parent / f).is_file() for f in ('config.json', 'model.py', 'features.py'))]
    hits = sorted(set(hits), key=lambda p: (len(p.parts), str(p)))
    if not hits:
        raise SystemExit('F7 FATAL: sanpier/casmi-spectrum-model-v1 not mounted (no folder with sha256.json + model.py)')
    return hits[0]


def step_bundle():
    log('== 1. published bundle')
    B = find_bundle()
    pins = json.loads((B / 'sha256.json').read_text())
    rows = {}
    for name, want in sorted(pins.items()):
        p = B / name
        got = sha256(p) if p.is_file() else None
        rows[name] = 'absent' if got is None else ('match' if got == want else f'MISMATCH {got}')
    unlisted = sorted(p.name for p in B.iterdir() if p.name not in pins and p.name != 'sha256.json')
    n_match = sum(v == 'match' for v in rows.values())
    check('published_sha256', n_match == len(pins), bundle=str(B), matched=n_match, listed=len(pins), files=rows,
          unlisted_files=unlisted)
    missing = [f for f in SOURCES if not (B / f).is_file()]
    check('published_sources', not missing, present=[f for f in SOURCES if f not in missing], missing=missing,
          note='train.py / prepare.py are in the dataset itself (casmi_spectrum_model_v1/), pinned by its sha256.json')
    if missing:
        raise SystemExit(f'F7 FATAL: published training sources missing from the bundle: {missing}')
    SRC.mkdir(parents=True, exist_ok=True)
    for f in SOURCES:
        shutil.copy2(B / f, SRC / f)
    REPORT['bundle_dir'] = str(B)
    return B


# ── 2. Environment ───────────────────────────────────────────────────────────────────────────────────────────
PROBE = ('import json, os, sys, torch, numpy, pandas, pyarrow\nfrom rdkit import rdBase\n'
         'print(json.dumps(dict(python=sys.version, torch=torch.__version__, numpy=numpy.__version__,'
         ' pandas=pandas.__version__, pyarrow=pyarrow.__version__, rdkit=rdBase.rdkitVersion,'
         ' cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'
         ' cpus=len(os.sched_getaffinity(0)))))')


def step_env(B):
    log('== 2. environment')
    subprocess.run('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | head -2;'
                   f' df -h {WORKING} | tail -1', shell=True)
    try:
        rv = importlib.metadata.version('rdkit')
    except importlib.metadata.PackageNotFoundError:
        rv = ''
    wheel = None
    if rv.replace('.03.', '.3.') != '2026.3.3':
        tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
        whl = sorted(INPUT.rglob(f'rdkit-2026.3.3-{tag}-*.whl'))
        if not whl:
            raise SystemExit(f'F7 FATAL: image RDKit is {rv or "absent"} and no rdkit-2026.3.3 {tag} wheel is attached')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', str(whl[0])])
        wheel = whl[0].name
    out = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True)
    env = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else dict(error=out.stderr[-2000:])
    pub = json.loads((B / 'environment.json').read_text()) if (B / 'environment.json').is_file() else {}
    pv = pub.get('versions', {})
    norm = lambda v: str(v).replace('.03.', '.3.')
    same = {k: [pv.get(k), env.get(k)] for k in ('torch', 'numpy', 'pandas', 'pyarrow', 'rdkit')}
    REPORT['environment'] = dict(ours=env, published=pub, rdkit_wheel_installed=wheel)
    check('environment', None, gpu=env.get('gpu'), cpus=env.get('cpus'), python=[pub.get('python', '')[:7], env.get('python', '')[:7]],
          versions_published_vs_ours=same, rdkit_pinned=norm(env.get('rdkit')) == '2026.3.3',
          differs=[k for k, (a, b) in same.items() if norm(a) != norm(b)])
    if norm(env.get('rdkit')) != '2026.3.3':
        raise SystemExit('F7 FATAL: RDKit is not 2026.3.3 after the wheel install')
    return env


# ── 3. Patch and effective config ────────────────────────────────────────────────────────────────────────────
def step_patch():
    log('== 3. the documented patch (prepare.py SMILES alias) and the effective config')
    orig = (SRC / 'prepare.py').read_text()
    n = orig.count(ALIAS_OLD)
    if n != 1:
        check('patch', False, occurrences=n, expected=1)
        raise SystemExit('F7 FATAL: the published prepare.py does not contain the expected alias line exactly once')
    patched = orig.replace(ALIAS_OLD, ALIAS_NEW)
    (RW / 'prepare.py.published').write_text(orig)
    (SRC / 'prepare.py').write_text(patched)
    diff = ''.join(difflib.unified_diff(orig.splitlines(True), patched.splitlines(True),
                                        'a/prepare.py (published)', 'b/prepare.py (patched)'))
    (RW / 'prepare_normalized_smiles.diff').write_text(diff)
    REPORT['patch_diff'] = diff
    log(diff)
    body = [l for l in diff.splitlines() if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    changed = {f: sha256(SRC / f) for f in SOURCES}
    check('patch', len(body) == 2, changed_lines=body, prepare_sha256_published=hashlib.sha256(orig.encode()).hexdigest(),
          prepare_sha256_patched=changed['prepare.py'],
          other_sources_unchanged=all(changed[f] == sha256(Path(REPORT['bundle_dir']) / f) for f in SOURCES if f != 'prepare.py'))
    cfg = json.loads((SRC / 'config.json').read_text())
    eff = dict(cfg)
    if not Path(cfg['train_path']).is_file():
        hits = sorted((p for p in INPUT.rglob('train.parquet') if 'casmi26' in str(p).lower()), key=lambda p: len(str(p)))
        if not hits:
            raise SystemExit('F7 FATAL: competition train.parquet not mounted')
        eff['train_path'] = str(hits[0])
    overrides = {k: [cfg[k], eff[k]] for k in cfg if cfg[k] != eff[k]}
    (RW / 'effective_config.json').write_text(json.dumps(eff, indent=2))
    check('config', True, overrides=overrides, seed=eff['seed'], epochs=eff['epochs'], steps_per_epoch=eff['steps_per_epoch'],
          batch_size=eff['batch_size'], patience=eff['patience'], refit_full=eff['refit_full'], output=eff['output'],
          work=eff['work'], train_path=eff['train_path'])
    return eff


# ── 4. Launcher ──────────────────────────────────────────────────────────────────────────────────────────────
LAUNCHER = r'''"""Launcher for the published saner training code (sanpier/casmi-spectrum-model-v1, Apache-2.0), written by the
casmi26-saner-repro kernel. Entry points as published: prepare.prepare(cfg), then train.run(cfg); cfg = config.json
(effective_config.json differs only in the path overrides printed in the log, normally none).
Speed-only addition: features.structure (RDKit tautomer canonicalisation, ~15 ms per SMILES, ~70 CPU-minutes for the
~277,566 unique SMILES of train.parquet) runs for every unique SMILES in a fork pool BEFORE prepare(). prepare() then
gets the identical tuple from a memo through its module-level name `structure`. A SMILES that fails in the pool, and any
miss, falls through to the original function in-process, so prepare() raises the original exceptions. Set
SANER_STRUCT_POOL=0 for the plain two-line launcher. After prepare(), a random sample of memo entries is recomputed with
the original function and compared field by field (written to the check file)."""
import json, os, random, sys, time
from pathlib import Path
SRC, CFG, CHECK = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, SRC)
cfg = json.loads(Path(CFG).read_text())
import numpy as np
import prepare as P
import features as Fe

STATS = dict(pool=os.environ.get('SANER_STRUCT_POOL', '1') == '1', unique_smiles=0, pool_failed=0, memo_hits=0,
             fallback_calls=0)
MEMO = {}
_original = P.structure


def _quiet():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def _one(s):
    try:
        return s, Fe.structure(s, cfg['fp_bits'])
    except Exception:
        return s, None


def structure_memo(smiles, nbits=2048):
    r = MEMO.get(smiles) if nbits == cfg['fp_bits'] else None
    if r is None:
        STATS['fallback_calls'] += 1
        return _original(smiles, nbits)
    STATS['memo_hits'] += 1
    return r


_quiet()
if STATS['pool']:
    import multiprocessing as mp
    import pyarrow.parquet as pq
    t0 = time.time(); uniq = set()
    for path in [cfg['train_path']] + list(cfg['external_paths']):
        if not str(path).endswith('.parquet'):
            continue
        names = pq.ParquetFile(path).schema_arrow.names
        col = next((c for c in P.ALIASES['smiles'] if c in names), None)
        if col:
            uniq.update(str(x) for x in pq.read_table(path, columns=[col], use_threads=False).column(0).to_pylist())
    uniq = sorted(uniq); STATS['unique_smiles'] = len(uniq)
    procs = max(1, min(8, len(os.sched_getaffinity(0))))
    print('STRUCT POOL', len(uniq), 'unique SMILES,', procs, 'processes', flush=True)
    with mp.get_context('fork').Pool(procs, initializer=_quiet) as pool:
        for n, (s, r) in enumerate(pool.imap_unordered(_one, uniq, chunksize=64), 1):
            if r is None:
                STATS['pool_failed'] += 1
            else:
                MEMO[s] = r
            if n % 25000 == 0:
                print(f'STRUCT POOL {n}/{len(uniq)} {time.time() - t0:.0f} s', flush=True)
    STATS['pool_seconds'] = round(time.time() - t0, 1)
    print('STRUCT POOL done', json.dumps(STATS), flush=True)
    P.structure = structure_memo

t0 = time.time()
P.prepare(cfg)
STATS['prepare_seconds'] = round(time.time() - t0, 1)
if MEMO:
    sample = random.Random(0).sample(sorted(MEMO), min(int(os.environ.get('SANER_SPOT_CHECK', '2000')), len(MEMO)))
    bad = []
    for s in sample:
        a, b = MEMO[s], _original(s, cfg['fp_bits'])
        if not (a[:4] == b[:4] and a[4].dtype == b[4].dtype and np.array_equal(a[4], b[4])):
            bad.append(s)
    STATS.update(spot_checked=len(sample), spot_mismatch=len(bad), spot_mismatch_examples=bad[:5])
print('STRUCT MEMO', json.dumps(STATS), flush=True)
CHECK.write_text(json.dumps(STATS, indent=2))

import train
t0 = time.time()
result = train.run(cfg)
STATS['train_seconds'] = round(time.time() - t0, 1)
CHECK.write_text(json.dumps(STATS, indent=2))
print('TRAINING_RESULT', json.dumps(result), flush=True)
'''

SELFTEST = ('import json, sys\nsys.path.insert(0, sys.argv[1])\nimport runtime_test\n'
            'runtime_test.self_test(json.load(open(sys.argv[2])))')

SEEDREF_CODE = r'''"""Seed reference for casmi26-saner-repro: the published holdout stage (train.train_stage, refit=False) once more on
the same prepared data and split, with seed + offset. It measures the run-to-run spread of the published recipe."""
import json, os, sys, time
from pathlib import Path
SRC, CFG, OUT, OFFSET = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, SRC)
import torch
import train
cfg = json.loads(Path(CFG).read_text())
cfg['seed'] = cfg['seed'] + OFFSET
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(4, os.cpu_count() or 1))
data = train.Data(cfg)
result = train.train_stage(data, cfg, OUT, device, deadline=time.monotonic() + cfg['holdout_hours'] * 3600)
result['seed'] = cfg['seed']
(OUT / 'training_result.json').write_text(json.dumps(result, indent=2))
print('SEEDREF_RESULT', json.dumps(result), flush=True)
'''


def step_train(eff):
    log('== 4a. published runtime_test.self_test')
    rc, sec = run_logged([sys.executable, '-u', '-c', SELFTEST, str(SRC), str(RW / 'effective_config.json')],
                         RW / 'runtime_test.log', cwd=RW)
    check('runtime_test', rc == 0, exit=rc, seconds=sec)
    log('== 4b. launcher (saner_repro/launch.py):')
    (RW / 'launch.py').write_text(LAUNCHER)
    log(LAUNCHER)
    rc, sec = run_logged([sys.executable, '-u', str(RW / 'launch.py'), str(SRC), str(RW / 'effective_config.json'),
                          str(RW / 'struct_memo_check.json')], RW / 'train.log', cwd=RW)
    memo = json.loads((RW / 'struct_memo_check.json').read_text()) if (RW / 'struct_memo_check.json').is_file() else {}
    ok_memo = (not memo.get('pool')) or (memo.get('spot_mismatch') == 0 and memo.get('spot_checked', 0) > 0)
    check('struct_memo', ok_memo if memo else False, **memo)
    out = Path(eff['output'])
    res = json.loads((out / 'training_result.json').read_text()) if (out / 'training_result.json').is_file() else None
    check('training_run', rc == 0 and res is not None, exit=rc, minutes=round(sec / 60, 1), training_result=res)
    return rc == 0 and res is not None


def step_seedref(eff):
    if not SEED_REF:
        check('seed_reference', None, skipped='SEED_REF off'); return False
    if minutes() > SEED_REF_START_BY_MIN:
        check('seed_reference', None, skipped=f'main run ended at {minutes()} min > {SEED_REF_START_BY_MIN}'); return False
    log(f'== 5. seed reference: holdout stage with seed {eff["seed"]} + {SEED_REF_OFFSET}')
    (RW / 'seedref.py').write_text(SEEDREF_CODE)
    rc, sec = run_logged([sys.executable, '-u', str(RW / 'seedref.py'), str(SRC), str(RW / 'effective_config.json'),
                          str(SEEDREF), str(SEED_REF_OFFSET)], RW / 'seedref.log', cwd=RW)
    ok = rc == 0 and (SEEDREF / 'best_validation_model.pt').is_file()
    res = json.loads((SEEDREF / 'training_result.json').read_text()) if (SEEDREF / 'training_result.json').is_file() else None
    check('seed_reference', ok, exit=rc, minutes=round(sec / 60, 1), result=res)
    return ok


# ── 6. Evaluation ────────────────────────────────────────────────────────────────────────────────────────────
def mrr_vec(r):
    import numpy as np
    r = np.asarray(r, np.float64)
    return np.where(r <= 25, 1.0 / r, 0.0)


def summarize(r):
    import numpy as np
    r = np.asarray(r)
    return dict(mrr25=float(mrr_vec(r).mean()), top1=float((r == 1).mean()), recall25=float((r <= 25).mean()),
                n=int(len(r)), median_rank=float(np.median(r)))


def ranks_from_scores(scores, qs):
    import numpy as np
    return np.array([1 + int(np.sum(scores[q][1:] >= scores[q][0])) for q in qs])


def pair_stats(sa, sb, qs, seed=0):
    """Agreement of two checkpoints on the same candidate sets (scores: {query: per-candidate array, gold first})."""
    import numpy as np
    from scipy.stats import spearmanr
    sp, pe, t1, ov, za, zb = [], [], [], [], [], []
    for q in qs:
        a, b = np.asarray(sa[q], np.float64), np.asarray(sb[q], np.float64)
        sp.append(spearmanr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan)
        pe.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
        t1.append(int(np.argmax(a) == np.argmax(b)))
        ka, kb = np.argsort(-a, kind='stable')[:25], np.argsort(-b, kind='stable')[:25]
        ov.append(len(set(ka.tolist()) & set(kb.tolist())) / 25)
        za.append((a - a.mean()) / (a.std() or 1)); zb.append((b - b.mean()) / (b.std() or 1))
    ra, rb = ranks_from_scores(sa, qs), ranks_from_scores(sb, qs)
    d = mrr_vec(rb) - mrr_vec(ra)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    za, zb = np.concatenate(za), np.concatenate(zb)
    return dict(n_queries=len(qs), spearman_per_query_mean=float(np.nanmean(sp)), spearman_per_query_median=float(np.nanmedian(sp)),
                pearson_per_query_mean=float(np.nanmean(pe)), pearson_pooled_within_query_z=float(np.corrcoef(za, zb)[0, 1]),
                top1_candidate_agreement=float(np.mean(t1)), top25_overlap_mean=float(np.mean(ov)),
                gold_rank_identical=float(np.mean(ra == rb)), gold_rank_spearman=float(spearmanr(ra, rb)[0]),
                mrr25_a=float(mrr_vec(ra).mean()), mrr25_b=float(mrr_vec(rb).mean()), mrr25_b_minus_a=float(d.mean()),
                mrr25_diff_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                top1_both=int(((ra == 1) & (rb == 1)).sum()), top1_only_a=int(((ra == 1) & (rb != 1)).sum()),
                top1_only_b=int(((ra != 1) & (rb == 1)).sum()))


def score_sets(model, data, cands, device, cfg):
    """Per-candidate scores exactly as the published train.validate computes them (mean spectrum embedding and mean
    fingerprint logits over the molecule's competition spectra), kept per candidate instead of reduced to a rank."""
    import numpy as np, torch
    model.eval(); out = {}
    with torch.inference_mode():
        for i, ids in cands.items():
            sr = data.by_mol[i]; sr = sr[data.spec.source.values[sr] == 0]
            if not len(sr):
                continue
            zsum = 0; psum = 0; n = 0
            for start in range(0, len(sr), cfg['batch_size']):
                b = sr[start:start + cfg['batch_size']]
                z, p = model.encode_spectra(*data.batch(b, device)); zsum = zsum + z.sum(0); psum = psum + p.sum(0); n += len(b)
            fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
            s = (model.encode_molecules(fp) @ (zsum / n)) / cfg['temperature'] + cfg['fp_score_weight'] * (fp @ (psum / n)) / cfg['fp_bits']
            out[i] = s.float().cpu().numpy()
    return out


def state_sha(sd):
    import torch
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def compare_prepared(B, eff, data):
    import numpy as np, pandas as pd
    res = {}
    pa = json.loads((B / 'data_audit.json').read_text()); oa = json.loads((Path(eff['work']) / 'data_audit.json').read_text())
    fields = ['rejected', 'spectra', 'molecules', 'competition_identities', 'validation_identities',
              'excluded_external_validation_spectra', 'unique_external_only_identities']
    src_f = ['sha256', 'input_rows', 'accepted', 'columns']
    diffs = {f: [pa.get(f), oa.get(f)] for f in fields if pa.get(f) != oa.get(f)}
    diffs.update({f'sources[0].{f}': [pa['sources'][0].get(f), oa['sources'][0].get(f)] for f in src_f
                  if pa['sources'][0].get(f) != oa['sources'][0].get(f)})
    res['data_audit'] = dict(identical=not diffs, differences=diffs, train_sha256=oa['sources'][0]['sha256'],
                             accepted=oa['sources'][0]['accepted'], input_rows=oa['sources'][0]['input_rows'],
                             rejected=oa['rejected'], molecules=oa['molecules'], validation_identities=oa['validation_identities'])
    check('prepare_data_audit', not diffs, **res['data_audit'])
    pm, om = pd.read_parquet(B / 'molecules.parquet'), data.mol
    m = dict(published=len(pm), ours=len(om), only_published=len(set(pm.key) - set(om.key)), only_ours=len(set(om.key) - set(pm.key)))
    if len(pm) == len(om):
        for c in ('key', 'split', 'smiles', 'formula'):
            m[f'{c}_identical_rows'] = int(np.sum(pm[c].to_numpy(dtype=object) == om[c].to_numpy(dtype=object)))
        m['mass_max_abs_diff'] = float(np.abs(pm.mass.to_numpy(np.float64) - om.mass.to_numpy(np.float64)).max())
    m['identical'] = len(pm) == len(om) and all(m.get(f'{c}_identical_rows') == len(pm) for c in ('key', 'split', 'smiles', 'formula')) \
        and m.get('mass_max_abs_diff', 1) == 0
    check('prepare_molecules', m['identical'], **m)
    man = pd.read_parquet(B / 'spectrum_split_manifest.parquet', columns=['key', 'source', 'split'])
    s = dict(published=len(man), ours=len(data.spec))
    if len(man) == len(data.spec):
        for c in ('key', 'source', 'split'):
            s[f'{c}_identical_rows'] = int(np.sum(man[c].to_numpy(dtype=object) == data.spec[c].to_numpy(dtype=object)))
    s['identical'] = len(man) == len(data.spec) and all(s.get(f'{c}_identical_rows') == len(man) for c in ('key', 'source', 'split'))
    check('prepare_spectrum_manifest', s['identical'], **s)
    res.update(molecules=m, manifest=s)
    return res


def candidate_sets(B, eff, data):
    import numpy as np
    pk = np.asarray(__import__('pandas').read_parquet(B / 'molecules.parquet', columns=['key']).key.values, dtype=object)
    ok = np.asarray(data.mol.key.values, dtype=object)
    oi = {k: i for i, k in enumerate(ok)}
    pub = np.load(B / 'validation_candidates.npz', allow_pickle=False)
    pub_sets = {pk[int(q)]: pk[pub[q]] for q in pub.files}
    our_path = Path(eff['output']) / 'validation_candidates.npz'
    ours = np.load(our_path, allow_pickle=False) if our_path.is_file() else None
    our_sets = {ok[int(q)]: ok[ours[q]] for q in ours.files} if ours is not None else {}
    same = sum(1 for k, v in pub_sets.items() if k in our_sets and np.array_equal(our_sets[k], v))
    cands, drop_q, drop_c = {}, 0, 0
    for qk, keys in pub_sets.items():
        if qk not in oi or keys[0] != qk:
            drop_q += 1; continue
        ids = [oi[k] for k in keys if k in oi]
        drop_c += len(keys) - len(ids)
        cands[oi[qk]] = np.asarray(ids, np.int64)
    source = 'published validation_candidates.npz (mapped by InChIKey14)'
    if len(cands) < 0.95 * len(pub_sets) and ours is not None:
        cands = {int(q): np.asarray(ours[q], np.int64) for q in ours.files}
        source = 'FALLBACK: our validation_candidates.npz (published code, fixed seed-hash split of train.parquet)'
    info = dict(source=source, published_queries=len(pub_sets), our_queries=len(our_sets), identical_lists=same,
                eval_queries=len(cands), dropped_queries=drop_q, dropped_candidates=drop_c,
                candidates_per_query=sorted({len(v) for v in cands.values()}))
    check('validation_candidates', same == len(pub_sets) and len(our_sets) == len(pub_sets), **info)
    return cands, info


def history_table(B, eff):
    out = Path(eff['output'])
    rd = lambda p: json.loads(p.read_text()) if p.is_file() else []
    ph, rh, sh = rd(B / 'training_history.json'), rd(out / 'training_history.json'), rd(SEEDREF / 'training_history.json')
    pf, rf = rd(B / 'refit_history.json'), rd(out / 'refit_history.json')
    rows = []
    for e in range(1, max(len(ph), len(rh), len(sh), len(pf), len(rf)) + 1):
        g = lambda h, k: (h[e - 1].get(k) if len(h) >= e else None)
        rows.append(dict(epoch=e, pub_loss=g(ph, 'loss'), rep_loss=g(rh, 'loss'), ref_loss=g(sh, 'loss'),
                         pub_mrr=g(ph, 'diagnostic_mrr25'), rep_mrr=g(rh, 'diagnostic_mrr25'), ref_mrr=g(sh, 'diagnostic_mrr25'),
                         pub_s=g(ph, 'seconds'), rep_s=g(rh, 'seconds'),
                         pub_refit_loss=g(pf, 'loss'), rep_refit_loss=g(rf, 'loss')))
    f = lambda v, d=4: '   -   ' if v is None else f'{v:.{d}f}'
    log('epoch | loss pub / rep / seedref | holdout MRR pub / rep / seedref | s/epoch pub / rep | refit loss pub / rep')
    for r in rows:
        log(f"{r['epoch']:5d} | {f(r['pub_loss'])} {f(r['rep_loss'])} {f(r['ref_loss'])} | {f(r['pub_mrr'])} {f(r['rep_mrr'])} "
            f"{f(r['ref_mrr'])} | {f(r['pub_s'], 1)} {f(r['rep_s'], 1)} | {f(r['pub_refit_loss'])} {f(r['rep_refit_loss'])}")
    pr, rr = rd(B / 'training_result.json'), rd(out / 'training_result.json')
    e1 = abs(ph[0]['loss'] - rh[0]['loss']) if ph and rh else None
    check('training_history', None, published_result=pr, reproduced_result=rr, epoch1_loss_abs_diff=e1,
          best_epoch_pub_rep=[pr.get('best_epoch') if pr else None, rr.get('best_epoch') if rr else None],
          refit_epochs_pub_rep=[len(pf), len(rf)])
    return rows


def step_eval(B, eff):
    log('== 6. evaluation on the published validation candidates')
    import numpy as np, pandas as pd, torch
    sys.path.insert(0, str(SRC))
    import train as T
    from model import SpectrumCandidateModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(eff['output'])
    data = T.Data(eff)
    REPORT['prepared'] = guarded('compare_prepared', compare_prepared, B, eff, data)
    cands, cinfo = candidate_sets(B, eff, data)
    REPORT['candidates'] = cinfo
    REPORT['history'] = guarded('history', history_table, B, eff)
    ck = dict(pub_best=B / 'best_validation_model.pt', pub_last=B / 'last_validation_model.pt', pub_final=B / 'final_model.pt',
              rep_best=out / 'best_validation_model.pt', rep_last=out / 'last_validation_model.pt', rep_final=out / 'final_model.pt',
              ref_best=SEEDREF / 'best_validation_model.pt')
    qs = sorted(cands)
    scores, ranks, meta, by_state = {}, {}, {}, {}
    for name, path in ck.items():
        if not path.is_file():
            meta[name] = dict(path=str(path), absent=True); continue
        obj = torch.load(path, map_location='cpu', weights_only=True)
        sd = obj['state_dict']
        st = state_sha(sd)
        meta[name] = dict(path=str(path), file_sha256=sha256(path), state_sha256=st, epoch=obj.get('epoch'), stage=obj.get('stage'),
                          contaminated=name in ('pub_final', 'rep_final'))
        if st in by_state:
            meta[name]['same_state_as'] = by_state[st]
            scores[name], ranks[name] = scores[by_state[st]], ranks[by_state[st]]
            meta[name].update(summarize(ranks[name])); continue
        by_state[st] = name
        model = SpectrumCandidateModel(eff).to(device)
        model.load_state_dict(sd); model.eval()
        t = time.time()
        scores[name] = score_sets(model, data, cands, device, eff)
        qn = [q for q in qs if q in scores[name]]
        ranks[name] = ranks_from_scores(scores[name], qn)
        official, frame = T.validate(model, data, cands, device, eff)
        meta[name].update(summarize(ranks[name]), validate_mrr25=float(official),
                          validate_parity_abs_diff=abs(float(official) - float(mrr_vec(ranks[name]).mean())),
                          eval_seconds=round(time.time() - t, 1))
        log(f'F7 EVAL {name}: ' + json.dumps({k: meta[name][k] for k in ('mrr25', 'top1', 'recall25', 'n', 'validate_mrr25', 'epoch', 'stage')}, default=_js))
        del model
    REPORT['checkpoints'] = meta
    save_report()
    qn = [q for q in qs if all(q in scores[n] for n in scores)]
    pairs = [('pub_best', 'rep_best', 'PRIMARY: published vs reproduced holdout model (clean on these molecules)'),
             ('rep_best', 'ref_best', 'seed spread: reproduced vs seed+1000 holdout model, same code/data/GPU'),
             ('pub_best', 'ref_best', 'published vs seed+1000 holdout model'),
             ('pub_final', 'rep_final', 'published vs reproduced final (refit) model; both trained on these molecules'),
             ('pub_best', 'pub_final', 'published holdout vs published final (contaminated reference)'),
             ('rep_best', 'rep_final', 'reproduced holdout vs reproduced final (contaminated reference)')]
    REPORT['pairs'] = {}
    for a, b, what in pairs:
        if a in scores and b in scores and meta[a].get('state_sha256') != meta[b].get('state_sha256'):
            st = pair_stats(scores[a], scores[b], qn)
            REPORT['pairs'][f'{a}__{b}'] = dict(what=what, **st)
            log(f'F7 PAIR {a} vs {b} ({what}): ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}, default=_js))
    save_report()
    # Evaluation-harness parity: the published best model recomputed here vs the published per-molecule predictions.
    if 'pub_best' in ranks:
        pp = pd.read_csv(B / 'validation_predictions.csv')
        key_of = {i: data.mol.key.iloc[i] for i in qs}
        ours = pd.DataFrame(dict(key=[key_of[q] for q in qs if q in scores['pub_best']], rank_here=ranks['pub_best']))
        mm = pp.merge(ours, on='key', how='inner')
        mr = meta['pub_best']['mrr25']
        check('harness_parity', abs(mr - PUBLISHED_HOLDOUT_MRR) <= PUB_MRR_TOL,
              published_training_result=PUBLISHED_HOLDOUT_MRR, published_predictions_csv_mrr25=float(pp.mrr25.mean()),
              recomputed_here=mr, abs_diff=abs(mr - PUBLISHED_HOLDOUT_MRR), molecules_matched=len(mm),
              rank_identical=int((mm['rank'] == mm.rank_here).sum()), validate_parity_abs_diff=meta['pub_best'].get('validate_parity_abs_diff'))
    if 'rep_best' in meta and 'pub_best' in meta and not meta['rep_best'].get('absent'):
        p = REPORT['pairs'].get('pub_best__rep_best', {})
        sref = REPORT['pairs'].get('rep_best__ref_best', {})
        d = meta['rep_best']['mrr25'] - meta['pub_best']['mrr25']
        check('holdout_reproduction', abs(d) <= HOLDOUT_TOL, published_mrr25=meta['pub_best']['mrr25'],
              reproduced_mrr25=meta['rep_best']['mrr25'], diff=d, diff_ci95=p.get('mrr25_diff_ci95'), tolerance=HOLDOUT_TOL,
              published_top1=meta['pub_best']['top1'], reproduced_top1=meta['rep_best']['top1'],
              spearman_pub_rep=p.get('spearman_per_query_mean'), seedref_mrr25=meta.get('ref_best', {}).get('mrr25'),
              spearman_rep_seedref=sref.get('spearman_per_query_mean'),
              seed_spread_mrr25=sref.get('mrr25_b_minus_a'))
    if 'rep_final' in meta and not meta['rep_final'].get('absent'):
        p = REPORT['pairs'].get('pub_final__rep_final', {})
        check('final_model', None, new_final_model_sha256=meta['rep_final']['file_sha256'],
              new_final_state_sha256=meta['rep_final']['state_sha256'], published_final_sha256=meta.get('pub_final', {}).get('file_sha256'),
              published_final_mrr25_contaminated=meta.get('pub_final', {}).get('mrr25'),
              reproduced_final_mrr25_contaminated=meta['rep_final']['mrr25'], spearman_pub_rep=p.get('spearman_per_query_mean'),
              pearson_pooled=p.get('pearson_pooled_within_query_z'), top1_candidate_agreement=p.get('top1_candidate_agreement'),
              top25_overlap=p.get('top25_overlap_mean'))
    return meta


def step_finish(eff):
    out = Path(eff['output'])
    shas = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()} if out.is_dir() else {}
    shas.update({f'seedref/{p.name}': sha256(p) for p in sorted(SEEDREF.glob('*')) if p.is_file()})
    (RW / 'repro_sha256.json').write_text(json.dumps(shas, indent=2))
    REPORT['output_sha256'] = shas
    work = Path(eff['work'])
    freed = 0
    for f in ('peaks.bin', 'metadata.bin', 'fingerprints.npy', 'spectra.parquet'):
        p = work / f
        if p.is_file():
            freed += p.stat().st_size; p.unlink()
    REPORT['cache_deleted_bytes'] = freed
    c = REPORT['checks']
    must = ['published_sha256', 'patch', 'struct_memo', 'training_run', 'prepare_data_audit', 'prepare_molecules',
            'prepare_spectrum_manifest', 'validation_candidates', 'harness_parity', 'holdout_reproduction']
    REPORT['verdict'] = dict(all_required_ok=all(c.get(k, {}).get('ok') for k in must),
                             failed=[k for k in must if not c.get(k, {}).get('ok')], minutes=minutes())
    save_report()
    log('F7 SUMMARY ' + json.dumps(REPORT['verdict']))
    hr = c.get('holdout_reproduction', {})
    fm = c.get('final_model', {})
    log(f"F7 SUMMARY holdout MRR@25 published {hr.get('published_mrr25')} | reproduced {hr.get('reproduced_mrr25')} "
        f"(diff {hr.get('diff')}, CI {hr.get('diff_ci95')}) | seed ref {hr.get('seedref_mrr25')} | per-query Spearman pub-rep "
        f"{hr.get('spearman_pub_rep')} vs rep-seedref {hr.get('spearman_rep_seedref')}")
    log(f"F7 SUMMARY final model: new sha256 {fm.get('new_final_model_sha256')} | Spearman pub-rep {fm.get('spearman_pub_rep')} | "
        f"top-1 agreement {fm.get('top1_candidate_agreement')} | MRR (contaminated) pub {fm.get('published_final_mrr25_contaminated')} "
        f"rep {fm.get('reproduced_final_mrr25_contaminated')}")
    log(f'F7 DONE {minutes()} min; report {REPORT_PATH}')


def main():
    B = step_bundle()
    step_env(B)
    eff = step_patch()
    ok = step_train(eff)
    if ok:
        guarded('seed_reference', step_seedref, eff)
        guarded('evaluation', step_eval, B, eff)
    guarded('finish', step_finish, eff)


main()
