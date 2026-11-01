"""casmi26-saner-seeds (task #46): a seed ensemble of the published saner recipe (sanpier/casmi-spectrum-model-v1,
Apache-2.0), measured on the published holdout, then the full refits that a deployment in casmi26-pubm would load.
Nothing here feeds a submission.

WHY. The saner neural scorer (final_model.pt) drives the neural fusion of the 0.479 public line (our fork casmi26-pubm).
F7 (kernels/casmi26-saner-repro) retrained it bit-exactly from the published code and measured its seed spread: the
seed + 1000 holdout model has the same MRR@25 (0.4860 vs 0.4844) but a per-query Spearman of only 0.87 and a top-1
agreement of 0.43 with the published model. Averaging several seeds of the SAME recipe may therefore improve the scorer
without the domain shift that sank a longer-trained replacement (-0.019 LB) and our PF blend (-0.014).

STEPS. Each prints 'SEEDS CHECK <name>: OK | FAIL | INFO {...}'; the full record is /kaggle/working/seeds_report.json.
 1-3. As casmi26-saner-repro: every bundle file against the bundle's sha256.json; RDKit pinned to 2026.3.3 (offline
      wheel); the one documented patch ('normalized_smiles' added to the SMILES aliases of prepare.py); published
      config.json without overrides.
 4.   Published runtime_test.self_test; the speed-only structure memo (fork pool; 2,000 entries recomputed in-process);
      published prepare.prepare(cfg) ONCE. No train.run: the seed-20261023 holdout and refit models are the published
      files, which F7 reproduced bit-exactly.
 5.   Evaluator (this process): prepared data vs the published data_audit / molecules / spectrum manifest; the published
      validation_candidates.npz (1,000 validation molecules x 256 candidates, gold first) mapped by InChIKey14; the
      published holdout model scored (harness parity with its training_result.json).
 6.   Holdout stage for 20262023 (= F7's seed reference: determinism check), 20263023, 20264023, 20265023. Each runs in
      its own process: train.train_stage(data, cfg with seed s, ...) exactly as train.run's first call (published
      config otherwise: 16 x 500 steps, batch 96, patience 4, AMP). After each seed: its holdout model is scored with
      train.validate's metric (MRR@25 with pessimistic ties, top-1, recall@25; train.validate itself is rerun for
      parity) and the ensembles of the first k seeds are recomputed: (a) mean raw score, (b) mean within-query z-score
      (z over the 256 candidates of the query). Paired bootstrap (5,000 query resamples, shared across comparisons):
      ensemble minus the single published holdout model, and ensemble minus the mean of its members (the gain over a
      typical single seed). Lines 'SEEDS EVAL single seed=..' and 'SEEDS EVAL k=.. mrr .. mode=raw|z ..'.
 7.   'SEEDS SUMMARY decision ...' once the holdouts are done: largest k (>= 3), the better mode (raw on a tie, it keeps
      saner's score scale for pubm); GO if the CI95 of (ensemble - published) is above 0, LEAN_GO if the mean is above
      0 and the CI95 of (ensemble - mean member) is above 0, else NO_GAIN. Mode and k are picked on this same set.
 8.   Full refit for each extra seed whose holdout completed: train_stage(refit=True, epochs=that seed's holdout
      best_epoch) = train.run's second call (it seeds s + 1). Saved as /kaggle/working/final_model_seed<s>.pt with
      final_model_seed<s>.pt.sha256 and final_models_sha256.json (file + state-dict sha256). The published
      final_model.pt is copied as final_model_seed20261023.pt (sha256 checked against the bundle pin).
 9.   Refit-launcher parity: the published seed refit once more in the same standalone launcher; expected bit-exact
      with the published final_model.pt (as F7's in-run refit was). Then contaminated diagnostics of the final models
      (they were trained on the validation molecules: Spearman / top-1 agreement to the published final model, a
      sanity check, not a quality number). The 2.5 GB training cache is deleted at the end.
BUDGET. 8.5 h in total. A step starts only if its hard timeout still fits (prepare 120 min, holdout 35 min, refit
30 min, 8 min reserve for the summary); training stops itself 4 min before the hard kill (train_stage's own deadline),
and a stage stopped that way is reported but not used. Holdouts 4 and 5 also keep room for the refits of the seeds
already trained. Order: holdouts + evaluation first (the decision), then refits.
EXPECTED on a T4 (F7 timings): ~1.9 h. Structure memo 33 min + prepare 10 min; each holdout ~8.5 min (decision with
3 seeds at ~60 min, all 5 at ~80 min); each refit ~6 min (4 seeds + the parity refit ~30 min); evaluation ~5 min.
"""
import difflib, gc, hashlib, importlib.metadata, json, os, shutil, signal, subprocess, sys, threading, time, traceback
from pathlib import Path

T0 = time.time()


def _env_num(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else default


INPUT = Path(os.environ.get('SEEDS_INPUT', '/kaggle/input'))
WORKING = Path(os.environ.get('SEEDS_WORKING', '/kaggle/working'))
RW = WORKING / 'saner_seeds'
SRC = RW / 'src'
SCORES = RW / 'scores'
REPORT_PATH = WORKING / 'seeds_report.json'
FINALS_PATH = WORKING / 'final_models_sha256.json'

SEEDS = [20261023, 20262023, 20263023, 20264023, 20265023]   # first = the published seed (its models are the bundle files)
PUBLISHED_SEED = SEEDS[0]
BUDGET_MIN = _env_num('SEEDS_BUDGET_MIN', 510.)              # 8.5 h total
FINISH_RESERVE_MIN = _env_num('SEEDS_FINISH_RESERVE_MIN', 8.)
SELFTEST_TIMEOUT_MIN = 10.
PREP_TIMEOUT_MIN = _env_num('SEEDS_PREP_TIMEOUT_MIN', 120.)  # structure memo + prepare (F7: 43 min)
HOLDOUT_TIMEOUT_MIN = _env_num('SEEDS_HOLDOUT_TIMEOUT_MIN', 35.)  # per seed (F7 seed reference: 8.3 min)
REFIT_TIMEOUT_MIN = _env_num('SEEDS_REFIT_TIMEOUT_MIN', 30.)      # per seed (F7: ~6 min)
SOFT_MARGIN_MIN = 4.          # train_stage's own deadline ends training this long before the hard kill
DECISION_MIN_K = 3            # holdout models (published included) needed for the decision
REFIT_PARITY = True           # step 9a on/off
COPY_PUBLISHED_FINAL = True   # final_model_seed20261023.pt = the published final_model.pt (Apache-2.0)
BOOT = 5000
BOOT_SEED = 0
SPOT_CHECK = 2000             # memo entries recomputed with the original features.structure after prepare()
PUB_MRR_TOL = 0.002           # published holdout model recomputed here vs its training_result.json
FINAL_SANITY = dict(min_spearman_to_published_final=0.7, min_contaminated_mrr25=0.45)
SOURCES = ['config.json', 'features.py', 'model.py', 'prepare.py', 'train.py', 'inference.py', 'runtime_test.py']
ALIAS_OLD = "'smiles':['smiles','canonical_smiles']"
ALIAS_NEW = "'smiles':['smiles','canonical_smiles','normalized_smiles']"
# F7 (casmi26-saner-repro v1, T4): the seed-reference holdout model, i.e. the published holdout stage with seed 20262023.
REFS = dict(f7_seedref=dict(seed=20262023, mrr25=0.48597427882623506,
                            file_sha256='a781bd383ef01c45b30a46401ff45228a38ceaed7d713b52036ca8af4b48a4e9',
                            state_sha256='d79a96e644ec6963a2a06e3e401188ffffb6dcc0f715c21e421d7ddd17a59039'))
if os.environ.get('SEEDS_REFS'):          # local tests only
    REFS.update(json.loads(Path(os.environ['SEEDS_REFS']).read_text()))
REPORT = dict(kernel='casmi26-saner-seeds', task='saner seed ensemble on the published holdout + full refits (task #46)',
              bundle='sanpier/casmi-spectrum-model-v1', seeds=SEEDS, budget_min=BUDGET_MIN, checks={}, errors=[], timeline=[],
              holdout={}, ensembles=[], finals={}, decision=None)
FINALS = {}


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
    log(f'SEEDS CHECK {name}: {tag} ' + json.dumps(detail, default=_js)[:1500])
    save_report()


def guarded(name, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        REPORT['errors'].append(dict(step=name, error=repr(e), trace=traceback.format_exc()[-3000:]))
        log(f'SEEDS ERROR in {name}: {e!r}\n{traceback.format_exc()}')
        save_report()
        return None


def stamp(step, start, status):
    REPORT['timeline'].append(dict(step=step, start_min=start, end_min=minutes(), status=status))
    save_report()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def run_logged(args, logfile, timeout=None, **kw):
    """Run a child in its own process group; stream its output to this log and to `logfile`.
    Returns (exit code, seconds, killed). With `timeout` (seconds) the whole group (pool workers included) is killed."""
    t = time.time()
    killed = []
    with open(logfile, 'w') as fh:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True, **kw)

        def _kill():
            killed.append(round(time.time() - t, 1))
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        timer = threading.Timer(timeout, _kill) if timeout else None
        if timer:
            timer.daemon = True
            timer.start()
        try:
            for line in p.stdout:
                sys.stdout.write(line); fh.write(line)
            rc = p.wait()
        finally:
            if timer:
                timer.cancel()
    sys.stdout.flush()
    return rc, round(time.time() - t, 1), bool(killed)


def soft_seconds(timeout_min):
    """train_stage's own deadline for a step with this hard timeout (graceful stop before the kill)."""
    return max(30., timeout_min * 60 - min(SOFT_MARGIN_MIN * 60, 0.2 * timeout_min * 60))


def fits(step, need_min):
    """True if a step with this hard timeout still fits into the budget (minus the finish reserve)."""
    left = BUDGET_MIN - FINISH_RESERVE_MIN - minutes()
    if need_min <= left:
        return True
    check(f'budget_{step}', None, skipped=True, need_min=round(need_min, 1), left_min=round(left, 1), at_min=minutes())
    return False


# ── 1. Published bundle (as casmi26-saner-repro) ─────────────────────────────────────────────────────────────
def find_bundle():
    hits = [p.parent for p in INPUT.rglob('sha256.json')
            if all((p.parent / f).is_file() for f in ('config.json', 'model.py', 'features.py'))]
    hits = sorted(set(hits), key=lambda p: (len(p.parts), str(p)))
    if not hits:
        raise SystemExit('SEEDS FATAL: sanpier/casmi-spectrum-model-v1 not mounted (no folder with sha256.json + model.py)')
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
    need = SOURCES + ['best_validation_model.pt', 'final_model.pt', 'validation_candidates.npz', 'molecules.parquet',
                      'training_result.json', 'data_audit.json', 'spectrum_split_manifest.parquet']
    missing = [f for f in need if not (B / f).is_file()]
    check('published_sources', not missing, present=[f for f in need if f not in missing], missing=missing)
    if missing:
        raise SystemExit(f'SEEDS FATAL: published files missing from the bundle: {missing}')
    SRC.mkdir(parents=True, exist_ok=True)
    for f in SOURCES:
        shutil.copy2(B / f, SRC / f)
    REPORT['bundle_dir'] = str(B)
    REPORT['bundle_sha256'] = pins
    return B


# ── 2. Environment (as casmi26-saner-repro) ──────────────────────────────────────────────────────────────────
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
            raise SystemExit(f'SEEDS FATAL: image RDKit is {rv or "absent"} and no rdkit-2026.3.3 {tag} wheel is attached')
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
        raise SystemExit('SEEDS FATAL: RDKit is not 2026.3.3 after the wheel install')
    return env


# ── 3. Patch and effective config (as casmi26-saner-repro) ──────────────────────────────────────────────────
def step_patch():
    log('== 3. the documented patch (prepare.py SMILES alias) and the effective config')
    orig = (SRC / 'prepare.py').read_text()
    n = orig.count(ALIAS_OLD)
    if n != 1:
        check('patch', False, occurrences=n, expected=1)
        raise SystemExit('SEEDS FATAL: the published prepare.py does not contain the expected alias line exactly once')
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
            raise SystemExit('SEEDS FATAL: competition train.parquet not mounted')
        eff['train_path'] = str(hits[0])
    overrides = {k: [cfg[k], eff[k]] for k in cfg if cfg[k] != eff[k]}
    (RW / 'effective_config.json').write_text(json.dumps(eff, indent=2))
    check('config', not set(overrides) - {'train_path'}, overrides=overrides, seed=eff['seed'], epochs=eff['epochs'],
          steps_per_epoch=eff['steps_per_epoch'], batch_size=eff['batch_size'], patience=eff['patience'],
          refit_full=eff['refit_full'], work=eff['work'], train_path=eff['train_path'],
          published_seed_matches=eff['seed'] == PUBLISHED_SEED)
    if eff['seed'] != PUBLISHED_SEED:
        raise SystemExit(f'SEEDS FATAL: config.json seed {eff["seed"]} != {PUBLISHED_SEED}')
    return eff


# ── 4. Structure memo + prepare (F7's launcher without train.run) ──────────────────────────────────────────
PREP_LAUNCHER = r'''"""Prepare launcher for the published saner code (sanpier/casmi-spectrum-model-v1, Apache-2.0), written by the
casmi26-saner-seeds kernel: the first half of casmi26-saner-repro's launcher (prepare.prepare(cfg) only; the training
stages run separately in stage.py). cfg = config.json (effective_config.json: path overrides only, normally none).
Speed-only addition: features.structure (RDKit tautomer canonicalisation, ~15 ms per SMILES) runs for every unique
SMILES in a fork pool BEFORE prepare(). prepare() then gets the identical tuple from a memo through its module-level
name `structure`. A SMILES that fails in the pool, and any miss, falls through to the original function in-process, so
prepare() raises the original exceptions. After prepare(), a random sample of memo entries is recomputed with the
original function and compared field by field (written to the check file)."""
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
'''

STAGE_LAUNCHER = r'''"""Stage launcher written by casmi26-saner-seeds: ONE stage of the published train.train_stage
(sanpier/casmi-spectrum-model-v1, Apache-2.0) on the prepared data, exactly as train.run(cfg) calls it, with
cfg['seed'] = SEED (published config otherwise):
  holdout: train_stage(data, cfg, OUT, device, deadline=...)                            (train.run's first call)
  refit:   train_stage(data, cfg, OUT, device, refit=True, epochs=EPOCHS, deadline=...) (train.run's second call;
           train_stage seeds it with SEED + 1)
The deadline is train.run's (holdout_hours / refit_hours of config.json) capped by the kernel's per-step soft limit; a
stage it stops reports time_budget_exhausted=True and the kernel does not use it. Same process set-up as train.run and as
casmi26-saner-repro's seed reference (torch threads, then train.Data(cfg))."""
import json, os, sys, time
from pathlib import Path
T_START = time.monotonic()
SRC, CFG, OUT, STAGE = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
SEED, EPOCHS, SOFT_SEC = int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
assert STAGE in ('holdout', 'refit'), STAGE
sys.path.insert(0, SRC)
import torch
import train
cfg = json.loads(Path(CFG).read_text())
cfg['seed'] = SEED
OUT.mkdir(parents=True, exist_ok=True)
assert not list(OUT.glob('*.pt')), f'{OUT} already holds checkpoints; a fresh directory is required'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cfg['require_gpu'] and device.type != 'cuda':
    raise RuntimeError('Enable a Kaggle GPU accelerator before training')
torch.set_num_threads(min(4, os.cpu_count() or 1))
data = train.Data(cfg)
refit = STAGE == 'refit'
hours = cfg['refit_hours'] if refit else cfg['holdout_hours']
deadline = min(time.monotonic() + hours * 3600, T_START + SOFT_SEC)
if refit:
    result = train.train_stage(data, cfg, OUT, device, refit=True, epochs=EPOCHS, deadline=deadline)
else:
    result = train.train_stage(data, cfg, OUT, device, deadline=deadline)
result.update(seed=SEED, stage=STAGE, stage_seed=SEED + (1 if refit else 0), epochs_requested=EPOCHS if refit else cfg['epochs'],
              process_seconds=round(time.monotonic() - T_START, 1), device=str(device))
(OUT / 'training_result.json').write_text(json.dumps(result, indent=2))
print('STAGE_RESULT', json.dumps(result), flush=True)
'''

SELFTEST = ('import json, sys\nsys.path.insert(0, sys.argv[1])\nimport runtime_test\n'
            'runtime_test.self_test(json.load(open(sys.argv[2])))')


def step_prepare(eff):
    log('== 4a. published runtime_test.self_test')
    t = minutes()
    rc, sec, killed = run_logged([sys.executable, '-u', '-c', SELFTEST, str(SRC), str(RW / 'effective_config.json')],
                                 RW / 'runtime_test.log', timeout=SELFTEST_TIMEOUT_MIN * 60, cwd=RW)
    check('runtime_test', rc == 0, exit=rc, seconds=sec, killed=killed)
    (RW / 'launch_prepare.py').write_text(PREP_LAUNCHER)
    (RW / 'stage.py').write_text(STAGE_LAUNCHER)
    log('== 4b. launchers (saner_seeds/launch_prepare.py, saner_seeds/stage.py):')
    log(PREP_LAUNCHER)
    log(STAGE_LAUNCHER)
    if not fits('prepare', PREP_TIMEOUT_MIN):
        return False
    log(f'== 4c. structure memo + published prepare.prepare(cfg) (hard timeout {PREP_TIMEOUT_MIN} min)')
    rc, sec, killed = run_logged([sys.executable, '-u', str(RW / 'launch_prepare.py'), str(SRC), str(RW / 'effective_config.json'),
                                  str(RW / 'struct_memo_check.json')], RW / 'prepare.log', timeout=PREP_TIMEOUT_MIN * 60, cwd=RW,
                                 env={**os.environ, 'SANER_SPOT_CHECK': str(SPOT_CHECK)})
    memo = read_json(RW / 'struct_memo_check.json') or {}
    ok_memo = bool(memo) and ((not memo.get('pool')) or (memo.get('spot_mismatch') == 0 and memo.get('spot_checked', 0) > 0))
    check('struct_memo', ok_memo, **memo)
    work = Path(eff['work'])
    files = {f: (work / f).is_file() for f in ('molecules.parquet', 'spectra.parquet', 'peaks.bin', 'metadata.bin',
                                               'fingerprints.npy', 'data_audit.json')}
    ok = rc == 0 and all(files.values())
    check('prepare_run', ok, exit=rc, minutes=round(sec / 60, 1), killed=killed, files=files)
    stamp('prepare', t, 'ok' if ok else 'failed')
    return ok


# ── 5. Evaluation helpers (pure numpy; tested locally on synthetic data) ────────────────────────────────────
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
    """train.validate's rank: gold is index 0 and must strictly beat every other candidate (pessimistic ties)."""
    import numpy as np
    return np.array([1 + int(np.sum(np.asarray(scores[q])[1:] >= np.asarray(scores[q])[0])) for q in qs])


def per_query(r):
    import numpy as np
    r = np.asarray(r)
    return dict(mrr25=mrr_vec(r), top1=(r == 1).astype(np.float64), recall25=(r <= 25).astype(np.float64))


def zscore(a):
    import numpy as np
    a = np.asarray(a, np.float64)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > 0 else 1.0)


def combine(score_dicts, qs, mode):
    """Per-query ensemble of several models' candidate scores: 'raw' = mean score, 'z' = mean within-query z-score."""
    import numpy as np
    out = {}
    for q in qs:
        arrs = [np.asarray(s[q], np.float64) for s in score_dicts]
        if any(len(a) != len(arrs[0]) for a in arrs):
            raise ValueError(f'query {q}: candidate counts differ between models')
        if mode == 'z':
            arrs = [zscore(a) for a in arrs]
        elif mode != 'raw':
            raise ValueError(mode)
        out[q] = np.mean(arrs, 0)
    return out


def boot_index(n, b=BOOT, seed=BOOT_SEED):
    import numpy as np
    return np.random.default_rng(seed).integers(0, n, (b, n))


def boot_diff(d, idx):
    """Paired bootstrap of a per-query difference: mean, CI95 (percentile) and P(mean > 0)."""
    import numpy as np
    d = np.asarray(d, np.float64)
    b = d[idx].mean(1)
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                p_gt0=float((b > 0).mean()))


def ensemble_rows(member_scores, member_names, qs, idx, ks=None, modes=('raw', 'z')):
    """Ensembles of the first k members (member 0 = the reference, the published holdout model). For each k and mode:
    metrics + paired bootstrap of (ensemble - member 0) and (ensemble - mean of its k members)."""
    import numpy as np
    K = len(member_scores)
    singles = [per_query(ranks_from_scores(s, qs)) for s in member_scores]
    ref = singles[0]
    rows = []
    for k in (ks or range(1, K + 1)):
        mean_member = {m: np.mean([singles[j][m] for j in range(k)], 0) for m in ref}
        for mode in modes:
            r = ranks_from_scores(combine(member_scores[:k], qs, mode), qs)
            pq = per_query(r)
            rows.append(dict(k=k, mode=mode, members=list(member_names[:k]), **summarize(r),
                             vs_pub={m: boot_diff(pq[m] - ref[m], idx) for m in pq},
                             vs_mean_member={m: boot_diff(pq[m] - mean_member[m], idx) for m in pq},
                             mean_member_mrr25=float(mean_member['mrr25'].mean())))
    return rows


def decide(rows, n_members, min_k=DECISION_MIN_K):
    """Decision rule (see the docstring, step 7)."""
    if n_members < min_k:
        return dict(verdict='INSUFFICIENT', members=n_members, min_k=min_k)
    at = {r['mode']: r for r in rows if r['k'] == n_members}
    if not at:
        return dict(verdict='INSUFFICIENT', members=n_members, min_k=min_k, note='no rows at k = members')
    mode = max(at, key=lambda m: (at[m]['mrr25'], m == 'raw'))
    r = at[mode]
    vp, vm = r['vs_pub']['mrr25'], r['vs_mean_member']['mrr25']
    if vp['ci95'][0] > 0:
        verdict = 'GO'
    elif vp['diff'] > 0 and vm['ci95'][0] > 0:
        verdict = 'LEAN_GO'
    else:
        verdict = 'NO_GAIN'
    other = {m: dict(mrr25=at[m]['mrr25'], d_vs_pub=at[m]['vs_pub']['mrr25']) for m in at if m != mode}
    return dict(verdict=verdict, k=n_members, mode=mode, members=r['members'], mrr25=r['mrr25'], top1=r['top1'],
                recall25=r['recall25'], pub_mrr25=r['mrr25'] - vp['diff'], d_mrr25_vs_pub=vp, d_mrr25_vs_mean_member=vm,
                d_top1_vs_pub=r['vs_pub']['top1'], d_recall25_vs_pub=r['vs_pub']['recall25'], other_modes=other,
                note='k and mode chosen on the same 1,000 molecules; GO = CI95(ens - published) > 0; LEAN_GO = mean > 0 '
                     'and CI95(ens - mean member) > 0')


def fmt_d(b, nd=4):
    return f"{b['diff']:+.{nd}f} [{b['ci95'][0]:+.{nd}f}, {b['ci95'][1]:+.{nd}f}] P>0 {b['p_gt0']:.3f}"


def eval_line(r):
    return (f"SEEDS EVAL k={r['k']} mrr {r['mrr25']:.5f} mode={r['mode']} top1 {r['top1']:.4f} recall25 {r['recall25']:.4f} | "
            f"d_mrr vs pub {fmt_d(r['vs_pub']['mrr25'])} | d_top1 vs pub {fmt_d(r['vs_pub']['top1'], 3)} | "
            f"d_mrr vs mean member {fmt_d(r['vs_mean_member']['mrr25'])} | members {','.join(map(str, r['members']))}")


def pair_stats(sa, sb, qs, seed=0):
    """Agreement of two checkpoints on the same candidate sets (scores: {query: per-candidate array, gold first})."""
    import numpy as np
    from scipy.stats import spearmanr
    sp, t1, ov = [], [], []
    for q in qs:
        a, b = np.asarray(sa[q], np.float64), np.asarray(sb[q], np.float64)
        sp.append(spearmanr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan)
        t1.append(int(np.argmax(a) == np.argmax(b)))
        ka, kb = np.argsort(-a, kind='stable')[:25], np.argsort(-b, kind='stable')[:25]
        ov.append(len(set(ka.tolist()) & set(kb.tolist())) / min(25, len(a)))
    ra, rb = ranks_from_scores(sa, qs), ranks_from_scores(sb, qs)
    d = mrr_vec(rb) - mrr_vec(ra)
    bt = boot_diff(d, boot_index(len(d), 2000, seed))
    return dict(n_queries=len(qs), spearman_per_query_mean=float(np.nanmean(sp)), spearman_per_query_median=float(np.nanmedian(sp)),
                top1_candidate_agreement=float(np.mean(t1)), top25_overlap_mean=float(np.mean(ov)),
                gold_rank_identical=float(np.mean(ra == rb)), mrr25_a=float(mrr_vec(ra).mean()), mrr25_b=float(mrr_vec(rb).mean()),
                mrr25_b_minus_a=bt['diff'], mrr25_diff_ci95=bt['ci95'])


# ── 5. Evaluator (published code, GPU) ───────────────────────────────────────────────────────────────────────
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


def checkpoint_state_sha(path):
    import torch
    return state_sha(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])


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


def published_candidates(B, data):
    """The published validation_candidates.npz (indices into the published molecules.parquet) mapped to our molecule
    indices by InChIKey14; gold first."""
    import numpy as np, pandas as pd
    pk = np.asarray(pd.read_parquet(B / 'molecules.parquet', columns=['key']).key.values, dtype=object)
    ok = np.asarray(data.mol.key.values, dtype=object)
    oi = {k: i for i, k in enumerate(ok)}
    pub = np.load(B / 'validation_candidates.npz', allow_pickle=False)
    cands, drop_q, drop_c = {}, 0, 0
    for q in pub.files:
        keys = pk[pub[q]]
        qk = pk[int(q)]
        if qk not in oi or keys[0] != qk:
            drop_q += 1; continue
        ids = [oi[k] for k in keys if k in oi]
        drop_c += len(keys) - len(ids)
        cands[oi[qk]] = np.asarray(ids, np.int64)
    split = data.mol.split.values
    info = dict(source='published validation_candidates.npz (mapped by InChIKey14)', published_queries=len(pub.files),
                eval_queries=len(cands), dropped_queries=drop_q, dropped_candidates=drop_c,
                candidates_per_query=sorted({len(v) for v in cands.values()}),
                molecule_index_identical=bool(len(pk) == len(ok) and all(pk == ok)),
                queries_in_validation_split=int(sum(split[q] == 'validation' for q in cands)))
    check('validation_candidates', drop_q == 0 and drop_c == 0 and len(cands) == len(pub.files) and
          info['queries_in_validation_split'] == len(cands), **info)
    return cands


class Evaluator:
    """Holds the prepared data and the published candidate sets in this process; scores checkpoints on the GPU."""

    def __init__(self, B, eff):
        log('== 5. evaluator: prepared data checks, published candidates, published holdout model')
        import torch
        sys.path.insert(0, str(SRC))
        import train as T
        from model import SpectrumCandidateModel
        SCORES.mkdir(parents=True, exist_ok=True)
        self.T, self.M, self.B, self.eff = T, SpectrumCandidateModel, B, eff
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = T.Data(eff)
        REPORT['prepared'] = guarded('compare_prepared', compare_prepared, B, eff, self.data)
        self.cands = published_candidates(B, self.data)
        self.qs = sorted(self.cands)
        self.scores, self.meta, self.members = {}, {}, []   # members: [(seed, name)] in SEEDS order, holdout models
        self._idx = {}
        name = f'holdout_{PUBLISHED_SEED}'
        meta = self.score(name, B / 'best_validation_model.pt')
        tr = read_json(B / 'training_result.json') or {}
        want = tr.get('best_diagnostic_mrr25')
        ok = want is not None and abs(meta['mrr25'] - want) <= PUB_MRR_TOL and meta['validate_parity_abs_diff'] <= 1e-9
        check('harness_parity', ok, published_training_result=want, recomputed_here=meta['mrr25'],
              abs_diff=None if want is None else abs(meta['mrr25'] - want), validate_mrr25=meta['validate_mrr25'],
              validate_parity_abs_diff=meta['validate_parity_abs_diff'], top1=meta['top1'], recall25=meta['recall25'],
              state_sha256=meta['state_sha256'], best_epoch=tr.get('best_epoch'))
        REPORT['holdout'][str(PUBLISHED_SEED)] = dict(source='published best_validation_model.pt', result=tr, eval=meta)
        self.members.append((PUBLISHED_SEED, name))
        self.log_k()

    def idx(self, n):
        if n not in self._idx:
            self._idx[n] = boot_index(n)
        return self._idx[n]

    def score(self, name, path, contaminated=False, parity=True):
        import numpy as np, torch
        obj = torch.load(path, map_location='cpu', weights_only=True)
        sd = obj['state_dict']
        meta = dict(path=str(path), file_sha256=sha256(path), state_sha256=state_sha(sd), epoch=obj.get('epoch'),
                    stage=obj.get('stage'), contaminated=contaminated)
        model = self.M(self.eff).to(self.device)
        model.load_state_dict(sd); model.eval()
        t = time.time()
        sc = score_sets(model, self.data, self.cands, self.device, self.eff)
        qn = [q for q in self.qs if q in sc]
        meta.update(summarize(ranks_from_scores(sc, qn)))
        if parity:
            official, _ = self.T.validate(model, self.data, self.cands, self.device, self.eff)
            meta.update(validate_mrr25=float(official), validate_parity_abs_diff=abs(float(official) - meta['mrr25']))
        meta['eval_seconds'] = round(time.time() - t, 1)
        del model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        np.savez_compressed(SCORES / f'{name}.npz', **{str(q): sc[q] for q in qn})
        self.scores[name], self.meta[name] = sc, meta
        return meta

    def common_qs(self, names):
        return [q for q in self.qs if all(q in self.scores[n] for n in names)]

    def rows(self, ks=None):
        names = [n for _, n in self.members]
        qn = self.common_qs(names)
        return ensemble_rows([self.scores[n] for n in names], [s for s, _ in self.members], qn, self.idx(len(qn)), ks=ks)

    def log_k(self):
        k = len(self.members)
        for r in self.rows(ks=[k]):
            log(eval_line(r))
            REPORT['ensembles'] = [x for x in REPORT['ensembles'] if not (x['k'] == r['k'] and x['mode'] == r['mode'])] + [r]
        save_report()

    def add_holdout(self, h):
        seed = h['seed']
        name = f'holdout_{seed}'
        meta = self.score(name, h['path'])
        pub = f'holdout_{PUBLISHED_SEED}'
        qn = self.common_qs([pub, name])
        ps = pair_stats(self.scores[pub], self.scores[name], qn)
        train_mrr = (h.get('result') or {}).get('best_diagnostic_mrr25')
        meta.update(vs_published=ps, train_time_mrr25=train_mrr,
                    train_eval_parity_abs_diff=None if train_mrr is None else abs(train_mrr - meta['mrr25']))
        REPORT['holdout'][str(seed)]['eval'] = meta
        log(f"SEEDS EVAL single seed={seed} mrr {meta['mrr25']:.5f} top1 {meta['top1']:.4f} recall25 {meta['recall25']:.4f} | "
            f"d_mrr vs pub {ps['mrr25_b_minus_a']:+.4f} [{ps['mrr25_diff_ci95'][0]:+.4f}, {ps['mrr25_diff_ci95'][1]:+.4f}] | "
            f"spearman vs pub {ps['spearman_per_query_mean']:.4f} top1-agree {ps['top1_candidate_agreement']:.3f} | "
            f"best_epoch {(h.get('result') or {}).get('best_epoch')} | train-time mrr {train_mrr} | "
            f"validate parity {meta['validate_parity_abs_diff']:.2e}")
        self.members.append((seed, name))
        self.log_k()
        return meta

    def summary(self):
        log('== 7. decision (holdout ensembles on the published validation candidates)')
        rows = self.rows()
        REPORT['ensembles'] = rows
        names = [n for _, n in self.members]
        mat = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                qn = self.common_qs([names[i], names[j]])
                st = pair_stats(self.scores[names[i]], self.scores[names[j]], qn)
                mat[f'{self.members[i][0]}__{self.members[j][0]}'] = dict(
                    spearman=st['spearman_per_query_mean'], top1_agreement=st['top1_candidate_agreement'],
                    top25_overlap=st['top25_overlap_mean'])
        REPORT['holdout_pairs'] = mat
        dec = decide(rows, len(self.members))
        singles = [self.meta[n]['mrr25'] for n in names]
        import numpy as np
        dec.update(single_mrr25={str(s): self.meta[n]['mrr25'] for s, n in self.members},
                   single_mrr25_mean=float(np.mean(singles)), single_mrr25_sd=float(np.std(singles, ddof=1)) if len(singles) > 1 else None,
                   pairwise_spearman_mean=float(np.mean([v['spearman'] for v in mat.values()])) if mat else None,
                   pairwise_top1_agreement_mean=float(np.mean([v['top1_agreement'] for v in mat.values()])) if mat else None)
        REPORT['decision'] = dec
        save_report()
        log('SEEDS SUMMARY decision ' + json.dumps(dec, default=_js))
        return dec

    def score_finals(self):
        log('== 9b. final (refit) models: contaminated diagnostics (trained on these molecules; agreement, not quality)')
        pubname = f'final_{PUBLISHED_SEED}'
        for seed in sorted(FINALS, key=lambda s: (s != PUBLISHED_SEED, s)):
            name = f'final_{seed}'
            meta = self.score(name, WORKING / FINALS[seed]['file'], contaminated=True, parity=False)
            row = dict(contaminated_mrr25=meta['mrr25'], contaminated_top1=meta['top1'], state_sha256=meta['state_sha256'])
            if seed != PUBLISHED_SEED and pubname in self.scores:
                ps = pair_stats(self.scores[pubname], self.scores[name], self.common_qs([pubname, name]))
                row.update(spearman_to_published_final=ps['spearman_per_query_mean'],
                           top1_agreement_to_published_final=ps['top1_candidate_agreement'],
                           top25_overlap_to_published_final=ps['top25_overlap_mean'])
                ok = (ps['spearman_per_query_mean'] >= FINAL_SANITY['min_spearman_to_published_final']
                      and meta['mrr25'] >= FINAL_SANITY['min_contaminated_mrr25'])
            else:
                ok = None
            FINALS[seed]['diagnostics'] = row
            check(f'final_{seed}_sanity', ok, **row, thresholds=FINAL_SANITY)
        write_finals()
        names = [f'final_{s}' for s in SEEDS if f'final_{s}' in self.scores]
        if len(names) > 1:
            qn = self.common_qs(names)
            REPORT['finals_ensemble_contaminated'] = [
                dict(k=r['k'], mode=r['mode'], mrr25=r['mrr25'], top1=r['top1'], d_vs_published_final=r['vs_pub']['mrr25']['diff'])
                for r in ensemble_rows([self.scores[n] for n in names], names, qn, self.idx(len(qn)), ks=[len(names)])]
            save_report()


# ── 6. Holdout stage per seed ───────────────────────────────────────────────────────────────────────────────
def stage_dir(stage, seed):
    out = RW / f'{stage}_{seed}'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def run_stage(stage, seed, epochs, timeout_min):
    out = stage_dir(stage, seed)
    rc, sec, killed = run_logged([sys.executable, '-u', str(RW / 'stage.py'), str(SRC), str(RW / 'effective_config.json'),
                                  str(out), stage, str(seed), str(epochs), str(soft_seconds(timeout_min))],
                                 out / 'stage.log', timeout=timeout_min * 60, cwd=RW)
    return out, rc, sec, killed, read_json(out / 'training_result.json')


def step_holdout(B, seed):
    t = minutes()
    log(f'== 6. holdout stage, seed {seed} (train.train_stage as train.run calls it; hard timeout {HOLDOUT_TIMEOUT_MIN} min)')
    out, rc, sec, killed, res = run_stage('holdout', seed, 0, HOLDOUT_TIMEOUT_MIN)
    best = out / 'best_validation_model.pt'
    complete = rc == 0 and res is not None and best.is_file() and not res.get('time_budget_exhausted')
    cf = out / 'validation_candidates.npz'
    cand_same = cf.is_file() and sha256(cf) == sha256(B / 'validation_candidates.npz')
    # A model whose own candidate file differs from the published one was trained on different prepared data (its split
    # may overlap the evaluation molecules): it is reported but neither evaluated, ensembled nor refit.
    ok = complete and cand_same
    h = dict(seed=seed, ok=ok, path=best, result=res, exit=rc, minutes=round(sec / 60, 1), killed=killed,
             candidates_file_identical=cand_same, file_sha256=sha256(best) if best.is_file() else None,
             best_epoch=(res or {}).get('best_epoch'))
    check(f'holdout_{seed}', ok, **{k: v for k, v in h.items() if k not in ('path', 'ok')})
    REPORT['holdout'][str(seed)] = dict(source=f'trained here ({out.name})', **{k: v for k, v in h.items() if k != 'path'})
    stamp(f'holdout_{seed}', t, 'ok' if ok else 'failed')
    return h


def check_determinism(h, meta):
    """Seed 20262023 = casmi26-saner-repro's seed reference (same code, data, seed and launcher set-up)."""
    ref = REFS['f7_seedref']
    if h['seed'] != ref['seed']:
        return
    st = (meta or {}).get('state_sha256')
    mrr = (h.get('result') or {}).get('best_diagnostic_mrr25')
    same = h.get('file_sha256') == ref['file_sha256'] or (st is not None and st == ref['state_sha256'])
    check(f'determinism_seed{ref["seed"]}', bool(same), file_sha256=h.get('file_sha256'), f7_file_sha256=ref['file_sha256'],
          state_sha256=st, f7_state_sha256=ref['state_sha256'], best_diagnostic_mrr25=mrr, f7_mrr25=ref['mrr25'],
          note='OK = bit-exact with the F7 seed reference; a FAIL means run-to-run GPU nondeterminism, not a broken run')


# ── 8. Refits ───────────────────────────────────────────────────────────────────────────────────────────────
def write_finals():
    FINALS_PATH.write_text(json.dumps({f'final_model_seed{s}.pt': v for s, v in sorted(FINALS.items())}, indent=2, default=_js))
    REPORT['finals'] = {str(s): v for s, v in sorted(FINALS.items())}
    save_report()


def publish_final(seed, src, **info):
    name = f'final_model_seed{seed}.pt'
    dst = WORKING / name
    shutil.copy2(src, dst)
    fs = sha256(dst)
    (WORKING / f'{name}.sha256').write_text(f'{fs}  {name}\n')
    FINALS[seed] = dict(file=name, file_sha256=fs, state_sha256=checkpoint_state_sha(dst), seed=seed, refit_seed=seed + 1,
                        bytes=dst.stat().st_size, **info)
    write_finals()
    return FINALS[seed]


def step_copy_published_final(B):
    pin = (REPORT.get('bundle_sha256') or {}).get('final_model.pt')
    tr = read_json(B / 'training_result.json') or {}
    f = publish_final(PUBLISHED_SEED, B / 'final_model.pt', epochs=(tr.get('refit') or {}).get('completed_epochs'),
                      source='published sanpier/casmi-spectrum-model-v1 final_model.pt (Apache-2.0); F7 reproduced it bit-exactly')
    check(f'final_{PUBLISHED_SEED}', f['file_sha256'] == pin, file=f['file'], file_sha256=f['file_sha256'], bundle_pin=pin)


def step_refit(seed, epochs):
    t = minutes()
    log(f'== 8. full refit, seed {seed} (train_stage refit=True, seed {seed + 1}, {epochs} epochs; hard timeout {REFIT_TIMEOUT_MIN} min)')
    out, rc, sec, killed, res = run_stage('refit', seed, epochs, REFIT_TIMEOUT_MIN)
    fm = out / 'final_model.pt'
    complete = (rc == 0 and res is not None and fm.is_file() and res.get('completed_epochs') == epochs
                and not res.get('time_budget_exhausted'))
    detail = dict(exit=rc, minutes=round(sec / 60, 1), killed=killed, epochs=epochs, refit_seed=seed + 1, result=res)
    if complete:
        f = publish_final(seed, fm, epochs=epochs, source=f'refit trained here (published train_stage refit, seed {seed + 1})',
                          refit_minutes=round(sec / 60, 1))
        detail.update(file=f['file'], file_sha256=f['file_sha256'], state_sha256=f['state_sha256'])
    check(f'refit_{seed}', complete, **detail)
    stamp(f'refit_{seed}', t, 'ok' if complete else 'failed')
    return complete


def step_refit_parity(B):
    t = minutes()
    tr = read_json(B / 'training_result.json') or {}
    epochs = int(tr.get('best_epoch') or 0)
    log(f'== 9a. refit-launcher parity: seed {PUBLISHED_SEED} refit ({epochs} epochs) vs the published final_model.pt')
    out, rc, sec, killed, res = run_stage('refit', PUBLISHED_SEED, epochs, REFIT_TIMEOUT_MIN)
    fm = out / 'final_model.pt'
    got = sha256(fm) if fm.is_file() else None
    pub = sha256(B / 'final_model.pt')
    st = checkpoint_state_sha(fm) if fm.is_file() else None
    pst = checkpoint_state_sha(B / 'final_model.pt')
    check(f'refit_parity_seed{PUBLISHED_SEED}', rc == 0 and (got == pub or st == pst), exit=rc, minutes=round(sec / 60, 1),
          killed=killed, epochs=epochs, file_sha256=got, published_file_sha256=pub, state_sha256=st, published_state_sha256=pst,
          result=res, note='OK = the standalone refit launcher reproduces train.run\'s refit bit-exactly')
    stamp('refit_parity', t, 'ok' if rc == 0 else 'failed')


# ── finish ──────────────────────────────────────────────────────────────────────────────────────────────────
def step_finish(eff):
    shas = {}
    for d in sorted(RW.glob('holdout_*')) + sorted(RW.glob('refit_*')):
        shas.update({f'{d.name}/{p.name}': sha256(p) for p in sorted(d.iterdir()) if p.is_file()})
    (RW / 'stage_outputs_sha256.json').write_text(json.dumps(shas, indent=2))
    REPORT['stage_outputs_sha256'] = shas
    work = Path(eff['work']) if eff else None
    freed = 0
    for f in ('peaks.bin', 'metadata.bin', 'fingerprints.npy', 'spectra.parquet'):
        p = work / f if work else None
        if p is not None and p.is_file():
            freed += p.stat().st_size; p.unlink()
    REPORT['cache_deleted_bytes'] = freed
    c = REPORT['checks']
    must = ['published_sha256', 'patch', 'config', 'struct_memo', 'prepare_run', 'prepare_data_audit', 'prepare_molecules',
            'prepare_spectrum_manifest', 'validation_candidates', 'harness_parity']
    hold_ok = [s for s in SEEDS[1:] if c.get(f'holdout_{s}', {}).get('ok')]
    refit_ok = [s for s in SEEDS[1:] if c.get(f'refit_{s}', {}).get('ok')]
    dec = REPORT.get('decision') or {}
    REPORT['verdict'] = dict(setup_ok=all(c.get(k, {}).get('ok') for k in must), setup_failed=[k for k in must if not c.get(k, {}).get('ok')],
                             holdouts_ok=hold_ok, decision_ready=len(hold_ok) + 1 >= DECISION_MIN_K, decision=dec.get('verdict'),
                             refits_ok=refit_ok, refits_missing=[s for s in hold_ok if s not in refit_ok],
                             final_models=sorted(v['file'] for v in FINALS.values()),
                             determinism_seed20262023=c.get('determinism_seed20262023', {}).get('ok'),
                             refit_parity=c.get(f'refit_parity_seed{PUBLISHED_SEED}', {}).get('ok'),
                             errors=len(REPORT['errors']), minutes=minutes())
    save_report()
    log('SEEDS SUMMARY ' + json.dumps(REPORT['verdict'], default=_js))
    for s in SEEDS:
        e = (REPORT['holdout'].get(str(s)) or {}).get('eval') or {}
        if e:
            log(f"SEEDS SUMMARY single seed={s} mrr {e['mrr25']:.5f} top1 {e['top1']:.4f} recall25 {e['recall25']:.4f}"
                + (f" spearman vs pub {e['vs_published']['spearman_per_query_mean']:.4f}" if e.get('vs_published') else ''))
    for r in REPORT.get('ensembles') or []:
        log('SEEDS SUMMARY ' + eval_line(r)[len('SEEDS '):])
    if dec:
        log(f"SEEDS SUMMARY decision {dec.get('verdict')} k={dec.get('k')} mode={dec.get('mode')} mrr {dec.get('mrr25')} vs pub "
            f"{dec.get('pub_mrr25')} d {(dec.get('d_mrr25_vs_pub') or {}).get('diff')} CI95 {(dec.get('d_mrr25_vs_pub') or {}).get('ci95')} "
            f"| vs mean member {(dec.get('d_mrr25_vs_mean_member') or {}).get('diff')} CI95 {(dec.get('d_mrr25_vs_mean_member') or {}).get('ci95')}")
    for s, v in sorted(FINALS.items()):
        log(f"SEEDS SUMMARY final seed={s} {v['file']} sha256 {v['file_sha256']} epochs {v.get('epochs')} "
            f"diag {json.dumps(v.get('diagnostics'), default=_js)}")
    log(f'SEEDS DONE {minutes()} min; report {REPORT_PATH}; finals {FINALS_PATH}')


def main():
    RW.mkdir(parents=True, exist_ok=True)
    eff = None
    try:
        B = step_bundle()
        step_env(B)
        eff = step_patch()
        if not step_prepare(eff):
            log('SEEDS FATAL: prepare did not complete; nothing to train')
            return
        ev = guarded('evaluator', Evaluator, B, eff)
        # 6. holdouts first: they are the decision. Holdouts beyond the decision minimum keep room for the refits of the
        # seeds already trained.
        done = []
        for i, s in enumerate(SEEDS[1:], start=2):
            need = HOLDOUT_TIMEOUT_MIN + (REFIT_TIMEOUT_MIN * len(done) if i > DECISION_MIN_K else 0)
            if not fits(f'holdout_{s}', need):
                continue
            h = guarded(f'holdout_{s}', step_holdout, B, s)
            if not h:
                continue
            meta = guarded(f'eval_{s}', ev.add_holdout, h) if (ev and h['ok']) else None
            check_determinism(h, meta)
            if h['ok']:
                done.append(h)
        if ev:
            guarded('decision', ev.summary)
        # 8. refits of the seeds whose holdout completed (published recipe: epochs = that seed's holdout best_epoch)
        if COPY_PUBLISHED_FINAL:
            guarded('copy_published_final', step_copy_published_final, B)
        for h in done:
            if fits(f'refit_{h["seed"]}', REFIT_TIMEOUT_MIN):
                guarded(f'refit_{h["seed"]}', step_refit, h['seed'], int(h['best_epoch']))
        # 9. parity of the standalone refit launcher, then contaminated diagnostics of the final models
        if REFIT_PARITY and fits('refit_parity', REFIT_TIMEOUT_MIN):
            guarded('refit_parity', step_refit_parity, B)
        if ev and FINALS:
            guarded('finals', ev.score_finals)
        if ev:
            ev.data = None
        ev = None
        gc.collect()
    finally:
        guarded('finish', step_finish, eff)


if os.environ.get('SEEDS_NO_MAIN') != '1':
    main()
