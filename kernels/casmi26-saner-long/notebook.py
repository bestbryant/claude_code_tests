"""casmi26-saner-long: long-schedule seeds of the published saner recipe (sanpier/casmi-spectrum-model-v1, Apache-2.0),
measured on the published holdout, then the full refits a deployment would load. Nothing here feeds a submission.

WHAT. kernels/casmi26-saner-seeds with one change to the recipe: epochs 120 (LONG_EPOCHS) and patience 8 (LONG_PATIENCE)
instead of the published 16 / 4. Every other config.json value is as published. Three long seeds (20261023, 20262023,
20263023) are trained on the published holdout split. Each is scored with the published candidates, alone and in
ensembles with the published 16-epoch holdout model and with ahmedberatozer's long model, then refit on the full data for
its holdout best_epoch (the published recipe). The per-candidate holdout scores of every member are saved so ensemble
weights can be chosen offline.

WHY. bgunshui/casmi26-scorer-train-public ran the same kit unchanged except epochs 80 (holdout_hours 5, refit_hours 6).
Early stopping (patience 4) stopped it at epoch 56, best epoch 52, with holdout diagnostic MRR@25 0.6603, against 0.4844 for
the published 16-epoch config. Its training_history was still rising: 0.1304 @1 (top-1 0.055), 0.4925 @17, 0.5451 @25,
0.6094 @37, 0.6350 @45, 0.6532 @49, 0.6526 @53, 0.6603 @52 (best). Epochs 1-16 are the published 16-epoch history:
the schedule does not depend on cfg['epochs']. A public notebook with 0.5 z(published 16-epoch final) + 0.5 z(that
80-epoch final) scored 0.480 on the public LB (our pipeline base: 0.479). ahmedberatozer/casmi26-spectrum-model-long
(40 x 2000 steps, patience 5, seed 20261023) reports holdout MRR@25 0.718 / top-1 0.591. We want stronger long models,
seed ensembles of them, and per-candidate holdout scores for choosing weights offline.

STEPS. Each step prints 'SEEDS CHECK <name>: OK | FAIL | INFO {...}'. The full record is /kaggle/working/seeds_report.json.
 1-3. As casmi26-saner-seeds: every bundle file against sha256.json; RDKit 2026.3.3 (offline wheel); the one documented
      patch ('normalized_smiles' added to the SMILES aliases of prepare.py). Config: effective_config.json is the
      published config.json (+ train_path if the mount differs), used by the self-test, prepare and the evaluator.
      long_config.json is that config + epochs LONG_EPOCHS + patience LONG_PATIENCE, used by the training stages. The
      'config' check accepts exactly these overrides (+ train_path) and records them.
 4.   Published runtime_test.self_test; the speed-only structure memo; published prepare.prepare(cfg) ONCE.
 5.   Evaluator (this process): prepared data vs the published data_audit / molecules / spectrum manifest; the published
      validation_candidates.npz (1,000 validation molecules x 256 candidates, gold first) mapped by InChIKey14.
      'published16' = the published holdout model (harness parity with training_result.json, and per query, by key,
      with the published validation_predictions.csv).
 5b.  'ahmed_long' = best_validation_model.pt of ahmedberatozer/casmi26-spectrum-model-long (not trained here). If the
      dataset is not mounted or its architecture differs, the member is skipped with a check entry. INFO check: our
      MRR@25 vs his README 0.718 at 3 decimals (his cache had 2,430,328 spectra vs the published 2,430,327), plus a
      per-query comparison with his validation_predictions.csv (same validation keys = same split).
 6.   One long holdout stage per seed, in that order ('long_<seed>'), each in its own process:
      train.train_stage(data, long cfg with seed s, ...) exactly as train.run's first call. Checks:
      holdout_long_<s>; determinism_long_<s> (INFO: the first 16 epochs vs the 16-epoch run of the same seed, i.e. the
      published training_history.json for 20261023 and the casmi26-saner-seeds v1 histories for the others);
      bgunshui_history_long_20261023 (INFO: diagnostic MRR@25 at epochs 1, 17, 25, 37, 45, 49, 53 at 4 decimals, top-1 at
      epoch 1, epoch 52 = 0.6602795453044211 within 1e-6, and the full 56-epoch history as extra detail; same seed, data
      and code, but ours continues past epoch 56). Each completed holdout model is then scored with train.validate's
      metric (MRR@25 with pessimistic ties, top-1, recall@25), and the configuration table is recomputed.
      Selection optimism of the singles: a long model (and ahmed_long) is its best_validation_model.pt, the epoch with the
      highest diagnostic MRR@25 on these same 1,000 queries (published16 is its last epoch, still rising). So 'long_<s>
      vs published16' is biased up and 'ensemble vs long_<s>' biased down, beyond the bootstrap CI. Each long single also
      reports history_mrr25_best_pm2 (mean training-time diagnostic MRR@25 over epochs best-2..best+2), a less biased value.
 7.   Configurations on the published holdout. A paired bootstrap (5,000 query resamples, shared by all comparisons,
      including the CI of the 'SEEDS EVAL single' lines) puts every row against published16, long_20261023, ahmed_long and
      the mean of its own members:
      (a) singles; (b) our long seeds k = 2, 3 (raw, z); (c) published16 + one long model (each seed and ahmed_long), z
      0.5/0.5; (d) published16 + our long seeds, z, equal weights and 0.5 / 0.5 (published / mean of the long seeds);
      (e) ahmed_long + each seed (raw, z); all-long = ahmed_long + our seeds (raw, z); published16 + all-long (z, equal
      weights and 0.5 / 0.5). 'SEEDS SUMMARY best <config>' names the highest MRR@25, with its CI vs published16 and vs
      long_20261023 alone. It is picked on the same 1,000 molecules (selection optimism).
      /kaggle/working/holdout_scores.npz (layout in holdout_scores.json) holds the per-candidate scores of every holdout
      member, the query 'key' (InChIKey14, as in the kit's validation_predictions.csv), molecule indices, CSR offsets and
      gold labels. It is rewritten after every member.
 8.   Full refit for each completed long holdout: train_stage(refit=True, epochs = that seed's holdout best_epoch) =
      train.run's second call (seed s + 1). Saved as /kaggle/working/final_model_long<s>.pt with .sha256, and listed in
      final_models_sha256.json (file + state-dict sha256). A refit that cannot finish inside its soft deadline even at
      the speed of its own holdout (stage.py load time + the holdout's per-epoch training seconds of epochs 1..best) is
      skipped with a FAIL check instead of burning its timeout.
 9.   Contaminated diagnostics of the finals. They were trained on the validation molecules, so this is a sanity check,
      not a quality number: Spearman to the same seed's long holdout model, contaminated MRR@25, and agreement with the
      published final_model.pt (scored from the bundle, not copied). No refit-parity run and no copy of the published
      final (casmi26-saner-seeds showed both). At the end the stage checkpoints (sha256 kept in the report) and the
      training cache are deleted, EXCEPT a long holdout's best_validation_model.pt when that seed has no published final
      or the model is missing from holdout_scores.npz: it is kept as /kaggle/working/holdout_model_long<s>.pt (+ .sha256,
      report 'kept_holdout_models'). The output size is checked (< 50 MB, kept holdout models counted separately).
OUTPUTS. final_model_long<s>.pt (+ .sha256) per refit, final_models_sha256.json, seeds_report.json, holdout_scores.npz,
holdout_scores.json, holdout_model_long<s>.pt (+ .sha256) only on the failure paths above; small logs / configs /
histories under saner_long/.
BUDGET. 8.5 h in total. A step starts only if its hard timeout still fits: prepare 120 min, each long holdout 60 min (soft
deadline 56 min), each refit 55 min (soft 51 min), plus 8 min reserved for the summary. Training stops itself 4 min before
the hard kill (train_stage's own deadline). The third long holdout also keeps room for the refits of the seeds already
trained. Worst case with every hard timeout: 120 + 3 x 60 + 3 x 55 + 8 = 473 min. A holdout stopped by that deadline keeps
its best epoch so far and IS used, flagged time_stopped (SEEDS_USE_TIME_STOPPED_HOLDOUT=0 restores the seeds kernel's rule:
reported, not used); its refit has no validation and ~1.6 min less loading, so it fits the 51 min soft deadline unless
the GPU needs > ~63 s of training per epoch. A refit stopped that way is never published. Order: prepare, the three long
holdouts with evaluation, summary, refits, finals sanity.
EXPECTED on a T4. Prepare ~45 min (structure memo + prepare). Each long holdout <= ~45 min: casmi26-saner-seeds v1 measured
~2.6 min of data loading + ~20.9 s per epoch (16.6 s training + 4.3 s validation). Early stopping around epochs 60-90 gives
~24-34 min; all 120 epochs would take ~44.4 min, inside the 56 min soft deadline (up to ~22.4 s of training per epoch).
Each refit <= ~35 min (~1 min loading + ~16.7 s per epoch: ~18-27 min for best epochs 60-90, 34.4 min at 120; fits the
51 min soft deadline up to ~25 s per epoch). Evaluation ~1 min per member. Total ~3.5 h expected, ~4.8 h at the worst-case
epochs (45 + 3 x 44.4 + 3 x 34.4 + ~5 min).
"""
import difflib, gc, hashlib, importlib.metadata, json, os, shutil, signal, subprocess, sys, threading, time, traceback
from pathlib import Path

T0 = time.time()


def _env_num(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else default


INPUT = Path(os.environ.get('SEEDS_INPUT', '/kaggle/input'))
WORKING = Path(os.environ.get('SEEDS_WORKING', '/kaggle/working'))
RW = WORKING / 'saner_long'
SRC = RW / 'src'
REPORT_PATH = WORKING / 'seeds_report.json'
FINALS_PATH = WORKING / 'final_models_sha256.json'
SCORES_PATH = WORKING / 'holdout_scores.npz'
SCORES_LAYOUT_PATH = WORKING / 'holdout_scores.json'

PUBLISHED_SEED = 20261023                      # config.json seed: the published 16-epoch holdout / final models
LONG_SEEDS = [20261023, 20262023, 20263023]    # trained here with the long schedule, in this order
LONG_EPOCHS = int(_env_num('SEEDS_LONG_EPOCHS', 120))
LONG_PATIENCE = int(_env_num('SEEDS_LONG_PATIENCE', 8))
PUB = 'published16'                            # member names; long models are 'long_<seed>' (never the bundle's names)
AHMED = 'ahmed_long'
LONG_REF = f'long_{LONG_SEEDS[0]}'             # the bgunshui seed: second reference of every comparison
BUDGET_MIN = _env_num('SEEDS_BUDGET_MIN', 510.)              # 8.5 h total
FINISH_RESERVE_MIN = _env_num('SEEDS_FINISH_RESERVE_MIN', 8.)
SELFTEST_TIMEOUT_MIN = 10.
PREP_TIMEOUT_MIN = _env_num('SEEDS_PREP_TIMEOUT_MIN', 120.)  # structure memo + prepare (seeds v1: 41 min)
HOLDOUT_TIMEOUT_MIN = _env_num('SEEDS_HOLDOUT_TIMEOUT_MIN', 60.)  # per long seed (<= 120 epochs ~44-47 min + load; 60 keeps a margin)
REFIT_TIMEOUT_MIN = _env_num('SEEDS_REFIT_TIMEOUT_MIN', 55.)      # per long seed (best epoch 120 ~34.4 min)
#   soft 51 min: 120 epochs up to ~25 s/epoch, and the refit of a holdout stopped by its own 56 min soft deadline
REFIT_LOAD_SEC = 90.          # refit start-up assumed when the holdout's stage.py load_seconds is missing (seeds v1: ~58 s)
SOFT_MARGIN_MIN = 4.          # train_stage's own deadline ends training this long before the hard kill
MIN_LONG_HOLDOUTS = 2         # long holdouts that start without keeping room for refits (the k = 2 ensemble)
USE_TIME_STOPPED_HOLDOUT = _env_num('SEEDS_USE_TIME_STOPPED_HOLDOUT', 1.) == 1
BOOT = 5000
BOOT_SEED = 0
SPOT_CHECK = 2000             # memo entries recomputed with the original features.structure after prepare()
PUB_MRR_TOL = 0.002           # published holdout model recomputed here vs its training_result.json
FINAL_SANITY = dict(min_spearman_to_own_holdout=0.7, min_contaminated_mrr25=0.45)
MAX_OUTPUT_MB = 50.
ARCH_KEYS = ('fp_bits', 'max_peaks', 'width', 'heads', 'layers', 'embedding', 'dropout', 'temperature', 'fp_score_weight')
AHMED_INFO = dict(dataset='ahmedberatozer/casmi26-spectrum-model-long', seed=20261023, epochs=40, steps_per_epoch=2000,
                  patience=5, readme_best_epoch=36, readme_mrr25=0.718, readme_top1=0.591, cache_spectra=2430328,
                  published_spectra=2430327)
SOURCES = ['config.json', 'features.py', 'model.py', 'prepare.py', 'train.py', 'inference.py', 'runtime_test.py']
ALIAS_OLD = "'smiles':['smiles','canonical_smiles']"
ALIAS_NEW = "'smiles':['smiles','canonical_smiles','normalized_smiles']"
# Training-history references (diagnostic MRR@25 per epoch, epoch 1 first).
#  bgunshui: bgunshui/casmi26-scorer-train-public (same kit and data, seed 20261023, epochs 80, patience 4): the values the
#    task names at 4 decimals, plus its full training_history.json (56 epochs) for the extra detail.
#  seeds_v1_history: casmi26-saner-seeds v1 holdout stages (16 x 500 steps; 20262023 = F7's seed reference, 0.48597 @16).
REFS = dict(
    bgunshui=dict(
        kernel='bgunshui/casmi26-scorer-train-public', seed=20261023, epochs=80, patience=4, best_epoch=52, completed_epochs=56,
        best_diagnostic_mrr25=0.6602795453044211,
        mrr25_4dp={'1': 0.1304, '17': 0.4925, '25': 0.5451, '37': 0.6094, '45': 0.6350, '49': 0.6532, '53': 0.6526},
        top1_4dp={'1': 0.055},
        history_mrr25=[
            0.13037950905767032, 0.2014661339609748, 0.2505490978039026, 0.28247093042757637, 0.3301348751874051,
            0.3471402708405323, 0.3593061739641209, 0.3892383286618483, 0.3965897832918939, 0.4058016143936305,
            0.429486036393349, 0.43912239141622195, 0.45350572579108717, 0.46841309298977896, 0.4734470410171558,
            0.48440741342338733, 0.4924871153861842, 0.5026822856703876, 0.5128988364622171, 0.52751037947993,
            0.5324105140411555, 0.5253154791068252, 0.5457255777581924, 0.5399368223662488, 0.5450528704399353,
            0.5571680640195941, 0.5577088986836887, 0.5699470648018319, 0.569705175459253, 0.5831432672380267,
            0.5870797956807554, 0.590181200827074, 0.5997941519402884, 0.6082363014514239, 0.5990148536476461,
            0.613551799835012, 0.6094160211978087, 0.6088757410825515, 0.6301463479756585, 0.6186106511000703,
            0.6221327065281155, 0.6278273989293488, 0.6303692800373898, 0.6434828011161525, 0.635015327311185,
            0.6525203986055225, 0.6373719823668428, 0.6474128793998063, 0.6532135502904388, 0.6448857511623223,
            0.6438923358450824, 0.6602795453044211, 0.6525802534664524, 0.6570472361215159, 0.6580721662771728,
            0.6583287484389723]),
    seeds_v1_history={
        '20262023': [0.12207944212336323, 0.20173316337638522, 0.26719823170693446, 0.2851478581290309, 0.3248991365596982,
                     0.3475161681467005, 0.364403968998688, 0.3941216677524626, 0.40578788496557505, 0.4260536984054369,
                     0.4255828696061645, 0.4481923865784759, 0.4419262545996353, 0.47711309473359675, 0.476803326442257,
                     0.48597427882623506],
        '20263023': [0.14009763852538182, 0.19000568780029442, 0.2431124875020285, 0.2767225387867991, 0.3021612329325376,
                     0.36724347422604015, 0.36060101275689205, 0.3916482785178582, 0.41230192388133396, 0.41781323000256204,
                     0.4253582523243198, 0.433739647582193, 0.4545556370561058, 0.4549066981765636, 0.46928909065034213,
                     0.48162915462250466]})
if os.environ.get('SEEDS_REFS'):          # local tests only
    REFS.update(json.loads(Path(os.environ['SEEDS_REFS']).read_text()))
REPORT = dict(kernel='casmi26-saner-long',
              task='long-schedule saner seeds (published recipe, epochs 120 / patience 8) on the published holdout + full refits',
              bundle='sanpier/casmi-spectrum-model-v1', extra_member=AHMED_INFO['dataset'], long_seeds=LONG_SEEDS,
              long_epochs=LONG_EPOCHS, long_patience=LONG_PATIENCE, budget_min=BUDGET_MIN,
              timeouts_min=dict(prepare=PREP_TIMEOUT_MIN, holdout=HOLDOUT_TIMEOUT_MIN, refit=REFIT_TIMEOUT_MIN),
              use_time_stopped_holdout=USE_TIME_STOPPED_HOLDOUT, checks={}, errors=[], timeline=[], members=[], holdout={},
              ensembles=[], finals={}, best=None)
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


def check(name, ok, /, **detail):
    REPORT['checks'][name] = dict(ok=ok, **{k: v for k, v in detail.items() if k != 'ok'})
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
    # ahmedberatozer's long dataset carries copies of config.json / model.py / features.py next to config_long.json:
    # never take it for the published bundle.
    hits = [p.parent for p in INPUT.rglob('sha256.json')
            if all((p.parent / f).is_file() for f in ('config.json', 'model.py', 'features.py'))
            and not (p.parent / 'config_long.json').is_file()]
    hits = sorted(set(hits), key=lambda p: ('casmi-spectrum-model-v1' not in str(p), len(p.parts), str(p)))
    if not hits:
        raise SystemExit('SEEDS FATAL: sanpier/casmi-spectrum-model-v1 not mounted (no folder with sha256.json + model.py)')
    return hits[0]


def find_ahmed(B):
    """ahmedberatozer/casmi26-spectrum-model-long: the folder with best_validation_model.pt next to config_long.json (or
    under a path naming the dataset); never the published bundle."""
    hits = [p.parent for p in INPUT.rglob('best_validation_model.pt')
            if p.parent.resolve() != Path(B).resolve()
            and ((p.parent / 'config_long.json').is_file() or 'casmi26-spectrum-model-long' in str(p))]
    hits = sorted(set(hits), key=lambda p: (not (p / 'config_long.json').is_file(), len(p.parts), str(p)))
    return hits[0] if hits else None


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


# ── 3. Patch and the two configs ────────────────────────────────────────────────────────────────────────────
def step_patch():
    log('== 3. the documented patch (prepare.py SMILES alias), the effective config and the long config')
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
    lcfg = dict(eff, epochs=LONG_EPOCHS, patience=LONG_PATIENCE)
    path_overrides = {k: [cfg[k], eff[k]] for k in cfg if cfg[k] != eff[k]}
    overrides = {k: [cfg[k], lcfg[k]] for k in cfg if cfg[k] != lcfg[k]}
    allowed = {'train_path', 'epochs', 'patience'}
    (RW / 'effective_config.json').write_text(json.dumps(eff, indent=2))
    (RW / 'long_config.json').write_text(json.dumps(lcfg, indent=2))
    ok = (not set(path_overrides) - {'train_path'} and not set(overrides) - allowed and set(lcfg) == set(cfg)
          and lcfg['epochs'] == LONG_EPOCHS and lcfg['patience'] == LONG_PATIENCE)
    REPORT['long_config'] = dict(overrides=overrides, file='saner_long/long_config.json')
    check('config', ok, overrides=overrides, allowed_overrides=sorted(allowed), long_epochs=lcfg['epochs'],
          long_patience=lcfg['patience'], published_epochs=cfg['epochs'], published_patience=cfg['patience'], seed=eff['seed'],
          steps_per_epoch=eff['steps_per_epoch'], batch_size=eff['batch_size'], refit_full=eff['refit_full'],
          holdout_hours=eff['holdout_hours'], refit_hours=eff['refit_hours'], work=eff['work'], train_path=eff['train_path'],
          published_seed_matches=eff['seed'] == PUBLISHED_SEED,
          files=dict(selftest_prepare_evaluator='effective_config.json (published + paths)',
                     training_stages='long_config.json (+ epochs / patience)'))
    if eff['seed'] != PUBLISHED_SEED:
        raise SystemExit(f'SEEDS FATAL: config.json seed {eff["seed"]} != {PUBLISHED_SEED}')
    if not ok:
        raise SystemExit(f'SEEDS FATAL: the long config differs from config.json by more than {sorted(allowed)}: {overrides}')
    return eff, lcfg


# ── 4. Structure memo + prepare (F7's launcher without train.run) ──────────────────────────────────────────
PREP_LAUNCHER = r'''"""Prepare launcher for the published saner code (sanpier/casmi-spectrum-model-v1, Apache-2.0), written by the
casmi26-saner-long kernel: the first half of casmi26-saner-repro's launcher (prepare.prepare(cfg) only; the training
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

STAGE_LAUNCHER = r'''"""Stage launcher written by casmi26-saner-long: ONE stage of the published train.train_stage
(sanpier/casmi-spectrum-model-v1, Apache-2.0) on the prepared data, exactly as train.run(cfg) calls it, with
cfg = long_config.json (the published config.json with epochs / patience overridden by the kernel, paths as needed) and
cfg['seed'] = SEED:
  holdout: train_stage(data, cfg, OUT, device, deadline=...)                            (train.run's first call)
  refit:   train_stage(data, cfg, OUT, device, refit=True, epochs=EPOCHS, deadline=...) (train.run's second call;
           train_stage seeds it with SEED + 1)
The deadline is train.run's (holdout_hours / refit_hours of config.json) capped by the kernel's per-step soft limit; a
stage it stops reports time_budget_exhausted=True. Same process set-up as train.run and as casmi26-saner-repro's seed
reference (torch threads, then train.Data(cfg)). load_seconds = start-up time (imports + train.Data), for the kernel's
refit-time prediction."""
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
LOAD_SECONDS = round(time.monotonic() - T_START, 1)
refit = STAGE == 'refit'
hours = cfg['refit_hours'] if refit else cfg['holdout_hours']
deadline = min(time.monotonic() + hours * 3600, T_START + SOFT_SEC)
if refit:
    result = train.train_stage(data, cfg, OUT, device, refit=True, epochs=EPOCHS, deadline=deadline)
else:
    result = train.train_stage(data, cfg, OUT, device, deadline=deadline)
result.update(seed=SEED, stage=STAGE, stage_seed=SEED + (1 if refit else 0), epochs_requested=EPOCHS if refit else cfg['epochs'],
              cfg_epochs=cfg['epochs'], cfg_patience=cfg['patience'], process_seconds=round(time.monotonic() - T_START, 1),
              load_seconds=LOAD_SECONDS, device=str(device))
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
    log('== 4b. launchers (saner_long/launch_prepare.py, saner_long/stage.py):')
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


def combine(score_dicts, qs, mode, weights=None):
    """Per-query ensemble of several models' candidate scores: 'raw' = mean score, 'z' = mean within-query z-score
    (z over the query's candidates). weights: per-model weights, normalised to sum 1 (None = the plain mean)."""
    import numpy as np
    if weights is not None:
        w = np.asarray(weights, np.float64)
        if len(w) != len(score_dicts) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f'bad weights {weights} for {len(score_dicts)} models')
        w = w / w.sum()
    out = {}
    for q in qs:
        arrs = [np.asarray(s[q], np.float64) for s in score_dicts]
        if any(len(a) != len(arrs[0]) for a in arrs):
            raise ValueError(f'query {q}: candidate counts differ between models')
        if mode == 'z':
            arrs = [zscore(a) for a in arrs]
        elif mode != 'raw':
            raise ValueError(mode)
        out[q] = np.mean(arrs, 0) if weights is None else np.sum([wi * a for wi, a in zip(w, arrs)], 0)
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


def config_id(members, mode, weights=None):
    """Row label: the member name for a single model, else 'a+b+c <mode>' (+ ' w=..' for unequal weights)."""
    if len(members) == 1:
        return members[0]
    s = '+'.join(members) + f' {mode}'
    if weights is not None and len({round(float(x), 12) for x in weights}) > 1:
        s += ' w=' + '/'.join(f'{float(x):.3g}' for x in weights)
    return s


def ensemble_rows(member_scores, member_names, qs, idx, ks=None, modes=('raw', 'z'), refs=None, weights=None, group=None):
    """Ensembles of the first k members (k in ks; every prefix by default). For each k and mode: metrics, the paired
    bootstrap of (ensemble - reference) for every reference model (refs: {name: scores}; default: member 0) under 'vs',
    and of (ensemble - mean of its k members) under 'vs_mean_member'. weights: per-member weights (None = equal)."""
    import numpy as np
    K = len(member_scores)
    singles = [per_query(ranks_from_scores(s, qs)) for s in member_scores]
    refq = ({member_names[0]: singles[0]} if refs is None else
            {n: per_query(ranks_from_scores(s, qs)) for n, s in refs.items()})
    rows = []
    for k in (ks or range(1, K + 1)):
        mean_member = {m: np.mean([singles[j][m] for j in range(k)], 0) for m in singles[0]}
        w = None if weights is None else [float(x) for x in weights[:k]]
        for mode in modes:
            r = ranks_from_scores(combine(member_scores[:k], qs, mode, w), qs)
            pq = per_query(r)
            rows.append(dict(config=config_id(list(member_names[:k]), mode, w), group=group, k=k, mode=mode,
                             members=list(member_names[:k]), weights=w, **summarize(r),
                             vs={n: {m: boot_diff(pq[m] - rq[m], idx) for m in pq} for n, rq in refq.items()},
                             vs_mean_member={m: boot_diff(pq[m] - mean_member[m], idx) for m in pq},
                             mean_member_mrr25=float(mean_member['mrr25'].mean())))
    return rows


def pick_best(rows):
    """The configuration with the highest holdout MRR@25 (ties: fewer members, then z). Picked on the same molecules it is
    scored on, so its margin is optimistic."""
    if not rows:
        return None
    b = max(rows, key=lambda r: (r['mrr25'], -r['k'], r['mode'] == 'z'))
    top = sorted(rows, key=lambda r: -r['mrr25'])[:5]
    return dict(config=b['config'], group=b['group'], members=b['members'], mode=b['mode'], weights=b['weights'], k=b['k'],
                mrr25=b['mrr25'], top1=b['top1'], recall25=b['recall25'],
                d_mrr25_vs={n: v['mrr25'] for n, v in b['vs'].items()}, d_top1_vs={n: v['top1'] for n, v in b['vs'].items()},
                d_mrr25_vs_mean_member=b['vs_mean_member']['mrr25'] if b['k'] > 1 else None,
                top5=[dict(config=r['config'], mrr25=r['mrr25']) for r in top], n_configs=len(rows),
                note='picked on the same 1,000 molecules (selection optimism); each long single (and ahmed_long) is also its '
                     'best-epoch checkpoint picked on these molecules, so differences vs a long single are biased against '
                     'the ensemble and long singles vs published16 biased up; CI95: paired bootstrap of per-query MRR@25 '
                     'differences, 5,000 query resamples shared by all rows')


def fmt_d(b, nd=4):
    return f"{b['diff']:+.{nd}f} [{b['ci95'][0]:+.{nd}f}, {b['ci95'][1]:+.{nd}f}] P>0 {b['p_gt0']:.3f}"


def eval_line(r):
    vs = ' | '.join(f"d_mrr vs {n} {fmt_d(b['mrr25'])}" for n, b in r['vs'].items() if r['members'] != [n])
    mm = f" | d_mrr vs mean member {fmt_d(r['vs_mean_member']['mrr25'])}" if r['k'] > 1 else ''
    return (f"SEEDS EVAL {r['config']} [{r['group']}] mrr {r['mrr25']:.5f} top1 {r['top1']:.4f} recall25 {r['recall25']:.4f}"
            + (f' | {vs}' if vs else '') + mm)


def best_line(best):
    vs = best['d_mrr25_vs']
    parts = [f'vs {n} ' + (fmt_d(vs[n]) if n in vs else 'n/a') for n in (PUB, LONG_REF)]
    parts += [f'vs {n} {fmt_d(v)}' for n, v in vs.items() if n not in (PUB, LONG_REF)]
    flag = '' if best.get('ahmed_split_verified') in (None, True) else ' | WARNING ahmed_long split not verified'
    return (f"SEEDS SUMMARY best {best['config']} mrr {best['mrr25']:.5f} top1 {best['top1']:.4f} recall25 {best['recall25']:.4f}"
            f" | {' | '.join(parts)} | of {best['n_configs']} configurations (picked on this set; long singles are "
            f"best-epoch picks on it too){flag}")


def pair_stats(sa, sb, qs, idx=None):
    """Agreement of two checkpoints on the same candidate sets (scores: {query: per-candidate array, gold first}).
    idx: bootstrap index (default boot_index(len(qs)) = BOOT resamples, BOOT_SEED: the configuration table's index, so the
    CI of a pair equals the CI of the matching single row when both use the same queries)."""
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
    idx = boot_index(len(d)) if idx is None else idx
    bt = boot_diff(d, idx)
    return dict(n_queries=len(qs), spearman_per_query_mean=float(np.nanmean(sp)), spearman_per_query_median=float(np.nanmedian(sp)),
                top1_candidate_agreement=float(np.mean(t1)), top25_overlap_mean=float(np.mean(ov)),
                gold_rank_identical=float(np.mean(ra == rb)), mrr25_a=float(mrr_vec(ra).mean()), mrr25_b=float(mrr_vec(rb).mean()),
                mrr25_b_minus_a=bt['diff'], mrr25_diff_ci95=bt['ci95'], mrr25_diff_p_gt0=bt['p_gt0'],
                bootstrap_resamples=int(len(idx)))


def prefix_compare(ours, ref):
    """ours: {epoch: diagnostic MRR@25}; ref: list of the reference's values, epoch 1 first. Compared on shared epochs."""
    got = [(e, ours.get(e), v) for e, v in enumerate(ref, 1) if ours.get(e) is not None and v is not None]
    d = [abs(o - v) for _, o, v in got]
    complete = bool(got) and len(got) == len(ref)
    exact = bool(got) and max(d) == 0
    m4 = bool(got) and all(abs(round(o, 4) - round(v, 4)) <= 1e-6 for _, o, v in got)
    verdict = ('NO_OVERLAP' if not got else 'BIT_EXACT' if exact else 'MATCH_4DP' if m4 else 'DIFFERS')
    if got and not complete and verdict != 'DIFFERS':
        verdict += '_PARTIAL'          # identical on the epochs both runs have; ours stopped earlier
    return dict(compared=len(got), reference_epochs=len(ref), complete=complete, max_abs_diff=max(d) if d else None,
                first_diff_epoch=next((e for e, o, v in got if o != v), None), bit_exact=complete and exact,
                match_4dp=complete and m4, verdict=verdict)


def history_map(path, field='diagnostic_mrr25'):
    """{epoch: value} from a train_stage training_history.json (list of per-epoch records). An epoch cut short by the
    deadline (partial_epoch true) is left out: it is not comparable with a full epoch of another run."""
    h = read_json(path) or []
    return {int(r['epoch']): r.get(field) for r in h
            if isinstance(r, dict) and 'epoch' in r and r.get(field) is not None and not r.get('partial_epoch')}


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
    """Holds the prepared data and the published candidate sets in this process; scores checkpoints on the GPU.
    members: the holdout models in order (published16, ahmed_long, long_<seed>...), the rows of holdout_scores.npz."""

    def __init__(self, B, eff):
        log('== 5. evaluator: prepared data checks, published candidates, published holdout model')
        import torch
        sys.path.insert(0, str(SRC))
        import train as T
        from model import SpectrumCandidateModel
        self.T, self.M, self.B, self.eff = T, SpectrumCandidateModel, B, eff
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = T.Data(eff)
        REPORT['prepared'] = guarded('compare_prepared', compare_prepared, B, eff, self.data)
        self.cands = published_candidates(B, self.data)
        self.qs = sorted(self.cands)
        mk = self.data.mol.key.values
        self.keys = {q: str(mk[q]) for q in self.qs}   # InChIKey14 = the kit's validation_predictions.csv 'key'
        self.scores, self.meta, self.members, self.desc = {}, {}, [], {}
        self._idx = {}
        meta = self.score(PUB, B / 'best_validation_model.pt')
        tr = read_json(B / 'training_result.json') or {}
        want = tr.get('best_diagnostic_mrr25')
        ok = want is not None and abs(meta['mrr25'] - want) <= PUB_MRR_TOL and meta['validate_parity_abs_diff'] <= 1e-9
        check('harness_parity', ok, published_training_result=want, recomputed_here=meta['mrr25'],
              abs_diff=None if want is None else abs(meta['mrr25'] - want), validate_mrr25=meta['validate_mrr25'],
              validate_parity_abs_diff=meta['validate_parity_abs_diff'], top1=meta['top1'], recall25=meta['recall25'],
              state_sha256=meta['state_sha256'], best_epoch=tr.get('best_epoch'))
        pqc = self.per_query_compare(PUB, B / 'validation_predictions.csv')
        check('harness_parity_per_query', None if pqc is None else bool(pqc['keys_identical'] and pqc['rank_identical_frac'] >= 0.99),
              **(pqc or dict(note='validation_predictions.csv not in the bundle')),
              note_keys='holdout_scores.npz key = data.mol.key of the query; OK = same key set and ranks as the published '
                        'validation_predictions.csv')
        REPORT['holdout'][PUB] = dict(source='published best_validation_model.pt', result=tr, eval=meta)
        self.add_member(PUB, f'published sanpier best_validation_model.pt (16 x 500 steps, patience 4, seed {PUBLISHED_SEED})')

    def idx(self, n):
        if n not in self._idx:
            self._idx[n] = boot_index(n)
        return self._idx[n]

    def pair(self, a, b):
        """pair_stats(a, b) on their common queries with the shared bootstrap index (same CI as the configuration table)."""
        qs = self.common_qs([a, b])
        return pair_stats(self.scores[a], self.scores[b], qs, self.idx(len(qs)))

    def score(self, name, path, contaminated=False, parity=True):
        import torch
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
        self.scores[name], self.meta[name] = sc, meta
        return meta

    def common_qs(self, names):
        return [q for q in self.qs if all(q in self.scores[n] for n in names)]

    def per_query_compare(self, name, csv):
        """Our ranks of `name` vs a validation_predictions.csv written by train.validate (columns key, rank), joined by key:
        the identity check of holdout_scores.npz's 'key' column and a per-query parity check."""
        import numpy as np, pandas as pd
        if not Path(csv).is_file():
            return None
        f = pd.read_csv(csv, usecols=['key', 'rank'])
        theirs = dict(zip(f.key.astype(str), f['rank'].astype(int)))
        qn = [q for q in self.qs if q in self.scores[name]]
        ours = dict(zip([self.keys[q] for q in qn], ranks_from_scores(self.scores[name], qn).tolist()))
        common = sorted(set(ours) & set(theirs))
        same = sum(ours[k] == theirs[k] for k in common)
        return dict(file=str(csv), csv_rows=len(f), queries=len(ours), keys_identical=set(ours) == set(theirs),
                    keys_common=len(common), only_csv=len(set(theirs) - set(ours)), only_ours=len(set(ours) - set(theirs)),
                    rank_identical=same, rank_identical_frac=same / max(1, len(common)),
                    csv_mrr25=float(np.mean(mrr_vec(list(theirs.values())))) if theirs else None)

    def add_member(self, name, desc, new_rows=True):
        self.members.append(name)
        self.desc[name] = desc
        REPORT['members'] = list(self.members)
        guarded('holdout_scores', self.write_scores)
        if new_rows:
            self.log_configs(new=name)

    def add_ahmed(self):
        log(f'== 5b. extra holdout member {AHMED} ({AHMED_INFO["dataset"]}; not trained here)')
        A = find_ahmed(self.B)
        if A is None:
            check(f'{AHMED}_member', None, skipped=True, reason=f'dataset {AHMED_INFO["dataset"]} not mounted')
            return None
        cfg_a = read_json(A / 'config_long.json') or read_json(A / 'config.json') or {}
        arch = {k: [self.eff.get(k), cfg_a.get(k)] for k in ARCH_KEYS if cfg_a.get(k) != self.eff.get(k)}
        pins = REPORT.get('bundle_sha256') or {}
        code = {f: (sha256(A / f) == pins.get(f)) if (A / f).is_file() else None
                for f in ('model.py', 'features.py', 'train.py', 'prepare.py', 'config.json', 'inference.py')}
        sched = {k: cfg_a.get(k) for k in ('seed', 'epochs', 'steps_per_epoch', 'patience', 'batch_size', 'lr')}
        if not cfg_a or arch:
            check(f'{AHMED}_member', None, skipped=True, dir=str(A), reason='config_long.json missing or architecture differs',
                  architecture_differences=arch, code_identical_to_bundle=code)
            return None
        try:
            meta = self.score(AHMED, A / 'best_validation_model.pt')
        except Exception as e:
            check(f'{AHMED}_member', None, skipped=True, dir=str(A), reason=f'scoring failed: {e!r}', code_identical_to_bundle=code)
            REPORT['errors'].append(dict(step=AHMED, error=repr(e), trace=traceback.format_exc()[-3000:]))
            return None
        tr = read_json(A / 'training_result.json') or {}
        pqc = self.per_query_compare(AHMED, A / 'validation_predictions.csv')
        ps = self.pair(PUB, AHMED)
        split_ok = None if pqc is None else bool(pqc['keys_identical'])
        REPORT['ahmed_split_verified'] = split_ok
        meta.update(vs_published16=ps, training_result=tr, per_query_vs_his_predictions=pqc)
        check(f'{AHMED}_member', True, dir=str(A), schedule=sched, code_identical_to_bundle=code, checkpoint_epoch=meta['epoch'],
              file_sha256=meta['file_sha256'], state_sha256=meta['state_sha256'], split_identical=split_ok,
              note='split_identical = his validation_predictions.csv has exactly our 1,000 query keys (same holdout molecules)')
        want, t1 = AHMED_INFO['readme_mrr25'], AHMED_INFO['readme_top1']
        his = tr.get('best_diagnostic_mrr25')
        check(f'{AHMED}_readme_mrr', None, readme_mrr25=want, recomputed_mrr25=meta['mrr25'],
              match_3dp=round(meta['mrr25'], 3) == want, readme_top1=t1, recomputed_top1=meta['top1'],
              top1_match_3dp=round(meta['top1'], 3) == t1, training_result_best_mrr25=his,
              abs_diff_vs_training_result=None if his is None else abs(meta['mrr25'] - his), best_epoch=tr.get('best_epoch'),
              validate_parity_abs_diff=meta['validate_parity_abs_diff'], per_query=pqc,
              note=f"his cache had {AHMED_INFO['cache_spectra']:,} spectra vs the published {AHMED_INFO['published_spectra']:,}: "
                   'tiny differences are possible')
        REPORT['holdout'][AHMED] = dict(source=f'{AHMED_INFO["dataset"]} best_validation_model.pt ({A})', result=tr, eval=meta)
        log(f"SEEDS EVAL single {AHMED} mrr {meta['mrr25']:.5f} top1 {meta['top1']:.4f} recall25 {meta['recall25']:.4f} | "
            f"d_mrr vs {PUB} {ps['mrr25_b_minus_a']:+.4f} [{ps['mrr25_diff_ci95'][0]:+.4f}, {ps['mrr25_diff_ci95'][1]:+.4f}] "
            f"P>0 {ps['mrr25_diff_p_gt0']:.3f} | "
            f"spearman vs {PUB} {ps['spearman_per_query_mean']:.4f} top1-agree {ps['top1_candidate_agreement']:.3f} | "
            f"checkpoint epoch {meta['epoch']} | readme 0.718 | split verified {split_ok}")
        self.add_member(AHMED, f"{AHMED_INFO['dataset']} best_validation_model.pt ({AHMED_INFO['epochs']} x "
                               f"{AHMED_INFO['steps_per_epoch']} steps, patience {AHMED_INFO['patience']}, seed "
                               f"{AHMED_INFO['seed']}; not trained here)")
        return meta

    def add_holdout(self, h):
        name = h['name']
        meta = self.score(name, h['path'])
        ps = self.pair(PUB, name)
        ref = self.pair(LONG_REF, name) if LONG_REF in self.scores and name != LONG_REF else None
        res = h.get('result') or {}
        train_mrr = res.get('best_diagnostic_mrr25')
        # Selection optimism: best_epoch is the argmax of this same diagnostic MRR@25 on these 1,000 queries. The mean of
        # the training-time values over epochs best-2..best+2 (full epochs only) is a less biased single-model estimate.
        hm = history_map(Path(h['dir']) / 'training_history.json')
        be = res.get('best_epoch') or 0
        win = [hm[e] for e in range(be - 2, be + 3) if e in hm] if be else []
        meta.update(vs_published16=ps, vs_long_ref=ref, train_time_mrr25=train_mrr,
                    train_eval_parity_abs_diff=None if train_mrr is None else abs(train_mrr - meta['mrr25']),
                    history_mrr25_best_pm2=float(sum(win) / len(win)) if win else None, history_best_pm2_epochs=len(win),
                    selection_note='best-epoch checkpoint picked by diagnostic MRR@25 on these same queries (optimistic); '
                                   'history_mrr25_best_pm2 = mean diagnostic MRR@25 over epochs best-2..best+2')
        REPORT['holdout'][name]['eval'] = meta
        log(f"SEEDS EVAL single {name} mrr {meta['mrr25']:.5f} top1 {meta['top1']:.4f} recall25 {meta['recall25']:.4f} | "
            f"d_mrr vs {PUB} {ps['mrr25_b_minus_a']:+.4f} [{ps['mrr25_diff_ci95'][0]:+.4f}, {ps['mrr25_diff_ci95'][1]:+.4f}] "
            f"P>0 {ps['mrr25_diff_p_gt0']:.3f} | "
            f"spearman vs {PUB} {ps['spearman_per_query_mean']:.4f} top1-agree {ps['top1_candidate_agreement']:.3f} | "
            + (f"spearman vs {LONG_REF} {ref['spearman_per_query_mean']:.4f} | " if ref else '')
            + f"best_epoch {res.get('best_epoch')} completed {res.get('completed_epochs')} stopped_by {h.get('stopped_by')} | "
              f"train-time mrr {train_mrr} | best+-2 mean "
              + (f"{meta['history_mrr25_best_pm2']:.5f}" if win else 'n/a') + f" (n={len(win)}) | "
              f"validate parity {meta['validate_parity_abs_diff']:.2e}")
        self.add_member(name, f'trained here: published recipe with epochs {LONG_EPOCHS}, patience {LONG_PATIENCE}, seed '
                              f'{h["seed"]} (best epoch {res.get("best_epoch")}, completed {res.get("completed_epochs")}, '
                              f'stopped by {h.get("stopped_by")})')
        return meta

    def configs(self):
        """Every configuration of the members present (docstring step 7), on the queries all members share."""
        names = list(self.members)
        qn = self.common_qs(names)
        idx = self.idx(len(qn))
        refs = {n: self.scores[n] for n in (PUB, LONG_REF, AHMED) if n in names}
        longs = [n for n in names if n.startswith('long_')]
        ahm = [AHMED] if AHMED in names else []
        rows = []

        def add(group, members, k, modes, weights=None):
            rows.extend(ensemble_rows([self.scores[m] for m in members], members, qn, idx, ks=[k], modes=modes, refs=refs,
                                      weights=weights, group=group))

        for n in names:                                                         # (a) singles
            add('single', [n], 1, ('raw',))
        for k in range(2, len(longs) + 1):                                      # (b) our long seeds, k = 2..n
            add('long_seeds', longs, k, ('raw', 'z'))
        if PUB in names:
            for n in ahm + longs:                                               # (c) published16 + one long model
                add('pub16_plus_one', [PUB, n], 2, ('z',))
            if len(longs) >= 2:                                                 # (d) published16 + our long seeds
                add('pub16_plus_long_seeds', [PUB] + longs, len(longs) + 1, ('z',))
                add('pub16_plus_long_seeds', [PUB] + longs, len(longs) + 1, ('z',), [0.5] + [0.5 / len(longs)] * len(longs))
        if ahm:
            for n in longs:                                                     # (e) ahmed_long + one seed
                add('ahmed_plus_one', [AHMED, n], 2, ('raw', 'z'))
            if len(longs) >= 2:                                                 # all-long = ahmed_long + our seeds
                add('all_long', [AHMED] + longs, len(longs) + 1, ('raw', 'z'))
            if PUB in names and longs:                                          # published16 + all-long
                m = [PUB, AHMED] + longs
                add('pub16_plus_all_long', m, len(m), ('z',))
                add('pub16_plus_all_long', m, len(m), ('z',), [0.5] + [0.5 / (len(m) - 1)] * (len(m) - 1))
        return rows

    def log_configs(self, new=None):
        rows = self.configs()
        REPORT['ensembles'] = rows
        for r in rows:
            if new is None or new in r['members']:
                log(eval_line(r))
        save_report()
        return rows

    def write_scores(self):
        """holdout_scores.npz (+ holdout_scores.json layout): every holdout member's per-candidate scores on the queries all
        members share, in query-index order (the bootstrap's order); rewritten after every member."""
        import numpy as np
        names = list(self.members)
        qn = self.common_qs(names)
        lens = np.array([len(self.cands[q]) for q in qn], np.int64)
        offsets = np.r_[0, np.cumsum(lens)].astype(np.int64)
        cand = np.concatenate([np.asarray(self.cands[q], np.int64) for q in qn])
        label = (cand == np.repeat(np.asarray(qn, np.int64), lens)).astype(np.int8)
        gold_first = bool((label[offsets[:-1]] == 1).all() and int(label.sum()) == len(qn))
        keys = [self.keys[q] for q in qn]
        arr = dict(key=np.array(keys, dtype=str), query_mol_idx=np.asarray(qn, np.int32), offsets=offsets,
                   n_candidates=lens.astype(np.int32), candidate_mol_idx=cand.astype(np.int32), label=label,
                   members=np.array(names, dtype=str))
        for n in names:
            arr[f'score__{n}'] = np.concatenate([np.asarray(self.scores[n][q], np.float32) for q in qn])
        tmp = SCORES_PATH.with_name('holdout_scores.tmp.npz')
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, SCORES_PATH)
        z = np.load(SCORES_PATH, allow_pickle=False)
        o = z['offsets']
        rt = {}
        for n in names:
            s = z[f'score__{n}']
            r = np.array([1 + int(np.sum(s[o[i] + 1:o[i + 1]] >= s[o[i]])) for i in range(len(o) - 1)])
            want = float(mrr_vec(ranks_from_scores(self.scores[n], qn)).mean())
            rt[n] = dict(mrr25=float(mrr_vec(r).mean()), identical_to_evaluator=float(mrr_vec(r).mean()) == want)
        keys_ok = [str(k) for k in z['key']] == keys
        members = {}
        for n in names:
            m = self.meta[n]
            members[n] = dict(array=f'score__{n}', description=self.desc.get(n), mrr25=rt[n]['mrr25'], top1=m.get('top1'),
                              recall25=m.get('recall25'), checkpoint_epoch=m.get('epoch'), file_sha256=m.get('file_sha256'),
                              state_sha256=m.get('state_sha256'), source=m.get('path'))
        layout = dict(
            file=SCORES_PATH.name, kernel='casmi26-saner-long', written_min=minutes(), n_queries=len(qn),
            n_candidates_total=int(offsets[-1]), candidates_per_query=sorted({int(x) for x in lens}), members=members,
            arrays={
                'key': '(n_queries,) str: InChIKey first block (14 chars) of the query molecule = the "key" column of the '
                       "kit's validation_predictions.csv (data.mol.key); one row per query, aligned with offsets",
                'query_mol_idx': '(n_queries,) int32: row of the query in the prepared molecules.parquet (= the published '
                                 'molecules.parquet row order when check validation_candidates.molecule_index_identical)',
                'offsets': '(n_queries + 1,) int64: the candidates of query i are rows offsets[i]:offsets[i + 1] of every '
                           'per-candidate array',
                'n_candidates': '(n_queries,) int32',
                'candidate_mol_idx': '(N,) int32: molecules.parquet row of each candidate (published validation_candidates.npz '
                                     'order, gold first)',
                'label': '(N,) int8: 1 for the gold candidate (always the first of its query), else 0',
                'members': '(n_members,) str: member names in evaluation order',
                'score__<member>': "(N,) float32: the member's candidate score exactly as train.validate computes it"},
            metric='rank = 1 + number of non-gold candidates of the query with score >= the gold score (pessimistic ties, '
                   'train.validate); MRR@25 = mean over queries of (1 / rank if rank <= 25 else 0)',
            ensembles='raw = weighted mean of the members\' scores; z = per query z = (s - mean) / std over its candidates '
                      '(std 0 -> 1), then the weighted mean; weights normalised to sum 1',
            bootstrap=dict(resamples=BOOT, seed=BOOT_SEED, indices='np.random.default_rng(0).integers(0, n_queries, '
                                                                     '(5000, n_queries)), queries in the order of this file'),
            example=('import numpy as np\nz = np.load("holdout_scores.npz")\no, s = z["offsets"], z["score__published16"]\n'
                     'r = np.array([1 + int((s[o[i] + 1:o[i + 1]] >= s[o[i]]).sum()) for i in range(len(o) - 1)])\n'
                     'print(np.where(r <= 25, 1 / r, 0).mean())'),
            roundtrip=rt, gold_first=gold_first, keys_roundtrip=keys_ok)
        SCORES_LAYOUT_PATH.write_text(json.dumps(layout, indent=2, default=_js))
        ok = gold_first and keys_ok and all(v['identical_to_evaluator'] for v in rt.values())
        check('holdout_scores', ok, file=str(SCORES_PATH), bytes=SCORES_PATH.stat().st_size, members=names, n_queries=len(qn),
              candidates_total=int(offsets[-1]), gold_first=gold_first, keys_roundtrip=keys_ok,
              roundtrip_mrr25={n: v['mrr25'] for n, v in rt.items()})

    def summary(self):
        log('== 7. summary: holdout configurations on the published validation candidates')
        import numpy as np
        rows = self.configs()
        REPORT['ensembles'] = rows
        names = list(self.members)
        mat = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                st = self.pair(names[i], names[j])
                mat[f'{names[i]}__{names[j]}'] = dict(spearman=st['spearman_per_query_mean'], top1_agreement=st['top1_candidate_agreement'],
                                                     top25_overlap=st['top25_overlap_mean'])
        REPORT['holdout_pairs'] = mat
        best = pick_best(rows)
        if best:
            best['ahmed_split_verified'] = REPORT.get('ahmed_split_verified') if AHMED in best['members'] else None
            longs = [self.meta[n]['mrr25'] for n in names if n.startswith('long_')]
            best.update(single_mrr25={n: self.meta[n]['mrr25'] for n in names},
                        long_seed_mrr25_mean=float(np.mean(longs)) if longs else None,
                        long_seed_mrr25_sd=float(np.std(longs, ddof=1)) if len(longs) > 1 else None)
        REPORT['best'] = best
        save_report()
        if best:
            log(best_line(best))
        return best

    def score_finals(self):
        log('== 9. final (refit) models: contaminated diagnostics (trained on these molecules; agreement, not quality)')
        pubname = 'final_published16'
        pm = guarded(pubname, self.score, pubname, self.B / 'final_model.pt', contaminated=True, parity=False)
        if pm:
            pin = (REPORT.get('bundle_sha256') or {}).get('final_model.pt')
            check(pubname, None, file='bundle final_model.pt (scored in place, not copied)', file_sha256=pm['file_sha256'],
                  bundle_pin=pin, pin_match=pm['file_sha256'] == pin, contaminated_mrr25=pm['mrr25'], contaminated_top1=pm['top1'])
        for seed in sorted(FINALS):
            name, own = f'final_long_{seed}', f'long_{seed}'
            meta = self.score(name, WORKING / FINALS[seed]['file'], contaminated=True, parity=False)
            row = dict(contaminated_mrr25=meta['mrr25'], contaminated_top1=meta['top1'], state_sha256=meta['state_sha256'])
            ok = None
            if own in self.scores:
                ps = self.pair(own, name)
                row.update(spearman_to_own_holdout=ps['spearman_per_query_mean'], top1_agreement_to_own_holdout=ps['top1_candidate_agreement'],
                           own_holdout_clean_mrr25=self.meta[own]['mrr25'])
                ok = (ps['spearman_per_query_mean'] >= FINAL_SANITY['min_spearman_to_own_holdout']
                      and meta['mrr25'] >= FINAL_SANITY['min_contaminated_mrr25'])
            if pubname in self.scores:
                ps = self.pair(pubname, name)
                row.update(spearman_to_published_final=ps['spearman_per_query_mean'],
                           top1_agreement_to_published_final=ps['top1_candidate_agreement'],
                           top25_overlap_to_published_final=ps['top25_overlap_mean'])
            FINALS[seed]['diagnostics'] = row
            check(f'{name}_sanity', ok, **row, thresholds=FINAL_SANITY)
        write_finals()
        names = [f'final_long_{s}' for s in LONG_SEEDS if f'final_long_{s}' in self.scores]
        if len(names) > 1:
            qn = self.common_qs(names + ([pubname] if pubname in self.scores else []))
            refs = {pubname: self.scores[pubname]} if pubname in self.scores else None
            REPORT['finals_ensemble_contaminated'] = [
                dict(config=r['config'], mrr25=r['mrr25'], top1=r['top1'],
                     d_vs={n: v['mrr25']['diff'] for n, v in r['vs'].items()})
                for r in ensemble_rows([self.scores[n] for n in names], names, qn, self.idx(len(qn)), ks=[len(names)], refs=refs)]
            save_report()


# ── 6. Long holdout stage per seed ──────────────────────────────────────────────────────────────────────────
def stage_dir(stage, name):
    out = RW / f'{stage}_{name}'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def run_stage(stage, name, seed, epochs, timeout_min):
    out = stage_dir(stage, name)
    rc, sec, killed = run_logged([sys.executable, '-u', str(RW / 'stage.py'), str(SRC), str(RW / 'long_config.json'),
                                  str(out), stage, str(seed), str(epochs), str(soft_seconds(timeout_min))],
                                 out / 'stage.log', timeout=timeout_min * 60, cwd=RW)
    return out, rc, sec, killed, read_json(out / 'training_result.json')


def step_holdout(B, seed):
    t = minutes()
    name = f'long_{seed}'
    log(f'== 6. long holdout stage {name} (train.train_stage as train.run calls it; epochs {LONG_EPOCHS}, patience '
        f'{LONG_PATIENCE}; hard timeout {HOLDOUT_TIMEOUT_MIN} min)')
    out, rc, sec, killed, res = run_stage('holdout', name, seed, 0, HOLDOUT_TIMEOUT_MIN)
    best = out / 'best_validation_model.pt'
    time_stopped = bool((res or {}).get('time_budget_exhausted'))
    finished = rc == 0 and res is not None and best.is_file() and (res.get('best_epoch') or 0) >= 1
    cfg_applied = res is not None and res.get('cfg_epochs') == LONG_EPOCHS and res.get('cfg_patience') == LONG_PATIENCE
    cf = out / 'validation_candidates.npz'
    cand_same = cf.is_file() and sha256(cf) == sha256(B / 'validation_candidates.npz')
    # A model whose own candidate file differs from the published one was trained on different prepared data (its split
    # may overlap the evaluation molecules): it is reported but neither evaluated, ensembled nor refit.
    ok = finished and cfg_applied and cand_same and (USE_TIME_STOPPED_HOLDOUT or not time_stopped)
    stopped_by = None if res is None else ('deadline' if time_stopped else
                                           ('epochs' if res.get('completed_epochs') == LONG_EPOCHS else 'patience'))
    h = dict(seed=seed, name=name, ok=ok, path=best, dir=out, result=res, exit=rc, minutes=round(sec / 60, 1), killed=killed,
             time_stopped=time_stopped, stopped_by=stopped_by, config_applied=cfg_applied, candidates_file_identical=cand_same,
             file_sha256=sha256(best) if best.is_file() else None, best_epoch=(res or {}).get('best_epoch'),
             completed_epochs=(res or {}).get('completed_epochs'))
    check(f'holdout_{name}', ok, **{k: v for k, v in h.items() if k not in ('path', 'dir', 'ok')})
    REPORT['holdout'][name] = dict(source=f'trained here ({out.name})', **{k: v for k, v in h.items() if k not in ('path', 'dir')})
    stamp(f'holdout_{name}', t, 'ok' if ok else 'failed')
    return h


def check_history(B, h):
    """INFO checks on the long run's training_history.json. (1) determinism_<name>: the first 16 epochs vs the 16-epoch run
    of the same seed (bgunshui's 80-epoch run shows the schedule does not depend on cfg['epochs']). (2) For seed 20261023:
    bgunshui's 80-epoch history (epochs 1-56; ours continues: epochs 120, patience 8)."""
    seed, name = h['seed'], h['name']
    mrr = history_map(h['dir'] / 'training_history.json')
    top1 = history_map(h['dir'] / 'training_history.json', 'top1')
    res = h.get('result') or {}
    base = dict(epochs_recorded=len(mrr), best_epoch=res.get('best_epoch'), completed_epochs=res.get('completed_epochs'),
                stopped_by=h.get('stopped_by'))
    if seed == PUBLISHED_SEED:
        pub = history_map(Path(B) / 'training_history.json')
        ref = [pub[e] for e in sorted(pub)] if pub and sorted(pub) == list(range(1, len(pub) + 1)) else []
        src = f'published training_history.json (16 x 500 steps, seed {PUBLISHED_SEED})'
    else:
        ref = list((REFS.get('seeds_v1_history') or {}).get(str(seed)) or [])
        src = 'casmi26-saner-seeds v1 holdout history of this seed (16 x 500 steps)'
    cmp = prefix_compare(mrr, ref)
    verdict = 'NO_HISTORY' if not mrr else 'NO_REFERENCE' if not ref else cmp.pop('verdict')
    cmp.pop('verdict', None)
    check(f'determinism_{name}', None, verdict=verdict, reference=src, **cmp, **base,
          note='first epochs of the long run vs the 16-epoch run of the same seed; DIFFERS = GPU nondeterminism or a '
               'schedule that depends on cfg["epochs"], not a broken run')
    b = REFS.get('bgunshui') or {}
    if seed != b.get('seed'):
        return
    rows = {}
    for e, v in sorted((int(k), v) for k, v in b['mrr25_4dp'].items()):
        o = mrr.get(e)
        rows[e] = dict(ours=o, ours_4dp=None if o is None else round(o, 4), bgunshui=v,
                       match=o is not None and abs(round(o, 4) - v) <= 1e-6)
    t1 = {}
    for e, v in sorted((int(k), v) for k, v in (b.get('top1_4dp') or {}).items()):
        o = top1.get(e)
        t1[e] = dict(ours=o, bgunshui=v, match=o is not None and abs(round(o, 4) - v) <= 1e-6)
    full = prefix_compare(mrr, b.get('history_mrr25') or [])
    be = b['best_epoch']
    e52 = mrr.get(be)
    allr = list(rows.values()) + list(t1.values())
    have = [r for r in allr if r['ours'] is not None]
    verdict = ('NO_HISTORY' if not have else 'DIFFERS' if not all(r['match'] for r in have) else
               'MATCH' if len(have) == len(allr) else 'MATCH_PARTIAL')     # PARTIAL: ours stopped before the last epoch
    check(f'bgunshui_history_{name}', None, verdict=verdict, reference=b.get('kernel'), epochs=rows,
          top1=t1, **{f'epoch{be}_ours': e52, f'epoch{be}_bgunshui': b['best_diagnostic_mrr25'],
                      f'epoch{be}_within_1e6': e52 is not None and abs(e52 - b['best_diagnostic_mrr25']) <= 1e-6},
          full_history=full, bgunshui_run=dict(epochs=b['epochs'], patience=b['patience'], best_epoch=be,
                                               completed_epochs=b['completed_epochs'], best_mrr25=b['best_diagnostic_mrr25']),
          ours=base, note=f'same seed, data and code; ours runs on to epochs {LONG_EPOCHS} / patience {LONG_PATIENCE}, so '
                          f'only its first {b["completed_epochs"]} epochs are comparable; values compared at 4 decimals')


# ── 8. Refits ───────────────────────────────────────────────────────────────────────────────────────────────
def write_finals():
    FINALS_PATH.write_text(json.dumps({v['file']: v for _, v in sorted(FINALS.items())}, indent=2, default=_js))
    REPORT['finals'] = {f'long_{s}': v for s, v in sorted(FINALS.items())}
    save_report()


def publish_final(seed, src, **info):
    name = f'final_model_long{seed}.pt'
    dst = WORKING / name
    shutil.copy2(src, dst)
    fs = sha256(dst)
    (WORKING / f'{name}.sha256').write_text(f'{fs}  {name}\n')
    FINALS[seed] = dict(file=name, file_sha256=fs, state_sha256=checkpoint_state_sha(dst), seed=seed, refit_seed=seed + 1,
                        member=f'long_{seed}', long_epochs=LONG_EPOCHS, long_patience=LONG_PATIENCE,
                        bytes=dst.stat().st_size, **info)
    write_finals()
    return FINALS[seed]


def refit_prediction(h, epochs):
    """Seconds a refit of `epochs` epochs needs at the speed of the seed's own holdout: the holdout's start-up time
    (stage.py load_seconds: imports + train.Data, as in the refit process) + the training seconds of epochs 1..epochs
    (train_stage's per-epoch 'seconds', which leave out validation; the refit trains the same steps per epoch;
    casmi26-saner-seeds: refit 16.7 s vs holdout 16.6 s per epoch). An epoch the history lacks or holds only partially
    (deadline) counts at the mean of the full epochs. None without a usable history. fits: the prediction is inside the
    refit's soft deadline."""
    hist = read_json(Path(h['dir']) / 'training_history.json') or []
    full = {int(r['epoch']): float(r['seconds']) for r in hist
            if isinstance(r, dict) and 'epoch' in r and r.get('seconds') is not None and not r.get('partial_epoch')}
    if not full or not epochs:
        return None
    mean = sum(full.values()) / len(full)
    train = sum(full.get(e, mean) for e in range(1, int(epochs) + 1))
    load = (h.get('result') or {}).get('load_seconds')
    load = float(load) if load is not None else REFIT_LOAD_SEC
    soft = soft_seconds(REFIT_TIMEOUT_MIN)
    return dict(predicted_min=round((load + train) / 60, 2), train_min=round(train / 60, 2), load_seconds=load,
                epochs=int(epochs), epochs_measured=len(full), mean_epoch_seconds=round(mean, 2), soft_min=round(soft / 60, 2),
                fits=load + train <= soft)


def step_refit(seed, epochs, h=None):
    t = minutes()
    name = f'long_{seed}'
    pred = refit_prediction(h, epochs) if h else None
    log(f'== 8. full refit {name} (train_stage refit=True, seed {seed + 1}, {epochs} epochs = the long holdout best_epoch; '
        f'hard timeout {REFIT_TIMEOUT_MIN} min; predicted {None if pred is None else pred["predicted_min"]} min)')
    if pred is not None and not pred['fits']:
        # It would be stopped by its own deadline and thrown away: keep the GPU time (the holdout model is kept instead).
        check(f'refit_{name}', False, skipped=True, epochs=epochs, refit_seed=seed + 1, prediction=pred,
              reason='predicted refit time exceeds its soft deadline at the speed of its own holdout')
        stamp(f'refit_{name}', t, 'skipped')
        return False
    out, rc, sec, killed, res = run_stage('refit', name, seed, epochs, REFIT_TIMEOUT_MIN)
    fm = out / 'final_model.pt'
    complete = (rc == 0 and res is not None and fm.is_file() and res.get('completed_epochs') == epochs
                and not res.get('time_budget_exhausted'))
    detail = dict(exit=rc, minutes=round(sec / 60, 1), killed=killed, epochs=epochs, refit_seed=seed + 1, result=res,
                  prediction=pred)
    if complete:
        f = publish_final(seed, fm, epochs=epochs, source=f'refit trained here (published train_stage refit, seed {seed + 1}, '
                                                           f'epochs = holdout best_epoch)', refit_minutes=round(sec / 60, 1))
        detail.update(file=f['file'], file_sha256=f['file_sha256'], state_sha256=f['state_sha256'])
    check(f'refit_{name}', complete, **detail)
    stamp(f'refit_{name}', t, 'ok' if complete else 'failed')
    return complete


# ── finish ──────────────────────────────────────────────────────────────────────────────────────────────────
def keep_holdout_models():
    """Long holdout checkpoints that would otherwise be lost. A long holdout's best_validation_model.pt is copied to
    /kaggle/working/holdout_model_long<s>.pt (+ .sha256) when that seed has no published final (refit failed, stopped,
    skipped or never run) or when the model is not in holdout_scores.npz (evaluator or scoring failed, or the holdout was
    not used). Returns ({member: info}, {source paths whose copy failed: never deleted})."""
    lay = read_json(SCORES_LAYOUT_PATH) or {}
    scored = set(lay.get('members') or {})        # written after the npz: never lists a member the npz lacks
    c = REPORT['checks']
    kept, failed = {}, set()
    for s in LONG_SEEDS:
        name = f'long_{s}'
        src = RW / f'holdout_{name}' / 'best_validation_model.pt'
        if not src.is_file():
            continue
        reasons = ([] if s in FINALS else ['no published final (refit failed, stopped, skipped or not run)']) + \
                  ([] if name in scored else ['not in holdout_scores.npz'])
        if not reasons:
            continue
        dst = WORKING / f'holdout_model_long{s}.pt'
        try:
            shutil.copy2(src, dst)
            fs = sha256(dst)
            (WORKING / f'{dst.name}.sha256').write_text(f'{fs}  {dst.name}\n')
            try:
                ss = checkpoint_state_sha(dst)
            except Exception as e:          # e.g. a file cut short by a hard kill: kept anyway, flagged
                ss = f'unreadable: {e!r}'
            hc = c.get(f'holdout_{name}') or {}
            kept[name] = dict(file=dst.name, file_sha256=fs, state_sha256=ss, bytes=dst.stat().st_size, reasons=reasons,
                              holdout_ok=hc.get('ok'), time_stopped=hc.get('time_stopped'), best_epoch=hc.get('best_epoch'),
                              completed_epochs=hc.get('completed_epochs'), in_holdout_scores=name in scored,
                              final_published=s in FINALS,
                              note='holdout only, no successful refit' if s not in FINALS else
                                   'refit published; this holdout model is not in holdout_scores.npz')
        except Exception as e:
            failed.add(src)
            REPORT['errors'].append(dict(step=f'keep_{name}', error=repr(e), trace=traceback.format_exc()[-3000:]))
            log(f'SEEDS ERROR keeping {src}: {e!r} (left in place)')
    REPORT['kept_holdout_models'] = kept
    for n, v in kept.items():
        log(f"SEEDS KEPT {n} -> {v['file']} sha256 {v['file_sha256']} best_epoch {v['best_epoch']} | {'; '.join(v['reasons'])}")
    save_report()
    return kept, failed


def step_finish(eff):
    stage_dirs = sorted(RW.glob('holdout_*')) + sorted(RW.glob('refit_*'))
    shas = {}
    for d in stage_dirs:
        shas.update({f'{d.name}/{p.name}': sha256(p) for p in sorted(d.iterdir()) if p.is_file()})
    (RW / 'stage_outputs_sha256.json').write_text(json.dumps(shas, indent=2))
    REPORT['stage_outputs_sha256'] = shas
    # Long holdout models without a published final or without holdout scores are copied to /kaggle/working first. If that
    # step itself fails, no best_validation_model.pt is deleted.
    try:
        kept, keep_failed = keep_holdout_models()
    except Exception as e:
        REPORT['errors'].append(dict(step='keep_holdout_models', error=repr(e), trace=traceback.format_exc()[-3000:]))
        log(f'SEEDS ERROR in keep_holdout_models: {e!r}; every best_validation_model.pt is left in place')
        kept, keep_failed = {}, None
    # Output size: the stage checkpoints (sha256 above; the finals and kept holdout models are in /kaggle/working, the
    # holdout scores in holdout_scores.npz) and the whole training cache except data_audit.json are deleted.
    dropped = 0
    for d in stage_dirs:
        for p in sorted(d.glob('*.pt')) + sorted(d.glob('validation_candidates.npz')):
            if p.name == 'best_validation_model.pt' and (keep_failed is None or p in keep_failed):
                continue
            dropped += p.stat().st_size; p.unlink()
    REPORT['stage_files_deleted_bytes'] = dropped
    work = Path(eff['work']) if eff else None
    freed = 0
    if work is not None and work.is_dir():
        for p in sorted(work.iterdir()):
            if p.is_file() and p.name != 'data_audit.json':
                freed += p.stat().st_size; p.unlink()
    REPORT['cache_deleted_bytes'] = freed
    files = {str(p.relative_to(WORKING)): p.stat().st_size for p in WORKING.rglob('*') if p.is_file()}
    total = sum(files.values())
    kept_bytes = sum(files.get(v['file'], 0) for v in kept.values())
    # The limit is for the regular outputs; holdout models kept on a failure path (at most 3 x ~12 MB) are counted apart.
    check('output_size', total - kept_bytes <= MAX_OUTPUT_MB * 1e6, megabytes=round(total / 1e6, 2),
          kept_holdout_models_mb=round(kept_bytes / 1e6, 2), limit_mb=MAX_OUTPUT_MB, files=len(files),
          largest=sorted(files.items(), key=lambda x: -x[1])[:8])
    c = REPORT['checks']
    must = ['published_sha256', 'patch', 'config', 'struct_memo', 'prepare_run', 'prepare_data_audit', 'prepare_molecules',
            'prepare_spectrum_manifest', 'validation_candidates', 'harness_parity']
    hold_ok = [s for s in LONG_SEEDS if c.get(f'holdout_long_{s}', {}).get('ok')]
    refit_ok = [s for s in LONG_SEEDS if c.get(f'refit_long_{s}', {}).get('ok')]
    best = REPORT.get('best') or {}
    REPORT['verdict'] = dict(setup_ok=all(c.get(k, {}).get('ok') for k in must), setup_failed=[k for k in must if not c.get(k, {}).get('ok')],
                             members=REPORT.get('members'), ahmed_long=c.get(f'{AHMED}_member', {}).get('ok'),
                             ahmed_readme_match_3dp=c.get(f'{AHMED}_readme_mrr', {}).get('match_3dp'),
                             holdouts_ok=hold_ok, time_stopped=[s for s in LONG_SEEDS if c.get(f'holdout_long_{s}', {}).get('time_stopped')],
                             refits_ok=refit_ok, refits_missing=[s for s in hold_ok if s not in refit_ok],
                             final_models=sorted(v['file'] for v in FINALS.values()),
                             holdout_models_kept=sorted(v['file'] for v in kept.values()),
                             best_config=best.get('config'), best_mrr25=best.get('mrr25'),
                             bgunshui_history=c.get(f'bgunshui_history_{LONG_REF}', {}).get('verdict'),
                             determinism={f'long_{s}': c.get(f'determinism_long_{s}', {}).get('verdict') for s in LONG_SEEDS},
                             holdout_scores_ok=c.get('holdout_scores', {}).get('ok'),
                             output_mb=c.get('output_size', {}).get('megabytes'), errors=len(REPORT['errors']), minutes=minutes())
    save_report()
    log('SEEDS SUMMARY ' + json.dumps(REPORT['verdict'], default=_js))
    for n in REPORT.get('members') or []:
        hd = REPORT['holdout'].get(n) or {}
        e = hd.get('eval') or {}
        if e:
            vp = e.get('vs_published16')
            log(f"SEEDS SUMMARY single {n} mrr {e['mrr25']:.5f} top1 {e['top1']:.4f} recall25 {e['recall25']:.4f}"
                + (f" spearman vs {PUB} {vp['spearman_per_query_mean']:.4f}" if vp else '')
                + (f" best_epoch {hd['best_epoch']} completed {hd.get('completed_epochs')}" if hd.get('best_epoch') else '')
                + (f" best+-2 mean {e['history_mrr25_best_pm2']:.5f}" if e.get('history_mrr25_best_pm2') is not None else ''))
    for r in REPORT.get('ensembles') or []:
        log('SEEDS SUMMARY ' + eval_line(r)[len('SEEDS '):])
    if best:
        log(best_line(best))
    for s, v in sorted(FINALS.items()):
        log(f"SEEDS SUMMARY final long_{s} {v['file']} sha256 {v['file_sha256']} epochs {v.get('epochs')} "
            f"diag {json.dumps(v.get('diagnostics'), default=_js)}")
    for n, v in kept.items():
        log(f"SEEDS SUMMARY kept holdout model {n} {v['file']} sha256 {v['file_sha256']} best_epoch {v['best_epoch']} | {v['note']}")
    log(f'SEEDS DONE {minutes()} min; report {REPORT_PATH}; finals {FINALS_PATH}; holdout scores {SCORES_PATH}')


def main():
    RW.mkdir(parents=True, exist_ok=True)
    eff = None
    try:
        B = step_bundle()
        step_env(B)
        eff, _ = step_patch()
        if not step_prepare(eff):
            log('SEEDS FATAL: prepare did not complete; nothing to train')
            return
        ev = guarded('evaluator', Evaluator, B, eff)
        if ev:
            guarded(AHMED, ev.add_ahmed)
        # 6. long holdouts first (with evaluation); the third keeps room for the refits of the seeds already trained
        done = []
        for i, s in enumerate(LONG_SEEDS, start=1):
            need = HOLDOUT_TIMEOUT_MIN + (REFIT_TIMEOUT_MIN * len(done) if i > MIN_LONG_HOLDOUTS else 0)
            if not fits(f'holdout_long_{s}', need):
                continue
            h = guarded(f'holdout_long_{s}', step_holdout, B, s)
            if not h:
                continue
            guarded(f'history_long_{s}', check_history, B, h)
            if ev and h['ok']:
                guarded(f'eval_long_{s}', ev.add_holdout, h)
            if h['ok']:
                done.append(h)
        if ev:
            guarded('summary', ev.summary)
        # 8. refits of the long seeds whose holdout completed (published recipe: epochs = that seed's holdout best_epoch)
        for h in done:
            if fits(f'refit_long_{h["seed"]}', REFIT_TIMEOUT_MIN):
                guarded(f'refit_long_{h["seed"]}', step_refit, h['seed'], int(h['best_epoch']), h)
        # 9. contaminated diagnostics of the final models
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
