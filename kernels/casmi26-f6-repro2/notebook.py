"""casmi26-f6-repro2 (F6 K2: docs/f6_engine1_chain.md section 6, ranks 6 and 8; assets A7 / A8 / A9): tie every published
FPNet weight file of Ahmed Berat Ozer's Engine 1 chain to its own training log on a Kaggle T4, and recover the two FPNet
seeds he did not record from the step-0 batch lines of their logs. Nothing here feeds a submission. His code is run as
published: the repository layout is restored by copies + symlinks and no source file is edited.

Files under test (sha256 against casmi26-v4b-models/MANIFEST.json and the README list of casmi26-training-code):
  fe_A.pt = v4b fpnet_0.pt = v3 fpnet_0.pt   FPNet A, seed 7 (default), logs/log_train_a.txt, best step 38,000
  fe_B.pt = v3 fpnet_1.pt                     FPNet B, seed NOT recorded, logs/log_train_b.txt, best step 40,000
  fpnet_full1.pt (casmi26-fpnet-full1)        --holdout "" (reconstructed), seed NOT recorded, logs/log_train_full1.txt
  fe_D.pt (DreamsFP, optional part)           dreams_ft/logs/pilot_d1.txt + the frozen table dreams_ft/s2_eval/D20k_report.txt

Steps; each prints 'F6 CHECK <name>: OK | FAIL | INFO {...}'; the record is /kaggle/working/repro_report.json.
 1. Inputs: v4b MANIFEST (every file), README sha list (fe_A / fe_B / fe_D / fpnet_full1; v3 fpnet_0 / fpnet_1 must equal
    fe_A / fe_B), the torch archive name inside each .pt (= the file name it was saved under: fpnet_a / fpnet_b /
    fpnet_full1 / pilot_d1_step20k_ema), training-code per-file pins + whole-tree hash (as K1), D20k_eval_info.json names
    the same three sha256 values, spec-cache README list, v2-pool fp_bits pin, competition train.parquet (sanpier's
    data_audit value), and the casmi code copies (training code = v4b = v3).
 2. Environment + bf16 probe. His trainer and evaluators hard-code torch.autocast(bfloat16); a T4 (sm_75) has no bf16
    tensor cores. If the probe fails (or bf16 is emulated below BF16_MIN_TFLOPS), every bf16 autocast request runs as fp16
    through one printed runtime patch (and the report says so); otherwise nothing is patched.
 3. Layout, no source edits: src/casmi, scripts/, work/analysis/{fpnet_v2,dreams_ft} are copies (sha-checked); work/*
    are symlinks to spec-cache + v2-pool. v2lib.py hard-codes ROOT = 'C:/Users/omen/Desktop/envedaCasmi'; on Linux that
    is a relative path, so a directory of that name inside the repository (holding symlinks src and work) lets the file
    run as published.
 4. File against log, harness process 'eval' (his functions, our driver):
    - A / B / full1: traindata.window_mrr (his VAL metric) on the validation sets 06_train_fpnet.py builds (np lib-9
      spectra, nplib 2,000 and syn 1,500 drawn with rng 123) vs (a) the logged VAL line of the step the file holds and
      (b) the full-precision 'val' dict saved inside the file. Tolerance 0.002 (bf16 on another GPU: rank ties flip).
    - fe_D: VAL np-dev / np-dev-merged / nplib exactly as train_dreams_fp.validate vs the step-20,000 line (published
      file) and the step-18,000 line (the trainer's unpublished 'best'): it must match 20k. Header: step 20,000, val None,
      args = pilot_d1_summary.json args.
    - Frozen v2lib.Evaluator (unmodified) for A, B, D20k and the logit-sum ensembles: query-level table and the
      unweighted per-structure rows of D20k_report.txt. The weighted rows need truth_g.parquet, which is not published.
    - fp32 sensitivity (optional): A's VAL with autocast off, to size the numeric part of any difference.
 5. Step-0 replay of A with seed 7 by his own script, unmodified: scripts/06_train_fpnet.py --steps 1 (seed 7 is the
    default; --steps does not change anything before the step-0 print, and the lr there is 0). Its [Data] / params / val
    sets lines must equal log_train_a.txt; the step-0 line is compared numerically: dropout masks are drawn on the GPU
    with a launch grid that depends on the SM count, so a T4 cannot redraw his RTX 5090 masks. The comparison uses the
    dropout spread measured in step 6 (12 redraws).
 6. Seed recovery, harness process 'step0': per seed s exactly the script's path up to the step-0 print
    (torch.manual_seed(s); FPNet init on the CPU; Data.rng = default_rng(s); sample_ids; batch; autocast forward in train
    mode; bce / ctr / top1 / el). Data.__init__ draws no random number, so one Data object per holdout serves all seeds,
    and one model init serves both holdouts (torch.cuda.manual_seed(s) resets the dropout generator before each forward).
    The forward runs under torch.no_grad() (same values; a bf16-emulated T4 graph would need ~10 GB).
    --K per logged line: at init the scores are flat, so the step-0 ctr is ln(K + 1) + O(0.005) for any seed. A / full1
    / C give K 63 (the default); B's ctr 4.5678 = ln 96 + 0.0035 gives K 95, which the README command for B omits. Each
    target is scanned with its nearest K (B: 95; B@K96 as an info target); K only changes the negatives, which his batch()
    draws after the items, so the K variants of a holdout share one batch and one forward (negatives redrawn from the
    captured rng state).
    - Calibration: seed 7, 12 dropout redraws per dtype and holdout.
    - Scan: fp16 (his own fpnet.autocast_dtype policy for pre-Ampere GPUs; emulated bf16 is several times slower), one
      draw per seed. Targets: A (blind, must give 7), B, full1 (holdout ''), and his unpublished run C (info only).
    - Range (bounded): 0-1023 exhaustively + 2000-2030 + 10000-10031 + 20260901-20260930 + 30 common constants, ~1,130
      seeds, ~0.7 s each. Why: his recorded seeds are 0-3, 7, 11, 12, 21, 123, 2026, 2027 and 10000 + seed; a 32-bit
      space is out of reach. 1024-4095 is added only if B or full1 is unmatched and time remains.
    - Verification: the best scan candidates per target (top 3 + every scan d2 <= 40, at most 6) re-run in bf16 (his
      dtype) with 12 redraws. RECOVERED: the best candidate fits the logged line (chi2(4) <= 18.47, 99.9 %), holds >= 0.99
      of the posterior among the candidates, and a chance match by a seed OUTSIDE the range is excluded at 0.99 (local
      density of scanned seeds around the logged line, p_in_range). WEAK: only the last condition fails. AMBIGUOUS /
      NOT FOUND otherwise. A passes when 7 is the best seed (RECOVERED or WEAK: its seed is known to be in range).
      The decision rules were checked by simulation in the unit tests (no false RECOVERED over 90 null targets).
 7. Seeds found (RECOVERED / WEAK) re-run through his unmodified script (B: --seed s --K 95; full1: --seed s --holdout
    ""): the logged [Data] line must reappear (full1: 2,539,608 spectra) and the step-0 line must equal the harness line.
    A script that runs out of GPU memory in T4-emulated bf16 is retried once with the printed fp16 launcher patch.
Budget 150 min (F6_BUDGET_MIN); every subprocess has a hard timeout (its process group is killed) and a step starts only
if its estimate fits. Expected 45-75 min on a T4.
"""
import hashlib, json, math, os, re, shutil, signal, subprocess, sys, threading, time, traceback
import zipfile
from pathlib import Path

T0 = time.time()
INPUT = Path(os.environ.get('F6_INPUT', '/kaggle/input'))
WORKING = Path(os.environ.get('F6_WORKING', '/kaggle/working'))
DEVICE = os.environ.get('F6_DEVICE', 'cuda')          # 'cpu' only for the local dry run with fake modules
RW = WORKING / 'f6_repro2'
REPO = RW / 'repo'
LOGS = RW / 'logs'
RES = RW / 'results'
TMP = RW / 'tmp'
REPORT_PATH = WORKING / 'repro_report.json'
BUDGET_MIN = float(os.environ.get('F6_BUDGET_MIN', '150'))
RESERVE_MIN = 5
CHILD_ENV = {'PYTHONUNBUFFERED': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}   # allocator only
WIN_ROOT = 'C:/Users/omen/Desktop/envedaCasmi'      # v2lib.ROOT (a relative path on Linux)
HOLD_DEFAULT = 'np,nplib,syn,plusk,twin'           # 06_train_fpnet.py --holdout default

# ── pins ─────────────────────────────────────────────────────────────────────────────────────────────────────
TC_TREE_SHA256 = '85e254f6430e8202744ca18ccb14d3ade29324e62208309512e746ac67ab3057'   # 2026-10-19 version, 363 files
TC_TREE_FILES = 363
TC_PINS = {
    'LICENSE': '6f05399ac30469cae349febdefe2b1f1315af1b2ded8088de7d2304ef7ebddb7',
    'README.md': '1c93dab90f9171a194e665949792c992017981fd9a63b82c0f647d20c6690249',
    'scripts/06_train_fpnet.py': '20ae6ca4f78d4a003a33258995b972c40eed5c7964a8c15186bea33229e73057',
    'casmi/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'casmi/fpnet.py': '0ea6e372d1f045a877d350234c2ccf5d93fdd98cb23bd4c22376bf7e99547608',
    'casmi/traindata.py': 'cecb329d96362f11d16b40a26f1888f092610ff6202812408c13e2c293c6a2bc',
    'casmi/spectra.py': 'e2481761cdfb2515e8a945751f85f6b9e885680a8e974d7ac35c78bbf279ae74',
    'casmi/chem.py': '99474f23d8a67040b0c0f8c70e73592a6f7f656f50103e350ca473451e2169f1',
    'fpnet_v2/v2lib.py': '01fac520347fe6f9f141105c52628c7695e49d7f483f23120786e0a3b10ad3e5',
    'fpnet_v2/evalq.parquet': '0b3425f03b66ea7e0a6de385d828b703b33838cd72b327f3083cc261765faef6',
    'fpnet_v2/np_split.parquet': 'a231fae064b982108799ab827b6a5e06195490b68d366ef004f7ff33dd23f490',
    'dreams_ft/dreams_model.py': '36a7f3451d60a5ca3057a64287c66003fc778d2398374537b534b7033473ded3',
    'dreams_ft/review_fixes/train_dreams_fp.py': '7b4c8573f45c1f3e785528de70bb7ac528b7a17a458af4ad735b6e949b67bcdf',
    'dreams_ft/extract_step20k_ema.py': 'bcbbee1b263cf23dd4c77340ebd7125f862d404a3016a039459f4fbc5aa15b24',
    'logs/log_train_a.txt': 'c3506b549e6d952df5b1454dca77206416e5cce840db07f34e7e0d7b319b3aaa',
    'logs/log_train_b.txt': '63df95e877e9fbe05fe2c893eefbc0d4c9a30131ed4664766929af118d2a092b',
    'logs/log_train_c.txt': '466adf17af2fd965911992bc5309b14772f8c5616ce6c45a078765930e337017',
    'logs/log_train_full1.txt': '9e849456d29756304f4e1ac3289960f2ac92308b7dcd54bdb6a59beed5b3fe35',
    'dreams_ft/logs/pilot_d1.txt': 'c93ace6a8772cc1fe607d927826e5bc8ac6b9fe08b0b4a23d2cb9d0d60ba72bc',
    'dreams_ft/logs/pilot_d1_summary.json': '3c886b8974a9c004a5c8269aeb6ee329cf63c5ca80f8746c26407d910031f5ce',
    'dreams_ft/s2_eval/D20k_report.txt': '358ecdcfe337454b5136ee344736684e21278181f9d492bff7d27ccdb3be4d83',
    'dreams_ft/s2_eval/check_D20k.log': '45ecf0abeb0c49f3bf72a89a046f696fb4555226b5dcd7d162ebe0cd358796e4',
    'dreams_ft/s2_eval/D20k_eval_info.json': 'fb6e0e86cce99c244f7ffed5d5e0617ef9de2921fa5935866d877cd10ecd7582',
}
README_SHA = {   # casmi26-training-code README section 1
    'fe_A.pt': 'edccf479a2a2e53461a06bc5e1aee4341b22abb32302c4f6d2c45281000d742a',
    'fe_B.pt': '6f60a22b3c3b21aa48d47328bb9523bdf07cc004bfb72966e0f9713507c24f2e',
    'fe_D.pt': '8ae5bee2c978f4f01ea7cfe29b514cd3e9112006ab51aa8b13f20e26e9365362',
    'fpnet_full1.pt': 'd13a3611e499d7783c4a6fce5b04363eb297dcf70f2d7d00bb66f9ff1f547123',
}
ARCHIVE_NAME = {'fe_A.pt': 'fpnet_a', 'fe_B.pt': 'fpnet_b', 'fpnet_full1.pt': 'fpnet_full1',
                'fe_D.pt': 'pilot_d1_step20k_ema'}     # torch.save archive prefix = stem of the path it was saved to
SPEC_CACHE_SHA = {   # casmi26-spec-cache docs/README.md
    'spec_mz.npy': '482332fec3fabcc46e06ce1b23d79e633af7009b8190a3dcd0a675121b42342b',
    'spec_it.npy': '9e127f00eb9d4217403991304e69166adc9a36a5e624ce7fd1ddbf6b1417cc23',
    'spec_off.npy': 'af2be108a532a278ec4563c023808b7fa19e40fdf4ea7dc888e980bb6f618e6c',
    'spec_meta.parquet': '4f968592dc75b9bf2b6256dda394be42bad3cd07fd243ff0078101e066a44376',
    'split.parquet': 'd05a2f9f71cffd480b92581c458547f7aeb782436b07af4fbf8f696f96649656',
}
V2POOL_KNOWN = {'fp_bits.npy': 'c3869e86449d6fed6e79e9655f8a7d35bd55c3138ecddfc1dae5072627e78677'}
V2POOL_NEEDED = ['pool_meta.parquet', 'pool_fp.npy', 'fp_bits.npy', 'train_structs.parquet']
TRAIN_PARQUET_SHA = '9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98'   # sanpier data_audit.json
CODE_IDENTITY = ['fpnet.py', 'traindata.py', 'spectra.py', 'chem.py', '__init__.py']
HIS_ENV = dict(python='3.14.5', torch='2.14.0+cu130', cuda='13.0', gpu='NVIDIA GeForce RTX 5090 Laptop GPU')

# ── tolerances / statistics ──────────────────────────────────────────────────────────────────────────────────
BF16_MIN_TFLOPS = 1.0             # probe 4096^3 matmul: below this the emulated bf16 eval / verification would not fit
VAL_TOL = 0.002                   # |recomputed - logged| per VAL metric
EVAL_TOL = 0.002                  # frozen Evaluator cells
STEP0_KEYS = ('bce', 'ctr', 'top1_count', 'el')
ROUND_HALF = dict(bce=5e-5, ctr=5e-5, top1_count=0.0, el=5e-4)    # print rounding of the logged line (top1 -> exact count)
FLOOR_SD = dict(bce=2e-5, ctr=2e-4, top1_count=0.5, el=2e-4)      # minimum noise per quantity
CHI2_PASS = 18.47                 # chi2(4) 99.9 % quantile
SCAN_D2_MAX = 40.0                # scan shortlist (generous: the shortlist is re-tested in bf16 with redraws)
N_TOP_CAND = 3                    # always verify the best 3 scan seeds per target
MAX_CAND = 6
N_DRAWS = 12                      # dropout redraws for calibration / verification
POST_MIN = 0.99                   # posterior of the best verified seed among the candidates (uniform prior)
P_RANGE_MIN = 0.99                # P(the true seed is in the scanned range | best chi2, chance-match rate), prior 1/2
STEP0_TARGETS = {                 # target -> (training log, Data holdout config, role)
    'A': ('logs/log_train_a.txt', 'def', 'blind check: must recover the recorded seed 7'),
    'B': ('logs/log_train_b.txt', 'def', 'seed not recorded'),
    'full1': ('logs/log_train_full1.txt', 'full', 'seed not recorded (holdout "")'),
    'C': ('logs/log_train_c.txt', 'def', 'unpublished run C (info only)'),
}
CONFIGS = {'def': HOLD_DEFAULT, 'full': ''}       # base Data configs (holdout); K variants are added at run time
SEED_A = 7
K_DEFAULT = 63                    # 06_train_fpnet.py --K default (negatives per positive -> K + 1 candidates)
# At init the candidate scores are almost flat (|sc| ~ 0.06), so the step-0 ctr = ln(K + 1) + ~0.002 +- 0.004 whatever the
# seed (A / full1 / C: -0.0021 / +0.0045 / +0.0011 around ln 64). B's 4.5678 is ln 96 + 0.0035, i.e. 100 sigma off ln 64:
# B ran with --K 95 (its ctr stays ~ln(96/64) = 0.41 above A's for the first 1,000 steps), which the README command omits.
# Every K with |ctr - ln(K + 1)| <= K_OFF_MAX is scanned (nearest first; the others as info targets 'B@K96').
K_OFF_MAX = 0.009

REPORT = dict(kernel='casmi26-f6-repro2',
              task='F6 K2: FPNet / DreamsFP files against their logs + seed recovery (docs/f6_engine1_chain.md ranks 6, 8)',
              datasets=['ahmedberatozer/casmi26-training-code', 'ahmedberatozer/casmi26-v4b-models',
                        'ahmedberatozer/casmi26-v3-models', 'ahmedberatozer/casmi26-fpnet-full1',
                        'ahmedberatozer/casmi26-v2-pool', 'ahmedberatozer/casmi26-spec-cache',
                        'metric/rdkit-2026-3-3-wheel', 'competition enveda-CASMI26-molecule-id-mass-spectra'],
              checks={}, steps={}, results={}, errors=[])
ROOTS = {}
STATE = dict(autocast='bf16', scan_dtype='fp16')


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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=_js))


def check(name, ok, **detail):
    REPORT['checks'][name] = dict(ok=ok, **detail)
    tag = 'INFO' if ok is None else ('OK' if ok else 'FAIL')
    log(f'F6 CHECK {name}: {tag} ' + json.dumps(detail, default=_js)[:2000])
    save_report()


def guarded(name, fn, *a, **k):
    try:
        return fn(*a, **k)
    except SystemExit:
        raise
    except Exception as e:
        REPORT['errors'].append(dict(step=name, error=repr(e), trace=traceback.format_exc()[-3000:]))
        log(f'F6 ERROR in {name}: {e!r}\n{traceback.format_exc()}')
        save_report()
        return None


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def run_logged(args, logfile, timeout=None, **kw):
    """Run a child process in its own session, stream its output to this log and to `logfile`; return (exit code,
    seconds, timed_out). On timeout the whole process group is killed."""
    t = time.time()
    fired = []

    def _kill(p):
        fired.append(1)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            p.kill()
    with open(logfile, 'w') as fh:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True, **kw)
        timer = threading.Timer(timeout, _kill, (p,)) if timeout else None
        if timer:
            timer.start()
        for line in p.stdout:
            sys.stdout.write(line); fh.write(line)
        rc = p.wait()
        if timer:
            timer.cancel()
    sys.stdout.flush()
    return rc, round(time.time() - t, 1), bool(fired)


def can_start(est_min, elapsed_min, budget_min=None, reserve_min=None):
    budget_min = BUDGET_MIN if budget_min is None else budget_min
    reserve_min = RESERVE_MIN if reserve_min is None else reserve_min
    return elapsed_min + est_min + reserve_min <= budget_min


def step_timeout_min(est_min, elapsed_min, budget_min=None, reserve_min=None):
    budget_min = BUDGET_MIN if budget_min is None else budget_min
    reserve_min = RESERVE_MIN if reserve_min is None else reserve_min
    return max(1.0, min(max(2.5 * est_min, est_min + 8), budget_min - reserve_min - elapsed_min))


# ── pure helpers (unit-tested locally) ───────────────────────────────────────────────────────────────────────
def tree_hash(root):
    """(sha256 over sorted '<relpath>\\t<sha256>\\n' lines of every file under root, {relpath: sha256})."""
    root = Path(root)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    blob = ''.join(f'{r}\t{h}\n' for r, h in sorted(files.items())).encode()
    return hashlib.sha256(blob).hexdigest(), files


def torch_archive_prefix(path):
    """Top-level folder of a torch.save zip (= stem of the file name it was saved under); None if not a zip."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return None
    tops = sorted({n.split('/', 1)[0] for n in names if '/' in n})
    return tops[0] if len(tops) == 1 else (tops or None)


_STEP0 = re.compile(r'^step 0 bce ([\d.]+) ctr ([\d.]+) top1 ([\d.]+) el ([\d.]+)')


def parse_step0(text):
    """First 'step 0 ...' line of a 06_train_fpnet log -> dict(bce, ctr, top1, top1_count, el, line)."""
    for l in text.splitlines():
        m = _STEP0.match(l.strip())
        if m:
            bce, ctr, top1, el = map(float, m.groups())
            return dict(bce=bce, ctr=ctr, top1=top1, top1_count=int(round(top1 * 256)), el=el, line=l.strip())
    return None


def k_candidates(ctr, max_off=None):
    """K values whose flat-score ctr ln(K + 1) lies within max_off of a logged step-0 ctr, nearest first (at least the
    nearest one). [(K, ctr - ln(K + 1)), ...]."""
    max_off = K_OFF_MAX if max_off is None else max_off
    k0 = max(1, int(round(math.exp(ctr))) - 1)
    near = sorted(((abs(ctr - math.log(k + 1)), k) for k in range(max(1, k0 - 3), k0 + 4)))
    out = [(k, round(ctr - math.log(k + 1), 5)) for d, k in near if d <= max_off]
    return out or [(near[0][1], round(ctr - math.log(near[0][1] + 1), 5))]


def cfg_name(base, K):
    return base if K == K_DEFAULT else f'{base}_K{K}'


def build_step0_targets(lines):
    """lines: {target: parse_step0 dict}. -> (targets, configs). Each target gets the base Data config of STEP0_TARGETS
    and its nearest K; further K candidates become info targets '<t>@K<k>'. configs: {name: dict(holdout, K)}."""
    targets, configs = {}, {}
    for t, s0 in lines.items():
        if not s0:
            continue
        logf, base, role = STEP0_TARGETS[t]
        ks = k_candidates(s0['ctr'])
        for i, (K, off) in enumerate(ks):
            name = t if i == 0 else f'{t}@K{K}'
            cfg = cfg_name(base, K)
            configs[cfg] = dict(holdout=CONFIGS[base], K=K)
            targets[name] = dict(s0, cfg=cfg, base_cfg=base, K=K, ctr_minus_lnK1=off, base=t, log=logf,
                                 role=role if i == 0 else f'alternative K for {t} (info)', k_candidates=ks)
    for base, hold in CONFIGS.items():          # the base configs always run (Data lines, calibration)
        configs.setdefault(base, dict(holdout=hold, K=K_DEFAULT))
    return targets, configs


def head_lines(text):
    """The deterministic header lines of a 06_train_fpnet run: [Data], params, val sets (first occurrence each)."""
    out = {}
    for l in text.splitlines():
        s = l.strip()
        for key, pre in (('data', '[Data]'), ('params', 'params '), ('valsets', 'val sets:')):
            if key not in out and s.startswith(pre):
                out[key] = s
    return out


_VAL_FP = re.compile(r'VAL step (\d+): np ([\d.]+) np-merged ([\d.]+) nplib ([\d.]+) syn ([\d.]+)')
_VAL_D = re.compile(r'VAL step (\d+): np-dev ([\d.]+) np-dev-merged ([\d.]+) nplib ([\d.]+)')


def parse_val_lines(text):
    """{step: {metric: value}} of every VAL line (FPNet: np / npm / nplib / syn; DreamsFP: np-dev / np-dev_merged / nplib)."""
    out = {}
    for l in text.splitlines():
        m = _VAL_FP.search(l)
        if m:
            s, a, b, c, d = m.groups()
            out[int(s)] = dict(np=float(a), npm=float(b), nplib=float(c), syn=float(d))
            continue
        m = _VAL_D.search(l)
        if m:
            s, a, b, c = m.groups()
            out[int(s)] = {'np-dev': float(a), 'np-dev_merged': float(b), 'nplib': float(c)}
    return out


_EVAL_KV = re.compile(r'\b(np|nplib|npo)_(merged|molecule|single)=([\d.]+)')


def parse_evaluator_line(text, tag):
    """'[Evaluator] <tag> done ...  np_merged=0.6369 ...' -> {'np|merged': 0.6369, ...} (last such line for tag)."""
    out = None
    for l in text.splitlines():
        if l.startswith(f'[Evaluator] {tag} ') and ' done ' in l:
            out = {f'{a}|{b}': float(v) for a, b, v in _EVAL_KV.findall(l)}
    return out


def _num(t):
    try:
        return float(t)
    except ValueError:
        return None


def parse_d20k_report(text):
    """D20k_report.txt -> dict(struct={'set|level|weight': {col: v}}, query={'qset|level': {col: v}}, cols=[...]).
    Pandas multi-index print: a label is shown only where it changes, so missing leading labels are carried forward."""
    struct, query = {}, {}
    cols, mode = None, None
    cur = ['', '', '']
    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if s.startswith('== window-MRR@25') or s.startswith('(report only'):
            mode, cols, cur = 'struct', None, ['', '', '']
            continue
        if s.startswith('== query-level'):
            mode, cols, cur = 'query', None, ['', '', '']
            continue
        if s.startswith('==') or s.startswith('--'):
            mode = None if not s.startswith('== window') else mode
            continue
        if mode is None or not s:
            continue
        toks = s.split()
        if toks[0] == 'model' or (mode == 'query' and cols is None and all(_num(t) is None for t in toks)):
            cols = toks[1:] if toks[0] == 'model' else toks
            continue
        if toks[0] in ('set', 'qset'):
            continue
        if cols is None:
            continue
        vals = [_num(t) for t in toks]
        k = 0
        while k < len(vals) and vals[-1 - k] is not None:
            k += 1
        nums, labels = vals[len(vals) - k:], toks[:len(toks) - k]
        if mode == 'struct':
            if len(nums) != len(cols) or not labels:
                continue
            if len(labels) == 3:
                cur = list(labels)
            elif len(labels) == 2:
                cur = [cur[0]] + list(labels)
            else:
                cur = [cur[0], cur[1], labels[0]]
            struct['|'.join(cur)] = dict(zip(cols, nums))
        else:
            if len(nums) != len(cols) or not labels:
                continue
            if len(labels) == 2:
                cur = list(labels) + ['']
            else:
                cur = [cur[0], labels[0], '']
            query[f'{cur[0]}|{cur[1]}'] = dict(zip(cols, nums))
    return dict(struct=struct, query=query)


def val_compare(logged, recomputed, keys, tol=None, saved=None):
    """Per-metric logged / saved-in-file / recomputed values; status REPRODUCED (all |diff| <= tol), CLOSE (<= 2.5 tol)
    or DIFFERENT; saved_equals_log: the file's own full-precision val rounds to the logged 4 decimals."""
    tol = VAL_TOL if tol is None else tol
    rows, worst = {}, 0.0
    for k in keys:
        lg, rc = (logged or {}).get(k), (recomputed or {}).get(k)
        sv = (saved or {}).get(k) if saved else None
        r = dict(logged=lg, recomputed=rc, saved_in_file=sv)
        if lg is not None and rc is not None:
            r['diff'] = rc - lg
            worst = max(worst, abs(rc - lg))
        else:
            worst = float('inf')
        if sv is not None and rc is not None:
            r['diff_vs_saved'] = rc - sv
        rows[k] = r
    status = 'REPRODUCED' if worst <= tol + 1e-12 else ('CLOSE' if worst <= 2.5 * tol else 'DIFFERENT')
    saved_eq = None
    if saved:
        lg_ = logged or {}
        saved_eq = all(saved.get(k) is not None and lg_.get(k) is not None and abs(saved[k] - lg_[k]) <= 5e-5 + 1e-9
                       for k in keys)
    return dict(status=status, max_abs_diff=worst, tol=tol, metrics=rows, saved_equals_log=saved_eq)


def nearest_steps(val_lines, recomputed, keys, n=3):
    """Logged VAL steps ordered by max |logged - recomputed| (which logged checkpoint the file behaves like)."""
    sc = []
    for step, v in val_lines.items():
        if all(k in v for k in keys) and all(recomputed.get(k) is not None for k in keys):
            sc.append((max(abs(v[k] - recomputed[k]) for k in keys), step))
    sc.sort()
    return [dict(step=s, max_abs_diff=round(d, 5)) for d, s in sc[:n]]


def evaluator_compare(ours, report, tag, tol=None):
    """Compare our Evaluator summaries for one column (tag) with D20k_report: query-level cells and the unweighted
    per-structure cells (sets np-dev / nplib / np-lock / np / npo x levels)."""
    tol = EVAL_TOL if tol is None else tol
    cells = {}
    for key, v in (report.get('query') or {}).items():
        if tag in v and key in ours.get('query', {}):
            cells['query|' + key] = (v[tag], ours['query'][key])
    for key, v in (report.get('struct') or {}).items():
        st, lv, w = key.split('|')
        if w != 'unw' or tag not in v:
            continue
        o = ours.get('struct_unw', {}).get(f'{st}|{lv}')
        if o is not None:
            cells[f'struct|{st}|{lv}'] = (v[tag], o['mrr'])
    diffs = {k: round(b - a, 5) for k, (a, b) in cells.items()}
    worst = max((abs(d) for d in diffs.values()), default=float('inf'))
    status = 'REPRODUCED' if cells and worst <= tol + 1e-12 else ('CLOSE' if worst <= 2.5 * tol else 'DIFFERENT')
    return dict(status=status, n_cells=len(cells), max_abs_diff=worst, tol=tol,
                worst=sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:6])


def format_step0(m):
    return f"step 0 bce {m['bce']:.4f} ctr {m['ctr']:.4f} top1 {m['top1']:.3f} el {m['el']:.3f}"


def draw_stats(rows):
    """Mean / sd (ddof 1) / n over dropout redraws of the step-0 quantities."""
    n = len(rows)
    mean = {k: sum(r[k] for r in rows) / n for k in STEP0_KEYS}
    sd = {k: (math.sqrt(sum((r[k] - mean[k]) ** 2 for r in rows) / (n - 1)) if n > 1 else 0.0) for k in STEP0_KEYS}
    return dict(n=n, mean=mean, sd=sd)


def chi2_vs(target, stats):
    """chi2 of a logged step-0 line against a seed's redraw distribution (logged = one more independent draw)."""
    n = max(stats['n'], 1)
    z, chi = {}, 0.0
    for k in STEP0_KEYS:
        var = stats['sd'][k] ** 2 * (1 + 1 / n) + ROUND_HALF[k] ** 2 / 3 + FLOOR_SD[k] ** 2
        z[k] = (target[k] - stats['mean'][k]) / math.sqrt(var)
        chi += z[k] ** 2
    return chi, z


def loglik(target, stats):
    """Gaussian log-likelihood (up to a constant) of the logged line under a seed's redraw distribution."""
    n = max(stats['n'], 1)
    chi, ld = 0.0, 0.0
    for k in STEP0_KEYS:
        var = stats['sd'][k] ** 2 * (1 + 1 / n) + ROUND_HALF[k] ** 2 / 3 + FLOOR_SD[k] ** 2
        chi += (target[k] - stats['mean'][k]) ** 2 / var
        ld += math.log(var)
    return dict(chi2=chi, ll=-0.5 * chi - 0.5 * ld)


def scan_var(cal_scan, cal_ref=None):
    """Per-quantity variance for single-draw scan rows: two independent dropout draws (ours, his), plus the dtype shift
    (scan dtype vs bf16 at the calibration seed), print rounding and the floor."""
    var = {}
    for k in STEP0_KEYS:
        shift = (cal_scan['mean'][k] - cal_ref['mean'][k]) if cal_ref else 0.0
        sd = max(cal_scan['sd'][k], cal_ref['sd'][k] if cal_ref else 0.0)
        var[k] = 2 * sd ** 2 + shift ** 2 + ROUND_HALF[k] ** 2 / 3 + FLOOR_SD[k] ** 2
    return var


def ver_var(cal):
    """Per-quantity variance of the verification test (logged line vs a N_DRAWS-redraw mean)."""
    return {k: cal['sd'][k] ** 2 * (1 + 1 / N_DRAWS) + ROUND_HALF[k] ** 2 / 3 + FLOOR_SD[k] ** 2 for k in STEP0_KEYS}


def p_in_range(chi2, e_false, thr=None):
    """P(the logged line comes from the matching seed rather than from a seed outside the range that some scanned
    seed matches by chance), prior 1/2. A true match has chi2 ~ chi2(4) (pdf c/4 exp(-c/2)); a chance match occurs
    with rate e_false and is spread over the acceptance region. top1 is an integer count that a wrong seed often hits
    exactly, so the region is modelled as a 3-d ball (pdf 1.5 sqrt(c) / thr^1.5 on [0, thr]), which puts more chance
    matches at small chi2 than a 4-d ball would (conservative; checked by simulation in the unit tests)."""
    thr = CHI2_PASS if thr is None else thr
    if chi2 is None or e_false is None or chi2 > thr:
        return None
    f_true = chi2 / 4 * math.exp(-chi2 / 2)
    f_false = e_false * 1.5 * math.sqrt(chi2) / thr ** 1.5
    if f_true + f_false == 0:
        return 1.0
    return f_true / (f_true + f_false)


def scan_d2(target, row, var):
    return sum((target[k] - row[k]) ** 2 / var[k] for k in STEP0_KEYS)


def select_candidates(rows_by_seed, target, var, n_top=None, d2_max=None, cap=None):
    """Scan shortlist for one target: the n_top best seeds plus every seed with d2 <= d2_max, at most cap.
    Returns (candidates, ranking[(d2, seed)] best first, n_scan_pass = seeds with d2 <= CHI2_PASS)."""
    n_top = N_TOP_CAND if n_top is None else n_top
    d2_max = SCAN_D2_MAX if d2_max is None else d2_max
    cap = MAX_CAND if cap is None else cap
    ranking = sorted((scan_d2(target, r, var), s) for s, r in rows_by_seed.items())
    cand = []
    for i, (d2, s) in enumerate(ranking):
        if (i < n_top or d2 <= d2_max) and s not in cand:
            cand.append(s)
        if len(cand) >= cap or (i >= n_top and d2 > d2_max):
            break
    return cand, ranking, sum(1 for d2, _ in ranking if d2 <= CHI2_PASS)


def null_false_rate(rows_by_seed, var, thr=None, chunk=512):
    """Chance-match rate measured on the scan itself: every scanned seed's own single-draw row stands in for a logged
    line (it is distributed like one) and the OTHER seeds within d2 <= thr under the verification variance are counted
    (their rows stand in for their redraw means; the extra single-draw blur hardly changes the density of seed means
    near the target, which varies on a much larger scale). Returns (p_false per seed pair, expected chance matches for
    one target over the scanned range = p_false x (N - 1))."""
    import numpy as np
    thr = CHI2_PASS if thr is None else thr
    seeds = sorted(rows_by_seed)
    n = len(seeds)
    if n < 2:
        return None, None
    X = np.array([[rows_by_seed[s][k] / math.sqrt(var[k]) for k in STEP0_KEYS] for s in seeds], np.float64)
    hits = 0
    for i in range(0, n, chunk):
        d2 = ((X[i:i + chunk, None, :] - X[None, :, :]) ** 2).sum(-1)
        hits += int((d2 <= thr).sum()) - min(chunk, n - i)          # minus the self-pairs
    p = hits / (n * (n - 1))
    return p, p * (n - 1)


def local_chance_rate(rows_by_seed, target, var, exclude=(), widen=25.0, dim=3, pseudo=0.5):
    """Expected chance matches for THIS logged line: scanned seeds (minus the claimed one) inside a widened region
    (scan d2 <= widen x CHI2_PASS, radius x5) scaled down to the acceptance region by the volume ratio widen^(dim/2)
    (dim 3: top1 is a count), with a pseudo-count of 0.5 so an empty neighbourhood is not read as zero risk. A kernel
    density estimate around the target: an outlying line (B's ctr 4.5678) is judged by its own sparse neighbourhood.
    Returns (expected chance matches, seeds in the widened region)."""
    ex = set(exclude)
    n = sum(1 for s, r in rows_by_seed.items() if s not in ex and scan_d2(target, r, var) <= widen * CHI2_PASS)
    return (n + pseudo) * widen ** (-dim / 2), n


def seed_verdict(fits, e_false=None, expected=None):
    """fits: {seed: dict(chi2, ll)} of the verified candidates. Posterior over them (uniform prior; seeds outside the
    shortlist have scan d2 > SCAN_D2_MAX and are negligible).
    RECOVERED: best chi2 <= CHI2_PASS, posterior >= POST_MIN and P(in range) >= P_RANGE_MIN (p_in_range);
    WEAK: the same but a chance match of a seed outside the range is not excluded at that level;
    AMBIGUOUS: best passes but another seed keeps >= 1 - POST_MIN of the posterior; NOT FOUND: best chi2 > CHI2_PASS."""
    if fits and not isinstance(next(iter(fits.values())), dict):
        fits = {s: dict(chi2=c, ll=-0.5 * c) for s, c in fits.items()}
    order = sorted((f['chi2'], s) for s, f in fits.items())
    passing = sorted(s for c, s in order if c <= CHI2_PASS)
    best = order[0] if order else (None, None)
    second = order[1] if len(order) > 1 else (None, None)
    post = {}
    if fits:
        mx = max(f['ll'] for f in fits.values())
        z = sum(math.exp(f['ll'] - mx) for f in fits.values())
        post = {s: math.exp(f['ll'] - mx) / z for s, f in fits.items()}
    pb = post.get(best[1]) if best[1] is not None else None
    pr = p_in_range(best[0], e_false)
    if best[0] is None or best[0] > CHI2_PASS:
        status = 'NOT FOUND'
    elif pb is not None and pb >= POST_MIN:
        status = 'RECOVERED' if (pr is not None and pr >= P_RANGE_MIN) else 'WEAK'
    else:
        status = 'AMBIGUOUS'
    d = dict(status=status, seed=best[1] if status in ('RECOVERED', 'WEAK') else None, passing=passing,
             best_seed=best[1], best_chi2=best[0], best_posterior=pb, p_in_range=pr, second_seed=second[1],
             second_chi2=second[0], expected_chance_matches=e_false, post_min=POST_MIN, p_range_min=P_RANGE_MIN)
    if expected is not None:      # known seed (A): its range membership is not in question, so WEAK counts
        d['expected'] = expected
        d['ok'] = status in ('RECOVERED', 'WEAK') and best[1] == expected
    return d


SEED_CONSTANTS = [42, 69, 77, 99, 100, 101, 111, 123, 128, 222, 256, 314, 321, 333, 404, 420, 500, 512, 555, 666, 700,
                  777, 888, 999, 1000, 1024, 1234, 1337, 3407, 4242, 7777, 8888, 9999, 10007, 12345, 31337, 54321,
                  65535, 99999, 123456, 1234567, 2 ** 31 - 1, 2 ** 32 - 1]


def seed_plan(skip=(SEED_A,)):
    """(main list in priority order, extension list). Main: 0-31 and common constants first, then the rest of 0-1023,
    2000-2030, 10000-10031 and 20260901-20260930. Extension (only if needed): 1024-4095."""
    first = list(range(32)) + SEED_CONSTANTS
    rest = list(range(32, 1024)) + list(range(2000, 2031)) + list(range(10000, 10032)) + list(range(20260901, 20260931))
    seen, main = set(skip), []
    for s in first + rest:
        if s not in seen:
            seen.add(s); main.append(s)
    ext = [s for s in range(1024, 4096) if s not in seen]
    return main, ext


def scan_rows(rows, dtype, draw=0):
    """{cfg: {seed: metrics}} of the single-draw rows of one dtype."""
    out = {}
    for r in rows:
        if r.get('kind') == 'step0' and r.get('dtype') == dtype and r.get('draw') == draw:
            out.setdefault(r['cfg'], {})[r['seed']] = r
    return out


def draws_of(rows, seed, cfg, dtype):
    return [r for r in rows if r.get('kind') == 'step0' and r.get('seed') == seed and r.get('cfg') == cfg
            and r.get('dtype') == dtype]


def read_jsonl(path):
    out = []
    p = Path(path)
    if p.is_file():
        for l in p.read_text().splitlines():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def remove_links(root):
    """Delete every symlink under root without following any (os.walk does not descend into linked directories)."""
    n = 0
    root = Path(root)
    if not root.exists():
        return 0
    for dp, dns, fns in os.walk(root, topdown=True, followlinks=False):
        for name in list(dns) + fns:
            p = Path(dp) / name
            if p.is_symlink():
                p.unlink(); n += 1
    return n


# ── harness (written to RW and run as a subprocess from the repository root) ──────────────────────────────────
HARNESS = r'''
# casmi26-f6-repro2 harness (written by notebook.py; our code). It imports Ahmed's modules from the restored repository
# layout (cwd = repository root) and never edits them. Modes: eval | step0. Results: one JSON object per line.
import argparse, contextlib, io, json, math, os, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['eval', 'step0'])
ap.add_argument('--plan', required=True)
ap.add_argument('--out', required=True)
A = ap.parse_args()
PLAN = json.load(open(A.plan))
DEV = PLAN.get('device', 'cuda')
DEADLINE = float(PLAN.get('deadline', 1e18))
QUIET = int(PLAN.get('print_every', 0))
OUT = open(A.out, 'a', buffering=1)
T0 = time.time()


def _js(o):
    try:
        return float(o)
    except Exception:
        return str(o)


def emit(show=True, **rec):
    rec.setdefault('t', round(time.time() - T0, 2))
    s = json.dumps(rec, default=_js)
    OUT.write(s + '\n')
    if show:
        print('F6H ' + s[:700], flush=True)


sys.path.insert(0, 'src'); sys.path.insert(0, 'work/analysis/fpnet_v2'); sys.path.insert(0, 'work/analysis/dreams_ft')
import numpy as np, pandas as pd, torch, torch.nn.functional as F

if PLAN.get('autocast') == 'fp16-patched':
    _AC = torch.autocast

    class _Fp16Autocast(_AC):
        # casmi26-f6-repro2 runtime patch, active only because the bf16 probe failed: bf16 requests run as fp16
        def __init__(self, device_type, dtype=None, *args, **kw):
            if dtype is torch.bfloat16:
                dtype = torch.float16
            super().__init__(device_type, dtype, *args, **kw)
    torch.autocast = _Fp16Autocast
    print('F6H PATCH torch.autocast: bfloat16 -> float16 (the bf16 probe failed on this GPU)', flush=True)

from casmi import fpnet
from casmi.traindata import Data, window_mrr

DT = {'bf16': torch.bfloat16, 'fp16': torch.float16}


def autocast(dtype):
    return torch.autocast('cuda', dtype=dtype) if DEV == 'cuda' else contextlib.nullcontext()


def make_data(holdout, seed=0):
    buf = io.StringIO(); t = time.time()
    with contextlib.redirect_stdout(buf):
        D = Data(DEV, K=63, seed=seed, holdout_folds=tuple(holdout.split(',')))
    lines = [l for l in buf.getvalue().splitlines() if l.startswith('[Data]')]
    return D, (lines[-1] if lines else buf.getvalue().strip()[-300:]), round(time.time() - t, 1)


def load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True), True
    except Exception as e:
        print('F6H weights_only load refused (%r); his load path (weights_only=False) is used' % (e,), flush=True)
        return torch.load(path, map_location='cpu', weights_only=False), False


def header(ck):
    return {k: v for k, v in ck.items() if k not in ('model', 'ema', 'opt')}


def fpnet_from(ck):
    # = casmi.fpnet.load_model after its torch.load call
    m = fpnet.FPNet(ck['nbits'], d=ck['d'], layers=ck['layers'], heads=ck.get('heads', 8))
    m.load_state_dict(ck['model'])
    return m.to(DEV).eval()


def left():
    return DEADLINE - time.time()


# ------------------------------------------------------------------------------------------------ eval
def mode_eval():
    D, dline, secs = make_data(PLAN.get('holdout', 'np,nplib,syn,plusk,twin'), seed=7)
    emit(kind='data', cfg='def', line=dline, secs=secs)
    # 06_train_fpnet.py main(): the fixed validation sets (verbatim)
    rng = np.random.default_rng(123)
    np_ids = np.where((D.fold[D.sid] == 'np') & (D.lib == 9))[0]
    nplib_ids = np.where(D.fold[D.sid] == 'nplib')[0]; nplib_ids = rng.choice(nplib_ids, min(2000, len(nplib_ids)), replace=False)
    syn_ids = np.where(D.fold[D.sid] == 'syn')[0]; syn_ids = rng.choice(syn_ids, min(1500, len(syn_ids)), replace=False)
    emit(kind='valsets', line=f'val sets: np {len(np_ids)} nplib {len(nplib_ids)} syn {len(syn_ids)}')
    keep = {}
    for key, path in PLAN['fpnets'].items():
        if left() < 0:
            emit(kind='deadline', at='val ' + key); return
        t = time.time()
        ck, wo = load_ckpt(path); hd = header(ck); m = fpnet_from(ck); del ck
        r = dict(np=window_mrr(m, D, np_ids, DEV)[0], npm=window_mrr(m, D, np_ids, DEV, merged=True)[0],
                 nplib=window_mrr(m, D, nplib_ids, DEV)[0], syn=window_mrr(m, D, syn_ids, DEV)[0])
        emit(kind='val', model=key, header=hd, weights_only=wo, recomputed=r, secs=round(time.time() - t, 1))
        if key in ('A', 'B'):
            keep[key] = m
        else:
            del m
        if DEV == 'cuda':
            torch.cuda.empty_cache()
    mD = None
    if PLAN.get('fe_D') and left() > 0:
        try:
            t = time.time()
            import dreams_model as dm
            ck, wo = load_ckpt(PLAN['fe_D']); hd = header(ck)
            mD = dm.dreams_fp_from_ckpt(ck, DEV); del ck
            # train_dreams_fp.py: the fixed validation sets (verbatim)
            nps_path = 'work/analysis/fpnet_v2/np_split.parquet'
            np_all = np.where((D.fold[D.sid] == 'np') & (D.lib == 9))[0]
            nps = pd.read_parquet(nps_path)
            dev_sids = set(nps.sid.values[nps.split == 'dev'])
            np_dev_ids = np.array([i for i in np_all if D.sid[i] in dev_sids])
            rng_v = np.random.default_rng(123)
            nplib_d = np.where(D.fold[D.sid] == 'nplib')[0]
            nplib_d = rng_v.choice(nplib_d, min(2000, len(nplib_d)), replace=False)
            # train_dreams_fp.validate(m)
            r = {'np-dev': window_mrr(mD, D, np_dev_ids, DEV)[0],
                 'np-dev_merged': window_mrr(mD, D, np_dev_ids, DEV, merged=True)[0],
                 'nplib': window_mrr(mD, D, nplib_d, DEV)[0]}
            emit(kind='val', model='D', header=hd, weights_only=wo, recomputed=r,
                 n=dict(np_dev=int(len(np_dev_ids)), nplib=int(len(nplib_d))), secs=round(time.time() - t, 1))
        except Exception as e:
            emit(kind='error', at='fe_D', error=repr(e), trace=traceback.format_exc()[-2000:]); mD = None
    if PLAN.get('evaluator') and left() > 0:
        try:
            import v2lib

            class CachedEvaluator(v2lib.Evaluator):
                # his Evaluator, unmodified. For a model list his logits() adds each model's float32 logits to a zero
                # tensor in list order; here each model's logits are computed once by his own logits() and summed in
                # the same order onto a zero tensor, which is the same arithmetic, so ensembles cost no extra forward.
                cache = {}

                def logits(self, models, bs=128):
                    models = models if isinstance(models, (list, tuple)) else [models]
                    Z = None
                    for mm in models:
                        if id(mm) not in self.cache:
                            self.cache[id(mm)] = v2lib.Evaluator.logits(self, mm, bs)
                        if Z is None:
                            Z = torch.zeros_like(self.cache[id(mm)])
                        Z += self.cache[id(mm)]
                    return Z
            t = time.time()
            ev = CachedEvaluator(D, DEV)
            emit(kind='evaluator_init', n_queries=int(len(ev.Q)), n_windows=int(len(ev.groups)), secs=round(time.time() - t, 1))
            models = {'A': keep.get('A'), 'B': keep.get('B'), 'D20k': mD}
            combos = [('A',), ('B',), ('D20k',), ('A', 'B'), ('A', 'D20k'), ('B', 'D20k'), ('A', 'B', 'D20k')]
            for combo in combos:
                if any(models.get(k) is None for k in combo):
                    continue
                if left() < 0:
                    emit(kind='deadline', at='evaluator ' + '+'.join(combo)); break
                tag = '+'.join(combo); t = time.time()
                res = ev.run([models[k] for k in combo], tag=tag)
                q = {f'{a}|{b}': float(v) for (a, b), v in res.groupby(['qset', 'level']).rr.mean().items()}
                g = pd.DataFrame({'sid': np.unique(res.sid.values)})
                g['g'] = 0.0; g['w_full'] = 1.0; g['w_temp'] = 1.0      # weights unpublished: unweighted rows only
                ps = v2lib.per_structure(res, g=g)
                met = v2lib.metrics(ps)
                unw = {f'{r.set}|{r.level}': dict(mrr=float(r.mrr), n=int(r.n)) for r in met[met.weight == 'unw'].itertuples()}
                if PLAN.get('evals_dir'):
                    res.to_parquet(os.path.join(PLAN['evals_dir'], tag + '.parquet'), index=False)
                emit(kind='evaluator', tag=tag, query=q, struct_unw=unw, n_rows=int(len(res)), secs=round(time.time() - t, 1))
        except Exception as e:
            emit(kind='error', at='evaluator', error=repr(e), trace=traceback.format_exc()[-2000:])
    if PLAN.get('fp32_check') and keep.get('A') is not None and left() > 0:
        _AC2 = torch.autocast

        class _Off(_AC2):
            # numeric-sensitivity reference only: autocast disabled (float32)
            def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
                super().__init__(device_type, dtype=dtype, enabled=False, cache_enabled=cache_enabled)
        torch.autocast = _Off
        try:
            t = time.time(); m = keep['A']
            r = dict(np=window_mrr(m, D, np_ids, DEV)[0], npm=window_mrr(m, D, np_ids, DEV, merged=True)[0],
                     nplib=window_mrr(m, D, nplib_ids, DEV)[0], syn=window_mrr(m, D, syn_ids, DEV)[0])
            emit(kind='val_fp32', model='A', recomputed=r, secs=round(time.time() - t, 1))
        except Exception as e:
            emit(kind='error', at='fp32', error=repr(e))
        finally:
            torch.autocast = _AC2
    emit(kind='done')


# ------------------------------------------------------------------------------------------------ step0
# Both functions run under torch.no_grad(): the forward values do not depend on autograd (same kernels, same dropout
# draws), and without it a bf16-emulated T4 forward (math attention materialised in fp32) would hold ~10 GB of
# activations next to two Data pools. script_A's 'equals_harness_line' checks the identity against his script on the GPU.
def step0_forward(model, inp, dtype):
    # 06_train_fpnet.py main(): the forward of the step loop (verbatim apart from the dtype argument)
    with autocast(dtype):
        z, el = model(*inp)
    return z.float(), el.float()


def step0_losses(D, z, el, pos, cand):
    # 06_train_fpnet.py main(), from the forward to the step-0 print (verbatim)
    ypos = fpnet.unpack_gpu(D.PFP[pos], D.nbits)
    lb = F.binary_cross_entropy_with_logits(z, ypos)
    FPc = fpnet.unpack_gpu(D.PFP[cand], D.nbits)                 # (B,K+1,nbits)
    raw = torch.bmm(FPc, z.unsqueeze(-1)).squeeze(-1)
    sc = (raw - raw.mean(1, keepdim=True)) / math.sqrt(D.nbits)
    tgt = torch.zeros(len(pos), dtype=torch.long, device=DEV)
    lc = F.cross_entropy(sc, tgt)
    le = F.mse_loss(el, D.elem[pos])
    hit = (sc.argmax(1) == 0)
    return dict(bce=lb.item(), ctr=lc.item(), el=le.item(), top1=hit.float().mean().item(), top1_count=int(hit.sum().item()))


def batch_multi_k(D, ids, Ks):
    """Data.batch(ids, 0.5, True) as runs with --K Ks[0], Ks[1], ... would draw it. His batch() draws the items (all
    their rng use) before negatives(), and the items do not depend on K: so his batch() runs once with K = Ks[0], the rng
    state at its negatives() call is captured, and the other K redraw negatives() from that same state."""
    cap = {}
    neg = D.negatives

    def capture(pos_):
        cap['state'] = D.rng.bit_generator.state; cap['pos'] = pos_
        return neg(pos_)
    k0 = D.K
    try:
        D.negatives = capture                       # instance attribute; removed again below
        D.K = Ks[0]
        inp, pos, cand = D.batch(ids, 0.5, True)
        cands = {Ks[0]: cand}
        for K in Ks[1:]:
            D.rng.bit_generator.state = cap['state']; D.K = K
            negs = neg(cap['pos'])
            cands[K] = torch.as_tensor(np.concatenate([cap['pos'][:, None], negs], 1), device=D.device)   # = Data.batch
    finally:
        D.__dict__.pop('negatives', None); D.K = k0
    return inp, pos, cands


def draw_seed(s, j):
    return (int(s) + 1) * 100003 + int(j)       # dropout redraws j >= 1 (j = 0 is the faithful seed s)


def mode_step0():
    # configs: {name: holdout string (K 63) | dict(holdout, K)}; one Data object per holdout, shared by its K variants
    spec = {c: ((v, 63) if isinstance(v, str) else (v['holdout'], int(v.get('K', 63)))) for c, v in PLAN['configs'].items()}
    Ds, dlines = {}, {}
    for cfg, (hold, K) in spec.items():
        if hold not in Ds:
            D, dline, secs = make_data(hold)
            Ds[hold], dlines[hold] = D, (dline, secs)
        emit(kind='data', cfg=cfg, holdout=hold, K=K, line=dlines[hold][0], secs=dlines[hold][1])
    nbits = next(iter(Ds.values())).nbits
    n_done = 0
    for job in PLAN['jobs']:
        if left() < 0:
            emit(kind='deadline', next_seed=job['seed'], done=n_done); break
        s = int(job['seed']); t = time.time()
        many = int(job.get('draws', 1)) > 1
        show = many or not QUIET or n_done % QUIET == 0
        torch.manual_seed(s)
        model = fpnet.FPNet(nbits, d=512, layers=8, heads=8).to(DEV)   # 06_train_fpnet.py: FPNet(D.nbits, d=a.d, ...)
        groups = {}
        for cfg in job['configs']:
            groups.setdefault(spec[cfg][0], []).append(cfg)
        for hold, cfgs in groups.items():
            D = Ds[hold]
            D.rng = np.random.default_rng(s)        # = Data(..., seed=s): Data.__init__ draws no random number
            ids = D.sample_ids(256)
            Ks = list(dict.fromkeys(spec[c][1] for c in cfgs))
            inp, pos, cands = batch_multi_k(D, ids, Ks)
            for dt in job['dtypes']:
                for j in range(int(job.get('draws', 1))):
                    cs = s if j == 0 else draw_seed(s, j)
                    if DEV == 'cuda':
                        torch.cuda.manual_seed(cs)   # dropout generator at (seed, offset 0), as after torch.manual_seed
                    else:
                        torch.manual_seed(cs)
                    try:
                        with torch.no_grad():
                            z, el = step0_forward(model, inp, DT[dt])     # one forward serves every K of this holdout
                            res = {K: step0_losses(D, z, el, pos, cands[K]) for K in Ks}
                        del z, el
                        for c in cfgs:
                            emit(show=show, kind='step0', seed=s, cfg=c, K=spec[c][1], dtype=dt, draw=j, cuda_seed=cs,
                                 **res[spec[c][1]])
                    except Exception as e:
                        for c in cfgs:
                            emit(kind='error', seed=s, cfg=c, dtype=dt, draw=j, error=repr(e)[:500])
            del inp, pos, cands
        del model
        n_done += 1
        emit(show=show, kind='seed_done', seed=s, n_done=n_done, secs=round(time.time() - t, 2))
    emit(kind='done', n_done=n_done)


try:
    mode_eval() if A.mode == 'eval' else mode_step0()
except Exception as e:
    emit(kind='fatal', error=repr(e), trace=traceback.format_exc()[-3000:])
    raise
'''

SCRIPT_LAUNCHER = r'''
# casmi26-f6-repro2 launcher, used ONLY when the bf16 probe failed: patches torch.autocast (bf16 -> fp16) and then runs
# scripts/06_train_fpnet.py unmodified as __main__ with the given arguments.
import runpy, sys, torch
_AC = torch.autocast


class _Fp16Autocast(_AC):
    def __init__(self, device_type, dtype=None, *args, **kw):
        if dtype is torch.bfloat16:
            dtype = torch.float16
        super().__init__(device_type, dtype, *args, **kw)


torch.autocast = _Fp16Autocast
print('F6 PATCH torch.autocast: bfloat16 -> float16 (the bf16 probe failed on this GPU)', flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
'''


# ── 1. Inputs ────────────────────────────────────────────────────────────────────────────────────────────────
def find_inputs():
    def first(pattern, ok):
        hits = sorted((p for p in INPUT.rglob(pattern) if ok(p)), key=lambda p: (len(p.parts), str(p)))
        return hits[0] if hits else None
    tc = first('06_train_fpnet.py', lambda p: p.parent.name == 'scripts' and (p.parent.parent / 'casmi' / 'traindata.py').is_file())
    v4 = first('MANIFEST.json', lambda p: (p.parent / 'fe_A.pt').is_file())
    v3 = first('fpnet_1.pt', lambda p: (p.parent / 'fpnet_0.pt').is_file() and not (p.parent / 'fe_A.pt').exists())
    f1 = first('fpnet_full1.pt', lambda p: True)
    sc = first('spec_mz.npy', lambda p: (p.parent / 'split.parquet').is_file())
    vp = first('pool_fp.npy', lambda p: (p.parent / 'pool_meta.parquet').is_file())
    tp = first('train.parquet', lambda p: 'casmi26' in str(p).lower())
    roots = dict(training_code=tc.parent.parent if tc else None, v4b=v4.parent if v4 else None,
                 v3=v3.parent if v3 else None, full1=f1.parent if f1 else None,
                 spec_cache=sc.parent if sc else None, v2_pool=vp.parent if vp else None)
    missing = [k for k, v in roots.items() if v is None]
    ROOTS.update(roots)
    ROOTS['train_parquet'] = tp
    check('inputs_found', not missing, **{k: str(v) for k, v in roots.items()}, train_parquet=str(tp), missing=missing)
    if missing:
        raise SystemExit(f'F6 FATAL: datasets not mounted: {missing}')
    return roots


def step_published():
    log('== 1. published inputs')
    R = ROOTS
    man = json.loads((R['v4b'] / 'MANIFEST.json').read_text())
    rows, n_ok = {}, 0
    for rel, want in sorted(man['files'].items()):
        p = R['v4b'] / rel
        got = sha256(p) if p.is_file() else None
        rows[rel] = 'absent' if got is None else ('match' if got == want else f'MISMATCH {got}')
        n_ok += got == want
    check('v4b_manifest_sha256', n_ok == len(man['files']), matched=n_ok, listed=len(man['files']),
          mismatched={k: v for k, v in rows.items() if v != 'match'}, manifest_created=man.get('created'),
          fe_models=man.get('fe_models'))
    files = {'fe_A.pt': R['v4b'] / 'fe_A.pt', 'fe_B.pt': R['v4b'] / 'fe_B.pt', 'fe_D.pt': R['v4b'] / 'fe_D.pt',
             'fpnet_full1.pt': R['full1'] / 'fpnet_full1.pt'}
    got = {k: sha256(p) if p.is_file() else None for k, p in files.items()}
    v3 = {k: sha256(R['v3'] / k) if (R['v3'] / k).is_file() else None for k in ('fpnet_0.pt', 'fpnet_1.pt')}
    v4fp0 = man['files'].get('fpnet_0.pt')
    ok = all(got[k] == README_SHA[k] for k in README_SHA) and v3['fpnet_0.pt'] == README_SHA['fe_A.pt'] and \
        v3['fpnet_1.pt'] == README_SHA['fe_B.pt'] and v4fp0 == README_SHA['fe_A.pt']
    check('model_files_sha256', ok, readme=README_SHA, mounted=got, v3=v3, v4b_fpnet_0_manifest=v4fp0,
          note='fe_A = v4b fpnet_0 = v3 fpnet_0 (FPNet A); fe_B = v3 fpnet_1 (FPNet B)')
    REPORT['model_files'] = {k: str(p) for k, p in files.items()}
    arch = {k: torch_archive_prefix(p) for k, p in files.items()}
    check('model_archive_names', all(arch[k] == ARCHIVE_NAME[k] for k in ARCHIVE_NAME), archive_prefix=arch,
          expected=ARCHIVE_NAME, note='torch.save names the zip folder after the file it wrote: work/fpnet_a.pt, '
          'work/fpnet_b.pt, work/fpnet_full1.pt, .../pilot_d1_step20k_ema.pt (v4b MANIFEST fe_models sources)')
    tree, tfiles = tree_hash(R['training_code'])
    pins = {k: ('match' if tfiles.get(k) == v else ('absent' if k not in tfiles else f'MISMATCH {tfiles[k]}'))
            for k, v in TC_PINS.items()}
    bad = {k: v for k, v in pins.items() if v != 'match'}
    check('training_code_pins', not bad, pinned=len(TC_PINS), mismatched=bad, tree_sha256=tree, tree_files=len(tfiles),
          tree_equals_2026_10_19=tree == TC_TREE_SHA256)
    REPORT['training_code_tree'] = dict(sha256=tree, files=len(tfiles))
    info = json.loads((R['training_code'] / 'dreams_ft/s2_eval/D20k_eval_info.json').read_text())
    ev_sha = info.get('sha256', {})
    check('d20k_report_files', ev_sha.get('A') == README_SHA['fe_A.pt'] and ev_sha.get('B') == README_SHA['fe_B.pt'] and
          ev_sha.get('D20k') == README_SHA['fe_D.pt'], d20k_eval_info_sha256=ev_sha, gpu=info.get('gpu'),
          torch=info.get('torch'), note='the frozen Evaluator report was computed on exactly the published files')
    ident = {}
    for f in CODE_IDENTITY:
        tcs = tfiles.get('casmi/' + f)
        v3p = R['v3'] / 'code' / 'casmi' / f
        ident[f] = dict(training_code=tcs, v4b=man['files'].get('code/casmi/' + f),
                        v3=sha256(v3p) if v3p.is_file() else None)
    same = all(v['training_code'] == v['v4b'] and v['v3'] in (None, v['training_code']) for v in ident.values())
    check('code_identity_casmi', same, files=ident, note='casmi copy run here (training code) = the copy the fork runs '
          '(v4b code/casmi) = v3 code/casmi')
    sc = R['spec_cache']
    srows = {f: ('match' if (sc / f).is_file() and sha256(sc / f) == h else 'MISMATCH/absent') for f, h in SPEC_CACHE_SHA.items()}
    check('spec_cache_sha256', all(v == 'match' for v in srows.values()), files=srows)
    vp = R['v2_pool']
    vrows = {f: sha256(vp / f) if (vp / f).is_file() else None for f in V2POOL_NEEDED}
    check('v2_pool_sha256', all(vrows.values()) and all(vrows.get(k) == v for k, v in V2POOL_KNOWN.items()),
          known_pins=V2POOL_KNOWN, sha256=vrows)
    tp = ROOTS.get('train_parquet')
    if tp is not None:
        h = sha256(tp)
        check('competition_train_parquet', None, path=str(tp), sha256=h, equals_sanpier_data_audit=h == TRAIN_PARQUET_SHA,
              note='not read by K2; recorded to tie spec-cache / v2-pool to the competition file (K0 rebuild)')


# ── 2. Environment ───────────────────────────────────────────────────────────────────────────────────────────
PROBE = r'''
import importlib, json, os, platform, sys, time
def v(m):
    try:
        return getattr(importlib.import_module(m), '__version__', 'present')
    except Exception as e:
        return f'absent ({type(e).__name__})'
out = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=v('numpy'), pandas=v('pandas'),
           pyarrow=v('pyarrow'), numba=v('numba'), torch=v('torch'), cpus=len(os.sched_getaffinity(0)),
           mem_gb=round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1))
try:
    from rdkit import rdBase
    out['rdkit'] = rdBase.rdkitVersion
except Exception as e:
    out['rdkit'] = f'absent ({type(e).__name__})'
info = open('/proc/cpuinfo').read().splitlines()
out['cpu'] = next((l.split(':', 1)[1].strip() for l in info if l.startswith('model name')), None)
out['avx512f'] = 'avx512f' in next((l for l in info if l.startswith('flags')), '').split()
import torch, torch.nn.functional as F
out['cuda'] = torch.cuda.is_available()
if out['cuda']:
    p = torch.cuda.get_device_properties(0)
    out.update(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)), gpu_mem_gb=round(p.total_memory / 2 ** 30, 1),
               sm_count=p.multi_processor_count, n_gpus=torch.cuda.device_count(), torch_cuda=torch.version.cuda)
    for kw in ({}, {'including_emulation': False}):
        try:
            out['bf16_supported' + ('_native' if kw else '')] = torch.cuda.is_bf16_supported(**kw)
        except Exception as e:
            out['bf16_supported' + ('_native' if kw else '')] = repr(e)[:200]
    def probe(dt):
        r = {}
        try:
            torch.manual_seed(0)
            x = torch.randn(64, 161, 512, device='cuda', requires_grad=True)
            lin = torch.nn.Linear(512, 1536).cuda()
            pad = torch.zeros(64, 161, dtype=torch.bool, device='cuda'); pad[:, 120:] = True
            with torch.autocast('cuda', dtype=dt):
                q, k, vv = lin(x).view(64, 161, 3, 8, 64).permute(2, 0, 3, 1, 4)
                att = F.scaled_dot_product_attention(q, k, vv, attn_mask=(~pad)[:, None, None, :])
                y = F.dropout(F.gelu(att), 0.1, True)
            r['autocast_dtype'] = str(y.dtype)
            y.float().pow(2).mean().backward(); torch.cuda.synchronize()
            r['grad_finite'] = bool(torch.isfinite(x.grad).all())
            a = torch.randn(4096, 4096, device='cuda'); b = torch.randn(4096, 4096, device='cuda')
            with torch.autocast('cuda', dtype=dt):
                c = a @ b
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5):
                with torch.autocast('cuda', dtype=dt):
                    c = a @ b
            torch.cuda.synchronize(); r['matmul_tflops'] = round(5 * 2 * 4096 ** 3 / (time.time() - t) / 1e12, 2)
            r['ok'] = bool(r['grad_finite']) and r['autocast_dtype'] == str(dt)
        except Exception as e:
            r['ok'] = False; r['error'] = repr(e)[:500]
        return r
    out['probe'] = {n: probe(d) for n, d in (('bf16', torch.bfloat16), ('fp16', torch.float16))}
print(json.dumps(out))
'''


def step_env():
    log('== 2. environment')
    subprocess.run(f'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; '
                   f'free -g | head -2; df -h {WORKING} | tail -1', shell=True)
    try:
        out = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, timeout=600)
        env = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else \
            dict(error=out.stderr[-2000:])
    except Exception as e:
        env = dict(error=repr(e))
    pr = env.get('probe', {})
    why = None
    if env.get('cuda'):
        b = pr.get('bf16', {})
        if not b.get('ok'):
            why = 'bf16 refused: ' + str(b.get('error') or b)[:300]
        elif (b.get('matmul_tflops') or 0) < BF16_MIN_TFLOPS:
            why = f"bf16 emulated at {b.get('matmul_tflops')} TFLOPS < {BF16_MIN_TFLOPS}: too slow for the budget"
        STATE['autocast'] = 'fp16-patched' if why else 'bf16'
        STATE['scan_dtype'] = 'bf16' if (env.get('capability') or [0])[0] >= 8 else 'fp16'
    else:
        STATE['autocast'] = 'bf16'; STATE['scan_dtype'] = 'fp16'
    REPORT['environment'] = dict(ours=env, his=HIS_ENV, autocast=STATE['autocast'], scan_dtype=STATE['scan_dtype'],
                                 fp16_patch_reason=why)
    check('environment', None, ours=env, his=HIS_ENV, autocast_mode=STATE['autocast'], scan_dtype=STATE['scan_dtype'],
          fp16_patch_reason=why,
          note='autocast bf16 = his code as published; fp16-patched = bf16 refused (or emulated too slowly), one printed '
               'runtime patch')
    if DEVICE == 'cuda' and not env.get('cuda'):
        raise SystemExit('F6 FATAL: no CUDA device (this kernel needs the T4 accelerator)')


# ── 3. Layout (no source edits) ──────────────────────────────────────────────────────────────────────────────
LAYOUT = [('casmi', 'src/casmi'), ('scripts', 'scripts'), ('fpnet_v2', 'work/analysis/fpnet_v2'),
          ('dreams_ft', 'work/analysis/dreams_ft')]


def build_layout(tc_root, repo, data_dirs):
    """Copies of his code at the README section-4 paths, work/* symlinks to the data files, and the relative-ROOT alias
    'C:/Users/omen/Desktop/envedaCasmi' (a real directory holding symlinks src and work, no loop). Returns the map."""
    repo = Path(repo)
    if repo.exists():
        remove_links(repo); shutil.rmtree(repo)
    for src, rel in LAYOUT:
        shutil.copytree(Path(tc_root) / src, repo / rel)
    work = repo / 'work'
    linked = {}
    for d in data_dirs:
        for p in sorted(Path(d).iterdir()):
            if p.is_file() and not (work / p.name).exists():
                (work / p.name).symlink_to(p); linked[p.name] = str(p)
    alias = repo / WIN_ROOT
    alias.mkdir(parents=True)
    (alias / 'src').symlink_to(repo / 'src')
    (alias / 'work').symlink_to(repo / 'work')
    return linked


def layout_pins(repo):
    bad = []
    for rel, want in TC_PINS.items():
        top = rel.split('/', 1)[0]
        dst = dict(LAYOUT).get(top)
        if dst is None:
            continue
        p = Path(repo) / dst / rel.split('/', 1)[1]
        if not p.is_file() or sha256(p) != want:
            bad.append(str(p))
    return bad


def step_layout():
    log('== 3. layout: README section 4 restored by copies + symlinks; no source file is edited')
    for d in (LOGS, RES, TMP, RES / 'evals'):
        d.mkdir(parents=True, exist_ok=True)
    linked = build_layout(ROOTS['training_code'], REPO, [ROOTS['spec_cache'], ROOTS['v2_pool']])
    bad = layout_pins(REPO)
    need = list(SPEC_CACHE_SHA) + V2POOL_NEEDED
    layout = {'src/casmi': 'copy of casmi/', 'scripts': 'copy of scripts/', 'work/analysis/fpnet_v2': 'copy of fpnet_v2/',
              'work/analysis/dreams_ft': 'copy of dreams_ft/', 'work/*': 'symlinks to casmi26-spec-cache/cache/* + casmi26-v2-pool/*',
              WIN_ROOT + '/{src,work}': "symlinks: v2lib's hard-coded ROOT is relative on Linux"}
    log('source diff vs the published training code: NONE (layout only)')
    for k, v in layout.items():
        log(f'  {k:52s} {v}')
    check('layout', not bad and all(n in linked for n in need), source_edits=0, copies_differing_from_pins=bad,
          work_links=sorted(linked), layout=layout, autocast=STATE['autocast'])


# ── harness / script runner ──────────────────────────────────────────────────────────────────────────────────
def run_harness(name, mode, plan, est, timeout_min=None):
    st = REPORT['steps']
    if not can_start(est, minutes()):
        st[name] = dict(status='skipped', reason=f'budget: {minutes()} min + est {est} > {BUDGET_MIN}'); save_report()
        log(f'F6 STEP {name}: SKIPPED (budget)'); return None
    hp = RW / 'f6k2_harness.py'
    hp.write_text(HARNESS)
    plan = dict(plan, device=DEVICE, autocast=STATE['autocast'])
    pp = RES / f'{name}_plan.json'; pp.write_text(json.dumps(plan))
    out = RES / f'{name}.jsonl'
    if out.exists():
        out.unlink()
    tmo = timeout_min or step_timeout_min(est, minutes())
    log(f'== F6 STEP {name} ({minutes()} min; est {est} min, timeout {tmo:.0f} min): harness {mode}')
    rc, sec, to = run_logged([sys.executable, '-u', str(hp), mode, '--plan', str(pp), '--out', str(out)],
                             LOGS / f'{name}.log', timeout=tmo * 60, cwd=REPO,
                             env={**os.environ, **CHILD_ENV, 'PYTHONDONTWRITEBYTECODE': '1'})
    status = 'ok' if rc == 0 else ('timeout' if to else 'failed')
    st[name] = dict(status=status, exit=rc, seconds=sec, est_min=est, timeout_min=round(tmo, 1)); save_report()
    log(f'F6 STEP {name}: {status} exit {rc} in {sec / 60:.1f} min')
    return read_jsonl(out)


def run_script(name, args, est=6, force_fp16=False):
    """scripts/06_train_fpnet.py, unmodified, from the repository root (launcher only if bf16 was refused, or for the
    one retry after a bf16 out-of-memory, see run_script_oom)."""
    st = REPORT['steps']
    if not can_start(est, minutes()):
        st[name] = dict(status='skipped', reason='budget'); save_report()
        log(f'F6 STEP {name}: SKIPPED (budget)'); return None
    out_dir = TMP / name; out_dir.mkdir(parents=True, exist_ok=True)
    full = ['--steps', '1', *args, '--out', str(out_dir / f'{name}.pt')]
    patched = force_fp16 or STATE['autocast'] == 'fp16-patched'
    if patched:
        lp = RW / 'f6k2_launcher.py'; lp.write_text(SCRIPT_LAUNCHER)
        cmd = [sys.executable, '-u', str(lp), 'scripts/06_train_fpnet.py', *full]
    else:
        cmd = [sys.executable, '-u', 'scripts/06_train_fpnet.py', *full]
    tmo = step_timeout_min(est, minutes())
    shown = ' '.join(c if c != sys.executable else 'python' for c in cmd)
    log(f'== F6 STEP {name} ({minutes()} min; timeout {tmo:.0f} min): cd repo && {shown}')
    rc, sec, to = run_logged(cmd, LOGS / f'{name}.log', timeout=tmo * 60, cwd=REPO, env={**os.environ, **CHILD_ENV})
    status = 'ok' if rc == 0 else ('timeout' if to else 'failed')
    mode = 'fp16-patched' if patched else 'bf16'
    st[name] = dict(status=status, exit=rc, seconds=sec, cmd=shown, autocast=mode); save_report()
    shutil.rmtree(out_dir, ignore_errors=True)          # the 1-step checkpoints are not results
    log(f'F6 STEP {name}: {status} exit {rc} in {sec / 60:.1f} min (autocast {mode})')
    text = (LOGS / f'{name}.log').read_text(errors='replace')
    oom = 'out of memory' in text.lower() or 'outofmemoryerror' in text.lower()
    return dict(status=status, text=text, head=head_lines(text), step0=parse_step0(text), autocast=mode, oom=oom,
                args=' '.join(args))


def run_script_oom(name, args):
    """run_script; if his script ran out of GPU memory in (T4-emulated) bf16, one retry with the printed fp16 launcher
    patch. His step needs the bf16 math-attention graph of 256 x 161 tokens (~10 GB) next to the pool fingerprints;
    the retry is reported as such (its step-0 line is then an fp16 line, not comparable bit for bit with the bf16
    harness line)."""
    r = run_script(name, args)
    if r and r['status'] == 'failed' and r.get('oom') and STATE['autocast'] == 'bf16':
        shutil.copy(LOGS / f'{name}.log', LOGS / f'{name}_bf16_oom.log')
        log(f'F6 NOTE {name}: CUDA out of memory in bf16 -> one retry with the fp16 launcher patch')
        r = run_script(name, args, force_fp16=True)
        if r:
            r['retry_after_bf16_oom'] = True
    return r


def his(rel):
    return (ROOTS['training_code'] / rel).read_text(errors='replace')


# ── 4. VAL + Evaluator ───────────────────────────────────────────────────────────────────────────────────────
FP_KEYS = ('np', 'npm', 'nplib', 'syn')
D_KEYS = ('np-dev', 'np-dev_merged', 'nplib')
FP_LOGS = {'A': 'logs/log_train_a.txt', 'B': 'logs/log_train_b.txt', 'full1': 'logs/log_train_full1.txt'}


def step_eval():
    R = REPORT['model_files']
    plan = dict(fpnets={'A': R['fe_A.pt'], 'B': R['fe_B.pt'], 'full1': R['fpnet_full1.pt']}, fe_D=R['fe_D.pt'],
                evaluator=True, fp32_check=True, evals_dir=str(RES / 'evals'),
                deadline=time.time() + 40 * 60)
    rows = run_harness('eval', 'eval', plan, est=14, timeout_min=45)
    if rows is None:
        return
    REPORT['results']['eval_rows'] = [r for r in rows if r.get('kind') != 'evaluator'][:20]
    loga = his('logs/log_train_a.txt')
    hd = head_lines(loga)
    dl = next((r for r in rows if r.get('kind') == 'data'), {})
    vs = next((r for r in rows if r.get('kind') == 'valsets'), {})
    check('eval_data_lines', dl.get('line') == hd.get('data') and vs.get('line') == hd.get('valsets'),
          ours=[dl.get('line'), vs.get('line')], logged=[hd.get('data'), hd.get('valsets')])
    vals = {r['model']: r for r in rows if r.get('kind') == 'val'}
    out = {}
    for key, logf in FP_LOGS.items():
        r = vals.get(key)
        if r is None:
            check(f'val_{key}', False, error='not computed (see harness log)'); continue
        lines = parse_val_lines(his(logf))
        step = r['header'].get('step')
        cmp = val_compare(lines.get(step), r['recomputed'], FP_KEYS, saved=r['header'].get('val'))
        cmp.update(file_step=step, logged_best_step=max(lines, key=lambda s: 0.5 * lines[s]['np'] + 0.5 * lines[s]['npm'])
                   if lines else None, nearest_logged_steps=nearest_steps(lines, r['recomputed'], FP_KEYS),
                   header={k: v for k, v in r['header'].items() if k != 'val'}, weights_only_load=r.get('weights_only'),
                   secs=r.get('secs'), log=logf)
        out[key] = cmp
        check(f'val_{key}', cmp['status'] == 'REPRODUCED', **cmp)
    fp32 = next((r for r in rows if r.get('kind') == 'val_fp32'), None)
    if fp32 and 'A' in vals:
        check('val_A_fp32_sensitivity', None, bf16=vals['A']['recomputed'], fp32=fp32['recomputed'],
              max_abs_bf16_vs_fp32=max(abs(vals['A']['recomputed'][k] - fp32['recomputed'][k]) for k in FP_KEYS),
              note='numeric sensitivity of window_mrr on this GPU (rank ties): compare with val_A max_abs_diff')
    d = vals.get('D')
    if d is not None:
        lines = parse_val_lines(his('dreams_ft/logs/pilot_d1.txt'))
        summ = json.loads(his('dreams_ft/logs/pilot_d1_summary.json'))
        hist = {h['step']: h['val'] for h in summ.get('hist', []) if 'val' in h}
        c20 = val_compare(lines.get(20000), d['recomputed'], D_KEYS, saved=hist.get(20000))
        c18 = val_compare(lines.get(18000), d['recomputed'], D_KEYS)
        hdr = d['header']
        args_eq = hdr.get('args') == summ.get('args')
        check('fe_D_header', hdr.get('step') == 20000 and hdr.get('val') is None and args_eq and hdr.get('arch') == 'dreams_fp',
              step=hdr.get('step'), val=hdr.get('val'), arch=hdr.get('arch'), args_equal_pilot_d1_summary=args_eq,
              args=hdr.get('args'), fcfg=hdr.get('fcfg'))
        ok = c20['status'] == 'REPRODUCED' and c20['max_abs_diff'] < c18['max_abs_diff']
        out['D'] = dict(step20k=c20, step18k=c18)
        check('val_D', ok, vs_step20000=c20, vs_step18000_best_unpublished=dict(status=c18['status'],
              max_abs_diff=c18['max_abs_diff']), n=d.get('n'), nearest_logged_steps=nearest_steps(lines, d['recomputed'], D_KEYS),
              note='published fe_D must behave as the step-20k EMA (not the step-18k "best")')
    REPORT['results']['val'] = out
    evs = {r['tag']: r for r in rows if r.get('kind') == 'evaluator'}
    if evs:
        rep = parse_d20k_report(his('dreams_ft/s2_eval/D20k_report.txt'))
        chk = his('dreams_ft/s2_eval/check_D20k.log')
        pil = his('dreams_ft/logs/pilot_d1.txt')
        res = {}
        for tag, ours in evs.items():
            c = evaluator_compare(ours, rep, tag)
            res[tag] = c
            check(f'evaluator_{tag}', (c['status'] == 'REPRODUCED') if tag in ('A', 'B', 'D20k') else None, **c,
                  n_structures={k: v['n'] for k, v in ours.get('struct_unw', {}).items() if k.endswith('|merged')})
        if 'D20k' in evs:
            q = evs['D20k']['query']
            l20 = parse_evaluator_line(chk, 'D20k') or {}
            l18 = parse_evaluator_line(pil, 'pilot_d1') or {}
            m20 = max((abs(q[k] - v) for k, v in l20.items() if k in q), default=None)
            m18 = max((abs(q[k] - v) for k, v in l18.items() if k in q), default=None)
            check('evaluator_D20k_is_step20k', m20 is not None and m18 is not None and m20 <= EVAL_TOL and m20 < m18,
                  max_abs_vs_check_D20k_line=m20, max_abs_vs_pilot_d1_final_eval_step18k=m18,
                  note='check_D20k.log = the published file; pilot_d1.txt final eval = the trainer best (step 18k)')
        check('evaluator_weighted_rows', None, reproducible=False,
              reason='w_full / w_temp rows need fpnet_v2/truth_g.parquet (load_truth_g), which is not in the training-code dataset')
        REPORT['results']['evaluator'] = res


# ── 5 / 7. His script ────────────────────────────────────────────────────────────────────────────────────────
def compare_script_run(name, run, target_key, harness_line=None, stats=None):
    if run is None:
        return None
    logged_text = his(STEP0_TARGETS[target_key][0])
    lh, oh = head_lines(logged_text), run['head']
    keys = ['data', 'params', 'valsets']
    same = {k: oh.get(k) == lh.get(k) for k in keys}
    s0, lg = run['step0'], parse_step0(logged_text)
    d = dict(status=run['status'], header_equal=same, ours_header=oh, logged_header=lh,
             step0_ours=s0 and s0['line'], step0_logged=lg and lg['line'], script_args=run.get('args'),
             script_autocast=run.get('autocast'), harness_autocast=STATE['autocast'],
             retry_after_bf16_oom=bool(run.get('retry_after_bf16_oom')))
    if s0 and lg:
        d['diff'] = {k: round(s0[k] - lg[k], 5) for k in STEP0_KEYS}
        d['map_criterion_top1_exact_losses_2e-3'] = s0['top1_count'] == lg['top1_count'] and \
            all(abs(s0[k] - lg[k]) <= 2e-3 for k in ('bce', 'ctr', 'el'))
    if harness_line is not None and s0:
        d['equals_harness_line'] = format_step0(s0) == harness_line
        d['harness_line'] = harness_line
    if stats is not None and lg:
        chi, z = chi2_vs(lg, stats)
        d['chi2_logged_vs_redraws'] = round(chi, 2); d['z'] = {k: round(v, 2) for k, v in z.items()}
    ok = run['status'] == 'ok' and all(same.values()) and s0 is not None
    check(name, ok, **d)
    REPORT['results'][name] = d
    return d


# ── 6. Seed scan ─────────────────────────────────────────────────────────────────────────────────────────────
def step_seeds():
    targets, configs = build_step0_targets({t: parse_step0(his(logf)) for t, (logf, _, _) in STEP0_TARGETS.items()})
    STATE['targets'], STATE['configs'] = targets, configs
    REPORT['results']['step0_targets'] = targets
    kinf = {t: dict(ctr=tg['ctr'], K=tg['K'], ctr_minus_lnK1=tg['ctr_minus_lnK1'], cfg=tg['cfg'],
                    k_candidates=tg['k_candidates']) for t, tg in targets.items()}
    check('step0_K', targets.get('A', {}).get('K') == K_DEFAULT and len(targets['A']['k_candidates']) == 1, targets=kinf,
          configs=configs, k_off_max=K_OFF_MAX,
          note='step-0 ctr = ln(K + 1) + O(0.005) at init: K per logged line (A must be the default 63). B -> 95 means '
               'B ran with --K 95 (96 candidates), not the README command; its seed is scanned with that K')
    dt_scan = STATE['scan_dtype']
    dts_cal = sorted({dt_scan, 'bf16'})
    main_seeds, ext_seeds = seed_plan()
    REPORT['results']['seed_range'] = dict(main=f'{len(main_seeds) + 1} seeds: 0-1023, 2000-2030, 10000-10031, '
                                                f'20260901-20260930, constants {SEED_CONSTANTS}',
                                           extension='1024-4095 (only if B or full1 unmatched and time remains)',
                                           why='his recorded seeds: 0-3, 7, 11, 12, 21, 123, 2026, 2027, 10000+seed')
    # verification + 2 script runs + finish must still fit after the scan
    after = 12 + 2 * 6 + RESERVE_MIN
    scan_min = max(5.0, min(70.0, BUDGET_MIN - minutes() - after))
    cfgs_all = list(configs)
    jobs = [dict(seed=SEED_A, configs=cfgs_all, dtypes=dts_cal, draws=N_DRAWS)] + \
        [dict(seed=s, configs=cfgs_all, dtypes=[dt_scan], draws=1) for s in main_seeds]
    plan = dict(configs=configs, jobs=jobs, deadline=time.time() + scan_min * 60, print_every=50)
    rows = run_harness('scan', 'step0', plan, est=min(scan_min, 25), timeout_min=scan_min + 8)
    if rows is None:
        return
    allrows = list(rows)
    data_lines = {r['cfg']: r['line'] for r in rows if r.get('kind') == 'data'}
    for t in ('A', 'full1'):
        cfg = STEP0_TARGETS[t][1]
        lh = head_lines(his(STEP0_TARGETS[t][0]))
        check(f'data_line_{cfg}', data_lines.get(cfg) == lh.get('data'), ours=data_lines.get(cfg), logged=lh.get('data'),
              holdout=CONFIGS[cfg], log=STEP0_TARGETS[t][0])
    cal = {(cfg, dt): draw_stats(draws_of(allrows, SEED_A, cfg, dt)) for cfg in configs for dt in dts_cal
           if draws_of(allrows, SEED_A, cfg, dt)}
    if not cal:
        check('step0_calibration', False, error='no calibration rows'); return
    ref_dt = 'bf16'
    var = {cfg: scan_var(cal[(cfg, dt_scan)], cal.get((cfg, ref_dt))) for cfg in configs if (cfg, dt_scan) in cal}
    check('step0_calibration', None, seed=SEED_A, draws=N_DRAWS, scan_dtype=dt_scan,
          stats={f'{c}|{d}': v for (c, d), v in cal.items()}, scan_sd={c: {k: math.sqrt(v[k]) for k in v} for c, v in var.items()},
          note='spread of the step-0 quantities over dropout redraws (same init, same batch): the noise a T4 replay '
               'carries against his GPU masks')
    a7 = draws_of(allrows, SEED_A, 'def', ref_dt)
    if a7 and 'A' in targets:
        st = draw_stats(a7)
        chi, z = chi2_vs(targets['A'], st)
        h0 = next((r for r in a7 if r['draw'] == 0), None)
        d = dict(logged=targets['A']['line'], harness_seed7_draw0=format_step0(h0) if h0 else None,
                 redraw_mean={k: round(v, 5) for k, v in st['mean'].items()}, redraw_sd={k: round(v, 5) for k, v in st['sd'].items()},
                 chi2=round(chi, 2), z={k: round(v, 2) for k, v in z.items()}, chi2_pass=CHI2_PASS,
                 diff_draw0={k: round(h0[k] - targets['A'][k], 5) for k in STEP0_KEYS} if h0 else None)
        REPORT['results']['step0_A_seed7'] = d
        check('step0_A_seed7', chi <= CHI2_PASS, **d)
    scans = scan_rows(allrows, dt_scan)
    # extension (1024-4095) if B or full1 has no scan-level match and time remains
    need_ext = False
    for t in ('B', 'full1'):
        if t in targets:
            cfg = targets[t]['cfg']
            if cfg in var and scans.get(cfg):
                _, rk, npass = select_candidates(scans[cfg], targets[t], var[cfg])
                need_ext |= npass == 0
    done_main = any(r.get('kind') == 'done' for r in rows)
    ext_min = min(45.0, BUDGET_MIN - minutes() - after)
    if need_ext and done_main and ext_min >= 8:
        jobs = [dict(seed=s, configs=cfgs_all, dtypes=[dt_scan], draws=1) for s in ext_seeds]
        plan = dict(configs=configs, jobs=jobs, deadline=time.time() + ext_min * 60, print_every=100)
        rows2 = run_harness('scan_ext', 'step0', plan, est=min(ext_min, 20), timeout_min=ext_min + 6) or []
        allrows += rows2
        scans = scan_rows(allrows, dt_scan)
    scanned = sorted(scans.get('def', {}))
    REPORT['results']['scan_coverage'] = dict(n_seeds=len(scanned), main_list_done=done_main,
                                              n_main=len(main_seeds) + 1, extension_used=need_ext,
                                              last_seeds=scanned[-3:] if scanned else [])
    koff = {}
    for cfg, rws in scans.items():
        offs = [r['ctr'] - math.log(configs[cfg]['K'] + 1) for r in rws.values()]
        if len(offs) > 1:
            mu = sum(offs) / len(offs)
            koff[cfg] = dict(K=configs[cfg]['K'], n=len(offs), mean=round(mu, 5), min=round(min(offs), 5), max=round(max(offs), 5),
                             sd=round(math.sqrt(sum((o - mu) ** 2 for o in offs) / (len(offs) - 1)), 5))
    best63 = {t: round(min(scan_d2(tg, r, var['def']) for r in scans['def'].values()), 1) for t, tg in targets.items()
              if tg['base_cfg'] == 'def' and 'def' in var and scans.get('def')}
    check('step0_K_scan', None, ctr_minus_lnK1_over_scanned_seeds=koff, best_scan_d2_at_K63=best63,
          logged={t: dict(ctr=tg['ctr'], K=tg['K'], ctr_minus_lnK1=tg['ctr_minus_lnK1']) for t, tg in targets.items()},
          note='if ctr - ln(K + 1) stays within a few 0.001 over every scanned seed, no seed can give B its ctr 4.5678 '
               'under K 63 (its best K-63 scan distance is reported); K is then a recipe fact of B, not seed noise')
    null = {}
    for cfg in var:
        if scans.get(cfg):
            p, e = null_false_rate(scans[cfg], var[cfg])
            null[cfg] = dict(p_false=p, expected_chance_matches=e, n_seeds=len(scans[cfg]))
    REPORT['results']['scan_null'] = null
    check('scan_null_rate', None, **null, threshold=CHI2_PASS,
          note='each scanned seed\'s own row used as a logged line: expected number of OTHER seeds within the scan '
               'test (single-draw noise on both sides, conservative). Info; the verdicts use the local estimate '
               '(local_chance_rate) around each logged line')
    # shortlist per target, then bf16 redraws for the shortlist
    short = {}
    for t, tg in targets.items():
        cfg = tg['cfg']
        if cfg not in var or not scans.get(cfg):
            continue
        cand, ranking, npass = select_candidates(scans[cfg], tg, var[cfg])
        short[t] = dict(cfg=cfg, candidates=cand, n_scan_pass=npass, scan_top=[(round(d2, 2), s) for d2, s in ranking[:6]])
    REPORT['results']['scan_shortlist'] = short
    log('F6 SCAN shortlist ' + json.dumps(short))
    vjobs = {}
    for t, sh in short.items():
        for s in sh['candidates']:
            if s == SEED_A:
                continue                              # seed 7 already has bf16 redraws (calibration)
            vjobs.setdefault(s, set()).add(sh['cfg'])
    if vjobs:
        jobs = [dict(seed=s, configs=sorted(c), dtypes=[ref_dt], draws=N_DRAWS) for s, c in vjobs.items()]
        vmin = max(4.0, min(20.0, BUDGET_MIN - minutes() - 2 * 6 - RESERVE_MIN))
        plan = dict(configs=configs, jobs=jobs, deadline=time.time() + vmin * 60)
        allrows += run_harness('verify', 'step0', plan, est=min(vmin, 8), timeout_min=vmin + 6) or []
    verdicts = {}
    for t, sh in short.items():
        fits = {}
        for s in sh['candidates']:
            dr = draws_of(allrows, s, sh['cfg'], ref_dt)
            if len(dr) >= 3:
                fits[s] = loglik(targets[t], draw_stats(dr))
        best = min(fits, key=lambda s_: fits[s_]['chi2']) if fits else None
        e_loc, n_wide = local_chance_rate(scans[sh['cfg']], targets[t], var[sh['cfg']], exclude=(best,))
        v = seed_verdict(fits, e_false=e_loc, expected=SEED_A if t == 'A' else None)
        v.update(target_line=targets[t]['line'], role=targets[t]['role'], cfg=sh['cfg'], K=targets[t]['K'],
                 n_scan_pass=sh['n_scan_pass'],
                 neighbours_in_widened_region=n_wide,
                 global_chance_matches=null.get(sh['cfg'], {}).get('expected_chance_matches'),
                 scan_top=sh['scan_top'], seeds_scanned=len(scans.get(sh['cfg'], {})),
                 verified={s: round(f['chi2'], 2) for s, f in sorted(fits.items(), key=lambda kv: kv[1]['chi2'])})
        if v.get('seed') is not None:
            h0 = next((r for r in draws_of(allrows, v['seed'], sh['cfg'], ref_dt) if r['draw'] == 0), None)
            v['harness_line_draw0'] = format_step0(h0) if h0 else None
        verdicts[t] = v
        check(f'seed_{t}', v.get('ok') if t == 'A' else None, **{k: x for k, x in v.items() if k != 'ok'})
    REPORT['results']['seed_verdicts'] = verdicts
    return verdicts


# ── finish ───────────────────────────────────────────────────────────────────────────────────────────────────
def step_finish():
    c = REPORT['checks']
    must = ['inputs_found', 'v4b_manifest_sha256', 'model_files_sha256', 'training_code_pins', 'spec_cache_sha256',
            'v2_pool_sha256', 'layout', 'data_line_def', 'data_line_full', 'val_A', 'val_B', 'val_full1', 'script_A_seed7',
            'step0_A_seed7', 'seed_A']
    failed = [k for k in must if not c.get(k, {}).get('ok')]
    opt = ['step0_K', 'val_D', 'fe_D_header', 'evaluator_A', 'evaluator_B', 'evaluator_D20k', 'evaluator_D20k_is_step20k',
           'model_archive_names', 'code_identity_casmi', 'd20k_report_files', 'eval_data_lines']
    opt_failed = [k for k in opt if k in c and c[k].get('ok') is False]
    sv = REPORT['results'].get('seed_verdicts', {})
    REPORT['verdict'] = dict(all_required_ok=not failed, failed_required=failed, failed_optional=opt_failed,
                             not_run=[k for k in must + opt if k not in c],
                             seeds={t: dict(status=v.get('status'), seed=v.get('seed'), K=v.get('K'), best=v.get('best_seed'),
                                            best_chi2=v.get('best_chi2')) for t, v in sv.items()},
                             autocast=STATE['autocast'], minutes=minutes())
    save_report()
    log('F6 SUMMARY ' + json.dumps(REPORT['verdict'], default=_js))
    for k, v in REPORT['results'].get('val', {}).items():
        if k == 'D':
            v = v['step20k']; lg = {kk: m['logged'] for kk, m in v['metrics'].items()}
            log(f"F6 SUMMARY VAL fe_D step 20000: logged {lg} | recomputed "
                f"{ {kk: round(m['recomputed'], 4) for kk, m in v['metrics'].items()} } | max|d| {v['max_abs_diff']:.4f} {v['status']}")
            continue
        log(f"F6 SUMMARY VAL {k} (file step {v.get('file_step')}): logged "
            f"{ {kk: m['logged'] for kk, m in v['metrics'].items()} } | in file "
            f"{ {kk: (round(m['saved_in_file'], 4) if m['saved_in_file'] is not None else None) for kk, m in v['metrics'].items()} } "
            f"| recomputed { {kk: round(m['recomputed'], 4) for kk, m in v['metrics'].items()} } | max|d| "
            f"{v['max_abs_diff']:.4f} {v['status']}")
    for k, v in REPORT['results'].get('evaluator', {}).items():
        log(f"F6 SUMMARY EVALUATOR {k:9s} cells {v['n_cells']:2d} max|d| {v['max_abs_diff']:.4f} {v['status']}")
    a = REPORT['results'].get('step0_A_seed7')
    if a:
        log(f"F6 SUMMARY STEP0 A seed 7: logged '{a['logged']}' | harness draw0 '{a['harness_seed7_draw0']}' | "
            f"redraw chi2 {a['chi2']} (pass <= {CHI2_PASS})")
    for name in ('script_A_seed7', 'script_B', 'script_full1'):
        s = REPORT['results'].get(name)
        if s:
            log(f"F6 SUMMARY {name}: header equal {s['header_equal']} | ours '{s['step0_ours']}' | logged "
                f"'{s['step0_logged']}' | = harness {s.get('equals_harness_line')} | chi2 {s.get('chi2_logged_vs_redraws')}")
    for t, v in sv.items():
        r2 = lambda x, n=2: None if x is None else round(x, n)
        log(f"F6 SUMMARY SEED {t:8s} K {v.get('K')} {v['status']:10s} seed {v.get('seed')} | best {v.get('best_seed')} chi2 "
            f"{r2(v.get('best_chi2'))} posterior {r2(v.get('best_posterior'), 4)} | next {v.get('second_seed')} chi2 "
            f"{r2(v.get('second_chi2'))} | scan d2<=18.47: {v.get('n_scan_pass')} of {v.get('seeds_scanned')} seeds | "
            f"expected chance matches {r2(v.get('expected_chance_matches'), 4)} | P(in range) {r2(v.get('p_in_range'), 4)}")
    cov = REPORT['results'].get('scan_coverage')
    if cov:
        log(f'F6 SUMMARY scan coverage {json.dumps(cov)}')
    log(f'F6 DONE {minutes()} min; report {REPORT_PATH}')


def pick_verdict(verdicts, t):
    """The seed claim for target t among t and its alternative-K targets '<t>@K<k>': RECOVERED before WEAK, then the
    lower verified chi2 (the nearest-K target first on ties). -> (target name, verdict) or (None, None)."""
    rank = {'RECOVERED': 0, 'WEAK': 1}
    c = [(rank[v['status']], v.get('best_chi2', 0.0), n != t, n) for n, v in verdicts.items()
         if (n == t or n.startswith(t + '@K')) and v.get('status') in rank and v.get('seed') is not None]
    if not c:
        return None, None
    n = min(c)[3]
    return n, verdicts[n]


def cleanup():
    n = remove_links(RW)
    shutil.rmtree(TMP, ignore_errors=True)
    if REPO.exists():
        shutil.rmtree(REPO, ignore_errors=True)          # copies of his code only (pinned); nothing generated there
    return n


def main():
    RW.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    try:
        find_inputs()
        step_published()
        step_env()
        step_layout()
        guarded('eval', step_eval)
        run = guarded('script_A', run_script_oom, 'script_A_seed7', [])
        REPORT['results']['_script_A_run'] = run and dict(status=run['status'], head=run['head'], step0=run['step0'])
        verdicts = guarded('seeds', step_seeds) or {}
        rows = read_jsonl(RES / 'scan.jsonl') + read_jsonl(RES / 'verify.jsonl')
        h7 = next((r for r in draws_of(rows, SEED_A, 'def', 'bf16') if r['draw'] == 0), None)
        st7 = draw_stats(draws_of(rows, SEED_A, 'def', 'bf16')) if h7 else None
        guarded('compare script_A', compare_script_run, 'script_A_seed7', run, 'A',
                format_step0(h7) if h7 else None, st7)
        for t, name in (('B', 'script_B'), ('full1', 'script_full1')):
            tn, v = pick_verdict(verdicts, t)
            if v is None:
                continue
            s, tg = v['seed'], STATE['targets'][tn]
            extra = (['--holdout', ''] if tg['base_cfg'] == 'full' else []) + \
                (['--K', str(tg['K'])] if tg['K'] != K_DEFAULT else [])
            r = guarded(name, run_script_oom, name, ['--seed', str(s), *extra])
            dr = draws_of(rows, s, tg['cfg'], 'bf16')
            h0 = next((x for x in dr if x['draw'] == 0), None)
            guarded(f'compare {name}', compare_script_run, name, r, t, format_step0(h0) if h0 else None,
                    draw_stats(dr) if dr else None)
        guarded('finish', step_finish)
    finally:
        REPORT['symlinks_removed'] = guarded('cleanup', cleanup)
        save_report()


if os.environ.get('F6_IMPORT_ONLY') != '1':
    main()
