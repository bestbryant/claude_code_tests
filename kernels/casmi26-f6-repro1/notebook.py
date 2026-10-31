"""casmi26-f6-repro1 (F6 in docs/fork_eligibility.md, asset A10; A6 through it): rebuild the fe_v4 feature tables behind
Engine 1 of the frozen 0.465 pipeline from Ahmed Berat Ozer's published, MIT-licensed training code
(ahmedberatozer/casmi26-training-code) on a Kaggle CPU session, and compare every regenerated file with the copy that
ahmedberatozer/casmi26-v4b-models ships (and that the fork sha-checks against MANIFEST.json). Nothing here feeds a
submission. Chain map: scratchpad f6_map.md, section 3.2 and ranks 1-3 of section 6 ("K1").

Why these tables: ranker_v4b.pkl (A11) cannot be refit from published data (the 15,701 sim1x rows / features were never
published), so the cheapest decisive target is A10. Its gaps per Ahmed's README: res_fragnet.py and fams/fragnet.py
edited after the fits, fragnet `sets` and `fitA` without logs, `--nspec 32000` inferred, rec_bond.npy re-saved as uint8
without a recorded command, res_analog_struct.py edited after its rec outputs. Every one is tested here by running the
CURRENT (post-edit) scripts with the README commands and comparing outputs with the published files.

Steps; each prints 'F6 CHECK <name>: OK | FAIL | INFO {...}'; the record is /kaggle/working/repro_report.json.
 1. Inputs: sha256 of all 64 casmi26-v4b-models files against its MANIFEST.json; casmi26-spec-cache against its
    docs/README.md list; casmi26-v2-pool hashed and pinned (fp_bits.npy = c3869e86..., the value the pubm run mounted);
    casmi26-training-code: per-file pins of everything executed or compared (taken from the 2026-10-19 version, zip sha256
    41bf3f2d...) plus a whole-tree hash; the fe_v4 family code in the training code equals the v4b copy the fork runs.
 2. Environment: RDKit pinned to 2026.3.3 (offline wheel, as in the pubm lineage); his was python 3.14.5, lightgbm
    4.7.0, rdkit 2026.03.3. A lightgbm-4.7.0 wheel is installed if one is mounted; otherwise the image's version is
    recorded and the stageB comparison says so.
 3. Layout: NO source edits. The README's repository layout is restored by copying fe_v4/ to work/analysis/fe_v4/ and
    engine_v6_casmi/ to work/analysis/engine_v6/code/casmi/, and by symlinking the spec-cache and v2-pool files into
    work/. The copies are sha-checked against the pins. Outputs land in the copy's res/ (empty at start), never in /kaggle/input.
 4. Fit from the PUBLISHED intermediates (map rank 3, isolates the post-edit fit code from data-step drift), in a second
    copy (repo_pub) whose res/fragnet holds symlinks to the published countsA.npz / featB.npz:
      res_fragnet.py --step fitA            -> stageA.npz   vs published stageA.npz (arrays)
      res_fragnet.py --step fitB --procs 4  -> stageB.txt   vs published stageB.txt (tree text) + the 4 V MRR lines
    plus a seed reference: the same fitB with seed 0 + 1000 (launcher sets STAGEB_PARAMS['seed'] and OUT only).
 5. Rebuild from the data (README commands, run from the repository root, --procs 4):
      res_frag2.py --step caps | f1                       -> frag2/{pool_cap.npy, caps.txt, pool_f1_off.npy, pool_f1_mass.npy}
      res_fragnet.py --step rec | sets --nspec 32000 --seed 0 | countA | featB | fitA | fitB
                                                          -> fragnet/{rec_*.npy, sets.pkl, countsA.npz, featB.npz, stageA.npz, stageB.txt}
      res_analog_struct.py --step rec | str               -> analog_struct/{pool,train}_{hash,num,fvec,elsmi_*,coresmi_*}.npy
    Compared with the published files: sha256 against MANIFEST (.npy, caps.txt: np.save is byte-deterministic), content
    for sets.pkl (decoded with a numpy-only unpickler), arrays for the .npz files (zip containers / compression may
    differ), tree text for stageB.txt; the deterministic summary lines of his logs (timings stripped) must reappear.
    countA merges worker results with imap_unordered, so its float64 decoy weights are order-dependent in the last bits
    even on his machine: tolerance 1e-12 relative there, and everything downstream of it is judged at float tolerance /
    model agreement. featB uses our frag2 f1 table (published copy if the f1 step did not finish; recorded).
 6. Models on the PUBLISHED featB rows (his own _groups / _mrr harness): published stageB must give his logged V MRR
    (harness parity: fr_int 0.6465, frag2_mix 0.7022, stageA_pk_fcu 0.7549, fragnet 0.7712); refit-from-published (rep),
    seed+1000 reference (ref) and full-chain refit (chain) are compared with it (V MRR, per-group Spearman, top-1).
    A non-identical stageB counts as EQUIVALENT only if |V MRR - 0.7712| <= 0.002 and it agrees with the published model
    at least as well as the seed reference does.
Budget: 3 h wall clock (F6_BUDGET_MIN, default 170 min). A step starts only if its estimate fits in the remaining time,
dependent steps are skipped after a failure, and each step has a timeout (its process group is killed). Expected
~80-120 min on a 4-vCPU CPU session: the per-step estimates in STEPS are his laptop seconds (~24 min in total with 4
processes) x ~3, plus ~5 min of input hashing. Order: fits from the published files, frag2, fragnet chain, analog_struct
(so a slow session loses analog_struct first; it is independent of the rest).
"""
import difflib, hashlib, importlib.metadata, io, json, os, pickle, re, shutil, signal, subprocess, sys, threading, time
import traceback, zipfile
from pathlib import Path

T0 = time.time()
INPUT = Path(os.environ.get('F6_INPUT', '/kaggle/input'))
WORKING = Path(os.environ.get('F6_WORKING', '/kaggle/working'))
RW = WORKING / 'f6_repro1'
REPO = RW / 'repo'              # full rebuild from the data
REPO_PUB = RW / 'repo_pub'      # fitA / fitB from the published intermediates
SEEDREF = RW / 'seedref'        # fitB with seed + SEED_REF_OFFSET on the published featB
LOGS = RW / 'logs'
REPORT_PATH = WORKING / 'repro_report.json'
BUDGET_MIN = float(os.environ.get('F6_BUDGET_MIN', '170'))
RESERVE_MIN = 8                 # kept free for the model comparison and the report
PROCS = 4                       # README commands
SEED_REF_OFFSET = 1000
KEEP_MATCHED = os.environ.get('F6_KEEP_MATCHED', '0') == '1'    # keep regenerated files that equal the published ones
FE_REL = 'work/analysis/fe_v4'
V6_REL = 'work/analysis/engine_v6/code/casmi'
RES_KEY = 'code/fe_v4/res/'     # MANIFEST prefix of the published tables

# ── pins ─────────────────────────────────────────────────────────────────────────────────────────────────────
# casmi26-training-code as downloaded 2026-10-19 (zip sha256 41bf3f2d..., 363 files); tree = sha256 over sorted
# '<relpath>\t<sha256>\n' lines of every file.
TC_TREE_SHA256 = '85e254f6430e8202744ca18ccb14d3ade29324e62208309512e746ac67ab3057'
TC_TREE_FILES = 363
TC_PINS = {
    'LICENSE': '6f05399ac30469cae349febdefe2b1f1315af1b2ded8088de7d2304ef7ebddb7',
    'README.md': '1c93dab90f9171a194e665949792c992017981fd9a63b82c0f647d20c6690249',
    'engine_v6_casmi/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'engine_v6_casmi/build.py': '86776c14e237dc64dbf965d9bc200cc670dc7226bf946938380729d7e80c56f8',
    'engine_v6_casmi/chem.py': '99474f23d8a67040b0c0f8c70e73592a6f7f656f50103e350ca473451e2169f1',
    'engine_v6_casmi/derive.py': '301d3713c419020d3cec376b3e4fc29dcf48b1e144971b95ccafa96b986e5731',
    'engine_v6_casmi/dreams_model.py': '5c09c3d22e8d2c09678b796cb1d57a449442437847e094f49f6ed2619a73a83c',
    'engine_v6_casmi/engine.py': '78554b89d3709b554b6e7715c7bc9735654fd3f2406ffb8033020d26ce984a10',
    'engine_v6_casmi/fpnet.py': '32542bb0039b2d24d9911030559f13221e630cef7df0d10c7d65f54424da2633',
    'engine_v6_casmi/frag.py': '809f5582494cee4cf6dcace4b04699b8427e81022b78d42162cc0239a57773de',
    'engine_v6_casmi/graphs.py': '8dcbe16bbf3b34b7a08866234eb9abf2de713d21c93c653e7156e94e9debcd55',
    'engine_v6_casmi/library.py': '99eeaab9e02d3fedcc4b73db19aabb93e82b5096e8320f6bcff5b954074d0689',
    'engine_v6_casmi/pipeline.py': '5b5783ed30eb99f1d8d78673fa90fe277932c52bd676460a46d22cb3205b1f08',
    'engine_v6_casmi/pool.py': 'c1acad9a7ebdc4776918fbda444ee9464c2109c79acfb9b2243f98e3bb47ac6d',
    'engine_v6_casmi/pubchem.py': '4fd4a9702e0813f8651e3a3b99595f0803d7381c0434959abdfeee042049c12a',
    'engine_v6_casmi/ranker.py': '98b335c22893bc094c88a87ba6ca2451b794d06bbf1e7730a2c76c6a7fa0e619',
    'engine_v6_casmi/spectra.py': '41067c12e1287cff3a927d7940be78f1c2a0019310269fabc8ea161edf607930',
    'engine_v6_casmi/structs.py': '2c70df89e8f96b3e716651fe8815d0a05af3ad26e2b448e552ea53c26ede334a',
    'engine_v6_casmi/traindata.py': 'cecb329d96362f11d16b40a26f1888f092610ff6202812408c13e2c293c6a2bc',
    'engine_v6_casmi/xenc.py': '89fbb470e6fc085f7363bf79e4517c06d43eed3c1df197b7f374b65dacdb7310',
    'fe_v4/chemfeat.py': '8fedf317042f989b1569ca65c0d6d87a0387f9fc9934504420a95d8d973f8746',
    'fe_v4/ctxlib.py': 'c050e4b0ac937a7d65fa14ac55c6904e2232cb077830a20b653ac6df22922943',
    'fe_v4/derive6.py': '0e4b436d6b978b2e2e1af8767d6b3c32a6b1fc0e4b64960d5eeeaa8885ddf68a',
    'fe_v4/fams/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'fe_v4/fams/analog_struct.py': '2311219220c65bc83c2cd447a8cdb07db27a984124f6c88d687494556e7337b6',
    'fe_v4/fams/derivation.py': 'eb9aa474bef8e1d067cadccec0101042878ebb5aedf50d1168dc9dfd884dceae',
    'fe_v4/fams/frag2.py': 'bf17577ef673f8ab0121caa1c9335dc1e827e2f1eb44b3bbde0b0c0cbe858e6c',
    'fe_v4/fams/fragnet.py': 'd0952d34d227013b8486eff36c30a4233f491d2e32c3f816b9d000c493b7234e',
    'fe_v4/fams/model_views.py': 'aa964b12265bb70144d2d653b600a7f1a4972829032274bab76ea9e39afd7e79',
    'fe_v4/fragcaps.py': 'ba6280f9a46ca28cd72b8a7a0324724a73fd0c67a13acdd82a8193ac30d17a83',
    'fe_v4/res_analog_struct.log': 'c54d70a2c5a21ca347c323094f71e165f9804d75be43b143d7396dc97e988836',
    'fe_v4/res_analog_struct.py': '53251093d07f5dae06e26de9876c2244254f2767c23d68d95d8e40b143f201f8',
    'fe_v4/res_frag2.log': 'e26bc238bb78ecb34919c4c325f340eb677a38367a0cfe01a4a1df69a48d7148',
    'fe_v4/res_frag2.py': '191b3433b7d8ee199dafdc6a44014629b6ac9d55d84323a7cd8b832b8da58f2f',
    'fe_v4/res_frag2_f1.log': '08f9f4d6b113d317e267b2b638b77768efb7830bddf579cd0e667c51772f9878',
    'fe_v4/res_fragnet.py': '02c98941f6f100c267b84ad42afbf0b7ebe76c92d889646a6e40622d0d724378',
    'fe_v4/res_fragnet_countA.log': 'e6d457188b7b54d38b805605f98806be6a4587192a42b21a7d2bb206555575e6',
    'fe_v4/res_fragnet_featB.log': '62d594e29e9966b12b9865ec718e79fd2db2ffd1c83d20ed6b133a11dd57b6bd',
    'fe_v4/res_fragnet_fitB.log': '22f73ce545f853d077c658153217f28acd4941b2fc5f9ab5bc776a3279536aaa',
    'fe_v4/res_fragnet_rec.log': '1d75c362355617c9314cb4461e3b5efef2eea5701e40433fa7d56e80749a4f3f',
}
SPEC_CACHE_SHA = {   # casmi26-spec-cache docs/README.md
    'spec_mz.npy': '482332fec3fabcc46e06ce1b23d79e633af7009b8190a3dcd0a675121b42342b',
    'spec_it.npy': '9e127f00eb9d4217403991304e69166adc9a36a5e624ce7fd1ddbf6b1417cc23',
    'spec_off.npy': 'af2be108a532a278ec4563c023808b7fa19e40fdf4ea7dc888e980bb6f618e6c',
    'spec_meta.parquet': '4f968592dc75b9bf2b6256dda394be42bad3cd07fd243ff0078101e066a44376',
    'split.parquet': 'd05a2f9f71cffd480b92581c458547f7aeb782436b07af4fbf8f696f96649656',
}
V2POOL_KNOWN = {'fp_bits.npy': 'c3869e86449d6fed6e79e9655f8a7d35bd55c3138ecddfc1dae5072627e78677'}
V2POOL_NEEDED = ['pool_meta.parquet', 'train_structs.parquet', 'pool_frag_off.npy', 'pool_frag_mass.npy']
HIS_ENV = dict(python='3.14.5', lightgbm='4.7.0', rdkit='2026.03.3', torch='2.14.0+cu130', os='Windows (RTX 5090 laptop)')
PUB_V_MRR = {'fr_int': 0.6465, 'frag2_mix': 0.7022, 'stageA_pk_fcu': 0.7549, 'fragnet': 0.7712}   # res_fragnet_fitB.log
PUB_SETS = dict(n=32000, parts={'A': 16144, 'B': 12751, 'V': 3105})                             # fams/fragnet.py log
V_MRR_TOL = 0.002
F32_ULP_TOL = 2                 # stageA LLR tables: float32 ulps (np.log may differ in the last float64 bit across platforms)
COUNT_RTOL = 1e-12              # countsA float64 decoy weights (worker completion order)
FEAT_RTOL = 1e-5                # featB float aggregates downstream of countsA

# step -> (published table files under code/fe_v4/res/, published log, regex selecting that step's lines of the log)
TABLES = {
    'frag2_caps': (['frag2/pool_cap.npy', 'frag2/caps.txt'], 'res_frag2.log', r'^pool caps'),
    'frag2_f1': (['frag2/pool_f1_off.npy', 'frag2/pool_f1_mass.npy'], 'res_frag2_f1.log', None),
    'fragnet_rec': ([f'fragnet/rec_{k}.npy' for k in ('ok', 'aoff', 'atom', 'boff', 'bond')], 'res_fragnet_rec.log', None),
    'analog_rec': ([f'analog_struct/{s}_{k}.npy' for s in ('train', 'pool') for k in ('hash', 'num', 'fvec')],
                   'res_analog_struct.log', r'^(train|pool): \('),
    'analog_str': ([f'analog_struct/{s}_{f}smi_{b}.npy' for s in ('train', 'pool') for f in ('el', 'core') for b in ('buf', 'off')],
                   'res_analog_struct.log', r'^(train|pool) strings:'),
}
PACKED = {'frag2/pool_f1_mass.npy': 'frag2/pool_f1_off.npy', 'fragnet/rec_atom.npy': 'fragnet/rec_aoff.npy',
          'fragnet/rec_bond.npy': 'fragnet/rec_boff.npy',
          **{f'analog_struct/{s}_{f}smi_buf.npy': f'analog_struct/{s}_{f}smi_off.npy' for s in ('train', 'pool') for f in ('el', 'core')}}

# name, folder, script, args, estimated Kaggle minutes (4 vCPU; his laptop seconds x ~3), dependencies
STEPS = [
    ('pub_fitA', 'pub', 'res_fragnet.py', ['--step', 'fitA'], 1, []),
    ('pub_fitB', 'pub', 'res_fragnet.py', ['--step', 'fitB', '--procs', str(PROCS)], 4, []),
    ('seedref_fitB', 'seedref', None, [], 4, []),
    ('frag2_caps', 'repo', 'res_frag2.py', ['--step', 'caps', '--procs', str(PROCS)], 5, []),
    ('frag2_f1', 'repo', 'res_frag2.py', ['--step', 'f1', '--procs', str(PROCS)], 15, []),
    ('fragnet_rec', 'repo', 'res_fragnet.py', ['--step', 'rec', '--procs', str(PROCS)], 11, []),
    ('fragnet_sets', 'repo', 'res_fragnet.py', ['--step', 'sets', '--nspec', '32000', '--seed', '0'], 5, []),
    ('fragnet_countA', 'repo', 'res_fragnet.py', ['--step', 'countA', '--procs', str(PROCS)], 6, ['fragnet_rec', 'fragnet_sets']),
    ('fragnet_featB', 'repo', 'res_fragnet.py', ['--step', 'featB', '--procs', str(PROCS)], 10,
     ['fragnet_rec', 'fragnet_sets', 'fragnet_countA']),
    ('fragnet_fitA', 'repo', 'res_fragnet.py', ['--step', 'fitA'], 1, ['fragnet_countA']),
    ('fragnet_fitB', 'repo', 'res_fragnet.py', ['--step', 'fitB', '--procs', str(PROCS)], 4, ['fragnet_featB']),
    ('analog_rec', 'repo', 'res_analog_struct.py', ['--step', 'rec', '--procs', str(PROCS)], 18, []),
    ('analog_str', 'repo', 'res_analog_struct.py', ['--step', 'str', '--procs', str(PROCS)], 14, []),
]

REPORT = dict(kernel='casmi26-f6-repro1', task='F6 fe_v4 table rebuild (docs/fork_eligibility.md A10; f6_map.md K1 ranks 1-3)',
              datasets=['ahmedberatozer/casmi26-training-code', 'ahmedberatozer/casmi26-v4b-models',
                        'ahmedberatozer/casmi26-v2-pool', 'ahmedberatozer/casmi26-spec-cache'],
              checks={}, steps={}, artefacts={}, errors=[])
ROOTS = {}
MANIFEST = {}


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
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_report():
    REPORT['minutes'] = minutes()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=_js))


def check(name, ok, **detail):
    REPORT['checks'][name] = dict(ok=ok, **detail)
    tag = 'INFO' if ok is None else ('OK' if ok else 'FAIL')
    log(f'F6 CHECK {name}: {tag} ' + json.dumps(detail, default=_js)[:1800])
    save_report()


def artefact(name, status, **detail):
    """Per-artefact verdict: BIT-EXACT | CONTENT-EQUAL | NUMERIC-EQUAL | EQUIVALENT | DIFFERENT | SKIPPED | FAILED."""
    REPORT['artefacts'][name] = dict(status=status, **detail)
    save_report()


def guarded(name, fn, *a, **k):
    try:
        return fn(*a, **k)
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
    seconds, timed_out). On timeout the whole process group (multiprocessing workers included) is killed."""
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


# ── pure helpers (unit-tested locally on synthetic data) ─────────────────────────────────────────────────────
def tree_hash(root):
    """(sha256 over sorted '<relpath>\\t<sha256>\\n' lines of every file under root, {relpath: sha256})."""
    root = Path(root)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    blob = ''.join(f'{r}\t{h}\n' for r, h in sorted(files.items())).encode()
    return hashlib.sha256(blob).hexdigest(), files


_TIME = re.compile(r'(?<![\w.])\d+(?:\.\d+)?s\b')
_PROGRESS = re.compile(r'^\s*(rec|f1|countA|featB):?\s+\d+/\d+')


def norm_log_line(line):
    """A log line without its timings ('94s' -> '<t>') and with whitespace collapsed."""
    return re.sub(r'\s+', ' ', _TIME.sub('<t>', line.strip()))


def summary_lines(text, select=None):
    """Deterministic lines of a log: progress lines dropped, timings stripped; optional regex on the normalised line."""
    out = []
    for l in text.splitlines():
        if not l.strip() or _PROGRESS.match(l):
            continue
        n = norm_log_line(l)
        if select is None or re.search(select, n):
            out.append(n)
    return out


def compare_log(published_text, our_text, select=None):
    """Every deterministic line of the published log must reappear verbatim (timings stripped) in ours."""
    P = summary_lines(published_text, select)
    O = summary_lines(our_text)
    Os = set(O)
    missing = [l for l in P if l not in Os]
    near = {}
    for l in missing[:5]:
        cand = difflib.get_close_matches(l, O, n=1, cutoff=0.6)
        near[l] = cand[0] if cand else None
    return dict(ok=bool(P) and not missing, published_lines=len(P), reproduced=len(P) - len(missing),
                missing=missing[:5], ours_nearest=near)


def array_diff(a, b, max_examples=5):
    """Element-wise comparison of two arrays. Floats: NaN == NaN, max |diff|, max relative diff, max float ulps."""
    import numpy as np
    a = np.asarray(a); b = np.asarray(b)
    d = dict(dtype=[str(a.dtype), str(b.dtype)], shape=[list(a.shape), list(b.shape)])
    if a.shape != b.shape:
        d.update(equal=False, same_shape=False)
        return d
    fl = a.dtype.kind in 'fc' or b.dtype.kind in 'fc'
    if fl:
        x = a.astype(np.float64); y = b.astype(np.float64)
        eq = (x == y) | (np.isnan(x) & np.isnan(y))
    else:
        try:
            eq = np.asarray(a == b)
        except Exception:
            eq = np.zeros(a.shape, bool)
        if eq.shape != a.shape:
            eq = np.broadcast_to(eq, a.shape) if eq.ndim == 0 else np.zeros(a.shape, bool)
    eq = eq.reshape(a.shape)
    n_diff = int(eq.size - int(eq.sum()))
    d.update(same_shape=True, n=int(a.size), n_diff=n_diff, equal=n_diff == 0 and a.dtype == b.dtype)
    if n_diff and fl:
        m = ~eq
        xa, ya = x[m], y[m]
        fin = np.isfinite(xa) & np.isfinite(ya)
        ad = np.abs(xa[fin] - ya[fin])
        scale = np.maximum(np.abs(xa[fin]), np.abs(ya[fin]))
        rel = np.where(scale > 0, ad / np.where(scale > 0, scale, 1), 0.0)
        ft = a.dtype if a.dtype.kind == 'f' else (b.dtype if b.dtype.kind == 'f' else np.dtype(np.float64))
        sp = np.spacing(scale.astype(ft)).astype(np.float64)
        ulp = np.where(sp > 0, ad / np.where(sp > 0, sp, 1), 0.0)
        # scale floor: |diff| / largest magnitude of the element's column (2-D) or of the whole array (else)
        mag = np.maximum(np.where(np.isfinite(x), np.abs(x), 0.0), np.where(np.isfinite(y), np.abs(y), 0.0))
        if x.ndim == 2:
            colmax = mag.max(0) if len(mag) else np.zeros(x.shape[1])
            scl = np.broadcast_to(colmax[None, :], x.shape)[m][fin]
        else:
            scl = np.full(int(fin.sum()), float(mag.max()) if mag.size else 0.0)
        over = np.where(scl > 0, ad / np.where(scl > 0, scl, 1), np.where(ad > 0, np.inf, 0.0))
        d.update(max_abs=float(ad.max()) if ad.size else None, max_rel=float(rel.max()) if rel.size else None,
                 max_ulp=float(ulp.max()) if ulp.size else None, n_nonfinite_diff=int((~fin).sum()),
                 max_abs_over_scale=float(over.max()) if over.size else None)
    if n_diff:
        idx = np.argwhere(~eq)[:max_examples]
        d['first_diff_index'] = [list(map(int, i)) for i in idx]
    return d


def compare_named_arrays(A, B, float_rtol=0.0, ulp_tol=None, exact_keys=None):
    """Compare two {name: array} dicts. A float array passes if exactly equal, or if every difference is within float_rtol
    relative to the element or to the largest magnitude of its column (2-D) / array (1-D) (sums that cancel to ~0), or
    within ulp_tol float ulps;
    exact_keys must be exactly equal. Returns (status, detail): CONTENT-EQUAL | NUMERIC-EQUAL | DIFFERENT."""
    import numpy as np
    keys_a, keys_b = set(A), set(B)
    rows, worst = {}, 'CONTENT-EQUAL'
    if keys_a != keys_b:
        worst = 'DIFFERENT'
    for k in sorted(keys_a & keys_b):
        r = array_diff(A[k], B[k])
        if r['equal']:
            r['status'] = 'equal'
        elif (r.get('same_shape') and np.asarray(A[k]).dtype == np.asarray(B[k]).dtype and r.get('max_rel') is not None
              and not r.get('n_nonfinite_diff') and not (exact_keys and k in exact_keys)
              and ((float_rtol and (r['max_rel'] <= float_rtol or (r.get('max_abs_over_scale') is not None
                                                                  and r['max_abs_over_scale'] <= float_rtol)))
                   or (ulp_tol is not None and r['max_ulp'] <= ulp_tol))):
            r['status'] = 'within_tolerance'
            if worst == 'CONTENT-EQUAL':
                worst = 'NUMERIC-EQUAL'
        else:
            r['status'] = 'different'; worst = 'DIFFERENT'
        rows[k] = r
    detail = dict(only_published=sorted(keys_b - keys_a), only_ours=sorted(keys_a - keys_b),
                  n_keys=len(keys_a & keys_b), n_equal=sum(r['status'] == 'equal' for r in rows.values()),
                  n_within_tol=sum(r['status'] == 'within_tolerance' for r in rows.values()),
                  different={k: r for k, r in rows.items() if r['status'] == 'different'},
                  within_tol={k: {f: r.get(f) for f in ('n_diff', 'n', 'max_rel', 'max_ulp', 'max_abs_over_scale')}
                              for k, r in rows.items()
                              if r['status'] == 'within_tolerance'})
    return worst, detail


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the numpy array constructors a plain data pickle needs (sets.pkl: list of dicts of
    ints / str / int arrays; the object member 'chains' of stageA.npz). Anything else raises."""
    ALLOWED = {('numpy._core.numeric', '_frombuffer'), ('numpy.core.numeric', '_frombuffer'), ('numpy', 'dtype'),
               ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', '_reconstruct'),
               ('numpy', 'ndarray'), ('numpy._core.multiarray', 'scalar'), ('numpy.core.multiarray', 'scalar'),
               ('_codecs', 'encode')}   # protocol <= 2 numpy pickles carry raw bytes through _codecs.encode

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'blocked global {module}.{name}')
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            alt = module.replace('numpy._core', 'numpy.core') if module.startswith('numpy._core') else \
                module.replace('numpy.core', 'numpy._core')
            return super().find_class(alt, name)


def load_pickle_numpy_only(path):
    with open(path, 'rb') as f:
        return NumpyOnlyUnpickler(f).load()


def npz_load(path):
    """{name: array} of an .npz without pickle execution: plain members via allow_pickle=False, object members decoded
    with NumpyOnlyUnpickler."""
    import numpy as np
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename[:-4] if info.filename.endswith('.npy') else info.filename
            raw = zf.read(info)
            f = io.BytesIO(raw)
            ver = np.lib.format.read_magic(f)
            reader = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(f)
            if dtype.hasobject:
                out[name] = NumpyOnlyUnpickler(f).load()
            else:
                out[name] = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
    return out


def _plain(o):
    """Nested object arrays / lists -> nested lists of python scalars (for comparing pickled object members)."""
    import numpy as np
    if isinstance(o, np.ndarray):
        return [_plain(x) for x in o.tolist()] if o.dtype.hasobject or o.ndim else o.item()
    if isinstance(o, (list, tuple)):
        return [_plain(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def compare_sets(ours, pub):
    """fragnet sets.pkl content: per set sid / spec / pid / part equal, dec / dsrc equal arrays with equal dtype."""
    import numpy as np
    from collections import Counter
    n = min(len(ours), len(pub)); same = 0; first = None; fd = Counter()
    for i in range(n):
        x, y = ours[i], pub[i]; ok = True
        if set(x) != set(y):
            fd['keys'] += 1; ok = False
        for k in ('sid', 'spec', 'pid', 'part'):
            if x.get(k) != y.get(k):
                fd[k] += 1; ok = False
        for k in ('dec', 'dsrc'):
            u, v = np.asarray(x.get(k)), np.asarray(y.get(k))
            if u.dtype != v.dtype or not np.array_equal(u, v):
                fd[k] += 1; ok = False
        same += ok
        if not ok and first is None:
            first = i
    parts = lambda S: dict(sorted(Counter(s.get('part') for s in S).items()))
    dec = lambda S: float(np.mean([len(s['dec']) for s in S])) if S else None
    return dict(identical=same == len(ours) == len(pub), n_ours=len(ours), n_published=len(pub), identical_sets=same,
                field_diffs=dict(fd), first_diff=first, parts_ours=parts(ours), parts_published=parts(pub),
                decoys_mean_ours=dec(ours), decoys_mean_published=dec(pub))


def lgb_sections(text):
    """(header, tree section 'Tree=0' .. 'end of trees', parameters block) of a LightGBM model text."""
    i = text.find('Tree=0\n')
    j = text.find('end of trees', i if i >= 0 else 0)
    k = text.find('parameters:\n'); e = text.find('end of parameters', k if k >= 0 else 0)
    header = text[:i] if i >= 0 else text
    trees = text[i:j + len('end of trees')] if i >= 0 and j >= 0 else ''
    params = text[k:e] if k >= 0 and e >= 0 else ''
    return header, trees, params


def _kv(block):
    out = {}
    for l in block.splitlines():
        if '=' in l:
            k, v = l.split('=', 1); out[k.strip()] = v.strip()
    return out


_TREE_INT = ('num_leaves', 'num_cat', 'split_feature', 'decision_type', 'left_child', 'right_child', 'leaf_count',
             'internal_count')
_TREE_FLT = ('split_gain', 'threshold', 'leaf_value', 'leaf_weight', 'internal_value', 'internal_weight', 'shrinkage')


def lgb_trees(text):
    """List of per-tree dicts (ints / floats as python lists) parsed from a LightGBM model text."""
    _, trees, _ = lgb_sections(text)
    out = []
    for blk in re.split(r'\n(?=Tree=\d+\n)', trees.replace('end of trees', '')):
        blk = blk.strip()
        if not blk.startswith('Tree='):
            continue
        kv = _kv(blk)
        t = dict(index=int(kv['Tree']))
        for k in _TREE_INT:
            if k in kv:
                t[k] = [int(x) for x in kv[k].split()]
        for k in _TREE_FLT:
            if k in kv:
                t[k] = [float(x) for x in kv[k].split()]
        out.append(t)
    return out


def compare_lgb_text(ours, pub):
    """LightGBM model texts: whole file, header (feature names / infos), tree section byte equality; otherwise per tree
    structure (num_leaves, split_feature, decision_type, children) and max |diff| of thresholds / leaf values."""
    ho, to, po = lgb_sections(ours); hp, tp, pp = lgb_sections(pub)
    kho, khp = _kv(ho), _kv(hp)
    d = dict(file_identical=ours == pub, tree_section_identical=bool(to) and to == tp,
             header_identical=ho == hp, feature_names_equal=kho.get('feature_names') == khp.get('feature_names'),
             feature_infos_equal=kho.get('feature_infos') == khp.get('feature_infos'),
             params_lines_differing=[l for l in difflib.unified_diff(pp.splitlines(), po.splitlines(), lineterm='', n=0)
                                     if l[:1] in '+-' and not l.startswith(('+++', '---'))][:20])
    A, B = lgb_trees(ours), lgb_trees(pub)
    d['n_trees'] = [len(A), len(B)]
    same_struct, first, thr, leaf = 0, None, 0.0, 0.0
    for i, (a, b) in enumerate(zip(A, B)):
        s = all(a.get(k) == b.get(k) for k in ('num_leaves', 'split_feature', 'decision_type', 'left_child', 'right_child'))
        if s:
            same_struct += 1
            for k, acc in (('threshold', 'thr'), ('leaf_value', 'leaf')):
                x, y = a.get(k, []), b.get(k, [])
                m = max((abs(u - v) for u, v in zip(x, y)), default=0.0)
                if acc == 'thr':
                    thr = max(thr, m)
                else:
                    leaf = max(leaf, m)
        elif first is None:
            first = i
    d.update(trees_same_structure=same_struct, first_structure_diff_tree=first, threshold_max_abs_diff=thr,
             leaf_value_max_abs_diff=leaf)
    if d['tree_section_identical']:
        d['status'] = 'BIT-EXACT' if d['file_identical'] else 'CONTENT-EQUAL'
    elif len(A) == len(B) and same_struct == len(A) and thr == 0.0 and leaf <= 1e-9:
        d['status'] = 'NUMERIC-EQUAL'
    else:
        d['status'] = 'DIFFERENT'
    return d


def avg_rank(x):
    """1-based ranks of x ascending, ties averaged."""
    import numpy as np
    x = np.asarray(x, np.float64); n = len(x)
    if n == 0:
        return np.zeros(0)
    o = np.argsort(x, kind='mergesort'); xs = x[o]
    b = np.r_[True, xs[1:] != xs[:-1]]; bid = np.cumsum(b) - 1
    st = np.flatnonzero(b); en = np.r_[st[1:], n]
    r = np.empty(n); r[o] = ((st + en - 1) / 2.0 + 1.0)[bid]
    return r


def group_bounds(gid):
    import numpy as np
    gid = np.asarray(gid)
    if len(gid) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ch = np.r_[True, gid[1:] != gid[:-1]]
    st = np.flatnonzero(ch)
    return st, np.r_[st[1:], len(gid)]


def truth_rank(score, y, same):
    """Rank of the truth among same-formula candidates, ties counted half (as his _mrr)."""
    import numpy as np
    m = np.asarray(same, bool); sc = np.asarray(score)[m]; yy = np.asarray(y)[m]
    t = sc[yy > 0][0]; o = sc[yy == 0]
    return 1 + (o > t).sum() + 0.5 * (o == t).sum()


def pair_agreement(sa, sb, y, same, st, en, mask):
    """Agreement of two scorers on the same candidate groups (groups whose first row is in mask)."""
    import numpy as np
    sp, t1, rid, mra, mrb, mad = [], [], [], [], [], 0.0
    for a_, b_ in zip(st, en):
        if not mask[a_]:
            continue
        a, b = np.asarray(sa[a_:b_], np.float64), np.asarray(sb[a_:b_], np.float64)
        mad = max(mad, float(np.abs(a - b).max()) if len(a) else 0.0)
        if len(a) > 1 and a.std() > 0 and b.std() > 0:
            sp.append(float(np.corrcoef(avg_rank(a), avg_rank(b))[0, 1]))
        t1.append(int(np.argmax(a) == np.argmax(b)))
        ra = truth_rank(a, y[a_:b_], same[a_:b_]); rb = truth_rank(b, y[a_:b_], same[a_:b_])
        rid.append(ra == rb); mra.append(1.0 / ra); mrb.append(1.0 / rb)
    return dict(n_groups=len(t1), spearman_per_group_mean=float(np.mean(sp)) if sp else None,
                top1_agreement=float(np.mean(t1)) if t1 else None,
                truth_rank_identical=float(np.mean(rid)) if rid else None,
                mrr_a=float(np.mean(mra)) if mra else None, mrr_b=float(np.mean(mrb)) if mrb else None,
                max_abs_score_diff=mad)


def packed_diff(off_a, dat_a, off_b, dat_b, max_examples=5):
    """Entries (structures) whose packed slices differ between two (offsets, data) tables."""
    import numpy as np
    oa, ob = np.asarray(off_a), np.asarray(off_b)
    if len(oa) != len(ob):
        return dict(n_entries=[len(oa) - 1, len(ob) - 1], comparable=False)
    da, db = np.asarray(dat_a), np.asarray(dat_b)
    bad = []
    la, lb = np.diff(oa), np.diff(ob)
    for i in range(len(oa) - 1):
        if la[i] != lb[i] or not np.array_equal(da[oa[i]:oa[i + 1]], db[ob[i]:ob[i + 1]]):
            bad.append(i)
    return dict(n_entries=len(oa) - 1, comparable=True, entries_differing=len(bad), length_differs=int((la != lb).sum()),
                examples=bad[:max_examples])


def stageB_verdict(text_status, v_mrr, spearman_vs_pub, spearman_ref_vs_pub, logged=None, tol=None):
    """Final stageB status: the text comparison, upgraded from DIFFERENT to EQUIVALENT only when the V MRR is within tol
    of the logged value AND the model agrees with the published one at least as well as the seed+1000 reference does."""
    logged = PUB_V_MRR['fragnet'] if logged is None else logged
    tol = V_MRR_TOL if tol is None else tol
    if text_status != 'DIFFERENT':
        return text_status
    if v_mrr is not None and abs(v_mrr - logged) <= tol and spearman_ref_vs_pub is not None and \
            spearman_vs_pub is not None and spearman_vs_pub >= spearman_ref_vs_pub:
        return 'EQUIVALENT'
    return 'DIFFERENT'


def can_start(est_min, elapsed_min, budget_min=None, reserve_min=None):
    budget_min = BUDGET_MIN if budget_min is None else budget_min
    reserve_min = RESERVE_MIN if reserve_min is None else reserve_min
    return elapsed_min + est_min + reserve_min <= budget_min


def step_timeout_min(est_min, elapsed_min, budget_min=None, reserve_min=None):
    budget_min = BUDGET_MIN if budget_min is None else budget_min
    reserve_min = RESERVE_MIN if reserve_min is None else reserve_min
    return max(1.0, min(max(2.5 * est_min, est_min + 10), budget_min - reserve_min - elapsed_min))


# ── 1. Inputs ────────────────────────────────────────────────────────────────────────────────────────────────
def find_inputs():
    def first(pattern, ok):
        hits = sorted((p for p in INPUT.rglob(pattern) if ok(p)), key=lambda p: (len(p.parts), str(p)))
        return hits[0] if hits else None
    tc = first('res_fragnet.py', lambda p: p.parent.name == 'fe_v4' and (p.parent.parent / 'engine_v6_casmi' / 'library.py').is_file())
    v4 = first('MANIFEST.json', lambda p: (p.parent / 'ranker_v4b.pkl').is_file() or (p.parent / 'code' / 'fe_v4').is_dir())
    sc = first('spec_mz.npy', lambda p: (p.parent / 'split.parquet').is_file())
    vp = first('pool_frag_off.npy', lambda p: (p.parent / 'pool_meta.parquet').is_file())
    roots = dict(training_code=tc.parent.parent if tc else None, v4b=v4.parent if v4 else None,
                 spec_cache=sc.parent if sc else None, v2_pool=vp.parent if vp else None)
    missing = [k for k, v in roots.items() if v is None]
    check('inputs_found', not missing, **{k: str(v) for k, v in roots.items()}, missing=missing)
    if missing:
        raise SystemExit(f'F6 FATAL: datasets not mounted: {missing}')
    ROOTS.update(roots)
    return roots


def step_published():
    log('== 1. published inputs')
    R = ROOTS
    MANIFEST.update(json.loads((R['v4b'] / 'MANIFEST.json').read_text()))
    rows, n_ok = {}, 0
    for rel, want in sorted(MANIFEST['files'].items()):
        p = R['v4b'] / rel
        got = sha256(p) if p.is_file() else None
        rows[rel] = 'absent' if got is None else ('match' if got == want else f'MISMATCH {got}')
        n_ok += got == want
    need = [RES_KEY + f for fs, _, _ in TABLES.values() for f in fs] + \
        [RES_KEY + f'fragnet/{f}' for f in ('sets.pkl', 'countsA.npz', 'featB.npz', 'stageA.npz', 'stageB.txt')]
    check('v4b_manifest_sha256', n_ok == len(MANIFEST['files']) and all(k in MANIFEST['files'] for k in need),
          root=str(R['v4b']), matched=n_ok, listed=len(MANIFEST['files']),
          mismatched={k: v for k, v in rows.items() if v != 'match'}, tables_listed=all(k in MANIFEST['files'] for k in need),
          manifest_created=MANIFEST.get('created'), manifest_sha256=sha256(R['v4b'] / 'MANIFEST.json'))
    tree, files = tree_hash(R['training_code'])
    pins = {k: ('match' if files.get(k) == v else ('absent' if k not in files else f'MISMATCH {files[k]}')) for k, v in TC_PINS.items()}
    bad = {k: v for k, v in pins.items() if v != 'match'}
    check('training_code_pins', not bad, root=str(R['training_code']), pinned=len(TC_PINS), mismatched=bad,
          tree_sha256=tree, tree_files=len(files), tree_equals_2026_10_19=tree == TC_TREE_SHA256,
          note='per-file pins decide; the tree hash only says whether the dataset changed anywhere else')
    REPORT['training_code_tree'] = dict(sha256=tree, files=len(files))
    fam = {k: v for k, v in MANIFEST['files'].items() if k.startswith('code/fe_v4/') and k.endswith('.py')}
    same = {k: files.get('fe_v4/' + k[len('code/fe_v4/'):]) == v for k, v in fam.items()}
    check('code_identity_v4b', all(same.values()), compared=len(same), differing=[k for k, v in same.items() if not v],
          note='fe_v4 family code shipped in v4b (run by the fork) == the training-code copy run here')
    sc = R['spec_cache']
    srows = {f: ('match' if (sc / f).is_file() and sha256(sc / f) == h else 'MISMATCH/absent') for f, h in SPEC_CACHE_SHA.items()}
    check('spec_cache_sha256', all(v == 'match' for v in srows.values()), root=str(sc), files=srows)
    vp = R['v2_pool']
    vrows = {p.name: sha256(p) for p in sorted(vp.iterdir()) if p.is_file()}
    known = {k: vrows.get(k) == v for k, v in V2POOL_KNOWN.items()}
    check('v2_pool_sha256', all(known.values()) and all(f in vrows for f in V2POOL_NEEDED), root=str(vp),
          known_pins=known, needed_present={f: f in vrows for f in V2POOL_NEEDED}, sha256=vrows)
    REPORT['input_sha256'] = dict(v2_pool=vrows, spec_cache={f: SPEC_CACHE_SHA[f] for f in SPEC_CACHE_SHA if srows[f] == 'match'})


# ── 2. Environment ───────────────────────────────────────────────────────────────────────────────────────────
PROBE = r'''
import importlib, json, os, platform, sys
def v(m):
    try:
        return getattr(importlib.import_module(m), '__version__', 'present')
    except Exception as e:
        return f'absent ({type(e).__name__})'
try:
    from rdkit import rdBase
    rd = rdBase.rdkitVersion
except Exception as e:
    rd = f'absent ({type(e).__name__})'
mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
info = open('/proc/cpuinfo').read().splitlines()
cpu = next((l.split(':', 1)[1].strip() for l in info if l.startswith('model name')), None)
flags = next((l for l in info if l.startswith('flags')), '')
print(json.dumps(dict(python=sys.version.split()[0], platform=platform.platform(), numpy=v('numpy'), pandas=v('pandas'),
                      pyarrow=v('pyarrow'), lightgbm=v('lightgbm'), numba=v('numba'), scipy=v('scipy'), torch=v('torch'),
                      rdkit=rd, cpus=len(os.sched_getaffinity(0)), mem_gb=round(mem, 1), cpu=cpu,
                      avx512f='avx512f' in flags.split(), avx2='avx2' in flags.split())))
'''


def _installed(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ''


def step_env():
    log('== 2. environment')
    subprocess.run(f'nproc; free -g | head -2; df -h {WORKING} | tail -1', shell=True)
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    installed = {}
    if _installed('rdkit').replace('.03.', '.3.') != '2026.3.3':
        whl = sorted(INPUT.rglob(f'rdkit-2026.3.3-{tag}-*.whl'))
        if not whl:
            raise SystemExit(f'F6 FATAL: image RDKit is {_installed("rdkit") or "absent"} and no rdkit-2026.3.3 {tag} wheel is attached')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', str(whl[0])])
        installed['rdkit'] = whl[0].name
    if _installed('lightgbm') != HIS_ENV['lightgbm']:
        whl = sorted(INPUT.rglob(f'lightgbm-{HIS_ENV["lightgbm"]}-*.whl'))
        if whl:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', str(whl[0])])
            installed['lightgbm'] = whl[0].name
    out = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True)
    env = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else dict(error=out.stderr[-2000:])
    REPORT['environment'] = dict(ours=env, his=HIS_ENV, wheels_installed=installed)
    norm = lambda v: str(v).replace('.03.', '.3.')
    check('environment', None, ours=env, his=HIS_ENV, wheels_installed=installed,
          rdkit_pinned=norm(env.get('rdkit')) == '2026.3.3', lightgbm_same_as_his=env.get('lightgbm') == HIS_ENV['lightgbm'],
          note='lightgbm differing from 4.7.0 can change stageB bytes; the comparison then falls back to structure / '
               'agreement and says so')
    if norm(env.get('rdkit')) != '2026.3.3':
        raise SystemExit('F6 FATAL: RDKit is not 2026.3.3 after the wheel install')
    return env


# ── 3. Layout (no source edits) ──────────────────────────────────────────────────────────────────────────────
def _link_work(repo):
    work = repo / 'work'; work.mkdir(parents=True, exist_ok=True)
    linked = {}
    for src_dir in (ROOTS['spec_cache'], ROOTS['v2_pool']):
        for p in sorted(src_dir.iterdir()):
            if p.is_file():
                dst = work / p.name
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(p); linked[p.name] = str(p)
    return linked


def _copy_code(repo):
    tc = ROOTS['training_code']
    for src, rel in ((tc / 'fe_v4', FE_REL), (tc / 'engine_v6_casmi', V6_REL)):
        dst = repo / rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return repo / FE_REL


def step_layout():
    log('== 3. layout: README section 4 restored by copies + symlinks; no source file is edited')
    for d in (REPO, REPO_PUB, SEEDREF):
        if d.exists():
            shutil.rmtree(d)
    LOGS.mkdir(parents=True, exist_ok=True)
    linked = _link_work(REPO); _link_work(REPO_PUB)
    for repo in (REPO, REPO_PUB):
        _copy_code(repo)
    pub_res = ROOTS['v4b'] / 'code' / 'fe_v4' / 'res'
    fr = REPO_PUB / FE_REL / 'res' / 'fragnet'; fr.mkdir(parents=True, exist_ok=True)
    for f in ('countsA.npz', 'featB.npz'):
        (fr / f).symlink_to(pub_res / 'fragnet' / f)
    SEEDREF.mkdir(parents=True, exist_ok=True)
    (SEEDREF / 'featB.npz').symlink_to(pub_res / 'fragnet' / 'featB.npz')
    bad = []
    for repo in (REPO, REPO_PUB):
        for rel, want in TC_PINS.items():
            if rel.startswith('fe_v4/'):
                p = repo / FE_REL / rel[len('fe_v4/'):]
            elif rel.startswith('engine_v6_casmi/'):
                p = repo / V6_REL / rel[len('engine_v6_casmi/'):]
            else:
                continue
            if not p.is_file() or sha256(p) != want:
                bad.append(str(p))
    res_pre = sorted(str(p.relative_to(REPO)) for p in (REPO / FE_REL / 'res').rglob('*')) if (REPO / FE_REL / 'res').exists() else []
    layout = {'work/*': 'symlinks to casmi26-spec-cache/cache/* and casmi26-v2-pool/*',
              FE_REL: 'copy of casmi26-training-code/fe_v4', V6_REL: 'copy of casmi26-training-code/engine_v6_casmi',
              f'repo_pub/{FE_REL}/res/fragnet/{{countsA,featB}}.npz': 'symlinks to the published v4b files',
              'seedref/featB.npz': 'symlink to the published v4b file'}
    log('source diff vs the published training code: NONE (layout only)')
    for k, v in layout.items():
        log(f'  {k:60s} {v}')
    check('layout', not bad and not res_pre, source_edits=0, copies_differing_from_pins=bad, res_dir_before_run=res_pre,
          work_links=sorted(linked), layout=layout)


# ── 4-5. Steps ───────────────────────────────────────────────────────────────────────────────────────────────
SEEDREF_CODE = r'''"""Seed reference for casmi26-f6-repro1: res_fragnet.step_fitB (the published STAGEB_PARAMS, 300 rounds,
num_threads = procs) on the PUBLISHED featB.npz with seed + offset. This launcher changes only OUT (output folder) and
STAGEB_PARAMS['seed'] at run time; res_fragnet.py itself is not edited."""
import argparse, sys
FE, OUT, OFFSET, PROCS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, FE)
import res_fragnet as R
R.OUT = OUT
R.STAGEB_PARAMS = dict(R.STAGEB_PARAMS, seed=R.STAGEB_PARAMS['seed'] + OFFSET)
print('SEEDREF STAGEB_PARAMS', R.STAGEB_PARAMS, flush=True)
R.step_fitB(argparse.Namespace(procs=PROCS))
'''


def _step_cmd(name, folder, script, args):
    if name == 'seedref_fitB':
        (RW / 'seedref_launch.py').write_text(SEEDREF_CODE)
        return [sys.executable, '-u', str(RW / 'seedref_launch.py'), str(REPO_PUB / FE_REL), str(SEEDREF),
                str(SEED_REF_OFFSET), str(PROCS)], REPO_PUB
    repo = REPO_PUB if folder == 'pub' else REPO
    return [sys.executable, '-u', f'{FE_REL}/{script}', *args], repo


def run_step(name, folder, script, args, est, deps):
    st = REPORT['steps']
    bad_dep = [d for d in deps if st.get(d, {}).get('status') != 'ok']
    if bad_dep:
        st[name] = dict(status='skipped', reason=f'dependency not ok: {bad_dep}'); save_report()
        log(f'F6 STEP {name}: SKIPPED (dependency {bad_dep})'); return False
    if not can_start(est, minutes()):
        st[name] = dict(status='skipped', reason=f'budget: {minutes()} min elapsed + est {est} + reserve > {BUDGET_MIN}')
        save_report(); log(f'F6 STEP {name}: SKIPPED (budget)'); return False
    if name == 'fragnet_featB':
        _ensure_f1_table()
    cmd, cwd = _step_cmd(name, folder, script, args)
    tmo = step_timeout_min(est, minutes())
    log(f'== F6 STEP {name} ({minutes()} min; est {est} min, timeout {tmo:.0f} min): cd {cwd.name} && '
        + ' '.join(c if c != sys.executable else 'python' for c in cmd))
    rc, sec, timed_out = run_logged(cmd, LOGS / f'{name}.log', timeout=tmo * 60, cwd=cwd,
                                    env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    status = 'ok' if rc == 0 else ('timeout' if timed_out else 'failed')
    st[name] = dict(status=status, exit=rc, seconds=sec, est_min=est, cwd=str(cwd),
                    cmd=' '.join(c if c != sys.executable else 'python' for c in cmd))
    save_report()
    log(f'F6 STEP {name}: {status} exit {rc} in {sec / 60:.1f} min')
    return status == 'ok'


def _ensure_f1_table():
    """featB reads res/frag2/pool_f1_*.npy. Use ours when the f1 step finished, else link the published table."""
    d = REPO / FE_REL / 'res' / 'frag2'
    ours_ok = REPORT['steps'].get('frag2_f1', {}).get('status') == 'ok' and (d / 'pool_f1_off.npy').is_file()
    if not ours_ok:
        d.mkdir(parents=True, exist_ok=True)
        for f in ('pool_f1_off.npy', 'pool_f1_mass.npy'):
            if (d / f).exists() or (d / f).is_symlink():
                (d / f).unlink()
            (d / f).symlink_to(ROOTS['v4b'] / 'code' / 'fe_v4' / 'res' / 'frag2' / f)
    REPORT['featB_f1_table'] = 'regenerated (frag2_f1 step)' if ours_ok else 'PUBLISHED (frag2_f1 step not ok)'
    log(f'featB uses the {REPORT["featB_f1_table"]} frag2 f1 table')


def _pub(rel):
    return ROOTS['v4b'] / 'code' / 'fe_v4' / 'res' / rel


def _his_log(fname):
    p = ROOTS['training_code'] / 'fe_v4' / fname
    return p.read_text() if p.is_file() else ''


def _our_log(step):
    p = LOGS / f'{step}.log'
    return p.read_text(errors='replace') if p.is_file() else ''


def cmp_table_step(step):
    import numpy as np
    files, logname, sel = TABLES[step]
    out = REPO / FE_REL / 'res'
    rows, n_exact = {}, 0
    for rel in files:
        ours, pub = out / rel, _pub(rel)
        want = MANIFEST['files'].get(RES_KEY + rel)
        got = sha256(ours) if ours.is_file() else None
        r = dict(sha256=got, manifest=want, bit_exact=got is not None and got == want)
        n_exact += r['bit_exact']
        if got and not r['bit_exact']:
            if rel.endswith('.npy'):
                a, b = np.load(ours, mmap_mode='r', allow_pickle=False), np.load(pub, mmap_mode='r', allow_pickle=False)
                r['array'] = array_diff(a, b)
                if rel in PACKED:
                    r['per_structure'] = packed_diff(np.load(out / PACKED[rel]), a, np.load(_pub(PACKED[rel])), b)
            else:
                r['text_equal'] = ours.read_bytes() == pub.read_bytes()
        rows[rel] = r
    content = all(r['bit_exact'] or r.get('array', {}).get('equal') or r.get('text_equal') for r in rows.values())
    status = 'BIT-EXACT' if n_exact == len(files) else ('CONTENT-EQUAL' if content else 'DIFFERENT')
    lg = compare_log(_his_log(logname), _our_log(step), sel)
    check(step, status in ('BIT-EXACT', 'CONTENT-EQUAL'), status=status, files=rows, bit_exact=n_exact, n_files=len(files))
    log_ok = lg.pop('ok')
    check(f'{step}_log', log_ok, log=logname, **lg)
    artefact(step, status, files=files, log_reproduced=log_ok)


def cmp_stageA(step, path):
    if not path.is_file():
        check(step, False, error=f'{path} not written'); artefact(step, 'FAILED'); return
    A, B = npz_load(path), npz_load(_pub('fragnet/stageA.npz'))
    chains_equal = _plain(A.pop('chains', None)) == _plain(B.pop('chains', None))
    status, det = compare_named_arrays(A, B, ulp_tol=F32_ULP_TOL, exact_keys={k for k in A if k.startswith('k_')} | {'alpha'})
    if not chains_equal:
        status = 'DIFFERENT'
    file_same = sha256(path) == MANIFEST['files'][RES_KEY + 'fragnet/stageA.npz']
    if status == 'CONTENT-EQUAL' and file_same:
        status = 'BIT-EXACT'
    check(step, status in ('BIT-EXACT', 'CONTENT-EQUAL', 'NUMERIC-EQUAL'), status=status, file_sha256_equal=file_same,
          chains_equal=chains_equal, ulp_tol=F32_ULP_TOL, our_log_tail=_our_log(step).strip().splitlines()[-1:], **det)
    artefact(step, status, source='published countsA.npz' if step.startswith('pub') else 'regenerated countsA.npz')


def cmp_stageB_text(step, path):
    if not path.is_file():
        check(step, False, error=f'{path} not written'); artefact(step, 'FAILED'); return None
    d = compare_lgb_text(path.read_text(), _pub('fragnet/stageB.txt').read_text())
    d['file_sha256'] = sha256(path)
    lg = compare_log(_his_log('res_fragnet_fitB.log'), _our_log(step))
    d['fitB_log_reproduced'] = lg['ok']
    REPORT.setdefault('stageB_text', {})[step] = d
    # text verdict now; step 6 may upgrade DIFFERENT to EQUIVALENT (V MRR + agreement vs the seed reference)
    check(step, d['status'] != 'DIFFERENT', **d, note='text comparison; see the model check in step 6')
    ok_log = lg.pop('ok')
    check(f'{step}_log', ok_log if step == 'pub_fitB' else None, **lg)
    artefact(step, d['status'], stage='text comparison')
    return d


def cmp_sets():
    ours_p = REPO / FE_REL / 'res' / 'fragnet' / 'sets.pkl'
    if not ours_p.is_file():
        check('fragnet_sets', False, error='sets.pkl not written'); artefact('fragnet_sets', 'FAILED'); return
    d = compare_sets(load_pickle_numpy_only(ours_p), load_pickle_numpy_only(_pub('fragnet/sets.pkl')))
    d['file_sha256_equal'] = sha256(ours_p) == MANIFEST['files'][RES_KEY + 'fragnet/sets.pkl']
    d['our_summary_line'] = [l for l in _our_log('fragnet_sets').splitlines() if l.startswith('sets ')][-1:]
    d['decision_log_parts'] = PUB_SETS['parts']
    status = ('BIT-EXACT' if d['file_sha256_equal'] else 'CONTENT-EQUAL') if d['identical'] else 'DIFFERENT'
    check('fragnet_sets', d['identical'], status=status, **d)
    artefact('fragnet_sets', status, note='no published log (README gap); compared with the published sets.pkl')


def cmp_npz_step(step, fname, logname, rtol, exact):
    import numpy as np
    ours_p = REPO / FE_REL / 'res' / 'fragnet' / fname
    if not ours_p.is_file():
        check(step, False, error=f'{fname} not written'); artefact(step, 'FAILED'); return
    A = dict(np.load(ours_p, allow_pickle=False)); B = dict(np.load(_pub(f'fragnet/{fname}'), allow_pickle=False))
    status, det = compare_named_arrays(A, B, float_rtol=rtol, exact_keys=exact(A))
    file_same = sha256(ours_p) == MANIFEST['files'][RES_KEY + f'fragnet/{fname}']
    if status == 'CONTENT-EQUAL' and file_same:
        status = 'BIT-EXACT'
    lg = compare_log(_his_log(logname), _our_log(step))
    check(step, status in ('BIT-EXACT', 'CONTENT-EQUAL', 'NUMERIC-EQUAL'), status=status, file_sha256_equal=file_same,
          float_rtol=rtol, **det)
    log_ok = lg.pop('ok')
    check(f'{step}_log', log_ok, log=logname, **lg)
    artefact(step, status, log_reproduced=log_ok)


def compare_after(name):
    fr = lambda repo: repo / FE_REL / 'res' / 'fragnet'
    if name in TABLES:
        cmp_table_step(name)
    elif name == 'pub_fitA':
        cmp_stageA('pub_fitA', fr(REPO_PUB) / 'stageA.npz')
    elif name == 'pub_fitB':
        cmp_stageB_text('pub_fitB', fr(REPO_PUB) / 'stageB.txt')
    elif name == 'fragnet_sets':
        cmp_sets()
    elif name == 'fragnet_countA':
        # keys and truth counts (weight 1.0) exact; decoy counts carry imap_unordered summation order -> COUNT_RTOL
        cmp_npz_step(name, 'countsA.npz', 'res_fragnet_countA.log', COUNT_RTOL,
                     lambda A: {k for k in A if k.endswith('_k') or '_T_' in k})
    elif name == 'fragnet_featB':
        cmp_npz_step(name, 'featB.npz', 'res_fragnet_featB.log', FEAT_RTOL,
                     lambda A: {'set', 'part', 'y', 'same', 'dsrc', 'mode', 'lib'})
    elif name == 'fragnet_fitA':
        cmp_stageA('fragnet_fitA', fr(REPO) / 'stageA.npz')
    elif name == 'fragnet_fitB':
        cmp_stageB_text('fragnet_fitB', fr(REPO) / 'stageB.txt')


# ── 6. Models on the published featB ────────────────────────────────────────────────────────────────────────
def his_modules():
    fe = REPO_PUB / FE_REL
    for p in (str(REPO_PUB / 'work' / 'analysis' / 'engine_v6' / 'code'), str(fe / 'fams'), str(fe)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fragnet as F
    import res_fragnet as R
    return F, R


def step_models():
    log('== 6. stageB models on the published featB rows (his _groups / _mrr harness)')
    import numpy as np, lightgbm as lgb
    F, R = his_modules()
    z = np.load(_pub('fragnet/featB.npz'), allow_pickle=False)
    sets = z['set']; y = z['y']; same = z['same'].astype(bool); part = z['part']; md = z['mode']
    st, en = R._groups(sets)
    agg = z['agg_' + R.STAGEB_CFG].astype(np.float64)
    X = np.zeros((len(y), len(F.BNAMES)), np.float32)
    for a_, b_ in zip(st, en):
        X[a_:b_] = F.stageB_features(agg[a_:b_])
    V = part == 'V'
    raw = agg[:, F.AGG.index('pk')] + 0.02 * agg[:, F.AGG.index('fc_u')]
    refs = {'fr_int': z['fr_int'], 'frag2_mix': z['mix'], 'stageA_pk_fcu': raw}
    paths = dict(pub=_pub('fragnet/stageB.txt'), rep=REPO_PUB / FE_REL / 'res' / 'fragnet' / 'stageB.txt',
                 ref=SEEDREF / 'stageB.txt', chain=REPO / FE_REL / 'res' / 'fragnet' / 'stageB.txt')
    preds, meta = {}, {}
    for k, p in paths.items():
        if not p.is_file():
            meta[k] = dict(path=str(p), absent=True); continue
        bst = lgb.Booster(model_file=str(p))
        preds[k] = bst.predict(X, num_threads=PROCS)
        meta[k] = dict(path=str(p), sha256=sha256(p), trees=bst.num_trees(),
                       v_mrr=R._mrr(preds[k], y, same, st, en, V), v_mrr_pos=R._mrr(preds[k], y, same, st, en, V & (md > 0)),
                       v_mrr_neg=R._mrr(preds[k], y, same, st, en, V & (md < 0)))
        log(f'F6 MODEL {k}: ' + json.dumps(meta[k], default=_js))
    REPORT['models'] = meta
    here = {n: R._mrr(s, y, same, st, en, V) for n, s in refs.items()}
    if 'pub' in meta:
        here['fragnet'] = meta['pub']['v_mrr']
    parity = {n: [PUB_V_MRR[n], round(v, 4)] for n, v in here.items()}
    check('harness_parity', all(a == b for a, b in parity.values()) and len(parity) == 4, logged_vs_here=parity,
          v_groups=int(V[st].sum()), rows=len(y), note='res_fragnet_fitB.log V lines recomputed from the published files')
    pairs = [('pub', 'rep', 'published vs refit from the published featB (same data, same code)'),
             ('pub', 'ref', f'published vs seed+{SEED_REF_OFFSET} refit on the published featB (seed spread)'),
             ('pub', 'chain', 'published vs full-chain refit (our sets -> countA -> featB -> fitB)'),
             ('rep', 'ref', 'refit vs seed reference')]
    REPORT['pairs'] = {}
    for a, b, what in pairs:
        if a in preds and b in preds:
            s = pair_agreement(preds[a], preds[b], y, same, st, en, V)
            REPORT['pairs'][f'{a}__{b}'] = dict(what=what, **s)
            log(f'F6 PAIR {a} vs {b} ({what}): ' + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()}))
    save_report()
    ref_sp = REPORT['pairs'].get('pub__ref', {}).get('spearman_per_group_mean')
    for step, key in (('pub_fitB', 'rep'), ('fragnet_fitB', 'chain')):
        txt = REPORT.get('stageB_text', {}).get(step)
        if txt is None or key not in meta or meta[key].get('absent'):
            continue
        p = REPORT['pairs'].get(f'pub__{key}', {})
        dv = meta[key]['v_mrr'] - PUB_V_MRR['fragnet']
        status = stageB_verdict(txt['status'], meta[key]['v_mrr'], p.get('spearman_per_group_mean'), ref_sp)
        check(step, status != 'DIFFERENT', status=status, v_mrr=meta[key]['v_mrr'], v_mrr_published_log=PUB_V_MRR['fragnet'],
              v_mrr_diff=dv, tol=V_MRR_TOL, spearman_vs_published=p.get('spearman_per_group_mean'),
              spearman_seedref_vs_published=ref_sp, top1_agreement=p.get('top1_agreement'),
              max_abs_score_diff=p.get('max_abs_score_diff'), seedref_v_mrr=meta.get('ref', {}).get('v_mrr'),
              **{k: v for k, v in txt.items() if k != 'status'})
        artefact(step, status, v_mrr=meta[key]['v_mrr'])
    if 'ref' in meta and not meta['ref'].get('absent'):
        check('seed_reference', None, v_mrr=meta['ref']['v_mrr'], v_mrr_diff_vs_published=meta['ref']['v_mrr'] - PUB_V_MRR['fragnet'],
              spearman_vs_published=ref_sp, top1_agreement=REPORT['pairs'].get('pub__ref', {}).get('top1_agreement'))


# ── finish ───────────────────────────────────────────────────────────────────────────────────────────────────
ORDER = ['pub_fitA', 'pub_fitB', 'frag2_caps', 'frag2_f1', 'fragnet_rec', 'fragnet_sets', 'fragnet_countA',
         'fragnet_featB', 'fragnet_fitA', 'fragnet_fitB', 'analog_rec', 'analog_str']
WHAT = {'pub_fitA': 'stageA.npz from the published countsA', 'pub_fitB': 'stageB.txt from the published featB',
        'frag2_caps': 'frag2 pool_cap.npy + caps.txt', 'frag2_f1': 'frag2 pool_f1_{off,mass}.npy',
        'fragnet_rec': 'fragnet rec_*.npy (5)', 'fragnet_sets': 'fragnet sets.pkl (nspec 32000, seed 0)',
        'fragnet_countA': 'fragnet countsA.npz', 'fragnet_featB': 'fragnet featB.npz', 'fragnet_fitA': 'stageA.npz (full chain)',
        'fragnet_fitB': 'stageB.txt (full chain)', 'analog_rec': 'analog_struct {pool,train}_{hash,num,fvec}.npy',
        'analog_str': 'analog_struct {pool,train}_{el,core}smi_{buf,off}.npy'}


def step_finish():
    out = REPO / FE_REL / 'res'
    shas = {}
    for base, tag in ((out, 'repo'), (REPO_PUB / FE_REL / 'res', 'repo_pub'), (SEEDREF, 'seedref')):
        if base.exists():
            for p in sorted(base.rglob('*')):
                if p.is_file() and not p.is_symlink():
                    shas[f'{tag}/{p.relative_to(base).as_posix()}'] = sha256(p)
    (RW / 'repro_sha256.json').write_text(json.dumps(shas, indent=2))
    REPORT['output_sha256'] = shas
    freed = 0
    if not KEEP_MATCHED and out.exists():
        for p in out.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.stat().st_size > 20 << 20:
                rel = p.relative_to(out).as_posix()
                if MANIFEST.get('files', {}).get(RES_KEY + rel) == shas.get(f'repo/{rel}'):
                    freed += p.stat().st_size; p.unlink()
    REPORT['deleted_matched_bytes'] = freed
    c, a = REPORT['checks'], REPORT['artefacts']
    for s in ORDER:
        if s not in a:
            stp = REPORT['steps'].get(s, {})
            if stp.get('status') in (None, 'skipped'):
                artefact(s, 'SKIPPED', reason=stp.get('reason') or 'not reached')
            else:
                artefact(s, 'FAILED', reason='comparison error (see errors)' if stp.get('status') == 'ok' else stp.get('status'))
    must = ['inputs_found', 'v4b_manifest_sha256', 'training_code_pins', 'code_identity_v4b', 'spec_cache_sha256',
            'v2_pool_sha256', 'layout', 'harness_parity']
    ran = [s for s in ORDER if a.get(s, {}).get('status') not in ('SKIPPED',)]
    failed = [k for k in must if not c.get(k, {}).get('ok')] + \
        [s for s in ran if a[s]['status'] in ('DIFFERENT', 'FAILED')]
    REPORT['verdict'] = dict(all_required_ok=not failed, failed=failed,
                             skipped=[s for s in ORDER if a.get(s, {}).get('status') == 'SKIPPED'],
                             artefacts={s: a.get(s, {}).get('status') for s in ORDER}, minutes=minutes())
    save_report()
    log('F6 SUMMARY ' + json.dumps({k: v for k, v in REPORT['verdict'].items() if k != 'artefacts'}))
    for s in ORDER:
        stp = REPORT['steps'].get(s, {})
        log(f"F6 SUMMARY {s:15s} {a.get(s, {}).get('status', '?'):14s} {WHAT[s]:52s} "
            f"{(stp.get('seconds') or 0) / 60:5.1f} min")
    m = REPORT.get('models', {}); pr = REPORT.get('pairs', {})
    g = lambda k: (round(m[k]['v_mrr'], 4) if k in m and not m[k].get('absent') else None)
    log(f"F6 SUMMARY stageB V MRR: logged {PUB_V_MRR['fragnet']} | published file {g('pub')} | refit-from-published "
        f"{g('rep')} | seed+{SEED_REF_OFFSET} {g('ref')} | full chain {g('chain')} ; per-group Spearman vs published: "
        f"rep {pr.get('pub__rep', {}).get('spearman_per_group_mean')} ref {pr.get('pub__ref', {}).get('spearman_per_group_mean')} "
        f"chain {pr.get('pub__chain', {}).get('spearman_per_group_mean')}")
    log(f'F6 DONE {minutes()} min; report {REPORT_PATH}')


def remove_links():
    """Drop every symlink under RW (they point into /kaggle/input; the output collector must not see them)."""
    n = 0
    if RW.exists():
        for p in sorted(RW.rglob('*'), key=lambda q: -len(q.parts)):
            if p.is_symlink():
                p.unlink(); n += 1
    return n


def main():
    RW.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    try:
        find_inputs()
        step_published()
        step_env()
        step_layout()
        for name, folder, script, args, est, deps in STEPS:
            ok = guarded(f'run {name}', run_step, name, folder, script, args, est, deps)
            if ok:
                guarded(f'compare {name}', compare_after, name)
        guarded('models', step_models)
        guarded('finish', step_finish)
    finally:
        REPORT['symlinks_removed'] = remove_links()
        save_report()


if os.environ.get('F6_IMPORT_ONLY') != '1':
    main()
