"""casmi26-v18cx-build: stages C1, C2 and D of scripts/build_v18c.sh for the PF-x PeakFormer (pf_pfx_tpu.pt, md5 53b2c5eb,
output of kernel bryantyu/casmi26-pfx-tpu) on the v18c base -> stage-1 ranker pair v18cx (LightGBM seed 0) + v18cxr1 (seed 1)
for Engine 2 OURS. Moved here from a local run that a container recycle killed (/home/user/work/build_v18cx.log, Nov 2).

The repository's scripts/build_v18c.sh runs UNMODIFIED, in the three steps of the local wrapper (as scripts/build_pfswap_pair.sh,
STOP_AFTER=D):
  1. BASE=v18c REUSE_C1=0 STOP_AFTER=D bash scripts/build_v18c.sh $W/pfs/pf_pfx_tpu.pt $W/own_models/fp_model_v1r.pt v18cx
     C1 PeakFormer pf_* (scripts/add_fp_feats.py, 3 shards, RICHFP_CACHE empty) + casmi.own.formula_rel PF_FREL on the stage-B
     rows ret18c_{train,val}; C2 FP model fp_* (fp_model_v1r.pt, 3 shards); D scripts/train_ret_ranker.py (DOC_W=pool_v18c/
     doc_w.parquet DOC_ALPHA=1 RANKER_PARAMS='{"num_threads": 4}' --drop-artifact), scripts/oof_stage1.py, val_reach_c.
  2. ln ret18cx_{train,val}_{pf,fp}.parquet -> ret18cxr1_*   (same rows; val_reach_c is recomputed, as the wrapper does)
  3. BASE=v18c REUSE_C1=0 STOP_AFTER=D RANKER_PARAMS='{"num_threads": 4, "seed": 1}' bash scripts/build_v18c.sh ... v18cxr1
The script's hard-coded tree (/home/user/{claude_code_tests,work,data,ext}) is rebuilt from the inputs with symlinks:
  casmi/        <- bryantyu/casmi26-code (MANIFEST + per-module sha256 checked against repo HEAD)
  scripts/      <- bryantyu/casmi26-v18cx-inputs (git show of the repo commit named in v18cx_inputs.md5)
  pool_v18c/    <- bryantyu/casmi26-pool-v18c (md5 checked) + doc_w.parquet + .built
  ret18c_{train,val}.parquet, rank/val keys, train_store/spec_*, own_models/fp_model_v1r.pt, data/train_meta.parquet,
  ext/bio_raw/bio_structures_own.parquet <- bryantyu/casmi26-v18cx-inputs (md5 checked against v18cx_inputs.md5)
  pfs/pf_pfx_tpu.pt <- kernel output bryantyu/casmi26-pfx-tpu (md5 checked)
CPU kernel: casmi/peakformer.py scores on CPU only (PFScorer.score keeps candidate tensors on the CPU; add_fp_feats.py passes no
device), and the local builds ran on CPU.
Outputs (/kaggle/working, small): ret_ranker_v18cx{,r1}.txt + _feats.txt, build_v18cx{,r1}.log (stdout of build_v18c.sh with the
train_ret_ranker val table, OOF MRR, val queries), stage1_metrics.txt, shard_logs.txt (C1/C2 shard progress), md5.txt,
row_digest.json, env.json. The big row parquets stay in the container; KEEP_ROWS=1 also keeps ret_v18cx{,r1}_train_oof.parquet
and ret18cx_val_reach_c.parquet (stage E inputs, ~110 MB per tag).
Local plumbing test: V18CX_INPUT=<fake input tree> V18CX_HU=<fresh dir> V18CX_OUT=<dir> V18CX_TEST=1 python3 notebook.py
(V18CX_HU != /home/user relocates the scripts' /home/user/ paths into it; TEST: LightGBM / BLAS 1 thread).
"""
import glob, hashlib, json, os, re, shutil, subprocess, sys, threading, time

T0 = time.time()
KEEP_ROWS = int(os.environ.get('KEEP_ROWS', '0'))
INPUT = os.environ.get('V18CX_INPUT', '/kaggle/input')
OUT = os.environ.get('V18CX_OUT', '/kaggle/working')
HU = os.environ.get('V18CX_HU', '/home/user')
TEST = os.environ.get('V18CX_TEST') == '1'
PF_MD5 = '53b2c5ebb9c80c8e4236d318773ad3b6'
POOL_MD5 = {'pool.parquet': '8afda40aa9324130b800523c1e5f267b', 'pool_fp.npy': 'cef50fc519f985ddbf878b5ff5889ada',
            'coconut_prior.parquet': 'f8b83ce5512deabd27431a9ae8b48c87'}
CODE_MANIFEST_SHA = '4a360debfd8c83fd99f6c6b7697c480220a51e788d2ddb752fcb8273d6a02820'   # casmi26-code @1e31261 (Oct 30)
CODE_SHA = {   # repo HEAD sha256 of every casmi module these stages import (identical in casmi26-code @1e31261)
    '__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'analog.py': 'd709dc82474ecdbf02728e6508de681d22987986a82208917b838f9e115868a7',
    'docw.py': 'bbb84304ce2f684ec62c608fa491d964d56d19514933bc758870886c21461413',
    'dreams_feats.py': '645778571137ee21dfeff0ea1aae465eacefa40df4f01e85a894e6de9c8ef325',
    'entropy.py': '8ecc377071f6b8d7604d8420cd65f1f275dfa32b2c1d735eb26a1b56ff463faf',
    'fpmodel.py': '2930be30289b173669fff54ce0d9852a4fe117455f9f9cb1590b2900462a0760',
    'fpscore.py': '528195f3508b7cc4e3a74a05b522c399b275d08225efd1247b307852100ac598',
    'frag.py': '906d1462e12b20fa9291edd73158a7d57cac2dda28976bdaf42c01a4ea4172fb',
    'metric.py': '89919049e7689148b28245dfef455b39eb67797b85faaade26a56453081b415f',
    'own.py': 'fd73d4a73324e4d1c624649e45c34f3f07ef8776177f6e5c00b4f7c2e673cb28',
    'peakformer.py': '5e5bab5ddafc15db63951ec829bd61656de4d7ea649aceec407d91dfb53ecf62',
    'prep_meta.py': '3e312a4062d4d9904a39ef018dc0e676aa30eb38a3a2c74d0b54b081ee7d3719',
    'retrieval.py': 'e75fbc46e224e1c22315565c1f0574c71dc7ef4d6e9b8c1c351b8880251a5b86',
    'richfp.py': 'f7719ab099017dd5e5550ae2412d04ddd81c5c884d42cbbd3b8dfd4d0a702763',
    'spectra.py': 'b7014fb302c9c05ee7db34d0619ee07a913edbfe5a016921dc995aafc1d9497f',
}
FEATS_MD5 = '93d47b722066c83ed54f88d6885c9acd'     # the 58-feature list of every v18c-family ranker (v18c ... v18cur1)
SCRIPTS = ['build_v18c.sh', 'add_fp_feats.py', 'train_ret_ranker.py', 'oof_stage1.py']
TAGS = ['v18cx', 'v18cxr1']
times, problems = {}, []
os.makedirs(OUT, exist_ok=True)


def say(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def digest(p, algo='md5'):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def find(name):
    hits = sorted(glob.glob(f'{INPUT}/**/{name}', recursive=True), key=len)
    if not hits:
        raise SystemExit(f'input {name} not found under {INPUT}')
    if len(hits) > 1:
        say(f'note: {len(hits)} copies of {name}, using the shortest path:', hits)
    return hits[0]


# ---- 0. RDKit 2026.3.3 (the local reference version) from the wheel dataset, as the inference kernels do
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob(f'{INPUT}/**/rdkit-2026.3.3-*.whl', recursive=True) if tag in w]
if whl:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
elif not TEST:
    raise SystemExit(f'no rdkit-2026.3.3 {tag} wheel under {INPUT} (dataset metric/rdkit-2026-3-3-wheel)')

# ---- 1. inputs and checks
code = os.path.dirname(find('peakformer.py'))
inp = os.path.dirname(find('v18cx_inputs.md5'))
pool = os.path.dirname(find('pool_fp.npy'))
pf = find('pf_pfx_tpu.pt')
say('code', code, '| inputs', inp, '| pool', pool, '| pf', pf)
md5 = {}
for line in open(f'{inp}/v18cx_inputs.md5'):
    if line.startswith('#') or not line.strip():
        continue
    h, f = line.split()
    md5[f] = digest(f'{inp}/{f}')
    if md5[f] != h:
        problems.append(f'input {f}: md5 {md5[f]} != {h}')
for f, h in POOL_MD5.items():
    md5[f'pool_v18c/{f}'] = digest(f'{pool}/{f}')
    if md5[f'pool_v18c/{f}'] != h:
        problems.append(f'pool {f}: md5 {md5["pool_v18c/" + f]} != {h}')
md5['pf_pfx_tpu.pt'] = digest(pf)
if md5['pf_pfx_tpu.pt'] != PF_MD5:
    problems.append(f'pf_pfx_tpu.pt md5 {md5["pf_pfx_tpu.pt"]} != {PF_MD5}')
for f, h in CODE_SHA.items():
    got = digest(f'{code}/{f}', 'sha256') if os.path.exists(f'{code}/{f}') else 'missing'
    if got != h:
        problems.append(f'casmi/{f}: sha256 {got[:12]} != repo HEAD {h[:12]}')
man = f'{code}/MANIFEST.txt'
if os.path.exists(man):
    bad = [l.split()[1] for l in open(man) if not l.startswith('#') and l.strip()
           and digest(f'{code}/{l.split()[1]}', 'sha256') != l.split()[0]]
    say('casmi26-code', [l.strip() for l in open(man) if l.startswith('# commit')][:1], 'MANIFEST sha256',
        digest(man, 'sha256')[:12], '(expected', CODE_MANIFEST_SHA[:12] + ')', 'files failing it:', bad)
    if bad:
        problems.append(f'casmi26-code files failing its MANIFEST: {bad}')
else:
    say('casmi26-code has no MANIFEST.txt (per-module sha256 checked above)')
for f in SCRIPTS:
    if f not in md5:
        problems.append(f'script {f} not listed in v18cx_inputs.md5')
say('inputs checked:', len(md5), 'files;', 'problems:', problems or 'none')
if problems:
    raise SystemExit('input check failed:\n' + '\n'.join(problems))

# ---- 2. the /home/user tree of the build script
if 'V18CX_HU' not in os.environ:
    try:                        # Kaggle runs as root; if / were read-only, relocate the tree (same scripts, paths rewritten)
        os.makedirs(HU, exist_ok=True)
        open(f'{HU}/.probe', 'w').close(); os.remove(f'{HU}/.probe')
    except OSError as e:
        say(f'{HU} not writable ({e}); relocating the tree to /tmp/v18cx_hu')
        HU = '/tmp/v18cx_hu'
R, W = f'{HU}/claude_code_tests', f'{HU}/work'
if os.path.exists(W) or os.path.exists(R):
    raise SystemExit(f'refusing: {W} or {R} exists (this kernel builds its own tree; set V18CX_HU to a fresh directory)')
for d in (f'{R}/scripts', f'{W}/pool_v18c', f'{W}/own_models', f'{W}/pfs', f'{W}/train_store', f'{HU}/data',
          f'{HU}/ext/bio_raw', f'{HU}/bin'):
    os.makedirs(d)
shutil.copytree(code, f'{R}/casmi', ignore=shutil.ignore_patterns('__pycache__'))
for f in SCRIPTS:
    shutil.copy(f'{inp}/{f}', f'{R}/scripts/{f}')
WRAP_STEP2 = ('for s in train val; do for k in pf fp; do [ -f $W/ret18cxr1_${s}_$k.parquet ] || ln $W/ret18cx_${s}_$k.parquet '
              '$W/ret18cxr1_${s}_$k.parquet; done; done\n')      # = scripts/build_pfswap_pair.sh
R1_PARAMS = '{"num_threads": 4, "seed": 1}'
if HU != '/home/user':          # relocated tree (local test): point the literal /home/user/ paths of the copies into it
    for p in sorted(glob.glob(f'{R}/scripts/*') + glob.glob(f'{R}/casmi/*.py')):
        s = open(p).read()
        if '/home/user/' in s:
            open(p, 'w').write(s.replace('/home/user/', HU + '/'))
            say('relocated /home/user/ ->', HU, 'in', os.path.relpath(p, R))


def link(src, dst):
    os.symlink(os.path.abspath(src), dst)


for f in POOL_MD5:
    link(f'{pool}/{f}', f'{W}/pool_v18c/{f}')
link(f'{inp}/doc_w.parquet', f'{W}/pool_v18c/doc_w.parquet')
open(f'{W}/pool_v18c/.built', 'w').close()          # pool_v18c is the finished v18c pool (md5 checked above)
for f in ('ret18c_train.parquet', 'ret18c_val.parquet', 'rank_keys.txt', 'val_keys.txt'):
    link(f'{inp}/{f}', f'{W}/{f}')
for f in ('spec_mz.npy', 'spec_int.npy', 'spec_off.npy', 'spec_meta.parquet'):
    link(f'{inp}/{f}', f'{W}/train_store/{f}')
link(f'{inp}/fp_model_v1r.pt', f'{W}/own_models/fp_model_v1r.pt')
link(pf, f'{W}/pfs/pf_pfx_tpu.pt')
link(f'{inp}/train_meta.parquet', f'{HU}/data/train_meta.parquet')
link(f'{inp}/bio_structures_own.parquet', f'{HU}/ext/bio_raw/bio_structures_own.parquet')
with open(f'{HU}/bin/python3', 'w') as f:          # the script calls `python3`: this interpreter (with the RDKit wheel)
    f.write(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
os.chmod(f'{HU}/bin/python3', 0o755)
env = dict(os.environ, PATH=f'{HU}/bin:' + os.environ.get('PATH', ''))
for k in ('C1_DUP', 'DOC_TRANSPLANT', 'DOC_TRANSPLANT_KEEP', 'PRIOR_SHIFT', 'OWN_PF_INST', 'FP_PMI', 'FP_PMI_LAM', 'VENDOR_PATH',
          'CASMI_POOL', 'RICHFP_CACHE', 'RANKER_PARAMS', 'S2_PARAMS', 'DOC_W', 'DOC_ALPHA', 'BASE', 'REUSE_C1', 'STOP_AFTER',
          'GL_HAVE', 'GLW_N', 'GL_DEVICE'):     # env the scripts read: only build_v18c.sh / run() set them (local runs had none)
    env.pop(k, None)
if TEST:
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
               RANKER_PARAMS='{"num_threads": 1}')
    R1_PARAMS = '{"num_threads": 1, "seed": 1}'
vers = subprocess.run(['python3', '-c', 'import sys, numpy, pandas, pyarrow, torch, lightgbm, rdkit, numba, json; print(json.dumps('
                       'dict(python=sys.version.split()[0], numpy=numpy.__version__, pandas=pandas.__version__, pyarrow=pyarrow.__version__,'
                       ' torch=torch.__version__, lightgbm=lightgbm.__version__, rdkit=rdkit.__version__, numba=numba.__version__,'
                       ' torch_threads=torch.get_num_threads())))'], env=env, capture_output=True, text=True)
say('environment', vers.stdout.strip(), vers.stderr.strip()[-500:], 'cpus', os.cpu_count())
try:
    v = json.loads(vers.stdout)
except ValueError:
    raise SystemExit('python3 of the build cannot import the stack: ' + vers.stderr[-2000:])
say('local reference builds (v18c ... v18cur1): lightgbm 4.7.0, torch 2.14.1, pandas 3.0.5, numpy 2.5.3, pyarrow 26.0.0 '
    '(review test, 42 queries: pandas 2.3.3 / numpy 2.1.3 gave bit-identical pf_* columns and lightgbm 4.6.0 identical trees; '
    'torch and the CPU type can still change float bits)')
if v['rdkit'] != '2026.03.3':           # PeakFormer / FP candidate fingerprints come from RDKit: the local builds used 2026.03.3
    raise SystemExit(f'RDKit {v["rdkit"]} != 2026.03.3 (the local reference); refusing to build')

# ---- 3. progress monitor: last line of every C1 / C2 shard log
stop = threading.Event()


def monitor(every=60 if TEST else 600):
    while not stop.wait(every):
        logs = sorted(glob.glob(f'{W}/ret18cx*.parquet.[pf][0-9].log'))
        say('progress', ' | '.join(f'{os.path.basename(p)[5:]}: {(open(p).read().strip().splitlines() or ["-"])[-1]}' for p in logs))


threading.Thread(target=monitor, daemon=True).start()


def run(cmd, log, extra):
    say('RUN', ' '.join(f'{k}={v!r}' for k, v in extra.items()), cmd if isinstance(cmd, str) else ' '.join(cmd))
    t = time.time()
    with open(log, 'w') as fh:
        p = subprocess.Popen(cmd, cwd=R, env={**env, **extra}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             shell=isinstance(cmd, str), executable='/bin/bash' if isinstance(cmd, str) else None)
        for line in p.stdout:
            print(line, end='', flush=True)
            fh.write(line)
        rc = p.wait()
    times[os.path.basename(log)] = round(time.time() - t)
    say('exit', rc, 'after', times[os.path.basename(log)], 's')
    return rc


PFP, FPP = f'{W}/pfs/pf_pfx_tpu.pt', f'{W}/own_models/fp_model_v1r.pt'
base = dict(BASE='v18c', REUSE_C1='0', STOP_AFTER='D')
rc = run(['bash', 'scripts/build_v18c.sh', PFP, FPP, 'v18cx'], f'{OUT}/build_v18cx.log', base)
if rc == 0:
    rc = run(f'set -euo pipefail\nW={W}\n' + WRAP_STEP2, f'{OUT}/link_v18cxr1.log', {})
if rc == 0:
    rc = run(['bash', 'scripts/build_v18c.sh', PFP, FPP, 'v18cxr1'], f'{OUT}/build_v18cxr1.log', {**base, 'RANKER_PARAMS': R1_PARAMS})
stop.set()
if rc:
    problems.append(f'build exit {rc}')

# ---- 4. small outputs
with open(f'{OUT}/shard_logs.txt', 'w') as fh:
    for p in sorted(glob.glob(f'{W}/ret18cx*.parquet.[pf][0-9].log')):
        fh.write(f'== {os.path.basename(p)}\n{open(p).read()}\n')
PAT = re.compile(r"^(features \d+ |subset |(np|npx|other)\s+[12]\s|ALL cls |\{'|fold \d done|queries \d+ OOF|val queries)")
with open(f'{OUT}/stage1_metrics.txt', 'w') as fh:
    for t in TAGS:
        log = f'{OUT}/build_{t}.log'
        lines = [l for l in open(log)] if os.path.exists(log) else []
        fh.write(f'== {t}\n' + ''.join(l for l in lines if PAT.match(l)) + '\n')
out_md5 = {}
for t in TAGS:
    for suf in ('.txt', '_feats.txt'):
        src = f'{W}/own_models/ret_ranker_{t}{suf}'
        if os.path.exists(src):
            shutil.copy(src, OUT)
            out_md5[f'ret_ranker_{t}{suf}'] = digest(src)
        else:
            problems.append(f'missing ret_ranker_{t}{suf}')
    fp = f'{W}/own_models/ret_ranker_{t}_feats.txt'
    if os.path.exists(fp):
        n = len(open(fp).read().split())
        ok = out_md5[f'ret_ranker_{t}_feats.txt'] == FEATS_MD5
        say(f'{t}: {n} features; feature list', 'MATCHES' if ok else 'DIFFERS FROM', 'the v18c-family list (md5', FEATS_MD5[:8] + ')')
        if not ok:
            problems.append(f'{t} feature list differs from the v18c family')
rows = {}
row_files = [f'{W}/ret18cx_{s}_{k}.parquet' for s in ('train', 'val') for k in ('pf', 'fp')] + \
            [f'{W}/ret_{t}_train_oof.parquet' for t in TAGS] + [f'{W}/ret18cx_val_reach_c.parquet', f'{W}/ret18cxr1_val_reach_c.parquet']
for p in row_files:
    if os.path.exists(p):
        out_md5['work/' + os.path.basename(p)] = digest(p)
dg = subprocess.run(['python3', '-c', '''
import json, sys, pandas as pd
out = {}
for p in sys.argv[1:]:
    D = pd.read_parquet(p).sort_values(['qid', 'pid'])
    cols = [c for c in ('pf_cos', 'pf_bern', 'pf_cos_fz', 'pf_bern_fz', 'fp_cos', 'fp_bern', 's1_oof') if c in D]
    out[p.rsplit('/', 1)[1]] = dict(rows=len(D), queries=int(D.qid.nunique()), columns=len(D.columns),
                                   sums={c: float(D[c].astype('float64').sum()) for c in cols})
print(json.dumps(out))'''] + [p for p in row_files if os.path.exists(p)], env=env, capture_output=True, text=True)
try:
    rows = json.loads(dg.stdout)
except ValueError:
    rows = {'error': dg.stderr[-2000:]}
json.dump(rows, open(f'{OUT}/row_digest.json', 'w'), indent=1)
if KEEP_ROWS:
    for p in [f'{W}/ret_{t}_train_oof.parquet' for t in TAGS] + [f'{W}/ret18cx_val_reach_c.parquet']:
        if os.path.exists(p):
            shutil.copy(p, OUT)
with open(f'{OUT}/md5.txt', 'w') as fh:
    fh.write('# outputs and intermediate rows (work/ = not kept; parquet bytes depend on the pyarrow version)\n')
    fh.writelines(f'{h}  {f}\n' for f, h in out_md5.items())
    fh.write('# inputs (checked: v18cx_inputs.md5, pool / PF-x / casmi expected values)\n')
    fh.writelines(f'{h}  {f}\n' for f, h in md5.items())
    for t in TAGS:
        if os.path.exists(f'{W}/{t}.md5'):
            fh.write(f'# {W}/{t}.md5 (written by build_v18c.sh stage A)\n' + open(f'{W}/{t}.md5').read())
mem = next((l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal')), '?')
json.dump(dict(versions=v, cpus=os.cpu_count(), mem_kb=mem, seconds=round(time.time() - T0), steps=times, hu=HU,
               relocated=HU != '/home/user', test=TEST,
               keep_rows=KEEP_ROWS, inputs=dict(code=code, inputs=inp, pool=pool, pf=pf), problems=problems),
          open(f'{OUT}/env.json', 'w'), indent=1)
print(open(f'{OUT}/stage1_metrics.txt').read())
print(open(f'{OUT}/md5.txt').read())
say('outputs', sorted(os.listdir(OUT)))
if problems:
    raise SystemExit('FAILED: ' + '; '.join(problems))
say('V18CX_BUILD_DONE')
