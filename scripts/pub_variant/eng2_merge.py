"""Engine 2 = our stage 1 in the eligible fork elig1 (fix F2b of docs/fork_eligibility.md). Ours, MIT.

Written next to the pipeline modules by notebook cell 8 (SHA-256 pinned) and imported by the patched frozen cells:
  cell 32  run_engine2(): casmi.run_eng2 (our stage 1, from bryantyu/casmi26-elig-assets, on bryantyu/casmi26-pool-v18c)
           in its own process, where the original Engine 2 ran: before the PubChem channel and Engine 1, CPU only, in its
           own process group, with a time cap. Returns ({mid: {smiles, keys}}, report) and never raises.
  cell 39  verify_keys(): every key recomputed with the pipeline's own chem.score_key (exact identity for the ICE union
           and the fusions); merge(): ENG = our lists, and elig0's null-engine step-1 lists (Engine 1's top 40) for the
           molecules ours does not cover; read by the ICE / GLACIER input union of cell 41.
  cell 45  merge() with elig0's step-2 null lists (main list + Engine 1 tail) as the fallback; written to
           eng_lists.json. From here on every reader of Engine 2 (cells 45-53, export_evidence, outsiders) sees it.
A molecule without a list of ours (no usable spectra, no pool candidate, error, timeout, failed or missing run) gets
exactly elig0's content, so the coverage asserts of frozen cells 48 and 50 hold whatever happens in the subprocess.
Pure functions apart from run_engine2 / check_files (disk, subprocess) and write_json.
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ASSETS, POOL = 'casmi26-elig-assets', 'casmi26-pool-v18c'
DEFAULT_CFG = dict(cap_s=2700.0, grace_s=300.0, stall_s=900.0, procs=0, k=40, margin=8,
                   ranker='ret_ranker_v18cu,ret_ranker_v18cur1', fp='fp_model_v1r.pt', pf='pf_tpu_f.pt', lib_cap=10,
                   work='/tmp/eng2_ours', omp_threads=None, pool_sha256={})
# environment variables our stage-1 code reads (casmi/own.py, retrieval.py, fpscore.py): never inherited
_DROP = ('OWN_', 'PC_', 'FP_PMI', 'DREAMS_DIR', 'ZEN_', 'VENDOR_PATH', 'LIB_CAP', 'PYTHONPATH', 'PYTHONHOME',
         'PYTHONSTARTUP', 'PYTHONSAFEPATH')


def _dataset_dir(input_root, slug):
    root = Path(input_root)
    for c in [root / slug] + sorted(root.glob(f'datasets/*/{slug}')):
        if c.is_dir():
            return c
    return None


def _first(root, pattern):
    hits = sorted(root.rglob(pattern), key=lambda p: (len(str(p)), str(p)))
    return hits[0] if hits else None


def find_assets(input_root='/kaggle/input', ranker='ret_ranker_v18cu'):
    """Our two datasets: the code dir (the dir holding run_eng2.py and the other casmi modules; Kaggle mounts uploads
    flat, so it is usually the dataset root), the models dir and the pool dir, whatever the nesting."""
    a, p = _dataset_dir(input_root, ASSETS), _dataset_dir(input_root, POOL)
    if a is None or p is None:
        raise FileNotFoundError(f'not mounted: {[s for s, d in ((ASSETS, a), (POOL, p)) if d is None]}')
    entry = _first(a, 'run_eng2.py')
    model = _first(a, f"{ranker.split(',')[0]}.txt")
    pool = _first(p, 'pool.parquet')
    if entry is None or model is None or pool is None:
        raise FileNotFoundError(f'incomplete: run_eng2.py {entry}, ranker {model}, pool.parquet {pool}')
    manifest = _first(a, 'MANIFEST.sha256')
    return dict(assets=a, code=entry.parent, models=model.parent, pool=pool.parent, manifest=manifest)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_files(assets, pool_dir, pool_sha256):
    """sha256 of every file in the assets MANIFEST (sha256sum format) and of the pinned pool files; log only."""
    rep = dict(manifest_files=0, manifest_match=0, mismatches=[], missing=[], pool_match=0, pool_files=len(pool_sha256))
    if assets.get('manifest') is not None:
        for line in Path(assets['manifest']).read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            want, rel = line.split(None, 1)
            rel = rel.strip().lstrip('*')
            f = Path(assets['manifest']).parent / rel
            if not f.is_file():                   # nested differently on Kaggle: the unique file of that name
                hits = list(Path(assets['assets']).rglob(Path(rel).name))
                f = hits[0] if len(hits) == 1 else f
            rep['manifest_files'] += 1
            if not f.is_file():
                rep['missing'].append(rel)
            elif _sha256(f) == want:
                rep['manifest_match'] += 1
            else:
                rep['mismatches'].append(rel)
    for name, want in sorted(pool_sha256.items()):
        f = Path(pool_dir) / name
        if not f.is_file():
            rep['missing'].append(f'pool/{name}')
        elif _sha256(f) == want:
            rep['pool_match'] += 1
        else:
            rep['mismatches'].append(f'pool/{name}')
    rep['ok'] = not rep['mismatches'] and not rep['missing'] and rep['manifest_files'] > 0
    return rep


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def clean_env(base, pkg_root, omp_threads=None):
    env = {k: v for k, v in base.items() if not k.startswith(_DROP)}
    env.update(PYTHONPATH=str(pkg_root), PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', NUMBA_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='')
    if omp_threads:
        env['OMP_NUM_THREADS'] = str(int(omp_threads))
    return env


def read_output(path, k=40):
    """casmi.run_eng2's JSON -> (lists {mid: {smiles, keys}}, status, meta, n_invalid); invalid entries are dropped."""
    d = json.loads(Path(path).read_text())
    lists, bad = {}, 0
    for m, v in (d.get('lists') or {}).items():
        sm, ks = (v or {}).get('smiles'), (v or {}).get('keys')
        ok = (isinstance(sm, list) and isinstance(ks, list) and 0 < len(sm) == len(ks) <= k
              and all(isinstance(s, str) and s and s != 'CCO' and ';' not in s for s in sm)
              and all(isinstance(x, str) and x for x in ks) and len(set(ks)) == len(ks))
        if ok:
            lists[str(m)] = dict(smiles=list(sm), keys=list(ks))
        else:
            bad += 1
    return lists, dict(d.get('status') or {}), dict(d.get('meta') or {}), bad


def run_engine2(test, train, out, cfg=None, input_root='/kaggle/input', python=None, log=None):
    """Our stage 1 as Engine 2 (subprocess). Returns (lists, report); any failure gives ({}, report with the error)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    t0 = time.time()
    out = str(out)
    log = str(log or out.replace('.json', '') + '.log')
    rep = dict(status='failed', rc=None, seconds=0.0, molecules=0, ok=0, counts={}, invalid=0, cfg=cfg, out=out, log=log)
    lists = {}
    proc = None
    try:
        Path(out).unlink(missing_ok=True)
        assets = find_assets(input_root, cfg['ranker'])
        rep['assets'] = {k: str(v) for k, v in assets.items()}
        rep['files'] = check_files(assets, assets['pool'], cfg.get('pool_sha256') or {})
        work = Path(cfg['work'])
        pkg = work / 'pkg'
        if pkg.exists():
            shutil.rmtree(pkg)
        (pkg / 'casmi').mkdir(parents=True)      # the code dir's modules as package casmi (mounts are read-only: numba
        for f in sorted(Path(assets['code']).glob('*.py')):          # caches beside the modules; no models copied)
            shutil.copy2(f, pkg / 'casmi' / f.name)
        (pkg / 'casmi' / '__init__.py').touch()
        procs = int(cfg['procs']) or max(1, min(4, os.cpu_count() or 1))
        rep['procs'] = procs
        cmd = [python or sys.executable, '-m', 'casmi.run_eng2', str(test), str(train), str(assets['pool']),
               str(assets['models']), out, '--work', str(work / 'work'), '--procs', str(procs),
               '--cap', str(float(cfg['cap_s'])), '--stall', str(float(cfg['stall_s'])), '--k', str(int(cfg['k'])),
               '--margin', str(int(cfg['margin'])), '--ranker', cfg['ranker'], '--fp', cfg['fp'], '--pf', cfg['pf'],
               '--lib-cap', str(int(cfg['lib_cap']))] + [str(x) for x in cfg.get('extra_args', ())]
        with open(log, 'w') as fh:
            proc = subprocess.Popen(cmd, cwd=pkg, env=clean_env(os.environ, pkg, cfg.get('omp_threads')),
                                    stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rep['rc'] = proc.wait(timeout=float(cfg['cap_s']) + float(cfg['grace_s']))
            except subprocess.TimeoutExpired:          # its own cap failed: kill the group, keep the partial lists
                rep['rc'] = 'killed (cap + grace)'
                _kill_group(proc)
        if Path(out).is_file():
            lists, status, meta, rep['invalid'] = read_output(out, int(cfg['k']))
            pkg_used = str(meta.get('package', ''))
            if not pkg_used.startswith(str(pkg)):
                raise RuntimeError(f'run_eng2 imported casmi from {pkg_used!r}, not from {pkg}')
            counts = {}
            for st in status.values():
                counts[st.split(':')[0]] = counts.get(st.split(':')[0], 0) + 1
            rep.update(status=str(meta.get('status', 'unknown')), molecules=len(status), ok=len(lists), counts=counts,
                       meta={k: meta.get(k) for k in ('seconds_libs', 'seconds_load', 'seconds_stage1', 'seconds',
                                                      'seconds_per_molecule', 'max_rss_gb_main',
                                                      'max_rss_gb_largest_worker', 'versions', 'model_sha256',
                                                      'pool_rows', 'error', 'entries')},
                       errors=[f'{m}: {s}' for m, s in status.items() if s.startswith('error')][:10])
        else:
            rep['status'] = 'no output'
    except Exception as e:                        # Engine 2 never stops the pipeline: the null engine fills in
        lists = {}
        rep['error'] = f'{type(e).__name__}: {e}'[:500]
    finally:
        if proc is not None:
            _kill_group(proc)                    # pool workers left behind by a killed or crashed run
        shutil.rmtree(Path(cfg['work']) / 'work', ignore_errors=True)          # the ~1.2 GB libraries
    rep['seconds'] = round(time.time() - t0, 1)
    if rep.get('status') != 'done' or rep.get('error'):
        try:
            rep['log_tail'] = Path(log).read_text()[-1500:]
        except Exception:
            pass
    return lists, rep


def verify_keys(lists, key_of, head=None):
    """Recompute every key with key_of (the pipeline's chem.score_key): a different key is replaced, an entry whose key is
    falsy or already listed is dropped, a molecule left empty is removed (it falls back to the null engine).
    head = n: check only the first n entries of each list (the subprocess computed the keys with the same RDKit calls;
    the rerun saves ~20 ms per key); any mismatch among them -> every entry is checked as above."""
    if head:
        pairs = [(s, k) for v in lists.values() for s, k in list(zip(v['smiles'], v['keys']))[:int(head)]]
        if all(key_of(s) == k for s, k in pairs):
            return ({m: dict(smiles=list(v['smiles']), keys=list(v['keys'])) for m, v in lists.items()},
                    dict(entries=sum(len(v['smiles']) for v in lists.values()), checked=len(pairs), mismatches=0,
                         dropped=0, emptied=0, examples=[], mode=f'first {int(head)} per molecule'))
    out, rep = {}, dict(entries=0, checked=0, mismatches=0, dropped=0, emptied=0, examples=[], mode='all')
    for m, v in lists.items():
        sm, ks, seen = [], [], set()
        for s, k in zip(v['smiles'], v['keys']):
            rep['entries'] += 1
            rep['checked'] += 1
            k2 = key_of(s)
            if k2 != k:
                rep['mismatches'] += 1
                if len(rep['examples']) < 5:
                    rep['examples'].append(dict(molecule_id=m, smiles=s, key=k, score_key=k2))
            if not k2 or k2 in seen:
                rep['dropped'] += 1
                continue
            seen.add(k2)
            sm.append(s)
            ks.append(k2)
        if sm:
            out[m] = dict(smiles=sm, keys=ks)
        else:
            rep['emptied'] += 1
    return out, rep


def merge(mids, ours, null):
    """ENG for every molecule id (str, first occurrence order): our list where it is non-empty, else the null engine's."""
    eng, n_ours, fallback = {}, 0, []
    for mid in mids:
        m = str(mid)
        if m in eng:
            continue
        o = ours.get(m) if ours else None
        if o and o.get('smiles'):
            eng[m] = dict(smiles=list(o['smiles']), keys=list(o['keys']))
            n_ours += 1
        else:
            nl = (null or {}).get(m) or dict(smiles=[], keys=[])
            eng[m] = dict(smiles=list(nl['smiles']), keys=list(nl['keys']))
            fallback.append(m)
    return eng, dict(molecules=len(eng), ours=n_ours, fallback=len(fallback), fallback_ids=fallback[:20])


def fallback_ids(mids, ours):
    """Molecule ids (str, first occurrence order) that need the null engine."""
    seen, out = set(), []
    for mid in mids:
        m = str(mid)
        if m not in seen:
            seen.add(m)
            if not ((ours or {}).get(m) or {}).get('smiles'):
                out.append(m)
    return out


def report_line(rep):
    """Cell 32: one line on the subprocess."""
    c, meta, f = rep.get('counts') or {}, rep.get('meta') or {}, rep.get('files') or {}
    files = (f"assets MANIFEST {f.get('manifest_match')}/{f.get('manifest_files')}, pool {f.get('pool_match')}/"
             f"{f.get('pool_files')}{' MISMATCH ' + str(f.get('mismatches') + f.get('missing')) if f and not f.get('ok') else ''}"
             if f else 'files not checked')
    s = (f"ENGINE 2 OURS subprocess | status {rep.get('status')} | rc {rep.get('rc')} | molecules {rep.get('molecules')} | "
         f"ok {rep.get('ok')} | no_query {c.get('no_query', 0)} | no_keys {c.get('no_keys', 0)} | error {c.get('error', 0)} | "
         f"timeout {c.get('timeout', 0) + c.get('not_run', 0)} | invalid {rep.get('invalid', 0)} | libraries "
         f"{meta.get('seconds_libs')} s | stage 1 {meta.get('seconds_stage1')} s | {rep.get('seconds')} s | procs "
         f"{rep.get('procs')} | max RSS main {meta.get('max_rss_gb_main')} GB, worker {meta.get('max_rss_gb_largest_worker')} GB"
         f" | {files}")
    if rep.get('error'):
        s += f" | FAILED {rep['error'][:300]}"
    return s


def step_line(step, stats, rep, tail=''):
    """Cells 39 / 45: 'ENGINE 2 OURS | molecules N | ours N | fallback N | S s | <step> | <tail>'."""
    return (f"ENGINE 2 OURS | molecules {stats['molecules']} | ours {stats['ours']} | fallback {stats['fallback']} | "
            f"{(rep or {}).get('seconds', 0)} s | {step}" + (f' | {tail}' if tail else ''))


def write_json(obj, path):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(obj, handle, indent=1, default=str)
    os.replace(tmp, path)
    return path
