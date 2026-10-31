"""Engine 2 of the eligible public-lineage fork (elig1, fix F2b of docs/fork_eligibility.md): our stage 1 in its own
process -> per-molecule top-k lists in the fork's Engine 2 schema. Ours (MIT).

python -m casmi.run_eng2 <test.parquet> <train.parquet> <pool_dir> <model_dir> <out.json>
        [--work DIR] [--procs N] [--cap S] [--stall S] [--k 40] [--margin 8] [--ranker a,b] [--fp NAME] [--pf NAME]
        [--lib-cap 10] [--libs DIR]

Stage 1 exactly as kernels/casmi26-own-s2v18cue runs it inside casmi.run_own.main (OWN_STAGE2 set): analog library
(build_analog_lib per_struct 3) + direct-match library (LIB_CAP 10) built from train.parquet, Retrieval on pool v18c
(bryantyu/casmi26-pool-v18c: pool.parquet, pool_fp.npy, coconut_prior.parquet), FPScorer fp_model_v1r.pt, PFEnsemble
pf_tpu_f.pt, LightGBM seed ensemble ret_ranker_v18cu + ret_ranker_v18cur1 (casmi.run_own._ranker, mean score),
casmi.own.stage1. Differences: k = 40 (+ margin, see below) instead of 30, no stage 2 (the fork re-ranks the fused
lists with its own ICEBERG / GLACIER), LightGBM predicts with one thread per worker (same scores).

Output (written atomically, also every 25 molecules while running, so a killed run leaves its partial lists):
    {"meta": {...}, "lists": {molecule_id: {"smiles": [...], "keys": [...], "s1": [...]}},
     "status": {molecule_id: "ok" | "no_query" | "no_keys" | "error: ..." | "timeout" | "not_run"}}
Each list is the stage-1 order (seed-ensemble score s1, descending), at most k entries, one entry per key.

Keys: the frozen fork pipeline's chem.score_key, i.e. RDKit MolFromSmiles -> TautomerEnumerator().Canonicalize ->
MolToInchiKey[:14] WITHOUT removing stereo first (Engine 2's canon_key was the same function). Our casmi.metric.key14
(the competition metric) removes stereo before the tautomer step; the two can differ when stereo changes the canonical
tautomer. stage1 de-duplicates by key14; here the list is re-keyed by score_key, entries whose score_key is None or
already listed are dropped, and stage1 is asked for k + margin candidates so that the list still holds k entries.

Time cap: --cap seconds from the start (library build and model loading included); molecules not finished by then are
'timeout' (the caller fills them with the null engine). --stall: stop when no molecule finished for that long.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np
import pandas as pd

R = FPS = PFS = RK = FEATS = GROUPS = None
K = 40
MARGIN = 8
_TE = None


def score_key(smiles):
    """chem.score_key of the frozen pipeline: tautomer-canonical InChIKey14, stereo kept; None if RDKit fails."""
    global _TE
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    if _TE is None:
        RDLogger.DisableLog('rdApp.*')
        _TE = rdMolStandardize.TautomerEnumerator()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        ik = Chem.MolToInchiKey(_TE.Canonicalize(mol))
        return ik[:14] if ik else None
    except Exception:
        return None


def engine2_list(frame, n):
    """stage-1 frame (sorted by s1, one row per key14) -> (smiles, keys, s1) with unique score_key, at most n."""
    smiles, keys, s1, seen = [], [], [], set()
    for s, v in zip(frame.smiles.values, frame.s1.values):
        if not isinstance(s, str) or not s or s == 'CCO':
            continue
        k = score_key(s)
        if not k or k in seen:
            continue
        seen.add(k)
        smiles.append(s); keys.append(k); s1.append(float(v))
        if len(smiles) >= n:
            break
    return smiles, keys, s1


class _OneThread:
    """LightGBM predictions with one thread per worker process (identical scores; no oversubscription)."""

    def __init__(self, booster):
        self.b = booster

    def predict(self, X, **kw):
        kw.setdefault('num_threads', 1)
        return self.b.predict(X, **kw)


def _one(mid):
    from .own import stage1
    t = time.time()
    try:
        f = stage1(R, FPS, RK, FEATS, GROUPS[mid], k=K + MARGIN, pfs=PFS)
        if f is None or len(f) == 0:
            return mid, None, 'no_query', time.time() - t
        smiles, keys, s1 = engine2_list(f, K)
        if not smiles:
            return mid, None, 'no_keys', time.time() - t
        return mid, dict(smiles=smiles, keys=keys, s1=s1), 'ok', time.time() - t
    except Exception as e:                       # one molecule never stops the others
        return mid, None, f'error: {e!r}'[:300], time.time() - t


def _sha(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()[:n]


def _write(out, meta, lists, status):
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(dict(meta=meta, lists=lists, status=status), f)
    os.replace(tmp, out)


def _rss_gb():
    import resource
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(self_kb / 1e6, 2), round(child_kb / 1e6, 2)


def main(argv=None):
    global R, FPS, PFS, RK, FEATS, GROUPS, K, MARGIN
    ap = argparse.ArgumentParser(prog='python -m casmi.run_eng2')
    ap.add_argument('test'); ap.add_argument('train'); ap.add_argument('pool'); ap.add_argument('models'); ap.add_argument('out')
    ap.add_argument('--work', default='/kaggle/working/eng2_ours/work')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--cap', type=float, default=2700.0)
    ap.add_argument('--stall', type=float, default=900.0)
    ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--margin', type=int, default=8)
    ap.add_argument('--ranker', default='ret_ranker_v18cu,ret_ranker_v18cur1')
    ap.add_argument('--fp', default='fp_model_v1r.pt')
    ap.add_argument('--pf', default='pf_tpu_f.pt')
    ap.add_argument('--lib-cap', type=int, default=10)
    ap.add_argument('--libs', default='', help='directory with prebuilt analog_lib.npz / lib<cap>.npz (tests only)')
    ap.add_argument('--limit', type=int, default=0, help='first N molecules only (tests only)')
    a = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + a.cap
    K, MARGIN = int(a.k), int(a.margin)
    os.makedirs(a.work, exist_ok=True)
    import multiprocessing as mp
    import rdkit
    import torch
    import lightgbm as lgb
    from . import own, run_own
    meta = dict(engine='casmi.run_eng2 (our stage 1, elig1 F2b)', k=K, margin=MARGIN, procs=a.procs, cap_s=a.cap,
                stall_s=a.stall, ranker=a.ranker, fp=a.fp, pf=a.pf, lib_cap=a.lib_cap, pool=a.pool, models=a.models,
                test=a.test, key='chem.score_key semantics (tautomer-canonical InChIKey14, stereo kept)',
                versions=dict(python=sys.version.split()[0], rdkit=rdkit.__version__, torch=torch.__version__,
                              lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__),
                package=os.path.dirname(os.path.abspath(own.__file__)), status='starting')
    test = pd.read_parquet(a.test)
    test['molecule_id'] = test.molecule_id.astype(str)
    mids = list(dict.fromkeys(test.molecule_id))
    if a.limit:
        mids = mids[:a.limit]
        test = test[test.molecule_id.isin(mids)]
    status = {m: 'not_run' for m in mids}
    lists = {}
    meta['molecules'] = len(mids)
    _write(a.out, meta, lists, status)
    print(f'eng2 ours: {len(mids)} molecules, procs {a.procs}, cap {a.cap:.0f} s, k {K} (+{MARGIN}) | '
          f"rdkit {rdkit.__version__} torch {torch.__version__} lightgbm {lgb.__version__} | package {meta['package']}",
          flush=True)
    try:
        from .analog import AnalogLib
        from .build import build_analog_lib
        from .fpscore import FPScorer
        from .peakformer import PFEnsemble
        from .retrieval import Retrieval
        if a.libs:
            alib, flib = f'{a.libs}/analog_lib.npz', f'{a.libs}/lib{a.lib_cap}.npz'
        else:
            alib, flib = f'{a.work}/analog_lib.npz', f'{a.work}/lib{a.lib_cap}.npz'
            build_analog_lib(a.train, alib, per_struct=3)
            build_analog_lib(a.train, flib, per_struct=a.lib_cap)
        meta['seconds_libs'] = round(time.time() - t0, 1)
        print(f"eng2 ours: libraries {meta['seconds_libs']} s", flush=True)
        torch.set_num_threads(1)
        R = Retrieval(a.pool, AnalogLib(alib), f'{a.pool}/coconut_prior.parquet', lib_full=AnalogLib(flib))
        FPS = FPScorer(f'{a.models}/{a.fp}', np.load(f'{a.pool}/pool_fp.npy', mmap_mode='r'))
        rk, FEATS = run_own._ranker(a.models, a.ranker)
        RK = _OneThread(rk) if not isinstance(rk, run_own._MeanBooster) else \
            run_own._MeanBooster([_OneThread(b) for b in rk.m])
        assert 'pf_cos' in FEATS, 'the stage-1 ranker has no PeakFormer features'
        PFS = PFEnsemble([f'{a.models}/{n}' for n in a.pf.split(',')])
        GROUPS = {m: g for m, g in test.groupby('molecule_id', sort=False)}
        meta['model_sha256'] = {n: _sha(f'{a.models}/{n}') for n in
                                [a.fp] + a.pf.split(',') + [f'{r}.txt' for r in a.ranker.split(',')]}
        meta['pool_rows'] = int(len(R.P))
        meta['features'] = len(FEATS)
        meta['seconds_load'] = round(time.time() - t0, 1)
        print(f"eng2 ours: models loaded {meta['seconds_load']} s | pool {len(R.P):,} | features {len(FEATS)} | "
              f"ranker members {len(getattr(rk, 'm', [rk]))} | PeakFormer members {len(PFS.ms)}", flush=True)
    except Exception as e:
        meta.update(status='failed', error=f'{e!r}'[:500], seconds=round(time.time() - t0, 1))
        _write(a.out, meta, lists, status)
        print('eng2 ours FAILED before stage 1:', meta['error'], flush=True)
        return 0
    t1, last, last_write, n_done, stop = time.time(), time.time(), time.time(), 0, ''
    secs = []
    ctx = mp.get_context('fork')
    pool = ctx.Pool(max(1, a.procs))
    try:
        it = pool.imap_unordered(_one, mids, chunksize=1)
        while n_done < len(mids):
            now = time.time()
            wait = min(deadline - now, a.stall - (now - last))
            if wait <= 0:
                stop = 'cap' if deadline - now <= 0 else 'stall'
                break
            try:
                mid, lst, st, dt = it.next(timeout=wait)
            except mp.TimeoutError:
                stop = 'cap' if time.time() >= deadline else 'stall'
                break
            n_done += 1; last = time.time(); secs.append(dt)
            status[mid] = st
            if lst is not None:
                lists[mid] = lst
            if n_done % 25 == 0 or time.time() - last_write > 60:
                meta.update(status='running', done=n_done, seconds=round(time.time() - t0, 1))
                _write(a.out, meta, lists, status); last_write = time.time()
                print(f'eng2 ours: {n_done}/{len(mids)} molecules {time.time() - t0:.0f} s', flush=True)
    finally:
        pool.terminate()
        pool.join()
    for m in mids:
        if status[m] == 'not_run':
            status[m] = 'timeout'
    counts = {}
    for st in status.values():
        key = st.split(':')[0]
        counts[key] = counts.get(key, 0) + 1
    rss = _rss_gb()
    meta.update(status='done' if not stop else f'stopped ({stop})', done=n_done, counts=counts,
                ok=sum(1 for m in mids if m in lists), seconds_stage1=round(time.time() - t1, 1),
                seconds=round(time.time() - t0, 1), seconds_per_molecule=round(float(np.mean(secs)), 2) if secs else None,
                max_rss_gb_main=rss[0], max_rss_gb_largest_worker=rss[1],
                entries=sum(len(v['smiles']) for v in lists.values()))
    _write(a.out, meta, {m: lists[m] for m in mids if m in lists}, status)
    print(f"eng2 ours: {meta['status']} | molecules {len(mids)} | ok {meta['ok']} | {counts} | stage 1 "
          f"{meta['seconds_stage1']} s | total {meta['seconds']} s | max RSS main {rss[0]} GB, largest worker {rss[1]} GB",
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
