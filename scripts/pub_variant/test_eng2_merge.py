"""Unit tests of scripts/pub_variant/eng2_merge.py (ours; no public code is run).

python scripts/pub_variant/test_eng2_merge.py      (plain asserts; prints 'eng2_merge tests OK' and the count)

Covers the merge of our Engine 2 lists with elig0's null engine (scripts/pub_variant/null_engine.py) in both steps,
the key re-check, the output validation, the coverage asserts of frozen cells 48 / 50 (re-stated here), and
run_engine2's subprocess paths with stub entry points: normal run, cap + grace kill (partial lists kept, the process
group killed), crash, missing datasets, a package imported from the wrong place, and the assets / pool checks.
"""
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


em = _load('eng2_merge')
ne = _load('null_engine')
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def key_of(s):          # stand-in for chem.score_key: 'X~t' is a tautomer of 'X'; 'bad*' does not parse
    return None if s.startswith('bad') else s.split('~')[0].upper()


def _lists(n_mol=30, seed=0):
    rnd = random.Random(seed)
    mids = [f'm_{i:03d}' for i in range(n_mol)]
    base = {m: [f'b{m}{j}' for j in range(rnd.randint(1, 60))] for m in mids}
    ours = {}
    for m in mids:
        r = rnd.random()
        if r < 0.2:
            continue                                  # not covered (no usable spectra / timeout)
        if r < 0.25:
            ours[m] = dict(smiles=[], keys=[])        # covered but empty -> fallback too
            continue
        sm = [f'o{m}{j}' for j in range(rnd.randint(1, 40))]
        ours[m] = dict(smiles=sm, keys=[key_of(s) for s in sm])
    return mids, base, ours


@test
def merge_takes_ours_else_null():
    mids, base, ours = _lists()
    null = ne.null_engine_lists(mids, base, keys={m: [key_of(s) for s in v] for m, v in base.items()}, n=40)
    eng, st = em.merge(mids, ours, null)
    assert list(eng) == mids and st['molecules'] == len(mids)
    for m in mids:
        if ours.get(m, {}).get('smiles'):
            assert eng[m] == dict(smiles=ours[m]['smiles'], keys=ours[m]['keys'])
        else:
            assert eng[m] == null[m] and m in em.fallback_ids(mids, ours)
    assert st['ours'] + st['fallback'] == len(mids) and st['ours'] == sum(bool(v.get('smiles')) for v in ours.values())
    assert set(em.fallback_ids(mids, ours)) == {m for m in mids if not ours.get(m, {}).get('smiles')}
    eng[mids[0]]['smiles'].append('x')                # merge copies: callers may not alias ours / null
    assert 'x' not in (ours.get(mids[0]) or {}).get('smiles', []) and 'x' not in null[mids[0]]['smiles']


@test
def merge_ids_and_missing_null():
    eng, st = em.merge([1, '1', 2, 'm3'], {'1': dict(smiles=['A'], keys=['A'])}, {})
    assert list(eng) == ['1', '2', 'm3'] and eng['2'] == dict(smiles=[], keys=[]) and st['fallback'] == 2
    eng, st = em.merge(['a'], None, None)
    assert eng == {'a': dict(smiles=[], keys=[])} and st['ours'] == 0


@test
def both_steps_cover_every_molecule_like_elig0():
    """Step 1 (cell 39) and step 2 (cell 45) as patched: the null lists are built for the fallback molecules only and
    equal elig0's lists for them; the asserts of frozen cells 48 (coverage) and 50 (smoke: every list non-empty) hold."""
    for seed in range(50):
        mids, base, ours = _lists(seed=seed)
        fb = em.fallback_ids(mids, ours)
        base_keys = {m: [key_of(s) for s in v] for m, v in base.items()}
        null1 = ne.null_engine_lists(fb, base, keys=base_keys, n=40)
        eng1, st1 = em.merge(mids, ours, null1)
        full1 = ne.null_engine_lists(mids, base, keys=base_keys, n=40)              # elig0 step 1
        prefusion = {m: (['pc' + m] + v[:24]) for m, v in base.items()}
        null2 = ne.null_engine_lists(fb, prefusion, tails=base, key_of=lambda s: key_of(s) or s, n=40)
        eng2, st2 = em.merge(mids, ours, null2)
        full2 = ne.null_engine_lists(mids, prefusion, tails=base, key_of=lambda s: key_of(s) or s, n=40)  # elig0 step 2
        for eng, full in ((eng1, full1), (eng2, full2)):
            assert set(mids) <= set(map(str, eng))                                  # cell 48
            assert all(eng[m]['smiles'] for m in mids)                              # cell 50 (smoke; BASE non-empty)
            assert all(len(eng[m]['smiles']) == len(eng[m]['keys']) <= 40 for m in mids)
            for m in fb:
                assert eng[m] == full[m]                                            # exactly elig0's content
        assert st1 == st2


@test
def verify_keys_replaces_and_drops():
    lists = {'a': dict(smiles=['X', 'Y~t', 'Z', 'bad1', 'X~2'], keys=['X', 'WRONG', 'Z', 'BAD', 'X2']),
             'b': dict(smiles=['bad2'], keys=['K'])}
    out, rep = em.verify_keys(lists, key_of)
    assert out == {'a': dict(smiles=['X', 'Y~t', 'Z'], keys=['X', 'Y', 'Z'])}, out
    assert rep['entries'] == 6 and rep['mismatches'] == 4 and rep['dropped'] == 3 and rep['emptied'] == 1
    same, rep2 = em.verify_keys(out, key_of)
    assert same == out and rep2['mismatches'] == 0 and rep2['dropped'] == 0 and rep2['mode'] == 'all' and rep2['checked'] == 3


@test
def verify_keys_head_sample():
    good = {'a': dict(smiles=['X', 'Y~t', 'Z'], keys=['X', 'Y', 'Z']), 'b': dict(smiles=['Q', 'bad9'], keys=['Q', 'W'])}
    calls = []
    counted = lambda s: (calls.append(s), key_of(s))[1]
    out, rep = em.verify_keys(good, counted, head=1)          # first entry of each list only: no mismatch -> as is
    assert out == good and out['a'] is not good['a'] and rep['mode'] == 'first 1 per molecule' and rep['checked'] == 2
    assert calls == ['X', 'Q'] and rep['entries'] == 5 and rep['dropped'] == 0
    out, rep = em.verify_keys(good, key_of, head=2)           # 'bad9' (2nd of b) mismatches -> full check, dropped
    assert rep['mode'] == 'all' and rep['mismatches'] == 1 and out['b'] == dict(smiles=['Q'], keys=['Q'])


@test
def read_output_validates():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'o.json')
        good = dict(smiles=['A', 'B'], keys=['KA', 'KB'], s1=[2.0, 1.0])
        bad = {'len': dict(smiles=['A'], keys=[]), 'cco': dict(smiles=['CCO'], keys=['K']),
               'dup': dict(smiles=['A', 'B'], keys=['K', 'K']), 'empty': dict(smiles=[], keys=[]),
               'long': dict(smiles=[f'S{i}' for i in range(41)], keys=[f'K{i}' for i in range(41)]),
               'semi': dict(smiles=['A;B'], keys=['K']), 'nokey': dict(smiles=['A'], keys=['']), 'none': None}
        json.dump(dict(meta=dict(status='done'), lists=dict(good=good, **bad), status={'good': 'ok', 'x': 'timeout'}),
                  open(p, 'w'))
        lists, status, meta, n_bad = em.read_output(p, 40)
        assert lists == {'good': dict(smiles=['A', 'B'], keys=['KA', 'KB'])} and n_bad == len(bad)
        assert status == {'good': 'ok', 'x': 'timeout'} and meta['status'] == 'done'


@test
def clean_env_drops_our_switches():
    base = dict(PATH='/bin', OWN_PF_INST='1', OWN_SPEC_ONLY='pf_bern', VENDOR_PATH='/v', FP_PMI='/p', PYTHONPATH='/x',
                LIB_CAP='100000', CASMI_SRC_BITS='0', NUMBA_CACHE_DIR='/n', CUDA_VISIBLE_DEVICES='0')
    env = em.clean_env(base, '/pkg', omp_threads=3)
    assert env['PATH'] == '/bin' and env['PYTHONPATH'] == '/pkg' and env['NUMBA_CACHE_DIR'] == '/n'
    assert not any(k in env for k in ('OWN_PF_INST', 'OWN_SPEC_ONLY', 'VENDOR_PATH', 'FP_PMI', 'LIB_CAP'))
    assert env['CUDA_VISIBLE_DEVICES'] == '' and env['NUMBA_NUM_THREADS'] == '1' and env['OMP_NUM_THREADS'] == '3'
    assert env['PYTHONHASHSEED'] == '0' and env['CASMI_SRC_BITS'] == '0'
    assert 'OMP_NUM_THREADS' not in em.clean_env({}, '/pkg')


@test
def lines_have_the_logged_format():
    st = dict(molecules=12, ours=11, fallback=1, fallback_ids=['m_1'])
    line = em.step_line('step 2 (fused below)', st, dict(seconds=301.4), ne.summary({'a': dict(smiles=['A'], keys=['A'])}))
    assert line.startswith('ENGINE 2 OURS | molecules 12 | ours 11 | fallback 1 | 301.4 s | step 2'), line
    assert 'engine lists 1 ' in line                     # the kernel's compact-trace filter (KEY_LINE) shows it
    r = em.report_line(dict(status='done', rc=0, molecules=12, ok=12, counts=dict(ok=12), seconds=300.0, procs=4,
                            meta=dict(seconds_libs=250.0, seconds_stage1=40.0), files=dict(ok=True, manifest_files=23,
                            manifest_match=23, pool_files=3, pool_match=3, mismatches=[], missing=[])))
    assert r.startswith('ENGINE 2 OURS subprocess | status done | rc 0 | molecules 12 | ok 12') and 'MISMATCH' not in r
    r = em.report_line(dict(status='failed', error='FileNotFoundError: x', files={}))
    assert 'FAILED FileNotFoundError' in r


# ---- run_engine2 with stub entry points (a temporary /kaggle/input tree) ----
STUB = r'''
import json, os, sys, time, subprocess
a = sys.argv[1:]
out = a[4]
mode = os.environ.get('STUB_MODE', 'ok')
test = a[0]
mids = [l.strip() for l in open(test) if l.strip()]
meta = dict(status='done', package=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if mode != 'wrongpkg' else '/elsewhere',
            seconds=1.0, argv=a, env=dict(PYTHONPATH=os.environ.get('PYTHONPATH'), CUDA=os.environ.get('CUDA_VISIBLE_DEVICES')))
lists = {m: dict(smiles=[m + 'S0', m + 'S1'], keys=[m + 'K0', m + 'K1'], s1=[1.0, 0.5]) for m in mids[:2]}
status = {m: ('ok' if m in lists else 'no_query') for m in mids}
if mode == 'crash':
    raise SystemExit(3)
if mode == 'hang':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    open(os.environ['STUB_PIDFILE'], 'w').write(str(child.pid))
    meta['status'] = 'running'
    json.dump(dict(meta=meta, lists=lists, status={m: (s if m in lists else 'not_run') for m, s in status.items()}), open(out, 'w'))
    time.sleep(120)
json.dump(dict(meta=meta, lists=lists, status=status), open(out, 'w'))
'''


def _tree(d, with_pool=True, with_assets=True, nested=False):
    """A /kaggle/input tree: casmi26-elig-assets flat (as Kaggle mounts our uploads) or nested (casmi/, models/)."""
    import hashlib
    root = os.path.join(d, 'input')
    a = os.path.join(root, 'datasets', 'bryantyu', 'casmi26-elig-assets')
    if with_assets:
        code, models = (os.path.join(a, 'casmi'), os.path.join(a, 'models')) if nested else (a, a)
        os.makedirs(code, exist_ok=True)
        os.makedirs(models, exist_ok=True)
        open(os.path.join(code, '__init__.py'), 'w').close()
        open(os.path.join(code, 'run_eng2.py'), 'w').write(STUB)
        open(os.path.join(code, 'helper.py'), 'w').write('X = 1\n')
        open(os.path.join(models, 'ret_ranker_v18cu.txt'), 'w').write('tree\n')
        open(os.path.join(models, 'big_model.pt'), 'wb').write(b'0' * 1000)
        rels = ['run_eng2.py', 'helper.py', 'ret_ranker_v18cu.txt', 'big_model.pt']   # flat names (nested: by basename)
        lines = [f"{hashlib.sha256(open(os.path.join(code if r.endswith('.py') else models, r), 'rb').read()).hexdigest()}  {r}"
                 for r in rels]
        open(os.path.join(a, 'MANIFEST.sha256'), 'w').write('# test\n' + '\n'.join(lines) + '\n')
    if with_pool:
        p = os.path.join(root, 'datasets', 'bryantyu', 'casmi26-pool-v18c')
        os.makedirs(p)
        open(os.path.join(p, 'pool.parquet'), 'wb').write(b'pool')
    test = os.path.join(d, 'test.txt')
    open(test, 'w').write('m1\nm2\nm3\n')
    return root, test


def _run(d, mode, **cfg):
    root, test = _tree(d, **{k: cfg.pop(k) for k in ('with_pool', 'with_assets', 'nested') if k in cfg})
    os.environ['STUB_MODE'] = mode
    os.environ['STUB_PIDFILE'] = os.path.join(d, 'pid')
    os.environ['OWN_PF_INST'] = '1'                    # must not reach the subprocess
    try:
        c = dict(work=os.path.join(d, 'work'), cap_s=30, grace_s=5, pool_sha256={'pool.parquet': __import__('hashlib').sha256(b'pool').hexdigest()})
        c.update(cfg)
        return em.run_engine2(test, os.path.join(d, 'train.parquet'), os.path.join(d, 'eng2.json'), c, input_root=root)
    finally:
        for k in ('STUB_MODE', 'STUB_PIDFILE', 'OWN_PF_INST'):
            os.environ.pop(k, None)


@test
def run_engine2_normal():
    with tempfile.TemporaryDirectory() as d:
        lists, rep = _run(d, 'ok')
        assert lists == {m: dict(smiles=[m + 'S0', m + 'S1'], keys=[m + 'K0', m + 'K1']) for m in ('m1', 'm2')}, lists
        assert rep['status'] == 'done' and rep['rc'] == 0 and rep['molecules'] == 3 and rep['ok'] == 2
        assert rep['counts'] == dict(ok=2, no_query=1) and not rep.get('error')
        assert rep['files']['ok'] and rep['files']['manifest_match'] == 4 and rep['files']['pool_match'] == 1
        meta = json.load(open(os.path.join(d, 'eng2.json')))['meta']
        assert meta['env']['PYTHONPATH'] == os.path.join(d, 'work', 'pkg') and meta['env']['CUDA'] == ''
        assert '--cap' in meta['argv'] and meta['argv'][meta['argv'].index('--k') + 1] == '40'
        assert meta['argv'][3] == rep['assets']['models'] == rep['assets']['code']        # flat: models dir = code dir
        assert sorted(f for f in os.listdir(os.path.join(d, 'work', 'pkg', 'casmi')) if f != '__pycache__') == ['__init__.py', 'helper.py', 'run_eng2.py']
        assert not os.path.exists(os.path.join(d, 'work', 'work'))           # libraries removed after the run


@test
def run_engine2_nested_layout():
    with tempfile.TemporaryDirectory() as d:
        lists, rep = _run(d, 'ok', nested=True)
        assert len(lists) == 2 and rep['status'] == 'done' and rep['files']['ok'] and rep['files']['manifest_match'] == 4
        assert rep['assets']['code'].endswith('casmi') and rep['assets']['models'].endswith('models')


@test
def run_engine2_kill_keeps_partial_and_kills_group():
    with tempfile.TemporaryDirectory() as d:
        t0 = time.time()
        lists, rep = _run(d, 'hang', cap_s=2, grace_s=2)
        assert time.time() - t0 < 30, 'not killed'
        assert rep['rc'] == 'killed (cap + grace)' and set(lists) == {'m1', 'm2'} and rep['counts'].get('not_run') == 1
        pid = int(open(os.path.join(d, 'pid')).read())
        time.sleep(0.5)
        alive = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
        assert not alive or alive.startswith('Z'), f'grandchild {pid} survived: {alive}'


@test
def run_engine2_failures_give_empty_lists():
    with tempfile.TemporaryDirectory() as d:
        lists, rep = _run(d, 'crash')
        assert lists == {} and rep['rc'] == 3 and rep['status'] == 'no output' and 'log_tail' in rep
    with tempfile.TemporaryDirectory() as d:
        lists, rep = _run(d, 'ok', with_pool=False)
        assert lists == {} and rep['error'].startswith('FileNotFoundError') and 'casmi26-pool-v18c' in rep['error']
    with tempfile.TemporaryDirectory() as d:
        lists, rep = _run(d, 'ok', with_assets=False)
        assert lists == {} and 'casmi26-elig-assets' in rep['error']
    with tempfile.TemporaryDirectory() as d:
        lists, rep = _run(d, 'wrongpkg')
        assert lists == {} and 'imported casmi from' in rep['error']


@test
def run_engine2_reports_file_mismatch():
    with tempfile.TemporaryDirectory() as d:
        lists, rep = _run(d, 'ok', pool_sha256={'pool.parquet': '0' * 64, 'pool_fp.npy': '1' * 64})
        f = rep['files']
        assert not f['ok'] and f['mismatches'] == ['pool/pool.parquet'] and f['missing'] == ['pool/pool_fp.npy']
        assert len(lists) == 2                            # log only: the run itself goes ahead
        assert 'MISMATCH' in em.report_line(rep)


if __name__ == '__main__':
    for fn in TESTS:
        fn()
        print('ok', fn.__name__)
    print('eng2_merge tests OK', len(TESTS))
