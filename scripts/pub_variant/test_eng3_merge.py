"""Unit tests of scripts/pub_variant/eng3_merge.py and of OUR glue appended to frozen cells 32 / 39 and notebook cell 8
by scripts/pub_variant/eng3.py (ours; no public code is run: only the text blocks eng3.py generates are executed, with
stub modules and a temporary directory in place of /kaggle/working).

python scripts/pub_variant/test_eng3_merge.py      (plain asserts; prints 'eng3_merge tests OK' and the count)
"""
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import types
from fractions import Fraction
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m3 = _load('eng3_merge')
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def L(*names, key=None):
    """A list entry from names: SMILES 'x', key 'X' (key(s) when given)."""
    key = key or (lambda s: s.split('~')[0].upper())
    return dict(smiles=list(names), keys=[key(s) for s in names])


def reference(e, o, w, k=3, n=40):
    """Brute force with exact fractions: every identity, its score, sorted by (-score, e rank, o rank). An Engine 2
    entry without a key is its own identity (the slot r: ('#', r)); one of ours scores nothing."""
    ep = {}
    for r, x in enumerate(e['keys'], 1):
        ep.setdefault(x if x else ('#', r), r)
    op = {}
    for r, x in enumerate(o['keys'], 1):
        if x:
            op.setdefault(x, r)
    ids = set(ep) | set(op)
    sc = {x: (Fraction(1, k + ep[x]) if x in ep else 0) + (Fraction(w) / (k + op[x]) if x in op else 0) for x in ids}
    big = 10 ** 9
    order = sorted(ids, key=lambda x: (-sc[x], ep.get(x, big), op.get(x, big)))[:n]
    return [e['keys'][x[1] - 1] if isinstance(x, tuple) else x for x in order]


@test
def hand_example_w1():
    # A: 1/4 + 1/6 = 5/12; B: 1/5 + 1/4 = 9/20; C: 1/6; D: 1/5  ->  B, A, D, C
    out, st = m3.rrf_merge({'m': L('a', 'b', 'c')}, {'m': L('b', 'd', 'a')}, 1.0)
    assert out['m'] == dict(smiles=['b', 'a', 'd', 'c'], keys=['B', 'A', 'D', 'C']), out
    assert (st['molecules'], st['ours'], st['merged'], st['shared'], st['new_from_ours'], st['displaced'],
            st['top1_changed'], st['top1_from_ours']) == (1, 1, 1, 2, 1, 0, 1, 0), st


@test
def hand_example_w05():
    # A: 1/4 + 0.5/6 = 1/3; B: 1/5 + 0.5/4 = 13/40; C: 1/6; D: 0.5/5 = 1/10  ->  A, B, C, D (Engine 2's order kept)
    out, st = m3.rrf_merge({'m': L('a', 'b', 'c')}, {'m': L('b', 'd', 'a')}, 0.5)
    assert out['m']['keys'] == ['A', 'B', 'C', 'D'] and st['top1_changed'] == 0 and st['new_from_ours'] == 1


@test
def ties_engine2_rank_first():
    # disjoint lists, w 1: our rank r ties Engine 2 rank r -> Engine 2's entry first (interleave E1 O1 E2 O2)
    out, _ = m3.rrf_merge({'m': L('a', 'b')}, {'m': L('c', 'd')}, 1.0)
    assert out['m']['keys'] == ['A', 'C', 'B', 'D'], out
    # w 0.5: our rank 1 (0.5/4 = 1/8) ties Engine 2 rank 5 (1/8) -> after it, before Engine 2 rank 6
    e = L('e1', 'e2', 'e3', 'e4', 'e5', 'e6')
    out, st = m3.rrf_merge({'m': e}, {'m': L('x')}, 0.5)
    assert out['m']['keys'] == ['E1', 'E2', 'E3', 'E4', 'E5', 'X', 'E6'], out
    # two shared identities with equal exact scores: P (e 1, o 9) = 1/4 + 1/12 = 1/3, Q (e 3, o 3) = 1/6 + 1/6 = 1/3
    e = L('p', 'z1', 'q')
    o = L('y1', 'y2', 'q~t', 'y3', 'y4', 'y5', 'y6', 'y7', 'p~t')     # q at our rank 3, p at our rank 9
    out, _ = m3.rrf_merge({'m': e}, {'m': o}, 1.0)
    assert out['m']['keys'][:2] == ['P', 'Q'], out           # tie -> lower Engine 2 rank (P at 1) first
    assert out['m']['keys'] == reference(e, o, 1.0)
    # an identity Engine 2 does not list at all ranks after an equal-scoring one it lists (1/5 + 1/20 = 1/4 = ours rank 1)
    e = L('a', 'b')
    o = L('c', *[f'f{i}' for i in range(15)], 'b~t')         # b: Engine 2 rank 2 + our rank 17 -> 1/5 + 1/20 = 1/4
    out, _ = m3.rrf_merge({'m': e}, {'m': o}, 1.0)
    assert out['m']['keys'][:3] == ['A', 'B', 'C'], out       # A 1/4 (e 1), B 1/4 (e 2), C 1/4 (no Engine 2 rank)


@test
def ties_then_our_rank():
    # equal score, neither in Engine 2's list cannot happen with unique keys; equal score and equal Engine 2 rank
    # cannot either. The ours tie-break is exercised through the reference on random lists with many exact ties.
    rnd = random.Random(7)
    for trial in range(3000):
        pool = [f's{i}' for i in range(30)]
        e = L(*rnd.sample(pool, rnd.randint(1, 25)))
        o = L(*rnd.sample(pool, rnd.randint(1, 25)))
        w = rnd.choice([1.0, 0.5, 0.25, 2.0, 1.5, 0.1])
        n = rnd.choice([40, 5, 1])
        out, _ = m3.rrf_merge({'m': e}, {'m': o}, w, n=n)
        assert out['m']['keys'] == reference(e, o, w, n=n), (trial, e, o, w)


@test
def fallbacks():
    eng2 = {'m1': L('a', 'b'), 'm2': L('c'), 'm3': dict(smiles=[], keys=[]), 'm4': L('d')}
    ours = {'m1': L('x'), 'm2': dict(smiles=[], keys=[]), 'm3': L('y', 'z'), 'm5': L('v', 'u'), 'm6': dict(smiles=[], keys=[])}
    out, st = m3.rrf_merge(eng2, ours, 1.0)
    assert list(out) == ['m1', 'm2', 'm3', 'm4', 'm5'], list(out)        # Engine 2's order, then ours-only; m6 absent
    assert out['m2'] == eng2['m2'] and out['m4'] == eng2['m4']           # no (usable) list of ours: unchanged
    assert out['m2'] is not eng2['m2'] and out['m2']['smiles'] is not eng2['m2']['smiles']   # copies, no aliasing
    assert out['m3'] == L('y', 'z') and out['m5'] == L('v', 'u')          # no Engine 2 list: ours
    assert out['m1']['keys'] == ['A', 'X', 'B']
    assert (st['molecules'], st['engine2'], st['ours'], st['merged'], st['engine2_only'], st['ours_only']) == \
        (5, 3, 3, 1, 2, 2), st
    assert st['ours_only_ids'] == ['m3', 'm5']
    # empty / missing inputs
    assert m3.rrf_merge({'a': L('a')}, {}, 1.0)[0] == {'a': L('a')}
    assert m3.rrf_merge({'a': L('a')}, None, 1.0)[0] == {'a': L('a')}
    assert m3.rrf_merge({}, {'a': L('a')}, 1.0)[0] == {'a': L('a')}
    assert m3.rrf_merge(None, None, 1.0) == ({}, m3.rrf_merge({}, {}, 1.0)[1])
    # Engine 2's entry kept exactly, extra fields included; nothing mutated
    e = {'m': dict(smiles=['a', 'b'], keys=['A', 'B'], note='x')}
    before = json.dumps(e)
    out, _ = m3.rrf_merge(e, {'n': L('c')}, 1.0)
    assert out['m'] == e['m'] and json.dumps(e) == before


@test
def unchanged_when_ours_empty_everywhere():
    """Our subprocess failed (ENG3_OURS = {}): ENG equals the original Engine 2 lists exactly (pubm's content)."""
    rnd = random.Random(1)
    eng2 = {f'mol{i}': L(*[f'c{i}_{j}' for j in range(rnd.randint(1, 40))]) for i in range(50)}
    out, st = m3.rrf_merge(eng2, {}, 1.0)
    assert out == eng2 and list(out) == list(eng2) and st['merged'] == 0 and st['engine2_only'] == 50


@test
def key_identity():
    # same structure, different SMILES strings: one identity, Engine 2's SMILES kept; our SMILES for ours-only keys
    e = dict(smiles=['OCC', 'c1ccccc1'], keys=['ETHANOL', 'BENZENE'])
    o = dict(smiles=['C(O)C', 'CC(=O)O'], keys=['ETHANOL', 'ACETIC'])
    out, st = m3.rrf_merge({'m': e}, {'m': o}, 1.0)
    # ETHANOL 1/4 + 1/4; BENZENE (Engine 2 rank 2) 1/5 ties ACETIC (our rank 2) 1/5 -> Engine 2's first
    assert out['m'] == dict(smiles=['OCC', 'c1ccccc1', 'CC(=O)O'], keys=['ETHANOL', 'BENZENE', 'ACETIC']), out
    assert st['shared'] == 1 and st['new_from_ours'] == 1
    # a repeated key scores nothing; an Engine 2 entry without a key keeps its slot (positions as written)
    e = dict(smiles=['a', 'b', 'a2', 'c'], keys=['A', None, 'A', 'C'])
    o = dict(smiles=['c'], keys=['C'])
    out, _ = m3.rrf_merge({'m': e}, {'m': o}, 1.0)
    assert out['m'] == dict(smiles=['c', 'a', 'b'], keys=['C', 'A', None]), out   # C 1/7 + 1/4 > A 1/4 > b 1/5
    assert len(set(out['m']['keys'])) == len(out['m']['keys'])
    # molecule ids are str in the output (readers use ENG.get(str(mid)))
    out, _ = m3.rrf_merge({7: L('a')}, {'7': L('b')}, 1.0)
    assert list(out) == ['7'] and out['7']['keys'] == ['A', 'B']


@test
def keyless_engine2_entries():
    """Engine 2's canon_key writes '' when RDKit gives an empty InChIKey and eng_runner keeps that entry (only None is
    skipped): it is not dropped from the merged list, it keeps its own rank, SMILES and key as written."""
    e = dict(smiles=['EA', 'Ebad', 'EB'], keys=['A', '', 'B'])
    out, st = m3.rrf_merge({'m': e}, {'m': L('b')}, 1.0)
    # B: 1/6 + 1/4 = 5/12, A: 1/4, '' (rank 2): 1/5
    assert out['m'] == dict(smiles=['EB', 'EA', 'Ebad'], keys=['B', 'A', '']), out
    assert (st['engine2_keyless'], st['engine2_keyless_kept'], st['shared'], st['new_from_ours'], st['displaced']) == \
        (1, 1, 1, 0, 0), st
    assert st['per_molecule'][0]['engine2_keyless'] == 1
    # two keyless entries ('' and None) never merge with each other or with ours; w 0.5: our rank 1 after E2 rank 5
    e = dict(smiles=['e1', 'k1', 'e3', 'k2', 'e5', 'e6'], keys=['E1', '', 'E3', None, 'E5', 'E6'])
    out, st = m3.rrf_merge({'m': e}, {'m': dict(smiles=['x', 'y'], keys=['X', ''])}, 0.5)
    assert out['m'] == dict(smiles=['e1', 'k1', 'e3', 'k2', 'e5', 'x', 'e6'],
                            keys=['E1', '', 'E3', None, 'E5', 'X', 'E6']), out   # our keyless 'y' scores nothing
    assert st['engine2_keyless'] == 2 and st['engine2_keyless_kept'] == 2 and st['shared'] == 0
    # a keyless leader stays the leader at w 0.5 (1/4 vs our rank 1: 1/8), top1 not changed
    e = dict(smiles=['k', 'a'], keys=['', 'A'])
    out, st = m3.rrf_merge({'m': e}, {'m': L('z')}, 0.5)
    assert out['m']['keys'] == ['', 'A', 'Z'] and st['top1_changed'] == 0
    # a keyless entry displaced by the cap counts as displaced (and not kept)
    e = dict(smiles=[f'e{i}' for i in range(40)], keys=[f'E{i}' for i in range(39)] + [''])
    out, st = m3.rrf_merge({'m': e}, {'m': L(*[f'o{i}' for i in range(40)])}, 1.0)
    assert '' not in out['m']['keys'] and len(out['m']['keys']) == 40
    assert st['engine2_keyless'] == 1 and st['engine2_keyless_kept'] == 0 and st['displaced'] == 20
    # missing keys (keys shorter than smiles) on Engine 2's side: kept as keyless (key None); on ours: no score
    out, st = m3.rrf_merge({'m': dict(smiles=['a', 'b'], keys=['A'])}, {'m': dict(smiles=['c', 'd'], keys=['C'])}, 1.0)
    assert out['m'] == dict(smiles=['a', 'c', 'b'], keys=['A', 'C', None]), out
    # brute-force reference with keyless Engine 2 entries on random lists
    rnd = random.Random(11)
    for trial in range(1500):
        pool = [f's{i}' for i in range(30)]
        e = L(*rnd.sample(pool, rnd.randint(1, 25)))
        for i in range(len(e['keys'])):
            if rnd.random() < .15:
                e['keys'][i] = rnd.choice(['', None])
        o = L(*rnd.sample(pool, rnd.randint(1, 25)))
        w, n = rnd.choice([1.0, 0.5, 0.25, 2.0]), rnd.choice([40, 5, 1])
        out, st = m3.rrf_merge({'m': e}, {'m': o}, w, n=n)
        assert out['m']['keys'] == reference(e, o, w, n=n), (trial, e, o, w)
        # every SMILES is Engine 2's for an Engine 2 identity, the keyless ones included
        e_smi = {k: s for s, k in zip(e['smiles'], e['keys']) if k}
        got = dict(zip(out['m']['keys'], out['m']['smiles']))
        assert all(got[k] == e_smi[k] for k in got if k in e_smi)
        assert st['engine2_keyless'] == sum(1 for k in e['keys'] if not k)


@test
def cap_40():
    e = L(*[f'e{i}' for i in range(40)])
    o = L(*[f'o{i}' for i in range(40)])
    out, st = m3.rrf_merge({'m': e}, {'m': o}, 1.0)
    want = [x for i in range(20) for x in (f'E{i}', f'O{i}')]
    assert out['m']['keys'] == want and len(out['m']['smiles']) == 40, out['m']['keys'][:6]
    assert st['displaced'] == 20 and st['new_from_ours'] == 20 and st['shared'] == 0
    out, st = m3.rrf_merge({'m': e}, {'m': o}, 0.5)
    assert len(out['m']['keys']) == 40 and st['displaced'] == st['new_from_ours']
    # 0.5 / (3 + r) > 1 / (3 + s)  <=>  s > 2r + 3: our rank r enters ahead of Engine 2 rank 2r + 4
    assert out['m']['keys'].index('O0') == 5 and out['m']['keys'].index('O1') == 7 + 1
    # ours alone (no Engine 2 list): top n of ours
    out, _ = m3.rrf_merge({}, {'m': L(*[f'o{i}' for i in range(50)])}, 1.0, n=40)
    assert len(out['m']['keys']) == 40 and out['m']['keys'][-1] == 'O39'


@test
def weight_validation():
    for bad in (0, -1.0, float('inf'), float('nan')):
        try:
            m3.rrf_merge({}, {}, bad)
        except AssertionError:
            continue
        raise AssertionError(f'w {bad} accepted')
    out, st = m3.rrf_merge({'m': L('a')}, {'m': L('b')}, 1)
    assert st['w'] == 1.0 and out['m']['keys'] == ['A', 'B']


@test
def coverage_like_cell_48():
    """Every molecule of Engine 2 stays covered (frozen cell 48: active <= set(map(str, ENG))) with lists <= 40."""
    rnd = random.Random(3)
    for seed in range(40):
        mids = [f'm{i}' for i in range(rnd.randint(1, 30))]
        eng2 = {m: L(*[f'{m}e{j}' for j in range(rnd.randint(1, 40))]) for m in mids}
        ours = {m: L(*[f'{m}{rnd.choice("eo")}{j}' for j in range(rnd.randint(0, 40))]) for m in mids if rnd.random() < .8}
        out, st = m3.rrf_merge(eng2, ours, rnd.choice([1.0, 0.5]))
        assert set(mids) <= set(map(str, out)) and list(out)[:len(mids)] == mids
        for m in mids:
            v = out[m]
            real = [x for x in v['keys'] if x]
            assert v['smiles'] and len(v['smiles']) == len(v['keys']) <= 40 and len(set(real)) == len(real)
        assert st['merged'] + st['engine2_only'] == len(mids)


@test
def check_keys_log_only():
    lists = {'a': dict(smiles=['x', 'y', 'z'], keys=['X', 'WRONG', 'Z']), 'b': dict(smiles=[], keys=[])}
    before = json.dumps(lists)
    rep = m3.check_keys(lists, lambda s: s.upper())
    assert (rep['mode'], rep['molecules'], rep['checked'], rep['mismatches']) == ('all', 1, 3, 1) and json.dumps(lists) == before
    assert rep['examples'] == [dict(molecule_id='a', smiles='y', key='WRONG', score_key='Y')]
    rep = m3.check_keys(lists, lambda s: s.upper(), head=1)
    assert rep['checked'] == 1 and rep['mismatches'] == 0 and rep['mode'] == 'first 1 per molecule'


@test
def log_lines():
    out, st = m3.rrf_merge({'m': L('a', 'b', 'c')}, {'m': L('b', 'd', 'a')}, 1.0)
    line = m3.log_line(st, 253.6)
    assert line == 'ENG3 | w 1.0 | molecules 1 | ours 1 | merged 1 | shared identities 2 | new from ours 1 | 253.6 s', line
    assert len(line) < 180                                    # the notebook's compact trace prints 180 characters
    d = m3.detail_line(st)
    assert d.startswith('ENG3 merge | k 3 | top 40 | top1 changed 1 (from ours 0)') and len(d) < 180, d
    assert d.endswith('| ours only 0 | engine 2 keyless kept 0 / 0'), d
    big = dict(st, top1_changed=400, top1_from_ours=400, displaced=16000, engine2_only=400, ours_only=400,
               engine2_keyless=16000, engine2_keyless_kept=16000)     # a 400-molecule rerun's widest numbers
    assert len(m3.detail_line(big)) < 180 and len(m3.log_line(dict(big, molecules=400, ours=400, merged=400,
                                                                   shared=16000, new_from_ours=16000), 3000.0)) < 180
    k = m3.keys_line(dict(mode='all', entries=471, checked=471, mismatches=0, dropped=0, emptied=0),
                     dict(mode='all', checked=480, mismatches=0), 12.3)
    assert k == ('ENG3 keys | ours (all): entries 471, checked 471, mismatches 0, dropped 0, emptied 0 | engine 2 (log only, '
                 'all): checked 480, mismatches 0 | 12 s'), k
    assert len(k) < 180


@test
def write_lists_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'eng_lists.json')
        eng = {'m': L('a', 'b')}
        m3.write_lists(eng, p)
        assert json.load(open(p)) == eng and not os.path.exists(p + '.tmp')


# ---- our glue blocks (eng3.py), executed with stubs; the frozen cells' own text is never run ----
e3 = _load('eng3')


def _glue_scope(tmp, eng2, ours, rerun=False, rdkit_sub='2026.03.3', rdkit_main='2026.03.3', run_raises=False,
                merge_raises=False):
    def key_of(s):
        return None if s.startswith('bad') else s.split('~')[0].upper()

    stub2 = types.ModuleType('eng2_merge')
    real2 = _load('eng2_merge')
    calls = []

    def run_engine2(test, train, out, cfg):
        calls.append((test, train, out, cfg))
        if run_raises:
            raise RuntimeError('boom')
        return ours, dict(status='done', seconds=12.5, meta=dict(versions=dict(rdkit=rdkit_sub)), counts={'ok': len(ours)})
    stub2.run_engine2 = run_engine2
    for name in ('verify_keys', 'report_line', 'write_json'):
        setattr(stub2, name, getattr(real2, name))
    stub3 = _load('eng3_merge')
    if merge_raises:
        def boom(*a, **k):
            raise ValueError('merge boom')
        stub3.rrf_merge = boom
    sys.modules['eng2_merge'], sys.modules['eng3_merge'] = stub2, stub3
    scope = dict(ENG=eng2, COMP=str(tmp), os=os, json=json, time=__import__('time'), IS_RERUN=rerun,
                 rdkit=types.SimpleNamespace(__version__=rdkit_main), chem=types.SimpleNamespace(score_key=key_of))
    return scope, calls


def _run_glue(tmp, scope, w):
    for block in (e3.c32_block(w), e3.C39_BLOCK):
        code = block.replace('/kaggle/working', str(tmp))
        assert code != block or '/kaggle/working' not in block
        exec(compile(code, 'eng3 glue', 'exec'), scope)        # OUR appended text only
    return scope


@test
def glue_cells_32_39():
    eng2 = {'m1': L('a', 'b', 'c'), 'm2': L('d')}
    ours = {'m1': dict(smiles=['b', 'x', 'bad1', 'a'], keys=['B', 'X', 'BAD', 'WRONG'])}   # 2 keys fixed by score_key
    with tempfile.TemporaryDirectory() as d:
        scope, calls = _glue_scope(d, eng2, ours)
        _run_glue(d, scope, 1.0)
        assert calls and calls[0][2] == os.path.join(d, 'eng3_ours.json') and calls[0][3]['work'] == '/tmp/eng3_ours'
        assert calls[0][3]['cap_s'] == 2700.0 and calls[0][3]['stall_s'] == 900.0 and calls[0][3]['grace_s'] == 300.0
        assert scope['ENG3_ENGINE2'] is eng2 and scope['ENG3_W'] == 1.0 and scope['ENG3_K'] == 3 and scope['ENG3_N'] == 40
        # our list re-keyed: 'bad1' dropped (key None), 'a' re-keyed WRONG -> A
        assert scope['ENG3_OURS'] == {'m1': dict(smiles=['b', 'x', 'a'], keys=['B', 'X', 'A'])}
        assert scope['ENG3_KEYCHECK']['mismatches'] == 2 and scope['ENG3_KEYCHECK']['mode'] == 'all'
        # A: 1/4 + 1/6, B: 1/5 + 1/4, C: 1/6, X: 1/5 -> B, A, X, C; m2 unchanged
        assert scope['ENG'] == {'m1': dict(smiles=['b', 'a', 'x', 'c'], keys=['B', 'A', 'X', 'C']), 'm2': L('d')}
        assert json.load(open(os.path.join(d, 'eng_lists.json'))) == scope['ENG']
        assert json.load(open(os.path.join(d, 'eng_lists_engine2.json'))) == eng2
        rep = json.load(open(os.path.join(d, 'eng3_report.json')))
        assert rep['merge']['merged'] == 1 and rep['w'] == 1.0 and rep['engine2_key_check']['mismatches'] == 0
        assert json.load(open(os.path.join(d, 'eng3_ours_report.json')))['status'] == 'done'


@test
def glue_rerun_head_and_failures():
    eng2 = {'m1': L('a', 'b'), 'm2': L('d')}
    ours = {'m1': L('x', 'a')}
    with tempfile.TemporaryDirectory() as d:                 # rerun, same RDKit: first 10 keys per list checked
        scope, _ = _glue_scope(d, eng2, ours, rerun=True)
        _run_glue(d, scope, 0.5)
        assert scope['ENG3_KEYCHECK']['mode'] == 'first 10 per molecule'
        assert scope['ENG3_E2KEYS']['mode'] == 'first 3 per molecule'
        assert scope['ENG']['m1']['keys'] == ['A', 'B', 'X'] and scope['ENG']['m2'] == L('d')
    with tempfile.TemporaryDirectory() as d:                 # rerun, other RDKit in the subprocess: every key
        scope, _ = _glue_scope(d, eng2, ours, rerun=True, rdkit_sub='2025.03.6')
        _run_glue(d, scope, 0.5)
        assert scope['ENG3_KEYCHECK']['mode'] == 'all'
    with tempfile.TemporaryDirectory() as d:                 # subprocess call raises: original lists, pubm's content
        scope, _ = _glue_scope(d, eng2, ours, run_raises=True)
        _run_glue(d, scope, 1.0)
        assert scope['ENG3_OURS'] == {} and scope['ENG3_REPORT']['status'] == 'failed' and scope['ENG'] == eng2
        assert scope['ENG3_STATS']['merged'] == 0 and scope['ENG3_STATS']['engine2_only'] == 2
    with tempfile.TemporaryDirectory() as d:                 # merge raises: ENG is the original object, file restored
        scope, _ = _glue_scope(d, eng2, ours, merge_raises=True)
        _run_glue(d, scope, 1.0)
        assert scope['ENG'] is eng2 and scope['ENG3_STATS']['status'] == 'failed'
        assert json.load(open(os.path.join(d, 'eng_lists.json'))) == eng2
    with tempfile.TemporaryDirectory() as d:                 # original Engine 2 failed (ENG = {}): ours where it exists
        scope, _ = _glue_scope(d, {}, ours)
        _run_glue(d, scope, 1.0)
        assert scope['ENG'] == {'m1': L('x', 'a')} and scope['ENG3_STATS']['ours_only'] == 1


@test
def glue_cell8_setup():
    """Notebook cell 8's appendix: writes the pinned modules, refuses a cell-1 weight other than the baked one."""
    info = dict(original_sha256='0' * 64, patched_sha256='1' * 64, patched_cells=[32, 39])
    block = e3.cell8_block(info, 0.5)
    with tempfile.TemporaryDirectory() as d:
        code, inp = Path(d) / 'code', Path(d) / 'input'
        code.mkdir()
        (inp / 'datasets' / 'bryantyu' / 'casmi26-elig-assets').mkdir(parents=True)
        g = dict(CODE=code, INPUT=inp, FROZEN_SHA256='1' * 64, ENG3_W=0.5, __name__='cell8')
        exec(compile(block, 'eng3 cell 8', 'exec'), g)
        assert sorted(os.listdir(code)) == ['eng2_merge.py', 'eng3_merge.py']
        assert (code / 'eng3_merge.py').read_text() == open(os.path.join(HERE, 'eng3_merge.py')).read()
        assert (code / 'eng2_merge.py').read_text() == open(os.path.join(HERE, 'eng2_merge.py')).read()
        assert g['ENG3_SETUP']['mounted'] == {'bryantyu/casmi26-elig-assets': True, 'bryantyu/casmi26-pool-v18c': False}
        g = dict(CODE=code, INPUT=inp, FROZEN_SHA256='1' * 64, ENG3_W=1.0, __name__='cell8')
        try:
            exec(compile(block, 'eng3 cell 8', 'exec'), g)
        except AssertionError as exc:
            assert 'differs from the weight baked' in str(exc)
        else:
            raise AssertionError('a cell-1 ENG3_W other than the baked weight was accepted')


if __name__ == '__main__':
    for fn in TESTS:
        fn()
        print('ok', fn.__name__)
    print('eng3_merge tests OK', len(TESTS))
