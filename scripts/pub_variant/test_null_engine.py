"""Unit tests of scripts/pub_variant/null_engine.py on synthetic Engine 1 lists (ours; no public code is run).

python scripts/pub_variant/test_null_engine.py      (plain asserts; prints 'null_engine tests OK' and the count)

The fusion checks re-implement the documented formulas of frozen cells 45 (fuse2: sc[k] += w / (KRR + rank) over the
raw positions, keys chem.score_key(s) or s) and 49 / 52 (metric_fusion: one vote per identity, compact ranks) as our
own code; the frozen code itself is verified by the Kaggle commit run.
"""
import importlib.util
import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('null_engine', os.path.join(HERE, 'null_engine.py'))
ne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne)

ALPHA, KRR = 0.6, 3.0
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def key_of(s):          # stand-in for chem.score_key: 'X~t' is a tautomer of 'X' (same key); 'bad*' does not parse
    return None if s.startswith('bad') else s.split('~')[0].upper()


def fuse_keys(s):       # what fuse2 uses for the main list: chem.score_key(s) or s
    return key_of(s) or s


def identity_of(s):     # stand-in for metric_identity: also merges 'X@...' stereo variants; None when unparseable
    return None if s.startswith('bad') else s.split('~')[0].split('@')[0].upper()


def fuse2(v_smis, e_smis, e_keys, n=40):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = fuse_keys(s)
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order]


def metric_fusion(first, second, limit=40):
    scores, reps = {}, {}
    for cands, w in ((first, 1.0), (second, ALPHA)):
        seen = set()
        for c in cands:
            k = identity_of(c)
            if k is None or k in seen:
                continue
            seen.add(k)
            scores[k] = scores.get(k, 0.0) + w / (KRR + len(seen)); reps.setdefault(k, c)
    order = sorted(scores, key=lambda k: -scores[k])[:limit]
    return [reps[k] for k in order]


def own_order_fuse2(v):     # the main list fused with nothing but itself, by its own scores
    return fuse2(v, [], [])


def own_order_metric(v):
    return metric_fusion(v, [])


@test
def schema_and_coverage():
    mids = ['m1', 'm2', 'm3']
    eng = ne.null_engine_lists(mids, {'m1': ['A', 'B'], 'm2': []}, key_of=key_of)
    assert list(eng) == mids                                    # every molecule, in order
    for m, v in eng.items():
        assert isinstance(m, str) and set(v) == {'smiles', 'keys'}
        assert isinstance(v['smiles'], list) and isinstance(v['keys'], list) and len(v['smiles']) == len(v['keys'])
    assert eng['m1'] == {'smiles': ['A', 'B'], 'keys': ['A', 'B']}
    assert eng['m2'] == {'smiles': [], 'keys': []} and eng['m3'] == {'smiles': [], 'keys': []}
    json.loads(json.dumps(eng))                                 # JSON round trip, as eng_lists.json


@test
def placeholder_and_empty_entries_dropped():
    eng = ne.null_engine_lists(['m'], {'m': ['CCO']}, key_of=key_of)
    assert eng['m'] == {'smiles': [], 'keys': []}
    eng = ne.null_engine_lists(['m'], {'m': ['A', '', 'CCO', 'B']}, tails={'m': ['CCO', '', 'C']}, key_of=key_of)
    assert eng['m']['smiles'] == ['A', 'B', 'C']


@test
def head_is_verbatim_with_repeats_and_unparseable():
    head = ['A', 'B', 'a~t', 'bad1', 'C', 'A']                  # 'a~t' repeats key A; 'bad1' has no key; 'A' twice
    eng = ne.null_engine_lists(['m'], {'m': head}, key_of=fuse_keys)
    assert eng['m']['smiles'] == head and eng['m']['keys'] == ['A', 'B', 'A', 'bad1', 'C', 'A']


@test
def explicit_keys_are_kept():                                     # step 1: Engine 1's own keys, even None
    eng = ne.null_engine_lists(['m'], {'m': ['A', 'B', 'C']}, keys={'m': ['kA', None, 'kC']})
    assert eng['m'] == {'smiles': ['A', 'B', 'C'], 'keys': ['kA', None, 'kC']}
    try:
        ne.null_engine_lists(['m'], {'m': ['A', 'B']}, keys={'m': ['kA']})
        raise AssertionError('length mismatch accepted')
    except AssertionError as exc:
        assert 'length mismatch accepted' not in str(exc)


@test
def tail_appends_new_keys_only_in_order():
    eng = ne.null_engine_lists(['m'], {'m': ['A', 'B']}, tails={'m': ['B', 'b~t', 'C', 'bad2', 'D', 'C~x', 'E']},
                               key_of=key_of)
    assert eng['m']['smiles'] == ['A', 'B', 'C', 'D', 'E'] and eng['m']['keys'] == ['A', 'B', 'C', 'D', 'E']


@test
def cap_at_n():
    head = [f'H{i}' for i in range(30)]
    tail = [f'T{i}' for i in range(30)]
    eng = ne.null_engine_lists(['m'], {'m': head}, tails={'m': tail}, key_of=key_of, n=40)
    assert eng['m']['smiles'] == head + tail[:10]
    eng = ne.null_engine_lists(['m'], {'m': head + tail}, keys={'m': head + tail}, n=40)
    assert eng['m']['smiles'] == (head + tail)[:40]


@test
def mid_lookup_raw_or_str():
    eng = ne.null_engine_lists([7, 8], {'7': ['A'], 8: ['B']}, key_of=key_of)     # str(mid) or the raw id
    assert eng == {'7': {'smiles': ['A'], 'keys': ['A']}, '8': {'smiles': ['B'], 'keys': ['B']}}
    eng = ne.null_engine_lists(['m', 'm'], {'m': ['A']}, key_of=key_of)
    assert list(eng) == ['m']


@test
def step1_ice_union_adds_nothing():
    """Cell 41: es = engine entries whose key is not in the base top 60. With Engine 1's own keys it is empty."""
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(0, 60)
        smis = [f'S{rng.randint(0, 999)}' for _ in range(n)]
        keys = [None if rng.random() < 0.05 else f'K{i}' for i in range(n)]
        eng = ne.null_engine_lists(['m'], {'m': smis}, keys={'m': keys}, n=40)
        e = eng['m']
        bk = set(keys[:60])
        es = [s for s, k in zip(e['smiles'][:40], e['keys'][:40]) if k not in bk]
        assert es == [] and e['smiles'] == smis[:40]


def random_main(rng, base_n=60, pc_n=8):
    """Synthetic Engine 1 list (BASE, 60) + the gated PubChem merge of cell 43 into its top 25 (slots / promotion)."""
    pool = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    base = []
    while len(base) < base_n:                                    # unique by key, as cell 39 builds BASE
        s = rng.choice(pool) + str(rng.randint(0, 400)) + rng.choice(['', '~t', '@s', '@r'])
        if key_of(s) not in {key_of(x) for x in base}:
            base.append(s)
    pc = [f'P{rng.randint(0, 400)}{rng.choice(["", "~t"])}' for _ in range(pc_n)]
    slots = rng.choice([[2, 4, 6, 8, 10], [4, 8, 12, 16, 20], []])
    bk = {key_of(x) for x in base[:25]}
    pcs = [s for s in pc if key_of(s) not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, 26):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < 25:
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    if pc and rng.random() < 0.3:                                # promotion: the PubChem leader takes rank 1
        out = ([pc[0]] + [x for x in out if x != pc[0]])[:25]
    if rng.random() < 0.2:                                       # a tautomer repeat and an unparseable entry
        out[rng.randint(1, len(out) - 1)] = out[0].split('~')[0] + '~t'
    if rng.random() < 0.1:
        out[rng.randint(1, len(out) - 1)] = 'bad' + str(rng.randint(0, 9))
    if rng.random() < 0.1:
        out = out[:rng.randint(1, 24)]                           # short main list
    return base, out


@test
def step2_fusions_keep_the_main_order():
    """Cells 45 / 49 / 52 fuse the main list (PREFUSION_LISTS) with ENG; with the null lists the fused order is the
    main list's own order (top len(main)), and the tail only fills ranks after it."""
    rng = random.Random(7)
    for trial in range(3000):
        base, main = random_main(rng)
        eng = ne.null_engine_lists(['m'], {'m': main}, tails={'m': base}, key_of=fuse_keys, n=40)['m']
        assert eng['smiles'][:len(main)] == main and len(eng['smiles']) <= 40
        fused = fuse2(main, eng['smiles'], eng['keys'])
        own = own_order_fuse2(main)
        assert fused[:len(own)] == own, (trial, main, fused)
        tail_part = fused[len(own):]
        assert all(s not in main for s in tail_part) and tail_part == [s for s in eng['smiles'][len(main):]][:len(tail_part)]
        m_fused = metric_fusion(main, eng['smiles'])
        m_own = own_order_metric(main)
        assert m_fused[:len(m_own)] == m_own, (trial, main, m_fused)
        # PubChem inserts and a promoted PubChem leader keep their places (Engine 1's top 40 alone would demote them)
        if main and main[0].startswith('P'):
            assert fused[0] == main[0]


@test
def base_only_null_would_demote_pubchem():
    """Why step 2 uses the main list and not Engine 1's top 40 alone: a promoted PubChem leader loses rank 1."""
    base = [f'B{i}' for i in range(60)]
    main = ['P1'] + base[:24]                                    # promote(): PubChem leader, then Engine 1's list
    base_only = ne.null_engine_lists(['m'], {'m': base}, key_of=fuse_keys, n=40)['m']
    assert fuse2(main, base_only['smiles'], base_only['keys'])[0] == 'B0'
    ours = ne.null_engine_lists(['m'], {'m': main}, tails={'m': base}, key_of=fuse_keys, n=40)['m']
    assert fuse2(main, ours['smiles'], ours['keys'])[0] == 'P1'


@test
def summary_and_write():
    eng = ne.null_engine_lists(['a', 'b'], {'a': ['X', 'Y']}, key_of=key_of)
    line = ne.summary(eng)
    assert line.startswith('engine lists 2 | nonempty 1 | candidates 2'), line
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'eng_lists.json')
        ne.write_lists(eng, p)
        assert json.load(open(p)) == eng and not os.path.exists(p + '.tmp')


if __name__ == '__main__':
    for fn in TESTS:
        fn()
        print('ok', fn.__name__)
    print(f'null_engine tests OK ({len(TESTS)})')
    sys.exit(0)
