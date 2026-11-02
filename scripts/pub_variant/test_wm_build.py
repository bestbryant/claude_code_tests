"""Static checks of the fusion-weight options CUSTOM_W_TOP / CUSTOM_W_MID / MIDX_NEURAL_W (scripts/pub_variant/wm.py) and
of casmi26-pubm-sens-wm100 (SENS=5 raw + CUSTOM_W_MID=1.0 + MIDX_NEURAL_W=1.0) made by scripts/make_pub_variant.py (ours).

python scripts/pub_variant/test_wm_build.py [--regen]

Nothing of the notebooks is executed: cells are only parsed (ast / symtable / difflib on the text), the pinned modules are
read as text, and the generator (our code) is run into temporary directories.
  1. builds wm100, its base casmi26-pubm-sens and composition / boundary variants (CUSTOM_W_MID only; MIDX_NEURAL_W=0.75;
     w_top + midx on pubm; ELIG=1 + SENS + both; ENG3_W + midx; bounds 2 / 0 / 0) with the working generator; the
     defaults given explicitly (CUSTOM_W_TOP=0.25 CUSTOM_W_MID=0.75) are byte-identical to the build without them, stdout
     included; refusals: non-numbers, out of [0, 2], nan / inf, twice, VARIANT other than 'CUSTOM', FRAG_ON_V2 False
     (MIDX_NEURAL_W), CUSTOM as a KEY=VALUE setting;
  2. each build vs its base: changed cells exactly 0 (title suffix + note), 1 (when CUSTOM changes / MIDX_NEURAL_W is
     given) and the final layer (MIDX_NEURAL_W only); cell 1: only the CUSTOM line (old text -> the new one) and the two
     MIDX_NEURAL_W lines right after PF_MID_W's, every other statement ast-identical; notebook metadata: only
     casmi.ours.weights added; kernel-metadata: id / title only; CUSTOM / MIDX_NEURAL_W stored only by cell 1 (no
     `global`, absent from the frozen notebook);
  3. pinned-midx diff check: the final-layer cell of every build (the existing kernels' builds included) against
     midx_cell15.txt (SHA-256 pinned) has exactly the expected hunks (midx line range + SHA-256 of the new lines): the
     four of every build, the two F3 hunks of ELIG builds, and with MIDX_NEURAL_W the header, _score_cand and log-line
     hunks; with MIDX_NEURAL_W, after the name MIDX_NEURAL_W is replaced by the constant 0.75 the ast of pxm_merge equals
     the base's (the literal became a global read, nothing else) and _pxm_main differs only in its 'Final layer:' print;
     MIDX_NEURAL_W is a free global of _score_cand, read twice in the cell;
  4. the consensus path (text of our build): cell 9 copies CUSTOM w_top / w_mid into presets['CUSTOM']; the pinned
     consensus.py orders rank 1 by w_top and ranks 2-25 by w_mid (leader_safe_order); the final layer reads
     variants/submission_CUSTOM.csv;
  --regen: every existing pub-variant slug (all 15 kernels/ dirs with the ISO cell) regenerated with git HEAD's scripts/
     and with the working ones into throwaway folders, byte-compared (stdout too), and compared with kernels/<slug>; the
     SENS swap kernels (SWAP, test_sens_swap_build.py) too once git HEAD's sens.py has the swap mode.
"""
import ast
import copy
import difflib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import symtable
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(REPO, 'scripts', 'make_pub_variant.py')


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # our own generator / test code
    return m


W = _mod('pub_variant_wm_t', 'wm.py')
TSB = _mod('pub_variant_sens_build_wt', 'test_sens_build.py')     # gen / load / diff_cells / frozen_of / modules_of
gen, load, diff_cells, frozen_of = TSB.gen, TSB.load, TSB.diff_cells, TSB.frozen_of
PUBM = ['OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0']
SENS5 = ['SENS=5', 'SENS_MODE=raw']
EXISTING = dict(TSB.EXISTING, **{'casmi26-pube1-sens': ['ELIG=1'] + SENS5 + PUBM,
                                 'casmi26-pubm-eng3a-sens': ['ENG3_W=1.0'] + SENS5 + PUBM})
WM100 = 'casmi26-pubm-sens-wm100'
WM100_ARGS = SENS5 + ['CUSTOM_W_MID=1.0', 'MIDX_NEURAL_W=1.0'] + PUBM
SWAP = {  # SENS_MODE swap (seed-swap placebo, docs/plan.md NOV 3 SLOTS; checked by test_sens_swap_build.py)
    'casmi26-pubm-swap1': ['SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023]'] + PUBM,
    'casmi26-pubm-swap2': ['SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20263023]'] + PUBM,
    'casmi26-pube1-swap1': ['ELIG=1', 'SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023]'] + PUBM,
    'casmi26-pube1-swap2': ['ELIG=1', 'SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20263023]'] + PUBM}
NE = {'casmi26-pube1-ne': ['ELIG=1', 'NE480=1'] + PUBM}   # NE480 builds (checked by test_ne480_build.py)
SWAP_MODES = "MODES = ('raw', 'z', 'swap')"            # the sens.py line that marks a generator with the swap mode
BUILDS = {  # slug: (args, base slug, base args, w_top, w_mid, midx_w, elig)
    WM100: (WM100_ARGS, 'casmi26-pubm-sens', SENS5 + PUBM, 0.25, 1.0, 1.0, False),
    'casmi26-zz-midonly': (SENS5 + ['CUSTOM_W_MID=1.0'] + PUBM, 'casmi26-pubm-sens', SENS5 + PUBM, 0.25, 1.0, None, False),
    'casmi26-zz-midx075': (SENS5 + ['MIDX_NEURAL_W=0.75'] + PUBM, 'casmi26-pubm-sens', SENS5 + PUBM, 0.25, 0.75, 0.75,
                           False),
    'casmi26-zz-top05': (['CUSTOM_W_TOP=0.5', 'MIDX_NEURAL_W=0.5'] + PUBM, 'casmi26-pubm', PUBM, 0.5, 0.75, 0.5, False),
    'casmi26-zz-e1wm': (['ELIG=1'] + SENS5 + ['CUSTOM_W_MID=1.0', 'MIDX_NEURAL_W=1.0'] + PUBM, 'casmi26-pube1-sens',
                        ['ELIG=1'] + SENS5 + PUBM, 0.25, 1.0, 1.0, True),
    'casmi26-zz-e3wm': (['ENG3_W=1.0', 'MIDX_NEURAL_W=1.25'] + PUBM, 'casmi26-pubm-eng3a', ['ENG3_W=1.0'] + PUBM, 0.25,
                        0.75, 1.25, False),
    'casmi26-zz-bound': (['CUSTOM_W_TOP=2', 'CUSTOM_W_MID=0', 'MIDX_NEURAL_W=0'] + PUBM, 'casmi26-pubm', PUBM, 2.0, 0.0,
                         0.0, False)}
FINAL_ID = 'busyaprime-variant-m-merge'
OLD_CUSTOM = 'CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)'
WM100_CUSTOM = 'CUSTOM = dict(w_top=0.25, w_mid=1.0, rescue=0)  # ours: w_mid 0.75 -> 1.0 (scripts/make_pub_variant.py CUSTOM_W_MID=1.0)'
WM100_MIDX = ["MIDX_NEURAL_W = 1.0           # ours: neural weight w of the final layer's ranks 2-25 score",
              '                              # 1/(3 + b) + w/(3 + n) + prior (midx: the literal 0.75; the final layer reads this)']
PF_MID_LINES = ['PF_MID_W = 0.0                # ours: PF-syn in the ranks 2-25 neural rank of the final layer, per molecule',
                '                              # (1 - PF_MID_W) z(neural_score_saner) + PF_MID_W z(pf_fz); 0 = midx exactly']

# ---- 3. the expected hunks of the final layer against the pinned midx cell 15 (midx_cell15.txt) ----
# (first midx line, last midx line (first - 1 for an insertion), SHA-256[:16] of the new lines, what)
HUNKS_EVERY = [(2, 2, '307dc9e90f58f91e', 'header: box characters repaired + our two comment lines'),
               (25, 24, 'f1c508abff332b40', 'PF_MID_W hook _mid_neural_score inserted before pxm_merge'),
               (57, 57, 'e6d9edd2881cdc52', 'comment: en dash repaired'),
               (158, 158, 'd7286650d6b6cd47', 'neural rank from _mid_neural_score(g)')]
HUNKS_ELIG = [(16, 16, '7ad8e52382303ad6', 'ELIG F3: src_spectral_library column only with SPEC_W'),
              (19, 19, 'dc127bcc992ba7db', 'ELIG F3: spec_map only with SPEC_W')]
HUNKS_MIDX = [(2, 2, 'fab44497fa152896', 'header: box characters repaired + our comment lines naming both hooks'),
              (67, 67, '1b0e26bc23340b98', 'MIDX_NEURAL_W in place of the literal 0.75 (+ comment line)'),
              (230, 230, '73e1627292dcdfe1', "'Final layer:' log line names MIDX_NEURAL_W")]
MIDX_OLD = {67: '                    return 1.0 / (3.0 + br) + 0.75 / (3.0 + nr) + POP_W * (p_val / 20.0) + (SPEC_W if '
                'lib_hit else 0.0) * s_val + NP_W * np_val\n',
            230: "          f'library-hit molecules {rep.get(\"lib_hit_molecules\")}, POP_W {POP_W}, NP_W {NP_W}')\n"}
MIDX_NEW = {67: ["                    # ours: MIDX_NEURAL_W (cell 1) in place of midx's literal 0.75 (0.75 = midx exactly)\n",
                 '                    return 1.0 / (3.0 + br) + MIDX_NEURAL_W / (3.0 + nr) + POP_W * (p_val / 20.0) + (SPEC_W '
                 'if lib_hit else 0.0) * s_val + NP_W * np_val\n'],
            230: ["          f'library-hit molecules {rep.get(\"lib_hit_molecules\")}, POP_W {POP_W}, NP_W {NP_W}, '\n",
                  "          f'MIDX_NEURAL_W {MIDX_NEURAL_W}')\n"]}


def expected_hunks(midx_w, elig):
    exp = [h for h in HUNKS_EVERY if not (midx_w is not None and h[0] == 2)]
    exp += HUNKS_ELIG if elig else []
    exp += HUNKS_MIDX if midx_w is not None else []
    return sorted(exp)


def midx_text():
    t = open(os.path.join(HERE, 'midx_cell15.txt'), encoding='utf-8', newline='').read()
    assert hashlib.sha256(t.encode()).hexdigest() == '06093f589a8de6a5cf0f7cb1cd36f3d7cb6af8555f1528a2e0624bf3f85db42d'
    return t


def hunks(cell_src):
    a, b = midx_text().splitlines(True), cell_src.splitlines(True)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != 'equal':
            out.append((i1 + 1, i2, hashlib.sha256(''.join(b[j1:j2]).encode()).hexdigest()[:16], a[i1:i2], b[j1:j2]))
    return out


def check_final_hunks(src, midx_w, elig):
    got = hunks(src)
    exp = expected_hunks(midx_w, elig)
    assert [g[:3] for g in got] == [e[:3] for e in exp], ([g[:3] for g in got], [e[:3] for e in exp])
    if midx_w is not None:
        for g in got:
            if g[0] in MIDX_OLD:
                assert g[3] == [MIDX_OLD[g[0]]] and g[4] == MIDX_NEW[g[0]], g
    return [dict(midx_lines=f'{g[0]}-{g[1]}', new_lines=len(g[4]), sha=g[2], what=e[3]) for g, e in zip(got, exp)]


# ---- 3. ast: the hook is the literal 0.75 turned into a global read ----
class _Const(ast.NodeTransformer):
    def __init__(self, value):
        self.value = value

    def visit_Name(self, node):
        if node.id == 'MIDX_NEURAL_W':
            assert isinstance(node.ctx, ast.Load), 'MIDX_NEURAL_W assigned in the final layer'
            return ast.copy_location(ast.Constant(self.value), node)
        return node


def top_defs(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def check_final_ast(src, base_src):
    """The build's final layer vs the base's: MIDX_NEURAL_W := 0.75 gives the base's pxm_merge; _pxm_main differs only
    in one print; every other top-level statement identical; MIDX_NEURAL_W a free global of _score_cand."""
    t, b = ast.parse(src), ast.parse(base_src)
    reads = [n for n in ast.walk(t) if isinstance(n, ast.Name) and n.id == 'MIDX_NEURAL_W']
    assert len(reads) == 2 and all(isinstance(n.ctx, ast.Load) for n in reads), len(reads)
    assert not [n for n in ast.walk(t) if isinstance(n, ast.Global) and 'MIDX_NEURAL_W' in n.names]
    t75 = _Const(0.75).visit(copy.deepcopy(t))
    td, bd = top_defs(t75), top_defs(b)
    assert list(td) == list(bd), (list(td), list(bd))
    assert ast.dump(td['pxm_merge']) == ast.dump(bd['pxm_merge']), 'pxm_merge differs beyond the literal'
    for name in td:
        if name not in ('pxm_merge', '_pxm_main'):
            assert ast.dump(td[name]) == ast.dump(bd[name]), name
    other = lambda tree: [ast.dump(n) for n in tree.body if not isinstance(n, ast.FunctionDef)]
    assert other(t75) == other(b), 'module-level statements differ'
    ms, mb = td['_pxm_main'].body, bd['_pxm_main'].body
    assert len(ms) == len(mb)
    diff = [(x, y) for x, y in zip(ms, mb) if ast.dump(x) != ast.dump(y)]
    assert len(diff) == 1, len(diff)
    x, y = diff[0]
    assert isinstance(x, ast.Expr) and isinstance(x.value, ast.Call) and getattr(x.value.func, 'id', '') == 'print' and \
        isinstance(y, ast.Expr) and getattr(y.value.func, 'id', '') == 'print', ast.dump(x)[:200]
    ux, uy = ast.unparse(x), ast.unparse(y)
    assert uy[:-2].startswith('print(f') and 'Final layer: outsider fills' in uy and \
        ux == uy[:-2] + ', MIDX_NEURAL_W {0.75}' + uy[-2:], ux
    # the score: the literal of the base, the global here
    sc = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_score_cand')
    ret = lambda tree: next(n for n in sc(tree).body[0].body if isinstance(n, ast.Return))
    rb, rs = ast.unparse(ret(b)), ast.unparse(ret(t))
    assert '1.0 / (3.0 + br) + 0.75 / (3.0 + nr) + POP_W' in rb and rs == rb.replace('0.75 / (3.0', 'MIDX_NEURAL_W / (3.0'), rs
    # symtable: MIDX_NEURAL_W is a free global inside _score_cand (nested in pxm_merge), not bound in the cell
    top = symtable.symtable(src, 'final', 'exec')
    assert 'MIDX_NEURAL_W' not in {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    pm = next(c for c in top.get_children() if c.get_name() == 'pxm_merge')
    scs = next(c for c in pm.get_children() if c.get_name() == '_score_cand')
    s = scs.lookup('MIDX_NEURAL_W')
    assert s.is_global() and s.is_referenced() and not s.is_assigned() and not s.is_local(), s
    for c in (pm, next(c for c in top.get_children() if c.get_name() == '_pxm_main')):
        assert not ('MIDX_NEURAL_W' in [x.get_name() for x in c.get_symbols()] and c.lookup('MIDX_NEURAL_W').is_assigned())
    return dict(pxm_merge_equal_at_075=True, pxm_main_print_only=True)


# ---- 2. cell 1 ----
def assigns(src):
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out.setdefault(n.targets[0].id, []).append(n)
    return out


def check_cell1(c1, b1, w_top, w_mid, midx_w):
    removed = [l[2:] for l in difflib.ndiff(b1.split('\n'), c1.split('\n')) if l.startswith('- ')]
    added = [l[2:] for l in difflib.ndiff(b1.split('\n'), c1.split('\n')) if l.startswith('+ ')]
    changed = W.custom_changed(w_top, w_mid)
    assert removed == ([OLD_CUSTOM] if changed else []), removed
    exp_added = []
    if changed:
        m = re.fullmatch(r"CUSTOM = dict\(w_top=([0-9.]+), w_mid=([0-9.]+), rescue=0\)  # ours: (.+) "
                         r"\(scripts/make_pub_variant\.py (.+)\)", added[0])
        assert m and (float(m.group(1)), float(m.group(2))) == (w_top, w_mid), added[0]
        exp_added.append(added[0])
    if midx_w is not None:
        exp_added += [f'MIDX_NEURAL_W = {midx_w!r}'.ljust(29) + WM100_MIDX[0][29:], WM100_MIDX[1]]
    assert added == exp_added, (added, exp_added)
    lines = c1.split('\n')
    if changed:
        assert lines.index(added[0]) == b1.split('\n').index(OLD_CUSTOM)          # in place
    if midx_w is not None:
        i = lines.index(exp_added[-2])
        assert lines[i - 1] == PF_MID_LINES[1] and lines[i - 2].startswith('PF_MID_W = '), lines[i - 2:i]
    a, b = assigns(c1), assigns(b1)
    assert set(a) - set(b) == ({'MIDX_NEURAL_W'} if midx_w is not None else set()) and not set(b) - set(a)
    for k in b:
        if k != 'CUSTOM':
            assert [ast.dump(n) for n in a[k]] == [ast.dump(n) for n in b[k]], k
    if midx_w is not None:
        v = ast.literal_eval(a['MIDX_NEURAL_W'][0].value)
        assert type(v) is float and v == midx_w and len(a['MIDX_NEURAL_W']) == 1
    assert W.custom_of(c1) == (w_top, w_mid, 0) and W.custom_of(b1) == (W.W_TOP, W.W_MID, 0)
    nonassign = lambda s: [ast.dump(n) for n in ast.parse(s).body if not isinstance(n, ast.Assign)]
    assert nonassign(c1) == nonassign(b1)
    vals = {k: ast.literal_eval(v[0].value) for k, v in a.items() if k in ('VARIANT', 'FRAG_ON_V2', 'RRF_OFFSET')}
    assert vals == dict(VARIANT='CUSTOM', FRAG_ON_V2=True, RRF_OFFSET=3.0), vals
    return dict(removed=removed, added=added)


def stores(nb, names=('CUSTOM', 'MIDX_NEURAL_W')):
    """(cell, name) of every assignment / global declaration of names in the notebook's code cells and frozen cells."""
    out = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = c['source']
        if "FROZEN_B64 = '" in s:
            b = s.index("FROZEN_B64 = '") + 14
            s = s[:b] + s[s.index("'\n", b):]
        for n in ast.walk(ast.parse(s)):
            if isinstance(n, ast.Name) and n.id in names and not isinstance(n.ctx, ast.Load):
                out.append((i, n.id))
            if isinstance(n, ast.Global) and set(n.names) & set(names):
                out.append((i, 'global'))
    fz, _ = frozen_of(nb)
    for i, c in enumerate(fz['cells']):
        t = ''.join(c['source'])
        assert not any(re.search(rf'\b{x}\b', t) for x in names), ('frozen', i)
    return out


def check_build(d, slug, base, w_top, w_mid, midx_w, elig):
    nb, meta = load(d, slug)
    bnb, bmeta = load(d, base)
    cells, bcells = nb['cells'], bnb['cells']
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            ast.parse(c['source'], f'{slug} cell {i}')
    ids = [c.get('id') for c in cells]
    fi = ids.index(FINAL_ID)
    assert ids == [c.get('id') for c in bcells] and fi == (17 if 'ours-sens-ensemble' in ids else 16), ids
    changed = W.custom_changed(w_top, w_mid)
    exp = [0] + ([1] if changed or midx_w is not None else []) + ([fi] if midx_w is not None else [])
    cd = diff_cells(nb, bnb)
    assert cd == exp, (cd, exp)
    for i in cd:
        assert {k: v for k, v in cells[i].items() if k != 'source'} == {k: v for k, v in bcells[i].items() if k != 'source'}
    rep = dict(changed_cells=cd, final_cell=fi)
    rep['cell1'] = check_cell1(cells[1]['source'], bcells[1]['source'], w_top, w_mid, midx_w)
    # markdown cell 0: title suffix + note after the title, the rest the base's
    c0, b0 = cells[0]['source'], bcells[0]['source']
    t0 = b0[:b0.index('\n\n')]
    note = W.note(w_top, w_mid, midx_w, fi)
    assert c0 == t0 + f' + fusion weights ({W._desc(w_top, w_mid, midx_w)})\n\n' + note + b0[len(t0) + 2:], slug
    assert f'(cell {fi})' in note and ('MIDX_NEURAL_W ' + repr(midx_w) in note if midx_w is not None else
                                        'unchanged, ranks 2-25 re-sorted by 1/(3 + b) + 0.75/(3 + n)' in note)
    rep['title'] = c0.split('\n')[0]
    # final layer: pinned-midx hunks; ast equivalence at 0.75
    rep['final_hunks'] = check_final_hunks(cells[fi]['source'], midx_w, elig)
    check_final_hunks(bcells[fi]['source'], None, elig)
    if midx_w is not None:
        rep['final_ast'] = check_final_ast(cells[fi]['source'], bcells[fi]['source'])
    else:
        assert cells[fi]['source'] == bcells[fi]['source']
    # only cell 1 stores CUSTOM / MIDX_NEURAL_W
    st = stores(nb)
    assert st == [(1, 'CUSTOM')] + ([(1, 'MIDX_NEURAL_W')] if midx_w is not None else []), st
    # metadata
    m, mb = json.loads(json.dumps(nb['metadata'])), json.loads(json.dumps(bnb['metadata']))
    rec = m['casmi']['ours'].pop('weights')
    assert m == mb, 'notebook metadata beyond casmi.ours.weights'
    assert rec == W.record(w_top, w_mid, midx_w, fi) and rec['custom'] == dict(w_top=w_top, w_mid=w_mid, rescue=0)
    assert {k: v for k, v in meta.items() if k not in ('id', 'title')} == \
        {k: v for k, v in bmeta.items() if k not in ('id', 'title')} and meta['id'] == f'bryantyu/{slug}'
    if 'ours-sens-ensemble' in ids:
        fz, sha = frozen_of(nb)
        assert sha == frozen_of(bnb)[1] and cells[ids.index('ours-sens-ensemble')] == bcells[ids.index('ours-sens-ensemble')]
    return rep, nb


def consensus_path(nb):
    """Static: CUSTOM -> presets['CUSTOM'] (cell 9) -> consensus.leader_safe_order (pinned); final layer reads it."""
    cells = nb['cells']
    c9 = cells[next(i for i, c in enumerate(cells) if c.get('id') == 'ac4866c0')]['source']
    assert "        if VARIANT == 'CUSTOM':\n            presets['CUSTOM'] = dict(w_top=float(CUSTOM['w_top']), " \
           "w_mid=float(CUSTOM['w_mid']),\n" in c9
    assert 'evidence, BASELINE, VARIANTS_DIR, presets, RRF_OFFSET,' in c9
    cons = TSB.modules_of(nb)['consensus.py']
    for t in ("order = leader_safe_order(g.neural_score.values, cfg['w_top'], cfg['w_mid'], offset)",
              "first = np.asarray(rrf_order(s, w_top, offset), dtype=int)",
              "rest = [int(i) for i in rrf_order(s, w_mid, offset) if int(i) != lead]",
              "    if w_mid == w_top:\n        return first\n",
              "path = outdir / f'submission_{name}.csv'"):
        assert t in cons, t
    assert 'MODULE_SHA256' in TSB.modules_of(nb) and 'consensus.py' in TSB.modules_of(nb)['MODULE_SHA256']
    fl = cells[next(i for i, c in enumerate(cells) if c.get('id') == FINAL_ID)]['source']
    assert "v_path, c_path = ROOT / 'variants/submission_V2.csv', ROOT / 'variants/submission_CUSTOM.csv'" in fl
    assert '            rest.sort(key=_score_cand, reverse=True)\n' in fl
    return dict(cell9='presets CUSTOM from cell 1', consensus='rank 1 w_top, ranks 2-25 w_mid', final_reads='CUSTOM file')


def regen(out_head, out_work, extra=None):
    """Every existing slug (+ extra when git HEAD's scripts/ have wm.py): HEAD scripts/ vs the working ones."""
    raw = subprocess.run(['git', '-C', REPO, 'archive', 'HEAD', 'scripts'], capture_output=True, check=True).stdout
    res = {}
    with tempfile.TemporaryDirectory() as root:
        tarfile.open(fileobj=io.BytesIO(raw)).extractall(root, filter='data')
        os.symlink(os.path.join(REPO, 'kernels'), os.path.join(root, 'kernels'))
        head_gen = os.path.join(root, 'scripts', 'make_pub_variant.py')
        todo = dict(EXISTING, **(extra if os.path.isfile(os.path.join(root, 'scripts', 'pub_variant', 'wm.py')) else {}))
        if SWAP_MODES in open(os.path.join(root, 'scripts', 'pub_variant', 'sens.py'), encoding='utf-8').read():
            todo.update(SWAP)
        for slug, args in todo.items():
            rc1, o1 = gen(slug, args, out_head, head_gen)
            rc2, o2 = gen(slug, args, out_work)
            assert rc1 == 0 and rc2 == 0, (slug, o1[-300:], o2[-300:])
            files = ('notebook.ipynb', 'kernel-metadata.json')
            rd = lambda *p: open(os.path.join(*p), 'rb').read()
            res[slug] = dict(head_equals_working=all(rd(out_head, slug, f) == rd(out_work, slug, f) for f in files),
                             stdout_equal=o1.replace(out_head, '<out>') == o2.replace(out_work, '<out>'),
                             equals_committed=all(rd(REPO, 'kernels', slug, f) == rd(out_work, slug, f) for f in files))
            nb, _ = load(out_work, slug)                 # 3. the pinned-midx hunks of every existing build
            fi = [c.get('id') for c in nb['cells']].index(FINAL_ID)
            res[slug]['final_hunks_ok'] = bool(check_final_hunks(nb['cells'][fi]['source'], 1.0 if slug == WM100 else None,
                                                                 slug.startswith('casmi26-pube')))
    return res


def main():
    out = {}
    pub = {s for s in os.listdir(os.path.join(REPO, 'kernels'))
           if os.path.isfile(os.path.join(REPO, 'kernels', s, 'notebook.ipynb')) and
           'ours-iso-rerank' in open(os.path.join(REPO, 'kernels', s, 'notebook.ipynb'), encoding='utf-8').read()}
    assert pub - {WM100} - set(SWAP) - set(NE) == set(EXISTING), sorted((pub - set(SWAP) - set(NE)) ^ set(EXISTING))
    with tempfile.TemporaryDirectory() as d:
        bases = {b: a for _, (_, b, a, *_r) in BUILDS.items()}
        for slug, args in list(bases.items()) + [(s, v[0]) for s, v in BUILDS.items()]:
            rc, log = gen(slug, args, d)
            assert rc == 0, (slug, log)
            if slug in BUILDS:
                out.setdefault('generator_lines', {})[slug] = [l for l in log.split('\n') if l.startswith('WEIGHTS')]
            else:
                assert 'WEIGHTS' not in log, slug
        # the defaults given explicitly: byte-identical, stdout included
        for slug, args in (('casmi26-pubm-sens', SENS5 + PUBM + ['CUSTOM_W_TOP=0.25', 'CUSTOM_W_MID=0.75']),
                           ('casmi26-pubm', PUBM + ['CUSTOM_W_MID=0.75'])):
            rc, log = gen(slug, args, os.path.join(d, 'dflt'))
            rc0, log0 = gen(slug, [a for a in args if not a.startswith('CUSTOM_W_')], os.path.join(d, 'dflt0'))
            assert rc == rc0 == 0 and log.replace(os.path.join(d, 'dflt'), '<o>') == log0.replace(os.path.join(d, 'dflt0'), '<o>')
            for f in ('notebook.ipynb', 'kernel-metadata.json'):
                assert open(os.path.join(d, 'dflt', slug, f), 'rb').read() == open(os.path.join(d, slug, f), 'rb').read(), f
        out['defaults_byte_identical'] = True
        refusals = ((['CUSTOM_W_MID=x'], 'a number in'), (['CUSTOM_W_MID=2.5'], 'a finite weight in'),
                    (['CUSTOM_W_TOP=-0.1'], 'a finite weight in'), (['MIDX_NEURAL_W=nan'], 'a finite weight in'),
                    (['MIDX_NEURAL_W=inf'], 'a finite weight in'), (['MIDX_NEURAL_W=2.01'], 'a finite weight in'),
                    (['CUSTOM_W_MID=1', 'CUSTOM_W_MID=1.5'], 'given twice'),
                    (['MIDX_NEURAL_W=1', 'MIDX_NEURAL_W=1'], 'given twice'),
                    (['CUSTOM_W_MID=1.0', 'VARIANT=V2'], "needs to be 'CUSTOM'"),
                    (['CUSTOM_W_TOP=0.5', 'VARIANT=V4'], "needs to be 'CUSTOM'"),
                    (['MIDX_NEURAL_W=1.0', 'VARIANT=V1'], "needs to be 'CUSTOM'"),
                    (['MIDX_NEURAL_W=1.0', 'FRAG_ON_V2=False'], 'FRAG_ON_V2 = True'),
                    (['CUSTOM=1'], 'not a cell-1 setting'))
        for args, why in refusals:
            rc, log = gen('casmi26-zz-refused', args + PUBM, os.path.join(d, 'refused'))
            assert rc != 0 and why in log, (args, rc, log[-400:])
            assert not os.path.exists(os.path.join(d, 'refused', 'casmi26-zz-refused')), args
        out['refusals'] = len(refusals)
        for slug, (args, base, _, w_top, w_mid, midx_w, elig) in BUILDS.items():
            rep, nb = check_build(d, slug, base, w_top, w_mid, midx_w, elig)
            out[slug] = rep
        wnb, _ = load(d, WM100)
        out['consensus_path'] = consensus_path(wnb)
        assert out[WM100]['cell1']['added'] == [WM100_CUSTOM] + WM100_MIDX, out[WM100]['cell1']
        kd = os.path.join(REPO, 'kernels', WM100)
        if os.path.isdir(kd):
            out['kernels_dir_equals_build'] = all(open(os.path.join(kd, f), 'rb').read() ==
                                                  open(os.path.join(d, WM100, f), 'rb').read()
                                                  for f in ('notebook.ipynb', 'kernel-metadata.json'))
            assert out['kernels_dir_equals_build']
    if '--regen' in sys.argv:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            out['regen'] = regen(a, b, {WM100: WM100_ARGS})
            assert all(v['head_equals_working'] and v['stdout_equal'] and v['final_hunks_ok']
                       for v in out['regen'].values()), out['regen']
    print(json.dumps(out, indent=1, default=sorted))
    print('wm build checks OK')


if __name__ == '__main__':
    main()
