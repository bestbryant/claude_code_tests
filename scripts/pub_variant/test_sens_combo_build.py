"""Static checks of the SENS builds on the ELIG=1 and ENG3 bases (casmi26-pube1-sens, casmi26-pubm-eng3a-sens) made by
scripts/make_pub_variant.py ELIG=1 / ENG3_W=w + SENS=K (ours).

python scripts/pub_variant/test_sens_combo_build.py [--regen]

Nothing of the public pipeline is executed: notebook and frozen cells are only parsed. Executed are only OUR generated
functions, extracted from the build (the ELIG block of cell 8: literal constants + function definitions, checked with
ast; sens_source_dir of the SENS cell), on synthetic /kaggle/input trees in temporary folders.
  1. builds the bases (casmi26-pube1, casmi26-pubm-eng3a) and the combos (pube1-sens: ELIG=1 SENS=5 raw; eng3a-sens:
     ENG3_W=1.0 SENS=5 raw; ELIG=1 SENS=3 z; ENG3_W=0.5 SENS=2 z) with the working generator; refusals: SENS with ELIG=0,
     ELIG=1 + ENG3_W, OURS / PF_MID_W / ISO_ALPHA;
  2. combo vs its base: the inserted cell sits between PF-syn and the driver and is sens_cell.py (eng3) or sens_cell.py +
     SOURCE_ELIG1 (elig1); the frozen notebook (FROZEN_SHA256 / base64) and cell 9 are the base's; cell 1 = base + the SENS
     lines; driver = base + the SENS call; section 4 renumbered; cell 8 = base (eng3) or base + the kernel-source edits +
     the driver's new number (elig1); cell 0 changes only where documented; elig1 sections 5 / 6 / 8; notebook metadata
     (casmi.ours.sens, final_layer / iso numbers, elig1: casmi.elig kernel_sources / open / closed / cells edited) and
     kernel-metadata (kernel_sources added beside the base's dataset_sources); driver order run_public_pipeline <
     score_final_lists < score_outside_candidates < ours_pf_blend < ours_seed_ensemble < build_consensus; the patched
     frozen cells (ELIG 32 / 39 / 45, ENG3 32 / 39) all run inside run_public_pipeline, before the evidence export SENS
     reads; every global the SENS cell reads is bound before the call;
  3. elig1 mount check (cell 8, our code): with the kernel output mounted in either layout, elig0_setup (IS_RERUN: no
     hashing) prints 'kernel sources mounted 1 / 1 | detached ones present: none | extra mounts: none' and no WARNING;
     another notebook output is an EXTRA MOUNT; without it 'kernel sources mounted 0 / 1 (ABSENT ...)'; pube1's own cell 8
     flags the kernel output as EXTRA MOUNT (the edit is needed);
  4. SENS source (our code): elig1 takes final_models_sha256.json only from notebooks/bryantyu/casmi26-saner-seeds or
     /kaggle/input/casmi26-saner-seeds (a dataset of that name, another folder: FileNotFoundError); eng3 keeps the
     published preference rule (sens_cell.py unchanged);
  5. collisions: every by-name lookup of the combo (frozen + notebook cells) globbed on the casmi26-saner-seeds output tree
     (both layouts): only the documented final_model.pt; the SENS lookup (final_models_sha256.json) on every other mount
     of the build (our two datasets, the upstream datasets / pubm container from the v43 rows, sanpier's bundle names,
     the local copies of casmi26-code / -models for eng3): no hit;
  --regen: every existing pub-variant slug regenerated with git HEAD's scripts/ and with the working ones, byte-compared;
     the kernels/ copies of the combos equal their builds.
"""
import ast
import contextlib
import difflib
import glob
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import time
import hashlib
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # our own generator / test code
    return m


S = _mod('pub_variant_sens_ct', 'sens.py')
EL = _mod('pub_variant_elig_ct', 'elig.py')
E3 = _mod('pub_variant_eng3_ct', 'eng3.py')
TSB = _mod('pub_variant_sens_build_ct', 'test_sens_build.py')   # gen / load / diff_cells / frozen_of / globals / collisions
E3T = TSB.E3T
gen, load, diff_cells, frozen_of = TSB.gen, TSB.load, TSB.diff_cells, TSB.frozen_of
PUBM = ['OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0']
BASES = {'casmi26-pube1': ['ELIG=1'] + PUBM, 'casmi26-pubm-eng3a': ['ENG3_W=1.0'] + PUBM,
         'casmi26-zz-eng3b': ['ENG3_W=0.5'] + PUBM}
COMBOS = {  # slug: (args, base slug, base kind, k, mode)
    'casmi26-pube1-sens': (['ELIG=1', 'SENS=5', 'SENS_MODE=raw'] + PUBM, 'casmi26-pube1', 'elig1', 5, 'raw'),
    'casmi26-pubm-eng3a-sens': (['ENG3_W=1.0', 'SENS=5', 'SENS_MODE=raw'] + PUBM, 'casmi26-pubm-eng3a', 'eng3', 5, 'raw'),
    'casmi26-zz-e1sens3z': (['ELIG=1', 'SENS=3', 'SENS_MODE=z'] + PUBM, 'casmi26-pube1', 'elig1', 3, 'z'),
    'casmi26-zz-e3bsens2z': (['ENG3_W=0.5', 'SENS=2', 'SENS_MODE=z'] + PUBM, 'casmi26-zz-eng3b', 'eng3', 2, 'z')}
# lookups that only enumerate mounts (ELIG cell 8: _elig_extra_mounts INPUT/{datasets,notebooks,models}/*/*,
# _elig_detached_present INPUT/datasets/*/<detached slug>); section 3 runs that check on the kernel output itself
ELIG_SCOPED = {'*/*': 'ELIG cell 8 mount enumeration (EXTRA MOUNT / detached checks), tested in section 3'}
KERNEL_DIRS = ('casmi26-pube1-sens', 'casmi26-pubm-eng3a-sens')
ELIG_HEAD = '# ── Eligibility (ours, elig1)'
SANPIER_FILES = ['sha256.json', 'config.json', 'model.py', 'features.py', 'train.py', 'prepare.py', 'inference.py',
                 'runtime_test.py', 'final_model.pt', 'best_validation_model.pt', 'last_validation_model.pt',
                 'validation_candidates.npz', 'validation_predictions.csv', 'training_result.json',
                 'training_history.json', 'data_audit.json', 'molecules.parquet', 'spectrum_split_manifest.parquet']


# ---- 2. combo vs base ----
def check_combo(d, slug, base_slug, kind, k, mode):
    nb, meta = load(d, slug)
    bnb, bmeta = load(d, base_slug)
    cells, bcells = nb['cells'], bnb['cells']
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            ast.parse(c['source'], f'{slug} cell {i}')
    ids = [c.get('id') for c in cells]
    ic = ids.index(S.CELL_ID)
    assert ic == 12 and ids[ic - 1] == S.PF_CELL_ID and ids[ic + 1] == S.DRIVER_ID and len(cells) == len(bcells) + 1, ids
    cell = cells[ic]
    want = S.cell_source() if kind == 'eng3' else S.splice(S.cell_source(), S.SOURCE_ELIG1, 'elig1 source')
    assert cell['source'] == want and cell['outputs'] == [] and cell['execution_count'] is None
    assert cell['metadata'] == cells[ic - 1]['metadata']
    rest = dict(nb, cells=cells[:ic] + cells[ic + 1:])
    cd = diff_cells(rest, bnb)
    i4 = next(i for i, c in enumerate(bcells) if c['cell_type'] == 'markdown' and c['source'].startswith(S.SEC4))
    md = lambda h: next(i for i, c in enumerate(bcells) if c['cell_type'] == 'markdown' and c['source'].startswith(h))
    i5, i6, i8c = md('## 5. Every submission'), md('## 6. Run it yourself'), md('## 8. Credits')
    expect = [0, 1, 8, 12, i4, i5, i6, i8c] if kind == 'elig1' else [0, 1, 12, i4]
    assert cd == expect, (cd, expect)
    # frozen notebook and stage machinery: the base's (ELIG / ENG3 patches untouched by SENS)
    fz, sha = frozen_of(nb)
    bfz, bsha = frozen_of(bnb)
    assert sha == bsha == bnb['metadata']['casmi']['frozen_source_sha256'] == nb['metadata']['casmi']['frozen_source_sha256']
    assert fz == bfz and cells[9] == bcells[9]
    # cell 1: the base's + the SENS lines (after the ISO knobs; the ENG3 line follows them)
    block = S.cell1_block(k, mode)
    assert cells[1]['source'] == bcells[1]['source'].replace(S.C1_ANCHOR, S.C1_ANCHOR + block, 1)
    vals = TSB_values(cells[1]['source'])
    assert (vals['SENS'], vals['SENS_MODE'], vals['OURS'], vals['PF_MID_W'], vals['ISO_ALPHA']) == (k, mode, False, 0.0, 0.0)
    if kind == 'elig1':
        assert vals['ELIG'] == 1 and 'ENG3_W' not in vals, vals
    else:
        assert 'ELIG' not in vals and vals['ENG3_W'] > 0, vals
        assert nb['metadata']['casmi']['ours']['eng3'] == bnb['metadata']['casmi']['ours']['eng3']
    # driver: the base's + the SENS call; the order of the stages
    drv, bdrv = cells[ic + 1]['source'], bcells[ic]['source']
    assert drv == bdrv.replace('CONSENSUS_OK, REPORT, MOVES = build_consensus()\n',
                               S.DRIVER_BLOCK + 'CONSENSUS_OK, REPORT, MOVES = build_consensus()\n', 1)
    names = ['run_public_pipeline', 'score_final_lists', 'score_outside_candidates', 'ours_pf_blend', 'ours_seed_ensemble',
             'build_consensus']
    assert all(drv.count(n + '(') == 1 for n in names) and \
        sorted(names, key=lambda n: drv.index(n + '(')) == names, 'driver order'
    # the frozen pipeline (with the ELIG / ENG3 patches) runs inside run_public_pipeline, whose export SENS reads
    c9 = cells[9]['source']
    rpp = TSB.fn_source(c9, 'run_public_pipeline')
    assert "CODE / 'runner.py', FROZEN, FROZEN_SHA256, ROOT" in rpp
    sfl = TSB.fn_source(c9, 'score_final_lists')
    assert "ROOT / 'frozen_candidate_evidence.parquet'" in sfl and "'frozen_candidate_evidence.parquet'" in \
        TSB.fn_source(cell['source'], 'ours_seed_ensemble')
    patched = nb['metadata']['casmi']['elig']['frozen_patched_cells'] if kind == 'elig1' else \
        nb['metadata']['casmi']['ours']['eng3']['frozen_patched_cells']
    assert {32, 39} <= set(patched) and (kind == 'eng3' or 45 in patched)
    # globals of the SENS cell bound before its call
    need = TSB.free_globals(cell['source'])
    assert set(need) == {'BUNDLE', 'CODE', 'INPUT', 'MODEL_PINS', 'MODULE_SHA256', 'NB_T0', 'NEURAL_OK', 'ROOT',
                         'SENS_MAX_SECONDS', 'SENS_SEEDS'}, need
    before = drv[:drv.index('ours_seed_ensemble()')]
    bound = TSB.module_level_names(before[:before.rindex('try:')])
    for i in range(ic):
        if cells[i]['cell_type'] == 'code':
            bound |= TSB.module_level_names(cells[i]['source'])
    assert not set(need) - bound, set(need) - bound
    # section 4 and the final layer / ISO numbers
    assert cells[i4 + 1]['source'] == S.splice(bcells[i4]['source'], S.RENUMBER4, 'section 4')
    assert (ids.index(S.FINAL_ID), ids.index(S.ISO_ID)) == (17, 18)
    # cell 8
    if kind == 'eng3':
        assert cells[8] == bcells[8]
    else:
        assert cells[8]['source'] == S.splice(bcells[8]['source'], EL.cell8_kernel_source_edits([S.SOURCE_KERNEL]) +
                                              S.CELL8_ELIG1, 'cell 8')
        assert cells[8]['source'].count("ELIG_KERNEL_SOURCES = ['bryantyu/casmi26-saner-seeds']") == 1
    # cell 0
    c0, b0 = cells[0]['source'], bcells[0]['source']
    seeds = S.SEEDS[:k - 1]
    if kind == 'eng3':
        assert c0 == S.md0(b0, k, mode, seeds, 'eng3', w=vals['ENG3_W'])
    else:
        assert c0 == S.md0(b0, k, mode, seeds, 'elig1', elig_statement=EL.md0_elig1(False))
    removed = [l for l in difflib.ndiff(b0.split('\n'), c0.split('\n')) if l.startswith('- ')]
    allowed_removed = ['# Variant M core', '# Eligible fork elig1 ·', '> **Eligibility build elig1**',
                       '> docs/fork_eligibility.md of our repository): elig0', '> casmi26-pubm:',
                       '>   with OURS = False, PF-syn (cell 11) and ISO (cell 17)', '>   reports any mount that',
                       '> - Sections 6 and 8 name the datasets', '>   describe the forked notebook',
                       '> depositor flag naming the NIST MS', "> model. F8, the licence",
                       '> training-structure rows under the competition terms',
                       '> **This version** (made by', '> formula-gated swaps) plus our PF_MID_W hook']
    bad = [l for l in removed if not any(l[2:].startswith(a) for a in allowed_removed)]
    assert not bad and len(removed) <= (13 if kind == 'elig1' else 3), (bad, len(removed))
    for t in ('(cell 17) is\n> cell 15 of', 'cell 18 is\n> our ISO isomer reranker', f'SENS={k} SENS_MODE={mode}',
              'bryantyu/casmi26-saner-seeds', 'score_outside_candidates, then SENS'):
        assert t in c0, t
    assert '(cell 16) is' not in c0 and 'cell 17 is\n> our ISO' not in c0
    rep = dict(cells=len(cells), vs_base=cd, cell0_lines_removed=len(removed))
    if kind == 'elig1':
        for t in ('ISO (cell 18) are off', 'the ISO cell (cell 18', 'kernel_sources', 'Apache-2.0', 'train.parquet only',
                  'final_models_sha256.json', '**Closed since casmi26-pube1:** F7', 'bit-exact',
                  'never from a dataset or another mount', ', '.join(map(str, seeds))):
            assert t in c0, t
        assert 'F7, a demonstrated retrain' not in c0 and '(cell 17,' not in c0 and 'ISO (cell 17)' not in c0
        assert cells[i5 + 1]['source'] == S.splice(bcells[i5]['source'], S.MD5_ELIG1, 's5')
        assert cells[i6 + 1]['source'] == S.splice(bcells[i6]['source'], S.MD6_ELIG1, 's6')
        assert cells[i8c + 1]['source'] == S.splice(bcells[i8c]['source'], S.credits_elig1(seeds), 'credits')
        cr = cells[i8c + 1]['source']
        assert cr.index('**Datasets attached to this build**') < cr.index('**Kernel output attached to this build**') < \
            cr.index('**Upstream lineage, not attached here.**')
        assert f'| [{S.SOURCE_KERNEL}](https://www.kaggle.com/code/{S.SOURCE_KERNEL}) |' in cr
    # metadata
    m, mb = json.loads(json.dumps(nb['metadata'])), json.loads(json.dumps(bnb['metadata']))
    rec = m['casmi']['ours'].pop('sens')
    assert list(m['casmi']['ours']) == list(mb['casmi']['ours'])
    assert m['casmi']['ours']['final_layer'] == mb['casmi']['ours']['final_layer'].replace('(cell 16)', '(cell 17)') and \
        m['casmi']['ours']['iso'] == mb['casmi']['ours']['iso'].replace('(cell 17)', '(cell 18)')
    m['casmi']['ours'].update(final_layer=mb['casmi']['ours']['final_layer'], iso=mb['casmi']['ours']['iso'])
    assert rec['k'] == k and rec['mode'] == mode and rec['extra_seeds'] == seeds and rec['base'] == kind
    if kind == 'elig1':
        e, eb = m['casmi']['elig'], mb['casmi']['elig']
        assert e.pop('kernel_sources') == [S.SOURCE_KERNEL] and 'F7' in e.pop('closed')[0]
        assert e.pop('notebook_cells_edited') == [i + (i >= 12) for i in eb.pop('notebook_cells_edited')]
        op, opb = e.pop('open'), eb.pop('open')
        assert [x[:2] for x in op] == ['F5', 'F6', 'F8'] and op[:2] == opb[:2] and op[2].startswith(opb[3]) and \
            'casmi26-saner-seeds' in op[2]
        assert 'source_rule' in rec and 'F7 closed' in rec['eligibility']
    assert m == mb, 'notebook metadata beyond casmi.ours.sens / numbers / elig fields'
    assert meta['kernel_sources'] == [S.SOURCE_KERNEL] and bmeta['kernel_sources'] == []
    assert meta['dataset_sources'] == bmeta['dataset_sources'] and meta['id'] == f'bryantyu/{slug}' and meta['title'] == slug
    assert {x: v for x, v in meta.items() if x not in ('id', 'title', 'kernel_sources')} == \
        {x: v for x, v in bmeta.items() if x not in ('id', 'title', 'kernel_sources')}
    if kind == 'elig1':
        assert meta['dataset_sources'] == EL.UPSTREAM + EL.ELIG1_DATASETS
        assert not any(S.SOURCE_KERNEL.split('/')[1] in r for r in EL.DETACHED + EL.DETACHED_ELIG1)
    else:
        assert meta['dataset_sources'][-2:] == E3.DATASETS
    return rep, nb, fz


def TSB_values(c1):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(c1).body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in
            ('SENS', 'SENS_MODE', 'OURS', 'PF_MID_W', 'ISO_ALPHA', 'ELIG', 'ENG3_W')}


# ---- 3. the elig1 mount check (our cell-8 code) on synthetic /kaggle/input trees ----
def elig_block(nb):
    """The ELIG block of cell 8 (ours) as code: literal constants and function definitions, without the final call."""
    c8 = nb['cells'][8]['source']
    assert c8.count(ELIG_HEAD) == 1 and c8.index("FROZEN_B64 = '") < c8.index(ELIG_HEAD)
    blk = c8[c8.index(ELIG_HEAD):]
    tail = 'ELIG_REPORT = elig0_setup()\n'
    assert blk.endswith(tail)
    blk = blk[:-len(tail)]
    for n in ast.parse(blk).body:          # only literal constants and defs: exec runs nothing else
        assert isinstance(n, ast.FunctionDef) or (isinstance(n, ast.Assign) and ast.literal_eval(n.value) is not None), \
            ast.dump(n)[:200]
    return blk


def make_tree(root, datasets, extra=()):
    inp = Path(root) / 'input'
    for ref in datasets:
        o, s = ref.split('/')
        (inp / 'datasets' / o / s).mkdir(parents=True, exist_ok=True)
        (inp / 'datasets' / o / s / 'x.txt').write_text('x')
    comp = inp / 'competitions' / 'enveda-CASMI26-molecule-id-mass-spectra'
    comp.mkdir(parents=True, exist_ok=True)
    for rel in extra:
        p = inp / rel
        p.mkdir(parents=True, exist_ok=True)
        (p / 'final_models_sha256.json').write_text('{}')
    work = Path(root) / 'working'
    (work / 'code').mkdir(parents=True, exist_ok=True)
    return inp, comp, work


def run_setup(blk, inp, comp, work):
    ns = dict(INPUT=inp, COMP=comp, CODE=work / 'code', ROOT=work, IS_RERUN=True, ELIG=1, SPEC_W=0.0, Path=Path,
              hashlib=hashlib, json=json, time=time, __name__='elig_block_test')
    exec(compile(blk, 'elig block (ours)', 'exec'), ns)
    ns['FROZEN_SHA256'] = ns['ELIG_FROZEN']['patched_sha256']
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rep = ns['elig0_setup']()
    return out.getvalue(), rep


def mount_checks(nb, base_nb):
    blk, bblk = elig_block(nb), elig_block(base_nb)
    mounts = EL.UPSTREAM + EL.ELIG1_DATASETS
    res = {}
    with tempfile.TemporaryDirectory() as d:
        for name, extra in (('notebooks', ['notebooks/bryantyu/casmi26-saner-seeds']), ('flat', ['casmi26-saner-seeds'])):
            inp, comp, work = make_tree(os.path.join(d, name), mounts, extra)
            log, rep = run_setup(blk, inp, comp, work)
            line = next(l for l in log.split('\n') if l.startswith('ELIG 1 |'))
            assert 'F1 datasets mounted 12 / 12 | kernel sources mounted 1 / 1 | detached ones present: none | ' \
                   'extra mounts: none | F2b' in line, line
            assert 'WARNING' not in log and 'EXTRA MOUNT' not in log and 'ABSENT' not in log, log
            assert rep['kernel_sources'] == {S.SOURCE_KERNEL: str(inp / extra[0])} and rep['kernel_sources_absent'] == []
            assert json.loads((work / 'elig1_report.json').read_text())['kernel_sources'] == rep['kernel_sources']
            res[name] = line[:160]
            binp, bcomp, bwork = make_tree(os.path.join(d, name + '_base'), mounts, extra)
            blog, _ = run_setup(bblk, binp, bcomp, bwork)                     # pube1's own cell 8 on the same tree
            assert f'ELIG EXTRA MOUNT {binp / extra[0]}' in blog and 'ELIG WARNING' in blog, blog
        inp, comp, work = make_tree(os.path.join(d, 'other'), mounts, ['notebooks/bryantyu/casmi26-saner-seeds',
                                                                       'notebooks/bryantyu/casmi26-other'])
        log, rep = run_setup(blk, inp, comp, work)
        assert f"ELIG EXTRA MOUNT {inp / 'notebooks/bryantyu/casmi26-other'}" in log and 'ELIG WARNING' in log
        assert 'kernel sources mounted 1 / 1' in log and rep['extra_mounts'] == [str(inp / 'notebooks/bryantyu/casmi26-other')]
        inp, comp, work = make_tree(os.path.join(d, 'absent'), mounts)
        log, rep = run_setup(blk, inp, comp, work)
        assert f'kernel sources mounted 0 / 1 (ABSENT: {S.SOURCE_KERNEL})' in log and \
            f'ELIG KERNEL SOURCE ABSENT {S.SOURCE_KERNEL}' in log and rep['kernel_sources_absent'] == [S.SOURCE_KERNEL]
        res['absent'] = next(l for l in log.split('\n') if l.startswith('ELIG 1 |'))[:160]
    return res


# ---- 4. the SENS source finder (our code) ----
def source_finder(cell_src):
    ns = dict(glob=glob, Path=Path)
    tree = ast.parse(cell_src)
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'SENS_SOURCE':
            ns['SENS_SOURCE'] = ast.literal_eval(n.value)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sens_source_dir')
    exec(compile(ast.get_source_segment(cell_src, fn), 'sens_source_dir (ours)', 'exec'), ns)
    return ns


def source_checks(cell_elig, cell_eng3):
    res = {}
    with tempfile.TemporaryDirectory() as d:
        cases = {'notebooks': ['notebooks/bryantyu/casmi26-saner-seeds'], 'flat': ['casmi26-saner-seeds'],
                 'decoy_first': ['a/final', 'notebooks/bryantyu/casmi26-saner-seeds'],
                 'dataset_only': ['datasets/someone/casmi26-saner-seeds'], 'other_only': ['b'], 'none': []}
        for name, rels in cases.items():
            inp, _, _ = make_tree(os.path.join(d, name), [], rels)
            got = {}
            for kind, src in (('elig1', cell_elig), ('eng3', cell_eng3)):
                ns = source_finder(src)
                ns['INPUT'] = inp
                try:
                    got[kind] = os.path.relpath(ns['sens_source_dir'](), inp)
                except FileNotFoundError as exc:
                    got[kind] = 'FileNotFoundError' + (' (kernel output)' if 'kernel output' in str(exc) else '')
            res[name] = got
    assert res['notebooks'] == dict(elig1='notebooks/bryantyu/casmi26-saner-seeds', eng3='notebooks/bryantyu/casmi26-saner-seeds')
    assert res['flat'] == dict(elig1='casmi26-saner-seeds', eng3='casmi26-saner-seeds')
    assert res['decoy_first'] == dict(elig1='notebooks/bryantyu/casmi26-saner-seeds',
                                      eng3='notebooks/bryantyu/casmi26-saner-seeds')
    assert res['dataset_only'] == dict(elig1='FileNotFoundError (kernel output)', eng3='datasets/someone/casmi26-saner-seeds')
    assert res['other_only'] == dict(elig1='FileNotFoundError (kernel output)', eng3='b')
    assert res['none'] == dict(elig1='FileNotFoundError (kernel output)', eng3='FileNotFoundError')
    return res


# ---- 5. collisions ----
def other_mount_names(kind):
    """{mount: [relative file paths]} of every mount of the build besides the seeds output (synthetic listings)."""
    rows = json.load(open(os.path.join(HERE, EL.V43_FILE)))
    cont = 'datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets/'
    out = {'bryantyu/casmi26-elig-assets': E3.ASSET_FILES, 'bryantyu/casmi26-pool-v18c': E3.POOL_FILES,
           'sanpier/casmi-spectrum-model-v1': SANPIER_FILES}
    for r in rows:
        p = r['path'].split('/kaggle/input/', 1)[-1]
        if p.startswith(cont):
            parts = p[len(cont):].split('/')
            key = parts[0] if len(parts) > 1 else '(container root)'
            rel = '/'.join(parts[1:]) if len(parts) > 1 else parts[0]
            if kind == 'elig1':
                up = [u for u in EL.UPSTREAM if u.split('/')[1] == key]
                if not up:
                    continue                # container folders not mounted in elig1
                key = up[0]
            else:
                key = 'dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'
                rel = p[len(cont):]
        elif p.startswith('datasets/'):
            parts = p.split('/')
            key, rel = f'{parts[1]}/{parts[2]}', '/'.join(parts[3:])
            if kind == 'elig1' and key not in EL.UPSTREAM:
                continue
        else:
            key, rel = 'competition', p
        out.setdefault(key, []).append(rel)
    if kind == 'eng3':                       # pubm's own two datasets (local copies)
        for ref in ('bryantyu/casmi26-code', 'bryantyu/casmi26-models'):
            loc = f'/home/user/ds/{ref.split("/")[1]}'
            if os.path.isdir(loc):
                out[ref] = [os.path.relpath(os.path.join(a, f), loc) for a, _, fs in os.walk(loc) for f in fs]
    return out


def collisions(nb, fz, kind):
    col = TSB.collision_check(nb, fz, ELIG_SCOPED if kind == 'elig1' else None)   # every lookup on the seeds tree
    assert not col['bad'], col['bad']
    names = other_mount_names(kind)
    hits = {m: [f for f in fs if os.path.basename(f) in ('final_models_sha256.json', 'seeds_report.json') or
                re.fullmatch(r'final_model_seed\d+\.pt(\.sha256)?', os.path.basename(f))] for m, fs in names.items()}
    assert not any(hits.values()), {m: h for m, h in hits.items() if h}
    if kind == 'elig1':
        assert set(names) - {'competition'} == set(EL.UPSTREAM + EL.ELIG1_DATASETS), sorted(names)
    return dict(seeds_tree=dict(patterns=len(col['patterns']), hits=sorted(col['hits'])),
                other_mounts={m: len(fs) for m, fs in sorted(names.items())})


# ---- regen: every existing slug, git HEAD scripts/ vs the working ones (test_sens_build.regen) ----
regen = TSB.regen


def main():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for slug, args in list(BASES.items()) + [(s, v[0]) for s, v in COMBOS.items()]:
            rc, log = gen(slug, args, d)
            assert rc == 0, log
            out.setdefault('generator_lines', {})[slug] = [l for l in log.split('\n') if l.startswith(('ELIG', 'ENG3', 'SENS'))]
        refusals = ((['ELIG=0', 'SENS=5'] + PUBM, 'not combined with ELIG=0'),
                    (['ELIG=1', 'ENG3_W=1.0', 'SENS=5'] + PUBM, 'not valid with ELIG'),
                    (['ELIG=1', 'SENS=5'], 'not combined with PF-syn'),
                    (['ELIG=1', 'SENS=5', 'OURS_MODE=columns', 'ISO_ALPHA=0.5'], 'not combined with PF-syn'),
                    (['ENG3_W=1.0', 'SENS=5', 'OURS=True'], 'no PF-syn re-weighting'),
                    (['ENG3_W=1.0', 'SENS=5', 'OURS_MODE=columns', 'PF_MID_W=0.5'], 'no PF-syn re-weighting'),
                    (['ELIG=1', 'SENS=1'] + PUBM, '2..5'), (['ENG3_W=1.0', 'SENS_MODE=z'] + PUBM, 'without SENS'))
        for args, why in refusals:
            rc, log = gen('casmi26-zz-refused', args, os.path.join(d, 'refused'))
            assert rc != 0 and why in log, (args, rc, log[-400:])
            assert not os.path.exists(os.path.join(d, 'refused', 'casmi26-zz-refused')), args
        out['refusals'] = len(refusals)
        for slug, (args, base, kind, k, mode) in COMBOS.items():
            rep, nb, fz = check_combo(d, slug, base, kind, k, mode)
            rep['collisions'] = collisions(nb, fz, kind)
            if kind == 'elig1' and slug == 'casmi26-pube1-sens':
                rep['mount_check'] = mount_checks(nb, load(d, base)[0])
            out[slug] = rep
        e_nb, _ = load(d, 'casmi26-pube1-sens')
        g_nb, _ = load(d, 'casmi26-pubm-eng3a-sens')
        out['source_finder'] = source_checks(e_nb['cells'][12]['source'], g_nb['cells'][12]['source'])
        for slug in KERNEL_DIRS:                  # the kernels/ copies (when written) equal these builds
            kd = os.path.join(REPO, 'kernels', slug)
            if os.path.isdir(kd):
                out[slug]['kernels_dir_equals_build'] = all(
                    open(os.path.join(kd, f), 'rb').read() == open(os.path.join(d, slug, f), 'rb').read()
                    for f in ('notebook.ipynb', 'kernel-metadata.json'))
                assert out[slug]['kernels_dir_equals_build'], slug
    if '--regen' in sys.argv:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            out['regen'] = regen(a, b)
            assert all(v['head_equals_working'] and v['stdout_equal'] for v in out['regen'].values()), out['regen']
    print(json.dumps(out, indent=1, default=sorted))
    print('sens combo build checks OK')


if __name__ == '__main__':
    main()
