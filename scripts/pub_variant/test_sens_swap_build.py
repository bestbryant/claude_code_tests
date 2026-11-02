"""Static checks of the SENS swap builds (seed-swap placebo, docs/plan.md NOV 3 SLOTS) made by scripts/make_pub_variant.py
SENS=2 SENS_MODE=swap SENS_SEEDS=[s] (ours): casmi26-pubm-swap1 / -swap2 (pubm base, seeds 20262023 / 20263023) and
casmi26-pube1-swap1 / -swap2 (ELIG=1 base).

python scripts/pub_variant/test_sens_swap_build.py [--regen]

Nothing of the notebooks is executed: cells are only parsed (ast / symtable), and the generator (our code) is run into
temporary folders; executed are only OUR functions taken from the builds (sens_source_dir of the inserted cell).
  1. builds the four swap kernels, their raw twins (the same command with SENS_MODE=raw: a K 2 ensemble of the same
     seed), the bases casmi26-pubm / casmi26-pube1 and an ENG3_W=1.0 swap build with its twin; refusals: swap with
     SENS 3 / 5, without SENS_SEEDS, with two seeds, with the published seed, with ELIG=0, with OURS / ISO_ALPHA, SENS_MODE
     without SENS;
  2. each swap build vs its raw twin: identical except cell 0 (title, SENS note, elig1 statement), cell 1 (the SENS
     block: SENS 2, SENS_MODE 'swap', SENS_SEEDS [s]), the inserted cell (= the twin's with exactly the SOURCE_SWAP
     splices) and, elig1, sections 5 and 8; notebook metadata: only casmi.ours.sens mode / rule / cell (+ SOURCE_SWAP);
     kernel-metadata: only
     id / title;
  3. each swap build vs its base: the inserted cell sits between PF-syn and the driver; driver = base + the SENS call
     (after score_final_lists / score_outside_candidates / ours_pf_blend, before build_consensus); cell 1 = base + the
     swap block with the build's seed; frozen notebook and cell 9 the base's; cell 8 the base's (pubm) or + the
     kernel-source edits (elig1); changed cells as the ensemble builds; every global the cell reads is bound before the
     call; kernel_sources = the seeds kernel, dataset_sources the base's; generator line 'SENS K 2 | mode swap |
     extra seeds [s]'; cell 0 names the build's seed and no other extra seed (elig1: but the seed-20262023 holdout
     sentence), and no averaging;
  4. collisions of the build's by-name lookups with the casmi26-saner-seeds output tree and of its SENS lookup with
     every other mount (test_sens_combo_build.collisions); the elig1 source rule of the inserted cell
     (test_sens_combo_build.source_checks on pube1-swap1 / the ENG3 swap cell);
  5. kernels/<slug> of the four (when written) equal these builds byte for byte;
  --regen: every existing pub-variant slug regenerated with git HEAD's scripts/ and with the working ones into
     throwaway folders and byte-compared, stdout too (test_wm_build.regen; the swap slugs join once HEAD has the swap
     mode).
"""
import ast
import difflib
import importlib.util
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # our own generator / test code
    return m


S = _mod('pub_variant_sens_swt', 'sens.py')
EL = _mod('pub_variant_elig_swt', 'elig.py')
TSB = _mod('pub_variant_sens_build_swt', 'test_sens_build.py')
TSC = _mod('pub_variant_sens_combo_swt', 'test_sens_combo_build.py')
TWM = _mod('pub_variant_wm_build_swt', 'test_wm_build.py')
gen, load, diff_cells, frozen_of = TSB.gen, TSB.load, TSB.diff_cells, TSB.frozen_of
PUBM = ['OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0']
SWAP = TWM.SWAP                                  # the four kernels: slug -> generator arguments
BASES = {'casmi26-pubm': PUBM, 'casmi26-pube1': ['ELIG=1'] + PUBM, 'casmi26-pubm-eng3a': ['ENG3_W=1.0'] + PUBM}
EXTRA = {'casmi26-zz-e3swap': ['ENG3_W=1.0', 'SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20263023]'] + PUBM}


def seed_of(args):
    return ast.literal_eval(next(a for a in args if a.startswith('SENS_SEEDS=')).split('=', 1)[1])[0]


def base_of(args):
    return 'elig1' if 'ELIG=1' in args else 'eng3' if any(a.startswith('ENG3_W=') for a in args) else 'pubm'


def twin_args(args):
    return [a.replace('SENS_MODE=swap', 'SENS_MODE=raw') for a in args]


def cell1_vals(c1):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(c1).body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in
            ('SENS', 'SENS_MODE', 'SENS_SEEDS', 'SENS_MAX_SECONDS', 'OURS', 'PF_MID_W', 'ISO_ALPHA', 'ELIG', 'ENG3_W')}


def md_index(cells, head):
    hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and c['source'].startswith(head)]
    assert len(hits) == 1, (head, hits)
    return hits[0]


# ---- 2. swap build vs its raw twin ----
def check_vs_twin(d, slug, args):
    s, kind = seed_of(args), base_of(args)
    nb, meta = load(d, slug)
    tnb, tmeta = load(d, slug + '-twin')
    cells, tcells = nb['cells'], tnb['cells']
    assert len(cells) == len(tcells) and [c.get('id') for c in cells] == [c.get('id') for c in tcells]
    ic = [c.get('id') for c in cells].index(S.CELL_ID)
    i5, i8c = md_index(cells, '## 5. Every submission'), md_index(cells, '## 8. Credits')
    want = [0, 1, ic] + ([i5, i8c] if kind == 'elig1' else [])
    assert diff_cells(nb, tnb) == want, (slug, diff_cells(nb, tnb), want)
    # the inserted cell: the twin's (sens_cell.py [+ SOURCE_ELIG1]) with exactly the SOURCE_SWAP splices
    assert cells[ic]['source'] == S.splice(tcells[ic]['source'], S.SOURCE_SWAP, 'swap cell vs twin')
    assert {k: v for k, v in cells[ic].items() if k != 'source'} == {k: v for k, v in tcells[ic].items() if k != 'source'}
    want_src = S.cell_source('swap')
    if kind == 'elig1':
        want_src = S.splice(want_src, S.SOURCE_ELIG1, 'elig1 source')
    assert cells[ic]['source'] == want_src
    # cell 1: only the SENS block's lines differ
    i0 = cells[1]['source'].index(S.C1_ANCHOR) + len(S.C1_ANCHOR)
    blk = cells[1]['source'][i0:].split('SENS_MAX_SECONDS = ', 1)[0]
    tblk = tcells[1]['source'][i0:].split('SENS_MAX_SECONDS = ', 1)[0]
    assert cells[1]['source'].replace(blk, '', 1) == tcells[1]['source'].replace(tblk, '', 1)      # the rest identical
    changed = [l for l in difflib.ndiff(tblk.split('\n'), blk.split('\n')) if l[:2] in ('- ', '+ ')]
    assert all(l[2:].startswith(('# ', 'SENS = ', 'SENS_MODE = ', 'SENS_SEEDS = ')) for l in changed), changed
    assert f'SENS_SEEDS = [{s}]' in blk and f'SENS_SEEDS = [{s}]' in tblk
    assert cell1_vals(cells[1]['source']) == dict(cell1_vals(tcells[1]['source']), SENS_MODE='swap')
    # sections 5 / 8 (elig1): the swap wording on the base's text
    if kind == 'elig1':
        assert 'This SENS swap build of casmi26-pube1' in cells[i5]['source'] and \
            f'SENS swap: final_model_seed{s}.pt' in cells[i8c]['source'] and \
            'here in its swap mode' in cells[i8c]['source']
    # metadata: only casmi.ours.sens mode / rule
    m, mt = json.loads(json.dumps(nb['metadata'])), json.loads(json.dumps(tnb['metadata']))
    rec, trec = m['casmi']['ours'].pop('sens'), mt['casmi']['ours'].pop('sens')
    assert m == mt, slug
    assert rec == dict(trec, mode='swap', rule=S.SWAP_RULE,                 # cell: + SOURCE_SWAP (before SOURCE_ELIG1)
                       cell=trec['cell'].replace('sens_cell.py', 'sens_cell.py + SOURCE_SWAP', 1)), (rec, trec)
    assert rec['cell'].startswith('scripts/pub_variant/sens_cell.py + SOURCE_SWAP ' +
                                  ('+ SOURCE_ELIG1 (' if kind == 'elig1' else '(')), rec['cell']
    assert (rec['k'], rec['extra_seeds'], rec['published_seed']) == (2, [s], S.PUBLISHED_SEED)
    assert {k: v for k, v in meta.items() if k not in ('id', 'title')} == \
        {k: v for k, v in tmeta.items() if k not in ('id', 'title')} and meta['id'] == f'bryantyu/{slug}' and \
        meta['title'] == slug
    return dict(vs_twin=want, cell1_lines_changed=len(changed))


# ---- 3. swap build vs its base ----
def check_vs_base(d, slug, args, base_slug):
    s, kind = seed_of(args), base_of(args)
    nb, meta = load(d, slug)
    bnb, bmeta = load(d, base_slug)
    cells, bcells = nb['cells'], bnb['cells']
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            ast.parse(c['source'], f'{slug} cell {i}')
    ids = [c.get('id') for c in cells]
    ic = ids.index(S.CELL_ID)
    assert ic == 12 and ids[ic - 1] == S.PF_CELL_ID and ids[ic + 1] == S.DRIVER_ID and len(cells) == len(bcells) + 1
    assert (ids.index(S.FINAL_ID), ids.index(S.ISO_ID)) == (17, 18)
    assert cells[ic]['metadata'] == cells[ic - 1]['metadata'] and cells[ic]['outputs'] == [] and \
        cells[ic]['execution_count'] is None
    rest = dict(nb, cells=cells[:ic] + cells[ic + 1:])
    i4 = md_index(bcells, S.SEC4)
    if kind == 'elig1':
        i5, i6, i8c = (md_index(bcells, h) for h in ('## 5. Every submission', '## 6. Run it yourself', '## 8. Credits'))
        expect = [0, 1, 8, 12, i4, i5, i6, i8c]
    else:
        expect = [0, 1, 12, i4]
    assert diff_cells(rest, bnb) == expect, (slug, diff_cells(rest, bnb), expect)
    fz, sha = frozen_of(nb)
    bfz, bsha = frozen_of(bnb)
    assert fz == bfz and sha == bsha == nb['metadata']['casmi']['frozen_source_sha256'] and cells[9] == bcells[9]
    # cell 1: the base's + the swap block with this build's seed (the comment column kept)
    block = S.cell1_block(2, 'swap')
    assert block.count(f'SENS_SEEDS = {S.SEEDS[:1]!r}') == 1
    block = block.replace(f'SENS_SEEDS = {S.SEEDS[:1]!r}', f'SENS_SEEDS = {[s]!r}')
    assert cells[1]['source'] == bcells[1]['source'].replace(S.C1_ANCHOR, S.C1_ANCHOR + block, 1)
    v = cell1_vals(cells[1]['source'])
    assert (v['SENS'], v['SENS_MODE'], v['SENS_SEEDS'], v['SENS_MAX_SECONDS'], v['OURS'], v['PF_MID_W'],
            v['ISO_ALPHA']) == (2, 'swap', [s], S.MAX_SECONDS, False, 0.0, 0.0), v
    assert (v.get('ELIG'), 'ENG3_W' in v) == ((1, False) if kind == 'elig1' else (None, kind == 'eng3')), v
    assert max(len(x) for x in block.split('\n')) <= 120
    # driver: the base's + the SENS call, in order
    drv, bdrv = cells[ic + 1]['source'], bcells[ic]['source']
    assert drv == bdrv.replace('CONSENSUS_OK, REPORT, MOVES = build_consensus()\n',
                               S.DRIVER_BLOCK + 'CONSENSUS_OK, REPORT, MOVES = build_consensus()\n', 1)
    names = ['run_public_pipeline', 'score_final_lists', 'score_outside_candidates', 'ours_pf_blend', 'ours_seed_ensemble',
             'build_consensus']
    assert all(drv.count(n + '(') == 1 for n in names) and sorted(names, key=lambda n: drv.index(n + '(')) == names
    # globals the cell reads: bound before the call
    need = TSB.free_globals(cells[ic]['source'])
    assert set(need) == {'BUNDLE', 'CODE', 'INPUT', 'MODEL_PINS', 'MODULE_SHA256', 'NB_T0', 'NEURAL_OK', 'ROOT',
                         'SENS_MAX_SECONDS', 'SENS_SEEDS'}, need
    before = drv[:drv.index('ours_seed_ensemble()')]
    bound = TSB.module_level_names(before[:before.rindex('try:')])
    for i in range(ic):
        if cells[i]['cell_type'] == 'code':
            bound |= TSB.module_level_names(cells[i]['source'])
    assert not set(need) - bound and {'SENS', 'SENS_MODE'} <= TSB.module_level_names(cells[1]['source'])
    # cell 8
    if kind == 'elig1':
        assert cells[8]['source'] == S.splice(bcells[8]['source'], EL.cell8_kernel_source_edits([S.SOURCE_KERNEL]) +
                                              S.CELL8_ELIG1, 'cell 8')
    else:
        assert cells[8] == bcells[8]
    # cell 0: the swap title / note, the build's seed only, no averaging
    c0 = cells[0]['source']
    stmt = EL.md0_elig1(False) if kind == 'elig1' else None
    assert c0 == S.md0(bcells[0]['source'], 2, 'swap', [s], kind, w=v.get('ENG3_W'), elig_statement=stmt)
    for t in (f'SENS=2 SENS_MODE=swap SENS_SEEDS=[{s}]', f'final_model_seed{s}.pt', 'placebo', 'nothing is averaged',
              '(cell 17) is\n> cell 15 of', 'cell 18 is\n> our ISO isomer reranker', 'bryantyu/casmi26-saner-seeds'):
        assert t in c0, (slug, t)
    others = [x for x in S.SEEDS if x != s]
    stray = [m.group(0) for m in re.finditer(r'\d{8}', c0) if int(m.group(0)) in others]
    allowed = c0.count('seed-20262023 holdout') if kind == 'elig1' and 20262023 in others else 0
    assert kind != 'elig1' or c0.count('seed-20262023 holdout') == 1
    assert len(stray) == allowed, (slug, stray)
    for t in ('averages', 'seed\n> ensemble', 'members\' mean', 'saner seed ensemble (K'):
        assert t not in c0, (slug, t)
    if kind == 'elig1':
        assert c0.startswith('# Eligible fork elig1 + SENS swap · ') and 'ISO (cell 18) are off' in c0 and \
            '**Closed since casmi26-pube1:** F7' in c0 and f'The SENS swap refit (seed {s})' in c0
    else:
        assert c0.split('\n', 1)[0].endswith(f'+ SENS swap: the published saner model replaced by seed {s} (placebo)')
    # metadata / kernel-metadata
    m, mb = json.loads(json.dumps(nb['metadata'])), json.loads(json.dumps(bnb['metadata']))
    rec = m['casmi']['ours'].pop('sens')
    assert rec['k'] == 2 and rec['mode'] == 'swap' and rec['extra_seeds'] == [s] and rec['rule'] == S.SWAP_RULE
    assert rec.get('base', 'pubm') == kind and ('cell 12' in rec['cell']) and 'driver cell 13' in rec['call']
    assert meta['kernel_sources'] == [S.SOURCE_KERNEL] and bmeta['kernel_sources'] == []
    assert {k: x for k, x in meta.items() if k not in ('id', 'title', 'kernel_sources')} == \
        {k: x for k, x in bmeta.items() if k not in ('id', 'title', 'kernel_sources')}
    if kind == 'elig1':
        e = m['casmi']['elig']
        assert e['kernel_sources'] == [S.SOURCE_KERNEL] and [x[:2] for x in e['open']] == ['F5', 'F6', 'F8']
    return dict(vs_base=expect), nb, fz, kind


def main():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        todo = dict(BASES)
        for slug, args in list(SWAP.items()) + list(EXTRA.items()):
            todo[slug] = args
            todo[slug + '-twin'] = twin_args(args)
        for slug, args in todo.items():
            rc, log = gen(slug, args, d)
            assert rc == 0, (slug, log)
            if slug in SWAP or slug in EXTRA:
                line = [l for l in log.split('\n') if l.startswith('SENS ')]
                assert len(line) == 1 and line[0].startswith(f'SENS K 2 | mode swap | extra seeds [{seed_of(args)}] | '
                                                             'cell inserted 12 | '), line
                out.setdefault('generator_lines', {})[slug] = line[0]
        refusals = ((['SENS=3', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023]'] + PUBM, 'SENS=2, not 3'),
                    (['SENS=5', 'SENS_MODE=swap'] + PUBM, 'SENS=2, not 5'),
                    (['SENS=2', 'SENS_MODE=swap'] + PUBM, 'name the one extra seed'),
                    (['ELIG=1', 'SENS=2', 'SENS_MODE=swap'] + PUBM, 'name the one extra seed'),
                    (['SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023,20263023]'] + PUBM, 'ONE extra seed'),
                    (['SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20261023]'] + PUBM, 'without the published'),
                    (['SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[]'] + PUBM, 'needs 1 extra seeds'),
                    (['ELIG=0', 'SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023]'] + PUBM, 'not combined with ELIG=0'),
                    (['SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023]'], 'not combined with PF-syn'),
                    (['SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023]', 'OURS_MODE=columns', 'ISO_ALPHA=0.5'],
                     'not combined with PF-syn'),
                    (['SENS_MODE=swap', 'SENS_SEEDS=[20262023]'] + PUBM, 'without SENS'),
                    (['SENS=2', 'SENS_MODE=Swap', 'SENS_SEEDS=[20262023]'] + PUBM, 'raw or z (or swap with SENS=2)'))
        for args, why in refusals:
            rc, log = gen('casmi26-zz-refused', args, os.path.join(d, 'refused'))
            assert rc != 0 and why in log, (args, rc, log[-400:])
            assert not os.path.exists(os.path.join(d, 'refused', 'casmi26-zz-refused')), args
        out['refusals'] = len(refusals)
        cells_of = {}
        for slug, args in list(SWAP.items()) + list(EXTRA.items()):
            base = {'pubm': 'casmi26-pubm', 'elig1': 'casmi26-pube1', 'eng3': 'casmi26-pubm-eng3a'}[base_of(args)]
            rep = check_vs_twin(d, slug, args)
            r2, nb, fz, kind = check_vs_base(d, slug, args, base)
            rep.update(r2)
            if slug in SWAP:
                rep['collisions'] = TSC.collisions(nb, fz, kind)
            cells_of[slug] = nb['cells'][12]['source']
            out[slug] = rep
        out['source_finder'] = TSC.source_checks(cells_of['casmi26-pube1-swap1'], cells_of['casmi26-zz-e3swap'])
        # the two seeds of a base differ only in the seed (cell 0 / 1 text, metadata extra_seeds)
        for b in ('pubm', 'pube1'):
            n1, m1 = load(d, f'casmi26-{b}-swap1')
            n2, m2 = load(d, f'casmi26-{b}-swap2')
            assert diff_cells(n1, n2) == [0, 1] + ([24] if b == 'pube1' else []), (b, diff_cells(n1, n2))
            for i in diff_cells(n1, n2):
                assert n1['cells'][i]['source'].replace('20262023', 'SEED') == \
                    n2['cells'][i]['source'].replace('20263023', 'SEED').replace('20262023', 'SEED'), (b, i)
        for slug in SWAP:                         # the kernels/ copies (when written) equal these builds
            kd = os.path.join(REPO, 'kernels', slug)
            if os.path.isdir(kd):
                out[slug]['kernels_dir_equals_build'] = all(
                    open(os.path.join(kd, f), 'rb').read() == open(os.path.join(d, slug, f), 'rb').read()
                    for f in ('notebook.ipynb', 'kernel-metadata.json'))
                assert out[slug]['kernels_dir_equals_build'], slug
    if '--regen' in sys.argv:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            out['regen'] = TWM.regen(a, b, {TWM.WM100: TWM.WM100_ARGS})
            assert all(v['head_equals_working'] and v['stdout_equal'] and v['final_hunks_ok']
                       for v in out['regen'].values()), out['regen']
    print(json.dumps(out, indent=1, default=sorted))
    print('sens swap build checks OK')


if __name__ == '__main__':
    main()
