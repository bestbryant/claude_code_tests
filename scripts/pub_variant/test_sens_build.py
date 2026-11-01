"""Static checks of the SENS builds (casmi26-pubm-sens) made by scripts/make_pub_variant.py SENS=K (ours).

python scripts/pub_variant/test_sens_build.py [--regen]

Nothing of the notebooks is executed: cells are only parsed (ast.parse / symtable), the pinned modules are read as text,
and the generator (our code) is run into temporary directories.
  1. builds casmi26-pubm-sens (SENS=5 raw), a z / K 3 variant and the casmi26-pubm configuration with the current
     generator; SENS=0 is byte-identical to the casmi26-pubm configuration; the generator refuses SENS with OURS /
     PF_MID_W / ISO_ALPHA / ELIG=0, bad K / modes / seeds / budgets (SENS with ELIG=1 / ENG3_W: test_sens_combo_build.py);
  2. every code cell parses; the inserted cell is scripts/pub_variant/sens_cell.py byte for byte, right after the PF-syn
     cell; the driver calls it after score_final_lists / score_outside_candidates / ours_pf_blend and before
     build_consensus; every global the cell reads is bound before that call; its scorer command line / cwd /
     environment are cell 9's (ast); the pinned neural_scores.py text has the interface the cell relies on;
  3. diff vs the casmi26-pubm configuration: the inserted cell, cells 0 and 1 (the SENS lines only), the driver cell
     (the SENS call only) and the section-4 markdown cell (cell numbers: final layer 17, ISO 18); notebook metadata:
     casmi.ours.sens plus the cell numbers in casmi.ours final_layer / iso; kernel-metadata: id / title / kernel_sources;
  4. diff vs the committed kernels/casmi26-pubm: the same plus cells 1 / 17 where the current generator's pubm
     configuration differs from the committed Nov 1 build (the ISO knobs ISO_COS_W / ISO_FREEZE_LEADER);
  5. collisions: every by-name input lookup of the pipeline is globbed on the output tree of
     bryantyu/casmi26-saner-seeds (both mount layouts); no hit except the documented ones;
  --regen: every existing pub-variant slug regenerated with the committed (git HEAD) scripts/ and with the working
     generator into throwaway folders, byte-compared (and compared with kernels/<slug>).
"""
import ast
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import symtable
import tarfile
import io
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(REPO, 'scripts', 'make_pub_variant.py')


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # our own generator / test code
    return m


S = _mod('pub_variant_sens_t', 'sens.py')
E3T = _mod('pub_variant_eng3_build_t', 'test_eng3_build.py')     # gen / load / diff_cells / frozen_of / lookup regexes
PUBM_ARGS = ['OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0']
SENS_ARGS = ['SENS=5', 'SENS_MODE=raw'] + PUBM_ARGS
EXISTING = dict(E3T.EXISTING, **{'casmi26-pubm-eng3a': ['ENG3_W=1.0'] + PUBM_ARGS,
                                 'casmi26-pubm-eng3b': ['ENG3_W=0.5'] + PUBM_ARGS, 'casmi26-pubm-sens': SENS_ARGS})
gen, load, diff_cells, frozen_of = E3T.gen, E3T.load, E3T.diff_cells, E3T.frozen_of


def module_level_names(src):
    """Names bound at module level by a cell (assignments, defs, imports, `global` in functions)."""
    out = set()
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        out.add(x.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split('.')[0] for a in n.names)
        elif isinstance(n, (ast.For, ast.With, ast.Try, ast.If)):
            for x in ast.walk(n):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    out.add(x.id)
    for x in ast.walk(tree):
        if isinstance(x, ast.Global):
            out.update(x.names)
    return out


def free_globals(src):
    """Globals read inside the cell's functions and not bound by the cell itself or builtins."""
    top = symtable.symtable(src, 'sens_cell', 'exec')
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    names = set()

    def walk(t):
        for s in t.get_symbols():
            if s.is_global() and s.is_referenced():
                names.add(s.get_name())
        for c in t.get_children():
            walk(c)
    for c in top.get_children():
        walk(c)
    import builtins
    return sorted(n for n in names - bound if not hasattr(builtins, n))


def call_list(fn_src, fname):
    """The first list literal passed as the first argument to fname(...) inside a function source."""
    for x in ast.walk(ast.parse(fn_src)):
        if isinstance(x, ast.Call) and getattr(x.func, 'id', getattr(x.func, 'attr', None)) == fname:
            if x.args and isinstance(x.args[0], ast.List):
                return x.args[0]
            if x.args and isinstance(x.args[0], ast.ListComp):
                return x.args[0]
    return None


def fn_source(cell_src, name):
    tree = ast.parse(cell_src)
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(cell_src, f)


def modules_of(nb):
    c8 = nb['cells'][next(i for i, c in enumerate(nb['cells']) if c.get('id') == '3e2e781f')]['source']
    out = {}
    for n in ast.parse(c8).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript):
            out[ast.literal_eval(n.targets[0].slice)] = ast.literal_eval(n.value)
        elif isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('MODULE_SHA256', 'MODEL_PINS'):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def check_build(d, slug, k, mode, pubm_nb, pubm_meta, committed_nb):
    nb, meta = load(d, slug)
    cells = nb['cells']
    ids = [c.get('id') for c in cells]
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            ast.parse(c['source'], f'{slug} cell {i}')
    assert ids.count(S.CELL_ID) == 1 and len(set(i for i in ids if i)) == len([i for i in ids if i]), ids
    i_cell = ids.index(S.CELL_ID)
    assert ids[i_cell - 1] == S.PF_CELL_ID and ids[i_cell + 1] == S.DRIVER_ID, ids
    cell = cells[i_cell]
    assert cell['source'] == open(os.path.join(HERE, 'sens_cell.py'), encoding='utf-8').read()
    assert cell['cell_type'] == 'code' and cell['outputs'] == [] and cell['execution_count'] is None
    assert cell['metadata'] == cells[i_cell - 1]['metadata']                # hidden in the viewer like PF-syn
    # 3. diff vs the pubm configuration (the inserted cell removed)
    rest = dict(nb, cells=cells[:i_cell] + cells[i_cell + 1:])
    cd = diff_cells(rest, pubm_nb)
    i4 = [i for i, c in enumerate(rest['cells']) if c['cell_type'] == 'markdown' and c['source'].startswith(S.SEC4)]
    assert len(i4) == 1 and i4[0] > i_cell, i4
    i4 = i4[0]
    assert cd == [0, 1, i_cell, i4], cd                                    # md 0, config, driver, section 4 (same index)
    assert rest['cells'][i4]['source'] == S.splice(pubm_nb['cells'][i4]['source'], S.RENUMBER4, 'section 4')
    ids_all = [c.get('id') for c in cells]
    assert (ids_all.index(S.FINAL_ID), ids_all.index(S.ISO_ID)) == (17, 18), ids_all
    assert '(cell 17) is' in cells[0]['source'] and 'cell 18 is\n> our ISO' in cells[0]['source']
    c1, c1p = cells[1]['source'], pubm_nb['cells'][1]['source']
    block = S.cell1_block(k, mode)
    assert c1 == c1p.replace(S.C1_ANCHOR, S.C1_ANCHOR + block, 1) and c1.count(block) == 1
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(c1).body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith(('SENS', 'OURS', 'PF_', 'ISO_'))}
    assert (vals['SENS'], vals['SENS_MODE'], vals['SENS_SEEDS'], vals['SENS_MAX_SECONDS']) == \
        (k, mode, S.SEEDS, S.MAX_SECONDS), vals
    assert (vals['OURS'], vals['PF_MID_W'], vals['ISO_ALPHA']) == (False, 0.0, 0.0)
    drv, drvp = cells[i_cell + 1]['source'], pubm_nb['cells'][i_cell]['source']
    assert drv == drvp.replace('CONSENSUS_OK, REPORT, MOVES = build_consensus()\n',
                               S.DRIVER_BLOCK + 'CONSENSUS_OK, REPORT, MOVES = build_consensus()\n', 1)
    assert drv.count('ours_seed_ensemble()') == 1
    calls = [x.func.id for x in ast.walk(ast.parse(drv)) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)]
    order = [c for c in calls if c in ('score_final_lists', 'score_outside_candidates', 'ours_pf_blend',
                                       'ours_seed_ensemble', 'build_consensus')]
    pos = {c: drv.index(c + '(') for c in order}
    assert sorted(pos, key=pos.get) == ['score_final_lists', 'score_outside_candidates', 'ours_pf_blend',
                                        'ours_seed_ensemble', 'build_consensus'], pos
    assert cells[0]['source'] == S.md0(pubm_nb['cells'][0]['source'], k, mode, S.SEEDS[:k - 1])
    # globals the cell reads: bound before the call (cells before it, or the driver above the call)
    need = free_globals(cell['source'])
    assert set(need) == {'BUNDLE', 'CODE', 'INPUT', 'MODEL_PINS', 'MODULE_SHA256', 'NB_T0', 'NEURAL_OK', 'ROOT',
                         'SENS_MAX_SECONDS', 'SENS_SEEDS'}, need
    before_call = drv[:drv.index('ours_seed_ensemble()')]
    before_call = before_call[:before_call.rindex('try:')]
    bound = module_level_names(before_call)
    for i in range(i_cell):
        if cells[i]['cell_type'] == 'code':
            bound |= module_level_names(cells[i]['source'])
    assert not set(need) - bound, set(need) - bound
    assert {'SENS', 'SENS_MODE'} <= module_level_names(c1)              # read with globals().get
    # the scorer command line / cwd / environment are cell 9's
    c9 = cells[next(i for i, c in enumerate(cells) if c.get('id') == 'ac4866c0')]['source']
    pub = call_list(fn_source(c9, 'score_final_lists'), 'run_stage')
    ours_src = fn_source(cell['source'], '_sens_run_scorer')
    ours = next(n.value for n in ast.walk(ast.parse(ours_src)) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') == 'args')
    pe = [ast.unparse(e) for e in pub.elts]
    oe = [ast.unparse(e) for e in ours.elts]
    assert pe == ['sys.executable', "CODE / 'neural_scores.py'", 'BUNDLE', "'final_model.pt'",
                  "ROOT / 'frozen_candidate_evidence.parquet'", "ROOT / 'effective_query_spectra.parquet'",
                  "ROOT / 'neural_candidate_features.parquet'", "'cuda'"], pe
    assert oe == ['sys.executable', "CODE / 'neural_scores.py'", 'bundle', "'final_model.pt'", 'evidence',
                  "ROOT / 'effective_query_spectra.parquet'", 'out', "'cuda'"], oe
    rs = fn_source(c9, 'run_stage')
    for line in ("env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONUNBUFFERED': '1'}", "env.pop('INC_PANEL', None)"):
        assert line in rs and line in ours_src, line
    assert 'cwd=ROOT' in rs and 'cwd=ROOT' in ours_src and '[str(a) for a in args]' in rs and \
        '[str(a) for a in args]' in ours_src
    so = fn_source(c9, 'score_outside_candidates')
    assert "ROOT / 'outsider_evidence.parquet', ROOT / 'effective_query_spectra.parquet',\n" \
           "                   ROOT / 'outsider_neural_features.parquet', 'cuda']" in so
    # the pinned scorer's interface (text only, nothing executed)
    mods = modules_of(nb)
    ns = mods['neural_scores.py']
    for t in ("frame.groupby('molecule_id',sort=False)", "g=g.sort_values('baseline_rank',kind='stable').copy()",
              "combined=cos/cfg['temperature']+cfg['fp_score_weight']*fps",
              "g['neural_cos']=cos.mean(0);g['neural_fp']=fps.mean(0);g['neural_score']=combined.mean(0)",
              "g['neural_ok']=True", "result.to_parquet(output,index=False)",
              "Path(str(output)+'.errors.json').write_text", "spec_from_file_location(package+'.'+name,Path(bundle)/(name+'.py'))",
              "json.loads((Path(bundle)/'config.json').read_text())", "torch.load(Path(bundle)/checkpoint",
              "if __name__=='__main__':score(*sys.argv[1:])"):
        assert t in ns, t
    assert set(mods['MODEL_PINS']) == {'features.py', 'model.py', 'config.json', 'final_model.pt'}
    assert mods['MODEL_PINS']['final_model.pt'] == nb['metadata']['casmi']['neural_final_model_sha256']
    assert 'neural_scores.py' in mods['MODULE_SHA256']
    # metadata
    m, mp = json.loads(json.dumps(nb['metadata'])), json.loads(json.dumps(pubm_nb['metadata']))
    rec = m['casmi']['ours'].pop('sens')
    assert list(m['casmi']['ours']) == list(mp['casmi']['ours'])            # key order kept
    assert m['casmi']['ours']['final_layer'] == mp['casmi']['ours']['final_layer'].replace('(cell 16)', '(cell 17)') and \
        m['casmi']['ours']['iso'] == mp['casmi']['ours']['iso'].replace('(cell 17)', '(cell 18)'), m['casmi']['ours']
    m['casmi']['ours'].update(final_layer=mp['casmi']['ours']['final_layer'], iso=mp['casmi']['ours']['iso'])
    assert m == mp, 'notebook metadata beyond casmi.ours.sens / final_layer / iso'
    assert rec['k'] == k and rec['mode'] == mode and rec['extra_seeds'] == S.SEEDS[:k - 1] and \
        rec['published_seed'] == 20261023 and rec['max_seconds'] == S.MAX_SECONDS
    assert meta['id'] == f'bryantyu/{slug}' and meta['title'] == slug and meta['kernel_sources'] == [S.SOURCE_KERNEL]
    assert {x: v for x, v in meta.items() if x not in ('id', 'title', 'kernel_sources')} == \
        {x: v for x, v in pubm_meta.items() if x not in ('id', 'title', 'kernel_sources')}
    assert pubm_meta['kernel_sources'] == []
    # 4. vs the committed casmi26-pubm (Nov 1 build, before the ISO knobs)
    known = diff_cells(pubm_nb, committed_nb)
    cdc = diff_cells(rest, committed_nb)
    assert known == [1, 17] and cdc == [0, 1, i_cell, 17, i4], (known, cdc)
    return dict(cells=len(cells), inserted=i_cell, driver=i_cell + 1, vs_pubm_config=cd, vs_committed=cdc,
                globals_read=need)


# ---- 5. collisions with the output of bryantyu/casmi26-saner-seeds ----
SEEDS_ALL = [20261023] + S.SEEDS


def seeds_output_tree():
    """Relative paths of every file the casmi26-saner-seeds kernel can leave in /kaggle/working (its notebook.py:
    WORKING / RW / SRC / SCORES / stage folders; the published config's work dir casmi_training_cache_v1 with the cache
    files of an unfinished run as well; __pycache__ of the copied sources)."""
    f = ['final_models_sha256.json', 'seeds_report.json']
    for s in SEEDS_ALL:
        f += [f'final_model_seed{s}.pt', f'final_model_seed{s}.pt.sha256']
    f += [f'casmi_training_cache_v1/{x}' for x in ('molecules.parquet', 'data_audit.json', 'spectra.parquet',
                                                   'peaks.bin', 'metadata.bin', 'fingerprints.npy')]
    f += [f'saner_seeds/{x}' for x in ('effective_config.json', 'launch_prepare.py', 'prepare.log', 'prepare.py.published',
                                        'prepare_normalized_smiles.diff', 'runtime_test.log', 'stage.py',
                                        'stage_outputs_sha256.json', 'struct_memo_check.json')]
    f += [f'saner_seeds/src/{x}' for x in ('config.json', 'features.py', 'inference.py', 'model.py', 'prepare.py',
                                            'runtime_test.py', 'train.py')]
    f += [f'saner_seeds/src/__pycache__/{x}.cpython-313.pyc' for x in ('features', 'model', 'prepare', 'runtime_test',
                                                                       'train')]
    for s in SEEDS_ALL:
        f += [f'saner_seeds/holdout_{s}/{x}' for x in ('best_validation_model.pt', 'last_validation_model.pt', 'stage.log',
                                                       'training_history.json', 'training_result.json',
                                                       'validation_candidates.npz', 'validation_predictions.csv')]
        f += [f'saner_seeds/refit_{s}/{x}' for x in ('final_model.pt', 'refit_history.json', 'stage.log',
                                                     'training_result.json')]
        f += [f'saner_seeds/scores/final_{s}.npz', f'saner_seeds/scores/holdout_{s}.npz']
    return sorted(set(f))


SIM = '/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/seeds/fw_review'
ALLOWED = {'final_model.pt': 'bundle.find_bundle (cell 8): a root needs config.json in the same folder and all four '
                             'MODEL_PINS; the refit folders hold no config.json, src/ holds no final_model.pt'}


def collision_check(nb, fz, extra_scoped=None):
    texts = {f'frozen {i}': ''.join(c['source']) for i, c in enumerate(fz['cells']) if c['cell_type'] == 'code'}
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            s = c['source']
            if "FROZEN_B64 = '" in s:
                b = s.index("FROZEN_B64 = '") + 14
                s = s[:b] + s[s.index("'\n", b):]
            texts[f'cell {i}'] = s
    found = {}
    for where, t in texts.items():
        for rx, fmt in ((E3T.LOOKUP, lambda m: m.group(3)), (E3T.JOINED, lambda m: f'{m.group(2)}/{m.group(4)}'),
                        (E3T.ANYGLOB, lambda m: m.group(2)), (E3T.DSDIR, lambda m: f'{m.group(2)}/{m.group(4)}')):
            for m in rx.finditer(t):
                found.setdefault(fmt(m), set()).add(where)
    tree = seeds_output_tree()
    if os.path.isdir(SIM):          # the latest local simulation of the seeds kernel: its files are all in the tree
        sim = {re.sub(r'\d{8}', '{s}', os.path.relpath(os.path.join(a, x), SIM)) for a, _, fs in os.walk(SIM) for x in fs}
        model = {re.sub(r'\d{8}', '{s}', x) for x in tree}
        assert sim <= model, sorted(sim - model)
    dirs = {}
    for x in tree:
        dirs.setdefault(os.path.dirname(x), set()).add(os.path.basename(x))
    assert not [d for d, fs in dirs.items() if {'final_model.pt', 'config.json'} <= fs], 'a bundle-like folder'
    hits, checked = {}, []
    with tempfile.TemporaryDirectory() as d:
        for layout in ('casmi26-saner-seeds', 'notebooks/bryantyu/casmi26-saner-seeds'):
            root = os.path.join(d, layout.replace('/', '_'))
            for x in tree:
                p = os.path.join(root, layout, x)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, 'w').close()
            for pat in sorted(found):
                if pat in E3T.GENERIC:
                    continue
                p = re.sub(r'^/kaggle/input/(\*\*/)?', '', pat)
                p = re.sub(r'\{[^}]*\}', '*', p)
                if pat.startswith('/') and not pat.startswith('/kaggle/input'):
                    continue
                if p in E3T.SCOPED or p in (extra_scoped or {}):
                    continue
                checked.append(p)
                h = glob.glob(os.path.join(root, '**', p), recursive=True)
                if h:
                    hits.setdefault(p, set()).update(os.path.relpath(x, root) for x in h)
    bad = {p: sorted(h) for p, h in hits.items() if p not in ALLOWED}
    c8 = next(t for k, t in texts.items() if "MODULES['bundle.py']" in t)
    assert "for path in Path(input_root).rglob('final_model.pt'):\\n        if (path.parent/'config.json').is_file():" \
        "roots.append(path.parent)" in c8, 'find_bundle text'
    assert "matching=[r for r in roots if all((r/n).is_file() and sha(r/n)==h for n,h in pins.items())]" in c8
    return dict(patterns=sorted(set(checked)), hits={p: sorted(h) for p, h in hits.items()}, bad=bad, tree_files=len(tree))


def regen(out_head, out_work):
    """Every existing slug with the committed scripts/ (git HEAD, extracted whole) and with the working generator."""
    raw = subprocess.run(['git', '-C', REPO, 'archive', 'HEAD', 'scripts'], capture_output=True, check=True).stdout
    res = {}
    with tempfile.TemporaryDirectory() as root:
        tarfile.open(fileobj=io.BytesIO(raw)).extractall(root, filter='data')
        os.symlink(os.path.join(REPO, 'kernels'), os.path.join(root, 'kernels'))
        head_gen = os.path.join(root, 'scripts', 'make_pub_variant.py')
        for slug, args in EXISTING.items():
            rc1, o1 = gen(slug, args, out_head, head_gen)
            rc2, o2 = gen(slug, args, out_work)
            assert rc1 == 0 and rc2 == 0, (slug, o1[-300:], o2[-300:])
            files = ('notebook.ipynb', 'kernel-metadata.json')
            rd = lambda *p: open(os.path.join(*p), 'rb').read()
            res[slug] = dict(head_equals_working=all(rd(out_head, slug, f) == rd(out_work, slug, f) for f in files),
                             stdout_equal=o1.replace(out_head, '<out>') == o2.replace(out_work, '<out>'),
                             equals_committed=all(rd(REPO, 'kernels', slug, f) == rd(out_work, slug, f) for f in files))
    return res


def main():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        builds = (('casmi26-pubm', PUBM_ARGS), ('casmi26-pubm-sens', SENS_ARGS),
                  ('casmi26-zz-sens3z', ['SENS=3', 'SENS_MODE=z'] + PUBM_ARGS))
        for slug, args in builds:
            rc, log = gen(slug, args, d)
            assert rc == 0, log
        rc, log = gen('casmi26-pubm', ['SENS=0'] + PUBM_ARGS, os.path.join(d, 'zero'))      # SENS=0: pubm exactly
        assert rc == 0, log
        for f in ('notebook.ipynb', 'kernel-metadata.json'):
            assert open(os.path.join(d, 'zero', 'casmi26-pubm', f), 'rb').read() == \
                open(os.path.join(d, 'casmi26-pubm', f), 'rb').read(), f
        refusals = (
            (['SENS=5'], 'not combined with PF-syn'),                                  # template default OURS = True
            (['SENS=5', 'OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0.5'], 'need OURS = True'),
            (['SENS=5', 'OURS_MODE=columns', 'ISO_ALPHA=0.5'], 'not combined with PF-syn'),
            (['SENS=5', 'OURS_MODE=columns', 'PF_MID_W=0.5'], 'not combined with PF-syn'),
            (['SENS=5', 'ELIG=0'] + PUBM_ARGS, 'not combined with ELIG=0'),
            (['SENS=1'] + PUBM_ARGS, '2..5'), (['SENS=6'] + PUBM_ARGS, '2..5'), (['SENS=-1'] + PUBM_ARGS, '2..5'),
            (['SENS=x'] + PUBM_ARGS, 'an integer'), (['SENS=2.5'] + PUBM_ARGS, 'an integer'),
            (['SENS=5', 'SENS_MODE=mean'] + PUBM_ARGS, 'raw or z'), (['SENS_MODE=z'] + PUBM_ARGS, 'without SENS'),
            (['SENS=5', 'SENS=4'] + PUBM_ARGS, 'given twice'),
            (['SENS=5', 'SENS_SEEDS=[20262023,20263023]'] + PUBM_ARGS, 'needs 4 extra seeds'),
            (['SENS=3', 'SENS_SEEDS=[20261023,20262023]'] + PUBM_ARGS, 'without the published'),
            (['SENS=3', 'SENS_SEEDS=[20262023,20262023]'] + PUBM_ARGS, 'distinct'),
            (['SENS=3', "SENS_SEEDS=['a','b']"] + PUBM_ARGS, 'distinct int'),
            (['SENS=3', 'SENS_MAX_SECONDS=0'] + PUBM_ARGS, 'SENS_MAX_SECONDS'),
            (['SENS_SEEDS=[20262023]'] + PUBM_ARGS, 'not a cell-1 setting'))
        for args, why in refusals:
            rc, log = gen('casmi26-zz-refused', args, os.path.join(d, 'refused'))
            assert rc != 0 and why in log, (args, rc, log[-400:])
            assert not os.path.exists(os.path.join(d, 'refused', 'casmi26-zz-refused')), args
        out['refusals'] = len(refusals)
        pubm_nb, pubm_meta = load(d, 'casmi26-pubm')
        committed_nb = json.load(open(os.path.join(REPO, 'kernels', 'casmi26-pubm', 'notebook.ipynb'), encoding='utf-8'))
        for slug, k, mode in (('casmi26-pubm-sens', 5, 'raw'), ('casmi26-zz-sens3z', 3, 'z')):
            out[slug] = check_build(d, slug, k, mode, pubm_nb, pubm_meta, committed_nb)
        nb, _ = load(d, 'casmi26-pubm-sens')
        fz, sha = frozen_of(nb)
        assert sha == frozen_of(pubm_nb)[1] == pubm_nb['metadata']['casmi']['frozen_source_sha256']   # untouched
        col = collision_check(nb, fz)
        assert not col['bad'], col['bad']
        out['collisions'] = dict(patterns_checked=len(col['patterns']), tree_files=col['tree_files'], hits=col['hits'])
        kd = os.path.join(REPO, 'kernels', 'casmi26-pubm-sens')            # the kernels/ copy equals this build
        if os.path.isdir(kd):
            out['kernels_dir_equals_build'] = all(open(os.path.join(kd, f), 'rb').read() ==
                                                  open(os.path.join(d, 'casmi26-pubm-sens', f), 'rb').read()
                                                  for f in ('notebook.ipynb', 'kernel-metadata.json'))
            assert out['kernels_dir_equals_build']
    if '--regen' in sys.argv:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            out['regen'] = regen(a, b)
            assert all(v['head_equals_working'] and v['stdout_equal'] for v in out['regen'].values()), out['regen']
    print(json.dumps(out, indent=1, default=sorted))
    print('sens build checks OK')


if __name__ == '__main__':
    main()
