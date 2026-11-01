"""Static checks of the ENG3 builds (casmi26-pubm-eng3a / -eng3b) made by scripts/make_pub_variant.py ENG3_W=... (ours).

python scripts/pub_variant/test_eng3_build.py [--regen]

Nothing of the notebooks is executed: cells are only parsed (ast.parse / compile), the frozen notebook is base64-decoded
and compared as text, and the generator (our code) is run into temporary directories.
  1. builds eng3a (ENG3_W=1.0) and eng3b (0.5) and the casmi26-pubm configuration with the current generator; ENG3_W=0
     is byte-identical to the casmi26-pubm configuration; the generator refuses ENG3_W with ELIG, with PF (OURS=True)
     and bad weights;
  2. every code cell parses, every frozen code cell parses, patched frozen cells 32 / 39 compile;
  3. diff vs the casmi26-pubm configuration: notebook cells 0, 1, 8 and 9 only (cell 1: the ENG3 lines only; cell 8:
     the frozen pin / base64 and the appended ENG3 block only; cell 9: the two trace edits only); inside the frozen
     notebook cells 32 and 39 only, each the original text + an appended block (the original Engine 2 run untouched);
     notebook metadata: frozen_source_sha256 and ours.eng3 only; kernel-metadata: id / title and our two datasets only;
     the weight baked into frozen cell 32, cell 8 and cell 1 agree; eng2_merge.py embedded = elig1's (pube1 pin);
  4. diff vs the committed kernels/casmi26-pubm: the same, plus cells 1 / 17 exactly where the current generator's
     casmi26-pubm configuration differs from the committed Nov 1 build (ISO knobs ISO_COS_W / ISO_FREEZE_LEADER);
  5. find() collisions: every by-name input lookup of the pipeline (frozen cells incl. the original Engine 2's, PubChem
     channel's and simulator glue's embedded sources, notebook cells, the pinned helper modules) is globbed on a tree of
     our two datasets' files; no hit except the documented ones;
  --regen: every existing pub-variant slug regenerated with the committed (git HEAD) generator and with the working
     one, byte-compared (and compared with kernels/<slug>).
"""
import ast
import base64
import difflib
import glob
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(REPO, 'scripts', 'make_pub_variant.py')
spec = importlib.util.spec_from_file_location('pub_variant_eng3_t', os.path.join(HERE, 'eng3.py'))
e3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e3)
PUBM_ARGS = ['OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0']
ELIG1_ENG2_MERGE_SHA256 = '40fe7a46d656369bfb9af5866bc318a47208185d7de4c4b6df045d1e63a7530d'   # pube1 cell 8 pin
EXISTING = {   # slug: generator arguments (as the kernels were built; scripts/make_pub_variant.py docstring / docs/plan.md)
    'casmi26-pube0': ['ELIG=0', 'OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0'],
    'casmi26-pube1': ['ELIG=1', 'OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0'],
    'casmi26-pube1-iso075all': ['ELIG=1', 'OURS_MODE=columns', 'ISO_ALPHA=0.75', 'ISO_SCOPE=all'],
    'casmi26-pubm': PUBM_ARGS,
    'casmi26-pubm-iso05cos': ['OURS_MODE=columns', 'ISO_ALPHA=0.5', 'ISO_SCOPE=all', 'ISO_COS_W=1.0'],
    'casmi26-pubm-iso075': ['OURS=True', 'OURS_MODE=columns', 'PF_MID_W=0', 'ISO_ALPHA=0.75', 'ISO_SCOPE=formula'],
    'casmi26-pubm-iso075all': ['OURS_MODE=columns', 'ISO_ALPHA=0.75', 'ISO_SCOPE=all'],
    'casmi26-pubm-iso150': ['OURS_MODE=columns', 'ISO_ALPHA=1.5'],
    'casmi26-pubm-pfall': [],
    'casmi26-pubm-pfmid': ['OURS_MODE=columns', 'PF_MID_W=0.5']}


def gen(slug, args, out, gen_path=GEN):
    r = subprocess.run([sys.executable, gen_path, slug, *args, f'--out-dir={out}'], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def load(d, slug):
    return (json.load(open(os.path.join(d, slug, 'notebook.ipynb'), encoding='utf-8')),
            json.load(open(os.path.join(d, slug, 'kernel-metadata.json'))))


def frozen_of(nb):
    c8 = nb['cells'][8]['source']
    sha = re.search(r"^FROZEN_SHA256 = '([0-9a-f]{64})'", c8, re.M).group(1)
    b = c8.index("FROZEN_B64 = '") + len("FROZEN_B64 = '")
    raw = base64.b64decode(c8[b:c8.index("'\n", b)])
    assert hashlib.sha256(raw).hexdigest() == sha, 'FROZEN_SHA256 does not pin the embedded notebook'
    return json.loads(raw), sha


def cell8_core(src):
    """Cell 8 without the frozen pin / base64 and without the ENG3 block (what must equal casmi26-pubm's)."""
    src = src.split('\n\n\n# ── ENG3 (ours, rank line)')[0]
    src = re.sub(r"^FROZEN_SHA256 = '[0-9a-f]{64}'.*$", 'FROZEN_SHA256 = <pin>', src, flags=re.M)
    b = src.index("FROZEN_B64 = '") + len("FROZEN_B64 = '")
    return src[:b] + '<b64>' + src[src.index("'\n", b):]


def diff_cells(a, b):
    return [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if x != y] + \
        list(range(min(len(a['cells']), len(b['cells'])), max(len(a['cells']), len(b['cells']))))


def check_build(d, slug, w, pubm_nb, pubm_meta, committed_nb):
    nb, meta = load(d, slug)
    rep = {}
    # 2. parsing
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            ast.parse(c['source'], f'{slug} cell {i}')
    fz, sha = frozen_of(nb)
    fz0, sha0 = frozen_of(pubm_nb)
    assert sha0 == e3.FROZEN_ORIGINAL_SHA256 and sha != sha0
    for i, c in enumerate(fz['cells']):
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']), f'{slug} frozen cell {i}')
    # 3. frozen notebook: cells 32 and 39 only, append-only
    fdiff = [i for i, (x, y) in enumerate(zip(fz['cells'], fz0['cells'])) if x != y]
    assert fdiff == [32, 39] and len(fz['cells']) == len(fz0['cells']) == 59, fdiff
    assert fz['metadata'] == fz0['metadata'] and {k: v for k, v in fz.items() if k != 'cells'} == \
        {k: v for k, v in fz0.items() if k != 'cells'}
    for i in (32, 39):
        new, old = ''.join(fz['cells'][i]['source']), ''.join(fz0['cells'][i]['source'])
        assert new.startswith(old) and len(new) > len(old), i
        compile(new, f'frozen cell {i}', 'exec', dont_inherit=True)
        assert {k: v for k, v in fz['cells'][i].items() if k != 'source'} == \
            {k: v for k, v in fz0['cells'][i].items() if k != 'source'}
    t32 = ''.join(fz['cells'][32]['source'])
    old32 = ''.join(fz0['cells'][32]['source'])
    assert "ENG = json.load(open('/kaggle/working/eng_lists.json'))" in old32 and t32.count('eng_runner.py') == \
        old32.count('eng_runner.py')                                   # the original Engine 2 run, unchanged, first
    added32 = ast.parse(t32[len(old32):])
    baked = [n for n in added32.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
             and [t.id for t in n.targets[0].elts] == ['ENG3_W', 'ENG3_K', 'ENG3_N']]
    assert len(baked) == 1 and ast.literal_eval(baked[0].value) == (w, 3, 40), ast.dump(baked[0].value)
    cfg = [n for n in added32.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ENG3_CFG']
    assert ast.literal_eval(cfg[0].value) == e3.ENG3_CFG and e3.ENG3_CFG['cap_s'] == 2700.0 and \
        e3.ENG3_CFG['stall_s'] == 900.0 and e3.ENG3_CFG['grace_s'] == 300.0
    t39 = ''.join(fz['cells'][39]['source'])
    old39 = ''.join(fz0['cells'][39]['source'])
    assert 'ENG = _e3_merged' in t39[len(old39):] and t39[len(old39):].count('ENG = ') == 2   # merged, or the original
    # 3. notebook cells: 0, 1, 8, 9 only
    cd = diff_cells(nb, pubm_nb)
    assert cd == [0, 1, 8, 9], cd
    c1, c1p = nb['cells'][1]['source'], pubm_nb['cells'][1]['source']
    block = e3.cell1_edits(w)[0][1][len(e3.C1_ANCHOR):]
    assert c1.replace(block, '', 1) == c1p and c1.count(block) == 1
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(c1).body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('ENG3_W', 'OURS', 'PF_MID_W', 'ISO_ALPHA')}
    assert vals == dict(ENG3_W=w, OURS=False, PF_MID_W=0.0, ISO_ALPHA=0.0), vals
    assert cell8_core(nb['cells'][8]['source']) == cell8_core(pubm_nb['cells'][8]['source'])
    g8 = ast.parse('# ── ENG3 (ours, rank line)' + nb['cells'][8]['source'].split('\n\n\n# ── ENG3 (ours, rank line)')[1])
    lit = {n.targets[0].id: ast.literal_eval(n.value) for n in g8.body if isinstance(n, ast.Assign)
           and isinstance(n.targets[0], ast.Name) and n.targets[0].id != 'ENG3_SETUP'}
    assert lit['ENG3_BAKED']['w'] == w and lit['ENG3_BAKED']['frozen_patched_sha256'] == sha
    assert lit['ENG3_MODULE_SHA256']['eng2_merge.py'] == ELIG1_ENG2_MERGE_SHA256
    for name, src in lit['ENG3_MODULE_SRC'].items():
        assert src == open(os.path.join(HERE, name), encoding='utf-8').read()
        assert hashlib.sha256(src.encode()).hexdigest() == lit['ENG3_MODULE_SHA256'][name]
    c9 = pubm_nb['cells'][9]['source']
    for old, new in e3.C9:
        c9 = c9.replace(old, new, 1)
    assert nb['cells'][9]['source'] == c9 and 'ENG3 |' in c9
    key_line = re.search(r"KEY_LINE = re\.compile\((r'.*?'\s*r'.*?')\)", c9, re.S).group(1)
    rx = re.compile(ast.literal_eval('(' + key_line + ')'))  # the two raw literals, concatenated by the parser
    for line in ('ENG3 | w 1.0 | molecules 12 | ours 12 | merged 12 | shared identities 1 | new from ours 2 | 3 s',
                 'ENG3 keys | ours (all): ...', 'ENG3 merge | k 3 ...', 'ENG3 OURS subprocess | status done',
                 'ENG3 setup | w 1.0', 'engine lists 12 715s'):
        assert rx.search(line), line
    assert nb['cells'][0]['source'].startswith(f'# Variant M core + midx final layer + ENG3: Engine 2 merged with our '
                                               f'stage 1 (w {w!r})')
    # metadata
    m, mp = json.loads(json.dumps(nb['metadata'])), json.loads(json.dumps(pubm_nb['metadata']))
    eng3 = m['casmi']['ours'].pop('eng3')
    assert m['casmi'].pop('frozen_source_sha256') == sha and mp['casmi'].pop('frozen_source_sha256') == sha0 and m == mp
    assert eng3['w'] == w and eng3['frozen_patched_sha256'] == sha and eng3['frozen_patched_cells'] == [32, 39]
    assert 'pf_tpu_f.pt (the PF-syn checkpoint)' in eng3['engine']['peakformer'] and 'not PF-free' in \
        eng3['engine']['peakformer']
    c0 = nb['cells'][0]['source']                                     # cell 0 says the added engine carries PF-syn
    assert 'not a PF-free contrast' in c0 and 'pf_tpu_f.pt (the PF-syn checkpoint)' in c0 and 'are off (OURS = False)' in c0
    assert 'PF-syn and ISO are off' not in c0
    assert meta['id'] == f'bryantyu/{slug}' and meta['title'] == slug
    assert meta['dataset_sources'] == pubm_meta['dataset_sources'] + ['bryantyu/casmi26-elig-assets',
                                                                       'bryantyu/casmi26-pool-v18c']
    assert {k: v for k, v in meta.items() if k not in ('id', 'title', 'dataset_sources')} == \
        {k: v for k, v in pubm_meta.items() if k not in ('id', 'title', 'dataset_sources')}
    # 4. vs the committed casmi26-pubm (Nov 1 build, before the ISO knobs)
    cdc = diff_cells(nb, committed_nb)
    known = diff_cells(pubm_nb, committed_nb)
    assert known == [1, 17] and cdc == [0, 1, 8, 9, 17], (known, cdc)
    assert nb['cells'][17] == pubm_nb['cells'][17]
    iso_lines = [l for l in difflib.unified_diff(committed_nb['cells'][1]['source'].split('\n'),
                                                  c1p.split('\n'), lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
    assert all(l.startswith('+ISO_COS_W') or l.startswith('+ISO_FREEZE_LEADER') for l in iso_lines), iso_lines
    rep.update(frozen_sha256=sha, cells=cd, frozen_cells=fdiff, vs_committed=cdc, iso_knob_lines=len(iso_lines))
    return rep, nb, fz


# ---- 5. find() collisions ----
LOOKUP = re.compile(r"\b(find|_find|rglob|iglob|glob|_first_input)\(\s*[rf]?\\*(['\"])(.*?)\\*\2")
JOINED = re.compile(r"\bfind\(\s*os\.path\.join\(\s*\\*(['\"])(.*?)\\*\1\s*,\s*\\*(['\"])(.*?)\\*\3")
ANYGLOB = re.compile(r"\\*(['\"])((?:/kaggle/input/)?\*\*/[^'\"\\\\]*)\\*\1")   # any '**/...' literal (e.g. INPUT / '**/x')
DSDIR = re.compile(r"\bdataset_dir\(\s*\\*(['\"])(.*?)\\*\1\s*,\s*\\*(['\"])(.*?)\\*\3")
GENERIC = {'/kaggle/input/**/{pattern}', '/kaggle/input/**/{slug}/{marker}'}     # bodies of find() / dataset_dir()
SCOPED = {'*': 'rglob over one known directory (frozen cell 9: the v4g container folders; cell 27: the v4b code copy; '
               'cell 54: the five V43_DATASET_SOURCES datasets, which do not include ours)',
          '*.parquet': 'PANEL.glob (frozen cell 16: the panel benchmark folder)',
          '*.py': 'ours (eng2_merge.run_engine2, cell 8 block): the modules of OUR code dir (casmi26-elig-assets)',
          'datasets/*/*': 'ours (eng2_merge._dataset_dir, cell 8 block): our two datasets by slug'}
ALLOWED = {'fp_*.pt': "the original Engine 2's eng_runner keeps only paths containing 'casmi26-fp-models-v2'",
           'pf_tpu_f.pt': "cell 11 PF-syn (OURS = False: not run) prefers 'casmi26-models' paths",
           'peakformer.py': "cell 11 PF-syn (OURS = False: not run) prefers 'casmi26-code' paths (and checks the code)"}


def collision_check(nb, fz):
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
        for m in LOOKUP.finditer(t):
            found.setdefault(m.group(3), set()).add(f'{where} {m.group(1)}')
        for m in JOINED.finditer(t):
            found.setdefault(f'{m.group(2)}/{m.group(4)}', set()).add(f'{where} find(os.path.join)')
        for m in ANYGLOB.finditer(t):
            found.setdefault(m.group(2), set()).add(f'{where} recursive glob literal')
        for m in DSDIR.finditer(t):
            found.setdefault(f'{m.group(2)}/{m.group(4)}', set()).add(f'{where} dataset_dir')
    hits, checked = {}, []
    with tempfile.TemporaryDirectory() as d:
        for layout in ('datasets/bryantyu/{}', '{}'):                 # /kaggle/input/datasets/<owner>/<slug> or /<slug>
            root = os.path.join(d, layout.split('/')[0] if '/' in layout else 'flat')
            for slug, files in (('casmi26-elig-assets', e3.ASSET_FILES), ('casmi26-pool-v18c', e3.POOL_FILES)):
                base = os.path.join(root, layout.format(slug)) if '/' in layout else os.path.join(root, slug)
                os.makedirs(base, exist_ok=True)
                for f in files:
                    open(os.path.join(base, f), 'w').close()
            for pat, where in sorted(found.items()):
                if pat in GENERIC:
                    continue
                p = re.sub(r'^/kaggle/input/(\*\*/)?', '', pat)
                p = re.sub(r'\{[^}]*\}', '*', p)
                if pat.startswith('/') and not pat.startswith('/kaggle/input'):
                    continue                                             # not an input lookup
                if p in SCOPED:
                    continue
                checked.append(p)
                h = glob.glob(os.path.join(root, '**', p), recursive=True)
                if h:
                    hits.setdefault(p, set()).update(os.path.relpath(x, root) for x in h)
    bad = {p: sorted(h) for p, h in hits.items() if p not in ALLOWED}
    # the documented reasons hold in the text: Engine 2's glob keeps casmi26-fp-models-v2 paths only (ours never
    # contain it); cell 11 (PF-syn) prefers casmi26-models / casmi26-code and is off (OURS = False, cell 1)
    t32 = texts['frozen 32']
    assert "glob.glob(\\'/kaggle/input/**/fp_*.pt\\', recursive=True) if \\'casmi26-fp-models-v2\\' in p)" in t32
    assert not any('casmi26-fp-models-v2' in x for x in hits.get('fp_*.pt', ()))
    c11 = next(s for s in texts.values() if "_first_input('pf_tpu_f.pt', 'casmi26-models')" in s)
    assert "key=lambda p: (prefer not in p, len(p))" in c11 and "_first_input('peakformer.py', 'casmi26-code', " in c11
    assert re.search(r'^OURS = False', next(s for k, s in texts.items() if k == 'cell 1'), re.M)
    return dict(patterns=sorted(set(checked)), hits={p: sorted(h) for p, h in hits.items()}, bad=bad,
                sources={p: sorted(w) for p, w in found.items()})


def regen(out_head, out_work):
    """Every existing slug with the committed (HEAD) generator and with the working one; byte comparison."""
    head_src = subprocess.run(['git', '-C', REPO, 'show', 'HEAD:scripts/make_pub_variant.py'], capture_output=True,
                              text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as root:              # a root whose kernels / pub_variant are the repo's own
        os.makedirs(os.path.join(root, 'scripts'))
        os.symlink(os.path.join(REPO, 'kernels'), os.path.join(root, 'kernels'))
        os.symlink(HERE, os.path.join(root, 'scripts', 'pub_variant'))
        head_gen = os.path.join(root, 'scripts', 'make_pub_variant.py')
        open(head_gen, 'w').write(head_src)
        res = {}
        for slug, args in EXISTING.items():
            rc1, o1 = gen(slug, args, out_head, head_gen)
            rc2, o2 = gen(slug, args, out_work)
            assert rc1 == 0 and rc2 == 0, (slug, o1[-300:], o2[-300:])
            same = all(open(os.path.join(out_head, slug, f), 'rb').read() == open(os.path.join(out_work, slug, f), 'rb').read()
                       for f in ('notebook.ipynb', 'kernel-metadata.json'))
            committed = all(open(os.path.join(REPO, 'kernels', slug, f), 'rb').read() ==
                            open(os.path.join(out_work, slug, f), 'rb').read() for f in ('notebook.ipynb', 'kernel-metadata.json'))
            res[slug] = dict(head_equals_working=same, equals_committed=committed)
    return res


def main():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for slug, args in (('casmi26-pubm', PUBM_ARGS), ('casmi26-pubm-eng3a', ['ENG3_W=1.0'] + PUBM_ARGS),
                           ('casmi26-pubm-eng3b', ['ENG3_W=0.5'] + PUBM_ARGS)):
            rc, log = gen(slug, args, d)
            assert rc == 0, log
        # ENG3_W=0 is the casmi26-pubm configuration, byte for byte
        rc, log = gen('casmi26-pubm', ['ENG3_W=0'] + PUBM_ARGS, os.path.join(d, 'zero'))
        assert rc == 0, log
        for f in ('notebook.ipynb', 'kernel-metadata.json'):
            assert open(os.path.join(d, 'zero', 'casmi26-pubm', f), 'rb').read() == \
                open(os.path.join(d, 'casmi26-pubm', f), 'rb').read(), f
        # refusals
        for args, why in ((['ENG3_W=1.0', 'ELIG=1', 'OURS=False'], 'not valid with ELIG'),
                          (['ENG3_W=1.0', 'OURS=True'], 'no PF-syn re-weighting'),
                          (['ENG3_W=1.0', 'OURS_MODE=columns', 'ISO_ALPHA=0.5'], 'no PF-syn re-weighting'),
                          (['ENG3_W=1.0', 'OURS=False', 'PF_MID_W=0.5'], 'need OURS = True'),
                          (['ENG3_W=-1'], 'finite weight'), (['ENG3_W=nan'], 'finite weight'), (['ENG3_W=inf'], 'finite weight'),
                          (['ENG3_W=x'], 'a number'), (['ENG3_W=1', 'ENG3_W=2'], 'given twice')):
            rc, log = gen('casmi26-zz-refused', args, os.path.join(d, 'refused'))
            assert rc != 0 and why in log, (args, rc, log[-300:])
            assert not os.path.exists(os.path.join(d, 'refused', 'casmi26-zz-refused')), args
        pubm_nb, pubm_meta = load(d, 'casmi26-pubm')
        committed_nb = json.load(open(os.path.join(REPO, 'kernels', 'casmi26-pubm', 'notebook.ipynb'), encoding='utf-8'))
        for slug, w in (('casmi26-pubm-eng3a', 1.0), ('casmi26-pubm-eng3b', 0.5)):
            rep, nb, fz = check_build(d, slug, w, pubm_nb, pubm_meta, committed_nb)
            col = collision_check(nb, fz)
            assert not col['bad'], col['bad']
            rep['collisions'] = dict(patterns_checked=len(col['patterns']), hits=col['hits'])
            out[slug] = rep
        out['lookup_patterns'] = col['patterns']
        # the kernels/ copies (when written) equal these builds
        for slug in ('casmi26-pubm-eng3a', 'casmi26-pubm-eng3b'):
            kd = os.path.join(REPO, 'kernels', slug)
            if os.path.isdir(kd):
                out[slug]['kernels_dir_equals_build'] = all(
                    open(os.path.join(kd, f), 'rb').read() == open(os.path.join(d, slug, f), 'rb').read()
                    for f in ('notebook.ipynb', 'kernel-metadata.json'))
                assert out[slug]['kernels_dir_equals_build'], slug
    if '--regen' in sys.argv:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            out['regen'] = regen(a, b)
            assert all(v['head_equals_working'] for v in out['regen'].values()), out['regen']
    print(json.dumps(out, indent=1, default=sorted))
    print('eng3 build checks OK')


if __name__ == '__main__':
    main()
