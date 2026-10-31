"""Build kernels/<slug>: casmi26-pub480pf with the midx final layer (+ our PF_MID_W hook), PF-syn OURS_MODE 'columns' and
our ISO isomer reranker; every kernel made here differs from the others only in the configuration cell (cell 1).

Usage: python scripts/make_pub_variant.py <slug> [KEY=VALUE ...]
  KEY is any `NAME = value` line of cell 1 (VALUE a Python literal, a bare word is a string), e.g.
  python scripts/make_pub_variant.py casmi26-pubm-iso075 OURS=True OURS_MODE=columns PF_MID_W=0 ISO_ALPHA=0.75 ISO_SCOPE=formula
Structural edits of kernels/casmi26-pub480pf (left unchanged), the same for every variant:
  - cell 16 (final layer) := cell 15 of baidalinadilzhan/casmi26-midx-exact-dedup-fill (team LB 0.479; verbatim in
    scripts/pub_variant/midx_cell15.txt, SHA-256 pinned) spliced with the PF_MID_W hook of the researcher's drop-in
    (_mid_neural_score: ranks 2-25 neural rank from (1 - w) z(neural_score_saner) + w z(pf_fz); 0 = midx exactly) and
    comment glue; its configuration lines (POP_W .. OUTSIDER_FILL) go to cell 1;
  - cell 11 (PF-syn): OURS_MODE 'columns' adds only neural_score_saner / pf_fz, neural_score stays byte-identical;
    'blend' / 'replace' cast neural_score to float64 first (pandas >= 3 refuses float64 values in a float32 column);
    the PF-SYN line and pf_blend_report.json carry the pandas version;
  - new cell 17 (scripts/pub_variant/iso_cell.py): ISO isomer reranker on the final submission.csv (ISO_* in cell 1),
    with a commit-run dry run of the rerank path and a trace of the PF_MID_W hook;
  - markdown cells 0 and 18: a note on what this version runs.
Writes kernels/<slug>/notebook.ipynb + kernel-metadata.json (id bryantyu/<slug>, private, GPU / image / datasets of
casmi26-pub480pf). Nothing is executed: the cells are only checked with ast.parse.
"""
import ast, hashlib, json, os, re, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
here = os.path.join(root, 'scripts', 'pub_variant')
slug, args = sys.argv[1], sys.argv[2:]
src = os.path.join(root, 'kernels', 'casmi26-pub480pf')
dst = os.path.join(root, 'kernels', slug)
assert slug not in ('casmi26-pub480pf', 'casmi26-pub480'), 'the templates stay unchanged'
ISO_ID = 'ours-iso-rerank'
if os.path.exists(dst):                                  # only a kernel made here may be rewritten
    nb_path = os.path.join(dst, 'notebook.ipynb')
    assert os.path.isfile(nb_path) and ISO_ID in open(nb_path, encoding='utf-8').read(), f'{dst} is not a pub variant'
MIDX_SHA256 = '06093f589a8de6a5cf0f7cb1cd36f3d7cb6af8555f1528a2e0624bf3f85db42d'
MOJI = lambda s: s.encode('utf-8').decode('cp1251')     # how the pulled midx source carries its box characters / dashes

# ---- cell 1: configuration (the midx final-layer lines, PF_MID_W and the ISO switches join our PF-syn block) ----
CFG = [("OURS_MODE = 'blend'           # 'blend' or 'replace' (neural term = z(PF-syn f.z) alone)\n",
        "OURS_MODE = 'blend'           # 'blend', 'replace' (neural term = z(PF-syn f.z) alone) or 'columns' (neural_score\n"
        "                              # unchanged; only the columns neural_score_saner / pf_fz, read by PF_MID_W and ISO)\n"),
       ("PF_MAX_SECONDS = 1200         # PF-syn scoring budget; it is also skipped past 8.0 h of session time\n",
        "PF_MAX_SECONDS = 1200         # PF-syn scoring budget; it is also skipped past 8.0 h of session time\n"
        "\n"
        "# ── Final layer (baidalinadilzhan/casmi26-midx-exact-dedup-fill, team LB 0.479) ─────────────────────────────\n"
        "# Documentation-prior weights on ranks 2-25 (MID: 0.10 / 0.05 / 0.02), spectral-library bonus only for molecules whose\n"
        "# best library match reaches LIB_HIT_GATE, the pipeline's leader pinned at rank 2 after a fragment-check leader move,\n"
        "# metric-exact deduplication (the scorer's own key) and empty slots filled with scored outside candidates.\n"
        "POP_W, SPEC_W, NP_W = 0.10, 0.05, 0.02\n"
        "LIB_HIT_GATE = 0.87\n"
        "PIN_OLD_LEADER = True\n"
        "OUTSIDER_FILL = True\n"
        "PF_MID_W = 0.0                # ours: PF-syn in the ranks 2-25 neural rank of the final layer, per molecule\n"
        "                              # (1 - PF_MID_W) z(neural_score_saner) + PF_MID_W z(pf_fz); 0 = midx exactly\n"
        "\n"
        "# Ours: ISO isomer reranker on the final answer lists (the cell after the final layer)\n"
        "ISO_ALPHA = 0.0               # per list z(-log rank) + ISO_ALPHA * z(PF-syn f.z); 0 = off (submission.csv untouched)\n"
        "ISO_LIB_SKIP = 0.9            # molecules whose best library match reaches this keep their list\n"
        "ISO_KEEP_TRAIN = True         # candidates whose InChIKey14 is in train.parquet keep their exact positions\n"
        "ISO_SCOPE = 'formula'         # 'formula': only the rank-1 candidate's formula is re-sorted; 'all': every candidate\n")]

# ---- cell 11: PF-syn OURS_MODE 'columns' (blend / replace, the atomic write, backup and fall-back are unchanged) ----
PF = [("# fragment check and Variant M read the blended order unchanged. Any failure keeps the saner-only files (warning).\n",
       "# fragment check and Variant M read the blended order unchanged. Any failure keeps the saner-only files (warning).\n"
       "# OURS_MODE 'columns': neural_score stays byte-identical; only the audit columns neural_score_saner (= neural_score)\n"
       "# and pf_fz are added, read by the final layer's PF_MID_W hook (ranks 2-25) and the ISO reranker after it.\n"),
      ("    Returns (new_main, new_outs, stats); neural_score_saner and pf_fz are added for the audit.\"\"\"\n"
       "    assert mode in ('blend', 'replace'), mode\n",
       "    mode 'columns': neural_score is left exactly as it is; only neural_score_saner and pf_fz are added.\n"
       "    Returns (new_main, new_outs, stats); neural_score_saner and pf_fz are added for the audit.\"\"\"\n"
       "    assert mode in ('blend', 'replace', 'columns'), mode\n"),
      ("    main = main.copy()\n"
       "    main['neural_score_saner'] = main.neural_score.astype(float)\n",
       "    main = main.copy()\n"
       "    main['neural_score_saner'] = main.neural_score.astype(float)\n"
       "    if mode != 'columns':     # float64 before the blend: the public scorer writes float32 when every molecule is\n"
       "        main['neural_score'] = main.neural_score.astype(float)   # processed, and pandas >= 3 refuses float64 in it\n"),
      ("        outs = outs.copy()\n"
       "        outs['neural_score_saner'] = outs.neural_score.astype(float)\n",
       "        outs = outs.copy()\n"
       "        outs['neural_score_saner'] = outs.neural_score.astype(float)\n"
       "        if mode != 'columns':\n"
       "            outs['neural_score'] = outs.neural_score.astype(float)\n"),
      ("        s = g.neural_score_saner.to_numpy(float)\n"
       "        zs_of, zp_of = zmap(s), zmap(p)\n"
       "        new = mix(zs_of(s), zp_of(p))\n"
       "        assert np.isfinite(new).all(), mid\n"
       "        main.loc[g.index, 'neural_score'] = new\n"
       "        maps[str(mid)] = (zs_of, zp_of)\n"
       "        stats['blended'] += 1\n"
       "        stats['candidates'] += len(g)\n"
       "        stats['top1_changed'] += int(np.argmax(new) != np.argmax(s))\n",
       "        s = g.neural_score_saner.to_numpy(float)\n"
       "        stats['blended'] += 1\n"
       "        stats['candidates'] += len(g)\n"
       "        if mode != 'columns':                     # 'columns': neural_score untouched, audit columns only\n"
       "            zs_of, zp_of = zmap(s), zmap(p)\n"
       "            new = mix(zs_of(s), zp_of(p))\n"
       "            assert np.isfinite(new).all(), mid\n"
       "            main.loc[g.index, 'neural_score'] = new\n"
       "            maps[str(mid)] = (zs_of, zp_of)\n"
       "            stats['top1_changed'] += int(np.argmax(new) != np.argmax(s))\n"),
      ("    stats['rho'] = round(float(np.mean(stats['rho'])), 3) if stats['rho'] else None\n",
       "    stats['rho'] = round(float(np.mean(stats['rho'])), 3) if stats['rho'] else None\n"
       "    stats['pf_rows'] = sum(int(np.isfinite(f.pf_fz.to_numpy(float)).sum()) for f in (main, outs)\n"
       "                           if f is not None and 'pf_fz' in f.columns)\n"),
      ('    """Score the neural-fusion candidates with PF-syn and blend into neural_score in place; never fatal."""\n',
       '    """Score the neural-fusion candidates with PF-syn and blend into neural_score in place (OURS_MODE \'columns\': add\n'
       '    the audit columns only, neural_score byte-identical); never fatal."""\n'),
      ("    rep = dict(status='off', mode=OURS_MODE, w=PF_W, molecules=0, candidates=0, failures=0)\n",
       "    rep = dict(status='off', mode=OURS_MODE, w=PF_W, molecules=0, candidates=0, failures=0, pandas=pd.__version__)\n"),
      ("        print('PF-SYN off (OURS = False): saner-only neural term', flush=True)\n",
       "        print(f'PF-SYN off (OURS = False): saner-only neural term | pandas {pd.__version__}', flush=True)\n"),
      ("f\"skipped {rep.get('skipped_no_pf', 0)} | failures {rep['failures']} | {rep['seconds']} s\", flush=True)\n",
       "f\"skipped {rep.get('skipped_no_pf', 0)} | failures {rep['failures']} | {rep['seconds']} s | \"\n"
       "          f\"pandas {pd.__version__}\", flush=True)\n"),
      ("        assert OURS_MODE in ('blend', 'replace'), f'OURS_MODE {OURS_MODE!r}'\n",
       "        assert OURS_MODE in ('blend', 'replace', 'columns'), f'OURS_MODE {OURS_MODE!r}'\n"),
      ("            assert np.isfinite(new.neural_score.astype(float)).all(), 'non-finite neural score'\n",
       "            assert np.isfinite(new.neural_score.astype(float)).all(), 'non-finite neural score'\n"
       "            if OURS_MODE == 'columns':            # every column read, neural_score included, exactly as it was\n"
       "                assert all(new[c].equals(old[c]) for c in old.columns) and \\\n"
       "                    new.neural_score.to_numpy().tobytes() == old.neural_score.to_numpy().tobytes(), 'a column changed'\n"),
      ("        for path, _ in writes:\n"
       "            os.replace(path.with_name(path.name + '.pf.tmp'), path)\n",
       "        if OURS_MODE == 'columns':                # the written neural_score reads back byte for byte\n"
       "            for path, _ in writes:\n"
       "                a = pd.read_parquet(bak / path.name, columns=['neural_score']).neural_score\n"
       "                b = pd.read_parquet(path.with_name(path.name + '.pf.tmp'), columns=['neural_score']).neural_score\n"
       "                assert a.dtype == b.dtype and a.to_numpy().tobytes() == b.to_numpy().tobytes(), path.name\n"
       "        for path, _ in writes:\n"
       "            os.replace(path.with_name(path.name + '.pf.tmp'), path)\n")]

# ---- cell 16: midx cell 15 + the PF_MID_W hook of the researcher's drop-in (pub480/researcher/dropin) + comment glue ----
HOOK = '''def _mid_neural_score(g):
    """Score that sets the neural rank nr of ranks 2-25. PF_MID_W = 0 (default): the column neural_score, as midx.
    PF_MID_W > 0: (1 - w) z(saner) + w z(PF-syn fz) per molecule, from the columns neural_score_saner (else
    neural_score) and pf_fz; any missing / non-finite PF score keeps the plain neural_score for that molecule."""
    base = g['neural_score'].to_numpy(float)
    w = float(globals().get('PF_MID_W', 0.0) or 0.0)
    if w <= 0 or 'pf_fz' not in g.columns or len(g) < 2:
        return base
    s = (g['neural_score_saner'] if 'neural_score_saner' in g.columns else g['neural_score']).to_numpy(float)
    p = g['pf_fz'].to_numpy(float)
    if not (np.isfinite(s).all() and np.isfinite(p).all()):
        return base
    zs = (s - s.mean()) / s.std() if s.std() > 1e-12 else np.zeros_like(s)
    zp = (p - p.mean()) / p.std() if p.std() > 1e-12 else np.zeros_like(p)
    return (1.0 - w) * zs + w * zp
'''
HEAD = '# ── Variant M: rank 1 from fragment check on V2, ranks 2-25 from CUSTOM + DocPrior + In-Pool Dedup ────────\n'
MIDX = [(MOJI(HEAD),
         HEAD + '# Cell 15 of baidalinadilzhan/casmi26-midx-exact-dedup-fill (team LB 0.479) unchanged, its configuration lines in\n'
                '# cell 1, plus one hook of ours: _mid_neural_score (PF_MID_W > 0: PF-syn joins the ranks 2-25 neural rank).\n'),
        (MOJI('        # Re-rank Ranks 2–25 by'), '        # Re-rank Ranks 2–25 by'),
        ('    return pop_map, spec_map, np_map\n\n\ndef pxm_merge(',
         '    return pop_map, spec_map, np_map\n\n\n' + HOOK + '\n\ndef pxm_merge('),
        ("order = np.argsort(-g['neural_score'].to_numpy(), kind='stable')",
         "order = np.argsort(-_mid_neural_score(g), kind='stable')")]


# ---- markdown (cells 0 and 17 of the template): say what this version runs ----
TITLE = ('# LB 0.480+ · Variant M Core + Calibrated Documentation Prior (w_pop=0.14) + Formula-Gated Isomer Swaps + '
         'In-Pool InChIKey14 Deduplication\n')
MD0 = [(TITLE,
        '# Variant M core + midx final layer + PF-syn + ISO isomer reranker\n\n'
        '> **This version** (made by scripts/make_pub_variant.py; every setting is in cell 1): the final layer (cell 16) is\n'
        '> cell 15 of baidalinadilzhan/casmi26-midx-exact-dedup-fill (team LB 0.479: documentation prior 0.10 / 0.05 / 0.02 on\n'
        '> ranks 2-25, library bonus only from a best library match of 0.87, the old leader pinned at rank 2 after a\n'
        '> fragment-check leader move, metric-exact deduplication, empty slots filled with scored outside candidates; no\n'
        '> formula-gated swaps) plus our PF_MID_W hook; cell 11 scores the candidates with our PF-syn PeakFormer; cell 17 is\n'
        '> our ISO isomer reranker. The notes below are the forked notebook\'s and describe its earlier final layer\n'
        '> (w_pop = 0.14, formula-gated isomer swaps), which this version does not run.\n')]
MD17 = [('and is not shown to anyone.',
         'and is not shown to anyone.\nIn this version the last row also holds the final layer\'s documentation prior, '
         'deduplication and outside-candidate fill\n(cell 16) and, when ISO_ALPHA > 0, the ISO reranker (cell 17); the '
         'MERGE and ISO log lines (merge_report.json, iso_report.json)\ngive their own counts.')]


def splice(text, edits, what):
    for old, new in edits:
        assert text.count(old) == 1, f'{what}: {old[:70]!r} found {text.count(old)} times'
        text = text.replace(old, new)
    return text


def config_values(text):
    out = {}
    for n in ast.parse(text).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def set_config(text, key, value):
    """Replace the value of the single-line `key = value` statement of cell 1, keeping its comment column."""
    hits = [n for n in ast.parse(text).body if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == key]
    assert len(hits) == 1, f'{key}: {len(hits)} assignments in cell 1'
    v = hits[0].value
    lines = text.split('\n')
    line = lines[v.lineno - 1]
    assert v.lineno == v.end_lineno and line.isascii(), key
    head, rest = line[:v.col_offset] + repr(value), line[v.end_col_offset:]
    m = re.fullmatch(r'(\s*)(#.*)?', rest)
    assert m, (key, rest)
    lines[v.lineno - 1] = head + (' ' * max(1, len(line) - len(m.group(2)) - len(head)) + m.group(2) if m.group(2) else '')
    return '\n'.join(lines)


def literal(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


nb = json.load(open(os.path.join(src, 'notebook.ipynb'), encoding='utf-8'))
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))
cells = nb['cells']
assert len(cells) == 23 and all(isinstance(c['source'], str) for c in cells), 'unexpected template layout'
for i, start in ((1, '# ── Configuration'), (11, '# ── PF-syn'), (16, '# ── Variant M'), (17, '## 4. What this run did')):
    assert cells[i]['source'].startswith(start), (i, cells[i]['source'][:60])

midx = open(os.path.join(here, 'midx_cell15.txt'), encoding='utf-8', newline='').read()
assert hashlib.sha256(midx.encode()).hexdigest() == MIDX_SHA256, 'midx_cell15.txt differs from the pulled source'
cells[16]['source'] = splice(midx, MIDX, 'cell 16 (midx)')
cells[11]['source'] = splice(cells[11]['source'], PF, 'cell 11 (PF-syn)')
cells[0]['source'] = splice(cells[0]['source'], MD0, 'cell 0 (title)')
cells[17]['source'] = splice(cells[17]['source'], MD17, 'cell 17 (section 4)')
c1 = splice(cells[1]['source'], CFG, 'cell 1 (config)')
defaults = config_values(c1)
settings = {}
for a in args:
    k, v = a.split('=', 1)
    assert k in defaults, f'{k}: not a cell-1 setting ({", ".join(sorted(defaults))})'
    v = literal(v)
    kind = lambda x: 'bool' if isinstance(x, bool) else 'number' if isinstance(x, (int, float)) else type(x).__name__
    assert kind(v) == kind(defaults[k]), f'{k}={v!r}: expected a {kind(defaults[k])}'
    v = float(v) if isinstance(defaults[k], float) else v      # ISO_ALPHA=0 -> 0.0, as the default is written
    settings[k] = v
    c1 = set_config(c1, k, v)
cfg = config_values(c1)
assert cfg['OURS_MODE'] in ('blend', 'replace', 'columns') and cfg['ISO_SCOPE'] in ('formula', 'all'), cfg
assert 0 <= cfg['PF_W'] <= 1 and 0 <= cfg['PF_MID_W'] <= 1 and cfg['ISO_ALPHA'] >= 0 and cfg['ISO_LIB_SKIP'] > 0, cfg
if (cfg['PF_MID_W'] > 0 or cfg['ISO_ALPHA'] > 0) and not cfg['OURS']:
    sys.exit('PF_MID_W / ISO_ALPHA read the PF-syn columns: they need OURS = True')
cells[1]['source'] = c1
cells.insert(17, {'cell_type': 'code', 'execution_count': None, 'id': ISO_ID,
                  'metadata': {'_kg_hide-input': True, 'jupyter': {'source_hidden': True}}, 'outputs': [],
                  'source': open(os.path.join(here, 'iso_cell.py'), encoding='utf-8').read()})
for i, c in enumerate(cells):
    if c['cell_type'] == 'code':
        ast.parse(c['source'], f'cell {i}')                   # syntax only; nothing here is executed
nb['metadata']['casmi']['ours'] = {
    'pf_checkpoint': 'bryantyu/casmi26-models/pf_tpu_f.pt', 'template': 'scripts/make_pub_variant.py',
    'final_layer': 'baidalinadilzhan/casmi26-midx-exact-dedup-fill cell 15 + PF_MID_W hook (cell 16)',
    'iso': 'scripts/pub_variant/iso_cell.py (cell 17)', 'config': 'cell 1'}
os.makedirs(dst, exist_ok=True)
with open(os.path.join(dst, 'notebook.ipynb'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(nb, ensure_ascii=False, separators=(',', ':')))
meta['id'] = f'bryantyu/{slug}'; meta['title'] = slug
with open(os.path.join(dst, 'kernel-metadata.json'), 'w') as f:
    f.write(json.dumps(meta, indent=2) + '\n')
print('wrote', dst, json.dumps({k: cfg[k] for k in ('OURS', 'OURS_MODE', 'PF_W', 'PF_MID_W', 'ISO_ALPHA', 'ISO_LIB_SKIP',
                                                     'ISO_KEEP_TRAIN', 'ISO_SCOPE')}), 'set', json.dumps(settings))
