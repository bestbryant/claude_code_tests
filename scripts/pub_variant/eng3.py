"""ENG3 builds of the rank line (ours): casmi26-pubm with Engine 2's lists := RRF(original Engine 2, our stage 1).

Called by scripts/make_pub_variant.py ENG3_W=<w> (w > 0, without ELIG) after its own structural edits; nothing here is
executed on public code: the frozen 0.465 notebook is base64-decoded, two of its cells get text appended (exact
anchors), and it is re-encoded and re-pinned, as scripts/pub_variant/elig.py does.
  frozen cell 32  the original Engine 2 runs exactly as in casmi26-pubm (its text is unchanged); then OUR stage 1
                  (casmi.run_eng2 from bryantyu/casmi26-elig-assets on bryantyu/casmi26-pool-v18c: the Engine 2 of
                  casmi26-pube1, run by eng2_merge.run_engine2 with elig1's configuration: cap 45 min from its start +
                  5 min grace, stall 15 min, CPU only, own process group, pinned code / model / pool hashes, log only).
                  Both outputs are kept: ENG3_ENGINE2 (= ENG as loaded from the original's eng_lists.json) and ENG3_OURS.
  frozen cell 39  (end; ENG's first reader is the ICE / GLACIER input union of cell 41, no cell in between reads it, and
                  `chem` is bound from cell 37 on) the single merge point: our keys re-checked with chem.score_key
                  as in elig1 (eng2_merge.verify_keys), Engine 2's keys checked against it (log only), ENG :=
                  eng3_merge.rrf_merge(ENG3_ENGINE2, ENG3_OURS, w, k 3, top 40); eng_lists.json rewritten with the merged
                  lists (the original's kept as eng_lists_engine2.json). Every later reader of ENG sees the merged lists.
  notebook cell 8 (appended)  writes eng2_merge.py (elig1's, byte-identical) and eng3_merge.py next to the pipeline
                  modules (SHA-256 pinned), checks cell 1's ENG3_W against the weight baked into the frozen cells, logs
                  the mounts of our two datasets.
  notebook cell 9  the compact trace shows the ENG3 lines (KEY_LINE) and labels frozen cell 32 as Engine 2 + ours.
  cell 1 ENG3_W line; markdown cell 0 statement; kernel-metadata dataset_sources = casmi26-pubm's + our two datasets.
"""
import base64
import hashlib
import importlib.util
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('pub_variant_elig_for_eng3', os.path.join(HERE, 'elig.py'))
elig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(elig)                 # our generator code (constants and helpers), not notebook code

FROZEN_ORIGINAL_SHA256 = elig.FROZEN_ORIGINAL_SHA256
DATASETS = list(elig.ELIG1_DATASETS)           # bryantyu/casmi26-elig-assets, bryantyu/casmi26-pool-v18c
K_RRF, N_TOP = 3, 40
# elig1's Engine 2 configuration (caps, procs, k + margin, rankers, FP / PF models, pinned assets and pool), own work dir
ENG3_CFG = dict(elig.ENG2_CFG, work='/tmp/eng3_ours')
OUT_JSON = '/kaggle/working/eng3_ours.json'     # eng2_merge.run_engine2 writes its log next to it: eng3_ours.log
# Files of our two datasets as published (make_elig_assets.py: 16 casmi modules + 6 model files + 4 text files; the
# pool as published): used by the collision check of scripts/pub_variant/test_eng3_build.py.
ASSET_FILES = sorted(elig.ELIG_ASSETS_SHA256) + ['README.md', 'LICENSE.txt', 'MODELS_LICENSE.txt', 'MANIFEST.sha256']
POOL_FILES = sorted(elig.POOL_V18C_SHA256)

C32_ANCHOR = ("except Exception as e:\n"
              "    print('ENGINE FAILED -> v4g lists kept:', repr(e))\n")
C39_ANCHOR = "BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}\n"


def c32_block(w):
    return (
        "\n"
        "# ── ENG3 (ours, rank line; scripts/make_pub_variant.py ENG3_W) ─────────────────────────────────────────────\n"
        "# The original Engine 2 above ran unchanged; its lists are ENG (eng_lists.json). OUR stage 1 runs next, in its own\n"
        "# process: casmi.run_eng2 from bryantyu/casmi26-elig-assets on bryantyu/casmi26-pool-v18c (stage 1 of\n"
        "# casmi26-own-s2v18cue, the Engine 2 of casmi26-pube1) with elig1's configuration: CPU only, own process group,\n"
        "# cap ENG3_CFG['cap_s'] from its start + grace, stall stop, pinned code / model / pool hashes (log only). Both\n"
        "# outputs are kept: ENG3_ENGINE2 (the original's lists) and ENG3_OURS. ENG is not read before cell 41 (ICE union);\n"
        "# the reciprocal-rank merge replaces it once, at the end of cell 39, where chem.score_key exists to re-check our\n"
        "# keys. Nothing here can stop the pipeline: any failure leaves ENG3_OURS empty and ENG the original's lists.\n"
        f"ENG3_W, ENG3_K, ENG3_N = {float(w)!r}, {K_RRF!r}, {N_TOP!r}\n"
        "ENG3_ENGINE2 = ENG\n"
        f"ENG3_CFG = {ENG3_CFG!r}\n"
        "ENG3_OURS, ENG3_REPORT = {}, dict(status='not run', seconds=0.0)\n"
        "try:\n"
        "    import eng2_merge as _eng2\n"
        f"    ENG3_OURS, ENG3_REPORT = _eng2.run_engine2(os.path.join(COMP, 'test.parquet'), os.path.join(COMP, 'train.parquet'),\n"
        f"                                               {OUT_JSON!r}, ENG3_CFG)\n"
        "    _eng2.write_json(ENG3_REPORT, '/kaggle/working/eng3_ours_report.json')\n"
        "    print(_eng2.report_line(ENG3_REPORT).replace('ENGINE 2 OURS subprocess', 'ENG3 OURS subprocess', 1), flush=True)\n"
        "    if ENG3_REPORT.get('log_tail'):\n"
        "        print(ENG3_REPORT['log_tail'], flush=True)\n"
        "except Exception as _e3_exc:                     # eng2_merge.run_engine2 never raises; an import error might\n"
        "    ENG3_OURS, ENG3_REPORT = {}, dict(status='failed', seconds=0.0, error=f'{type(_e3_exc).__name__}: {_e3_exc}'[:500])\n"
        "    print('ENG3 OURS subprocess FAILED (the original Engine 2 lists stay):', ENG3_REPORT['error'], flush=True)\n")


C39_BLOCK = (
    "\n"
    "# ── ENG3 (ours): the single point where Engine 2's lists change ─────────────────────────────────────────────────\n"
    "# ENG was created in cell 32 (the original Engine 2's eng_lists.json) and is first read by the ICE / GLACIER input\n"
    "# union of cell 41; no cell in between reads it. Here, with the pipeline's chem.score_key (chem is bound in cell 37):\n"
    "# our keys are re-checked as in elig1 (every key in the commit run; in the rerun the first 10 of each list when the\n"
    "# subprocess ran this RDKit, every key as soon as one differs; a different key is replaced, a falsy or repeated one\n"
    "# dropped), Engine 2's own keys are checked against it (log only), and ENG := per molecule the top ENG3_N identities\n"
    "# by 1 / (ENG3_K + r_engine2) + ENG3_W / (ENG3_K + r_ours) (eng3_merge.rrf_merge: ties by Engine 2 rank, then ours;\n"
    "# Engine 2's SMILES for shared identities; an Engine 2 entry without a key kept as its own identity at its rank;\n"
    "# Engine 2's entry unchanged where ours has no list). eng_lists.json then holds the merged lists (the original's:\n"
    "# eng_lists_engine2.json). Every later reader of Engine 2 -- ICE union (41), fuse2 (45), safe tail (48), metric-\n"
    "# identity fusion (49), postflight record (50) and the outsiders read from it, fragment gap (51), window-40 overlay\n"
    "# (52), rank-2 placement (53), receipt (54), export_evidence -- sees them. Any failure keeps the original lists.\n"
    "_t_e3 = time.time()\n"
    "ENG3_KEYCHECK, ENG3_E2KEYS, ENG3_STATS = {}, {}, {}\n"
    "try:\n"
    "    import eng2_merge as _eng2\n"
    "    import eng3_merge as _eng3\n"
    "    _e3_head = 10 if IS_RERUN and ((ENG3_REPORT.get('meta') or {}).get('versions') or {}).get('rdkit') == \\\n"
    "        rdkit.__version__ else None\n"
    "    ENG3_OURS, ENG3_KEYCHECK = _eng2.verify_keys(ENG3_OURS, chem.score_key, head=_e3_head)\n"
    "    ENG3_E2KEYS = _eng3.check_keys(ENG3_ENGINE2, chem.score_key, head=3 if IS_RERUN else None)\n"
    "    _e3_merged, ENG3_STATS = _eng3.rrf_merge(ENG3_ENGINE2, ENG3_OURS, ENG3_W, k=ENG3_K, n=ENG3_N)\n"
    "    _eng3.write_lists(ENG3_ENGINE2, '/kaggle/working/eng_lists_engine2.json')\n"
    "    _eng3.write_lists(_e3_merged, '/kaggle/working/eng_lists.json')\n"
    "    ENG = _e3_merged\n"
    "    print(_eng3.log_line(ENG3_STATS, ENG3_REPORT.get('seconds', 0.0)), flush=True)\n"
    "    print(_eng3.keys_line(ENG3_KEYCHECK, ENG3_E2KEYS, time.time() - _t_e3), flush=True)\n"
    "    print(_eng3.detail_line(ENG3_STATS), flush=True)\n"
    "except Exception as _e3_exc:\n"
    "    ENG = ENG3_ENGINE2\n"
    "    ENG3_STATS = dict(status='failed', error=f'{type(_e3_exc).__name__}: {_e3_exc}'[:500])\n"
    "    print('ENG3 merge FAILED (the original Engine 2 lists stay):', ENG3_STATS['error'], flush=True)\n"
    "    try:\n"
    "        with open('/kaggle/working/eng_lists.json', 'w') as _e3_f:\n"
    "            json.dump(ENG3_ENGINE2, _e3_f)\n"
    "    except Exception:\n"
    "        pass\n"
    "ENG3_SUMMARY = dict(w=ENG3_W, k=ENG3_K, n=ENG3_N, subprocess=ENG3_REPORT, key_check=ENG3_KEYCHECK,\n"
    "                    engine2_key_check=ENG3_E2KEYS, merge=ENG3_STATS, seconds_merge=round(time.time() - _t_e3, 1))\n"
    "try:\n"
    "    with open('/kaggle/working/eng3_report.json', 'w') as _e3_f:\n"
    "        json.dump(ENG3_SUMMARY, _e3_f, indent=1, default=str)\n"
    "except Exception as _e3_exc:\n"
    "    print('ENG3 report not written:', repr(_e3_exc), flush=True)\n")


def patch_frozen(raw, w):
    """Patched frozen notebook bytes (same JSON serialisation as the original), info and {cell: (old, new)}."""
    assert hashlib.sha256(raw).hexdigest() == FROZEN_ORIGINAL_SHA256, 'not the pinned frozen 0.465 notebook'
    nb = json.loads(raw)
    assert json.dumps(nb, separators=(',', ':'), ensure_ascii=True).encode() == raw, 'serialisation differs'
    cells = nb['cells']
    assert len(cells) == 59, len(cells)
    text = {i: ''.join(c['source']) for i, c in enumerate(cells)}
    assert text[32].startswith(elig.C32_HEAD) and 'eng_runner.py' in text[32]
    assert text[32].count("ENG = json.load(open('/kaggle/working/eng_lists.json'))") == 1      # where ENG is created
    assert text[32].endswith(C32_ANCHOR) and text[32].count(C32_ANCHOR) == 1, 'frozen cell 32 layout'
    assert text[39].endswith(C39_ANCHOR) and text[39].count(C39_ANCHOR) == 1, 'frozen cell 39 layout'
    readers = [i for i, t in text.items() if cells[i]['cell_type'] == 'code' and i != 32 and
               re.search(r"\bENG\b|'ENG'|\"ENG\"", t)]
    assert not [i for i in readers if i < 32] and min(i for i in readers if i > 32) == 41, \
        f'cells naming ENG besides 32: {readers}'           # created in 32, nothing names it in 33-40, first reader 41
    assert 'score_key' in text[45] and 'from casmi import fpnet, chem' in text[37], 'chem.score_key before cell 39'
    new = {32: text[32] + c32_block(w), 39: text[39] + C39_BLOCK}
    for i, t in new.items():
        assert cells[i]['cell_type'] == 'code', i
        compile(t, f'frozen cell {i}', 'exec', dont_inherit=True)        # syntax only (compile, never exec)
        cells[i]['source'] = elig.as_lines(t)
    out = json.dumps(nb, separators=(',', ':'), ensure_ascii=True).encode()
    info = dict(original_sha256=FROZEN_ORIGINAL_SHA256, patched_sha256=hashlib.sha256(out).hexdigest(),
                patched_cells=sorted(new), first_reader=min(i for i in readers if i > 32), readers=readers)
    return out, info, {i: (text[i], new[i]) for i in sorted(new)}


def module_sources():
    """{name: source} of the two runtime modules written by notebook cell 8 (ASCII, pinned)."""
    out = {n: open(os.path.join(HERE, n), encoding='utf-8').read() for n in ('eng2_merge.py', 'eng3_merge.py')}
    assert all(s.isascii() for s in out.values())
    return out


def cell8_block(info, w):
    mods = module_sources()
    pins = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in mods.items()}
    baked = dict(w=float(w), k=K_RRF, n=N_TOP, frozen_original_sha256=info['original_sha256'],
                 frozen_patched_sha256=info['patched_sha256'], frozen_patched_cells=info['patched_cells'])
    return (
        "# ── ENG3 (ours, rank line): Engine 2 merged with our stage 1, runtime modules ─────────────────────────────────\n"
        "# Generated by scripts/make_pub_variant.py ENG3_W (scripts/pub_variant/eng3.py, eng2_merge.py, eng3_merge.py).\n"
        "# Writes eng2_merge.py (elig1's Engine 2 runner, byte-identical) and eng3_merge.py (the reciprocal-rank merge) next\n"
        "# to the pipeline modules, SHA-256 pinned; patched frozen cells 32 and 39 import them. The weight is baked into the\n"
        "# patched frozen cells (pinned by FROZEN_SHA256): cell 1's ENG3_W must equal it (else this cell stops the run).\n"
        f"ENG3_BAKED = {baked!r}\n"
        f"ENG3_DATASETS = {DATASETS!r}\n"
        f"ENG3_MODULE_SHA256 = {pins!r}\n"
        f"ENG3_MODULE_SRC = {mods!r}\n"
        "\n"
        "\n"
        "def _eng3_setup():\n"
        "    import hashlib as _hl\n"
        "    w = globals().get('ENG3_W')\n"
        "    assert w == ENG3_BAKED['w'], (f'cell 1 ENG3_W {w!r} differs from the weight baked into the patched frozen cells '\n"
        "                                  f\"({ENG3_BAKED['w']!r}): rebuild with scripts/make_pub_variant.py ENG3_W=...\")\n"
        "    for name, src in ENG3_MODULE_SRC.items():\n"
        "        path = CODE / name\n"
        "        path.write_text(src, encoding='utf-8')\n"
        "        assert _hl.sha256(path.read_bytes()).hexdigest() == ENG3_MODULE_SHA256[name], f'{name} differs from its pin'\n"
        "    mounted = {}\n"
        "    for ref in ENG3_DATASETS:\n"
        "        owner, slug = ref.split('/', 1)\n"
        "        mounted[ref] = any(p.is_dir() for p in (INPUT / 'datasets' / owner / slug, INPUT / slug))\n"
        "    absent = [r for r, ok in mounted.items() if not ok]\n"
        "    print(f\"ENG3 setup | w {w!r} | k {ENG3_BAKED['k']} | top {ENG3_BAKED['n']} | frozen source {FROZEN_SHA256[:12]} \"\n"
        "          f\"(cells {ENG3_BAKED['frozen_patched_cells']} patched from {ENG3_BAKED['frozen_original_sha256'][:12]}) | \"\n"
        "          + ' | '.join(f'{n} {s[:12]}' for n, s in ENG3_MODULE_SHA256.items()) +\n"
        "          f\" | our datasets mounted {len(mounted) - len(absent)} / {len(mounted)}\"\n"
        "          + (f\" (ABSENT: {', '.join(absent)}: our stage 1 gets no list, Engine 2 stays the original)\" if absent else ''),\n"
        "          flush=True)\n"
        "    if FROZEN_SHA256 != ENG3_BAKED['frozen_patched_sha256']:\n"
        "        print('ENG3 WARNING: the pinned frozen source is not the one this cell was generated with', flush=True)\n"
        "    return dict(w=w, mounted=mounted, modules=ENG3_MODULE_SHA256, baked=ENG3_BAKED)\n"
        "\n"
        "\n"
        "ENG3_SETUP = _eng3_setup()\n")


C9 = [("    32: 'Engine 2 · two rankers', ", "    32: 'Engine 2 · two rankers + our stage 1', "),
      ("|OUTSIDERS|engine lists \\d+ |", "|OUTSIDERS|ENG3 |engine lists \\d+ |")]
C1_ANCHOR = ("ISO_FREEZE_LEADER = False     # True: rank 1 never moves (ranks 2-25 only; scope 'formula' keeps the leader's "
             "formula)\n")


def cell1_edits(w):
    return [(C1_ANCHOR, C1_ANCHOR +
             "\n"
             "# Ours (rank line, ENG3): Engine 2's lists := per molecule the top 40 by 1 / (3 + r_engine2) + ENG3_W / (3 + r_ours):\n"
             "# the original Engine 2 merged with our stage 1 (frozen cells 32 and 39; k 3 = the pipeline's own fusion offset\n"
             "# KRR). Baked into the patched frozen cells at build time (scripts/make_pub_variant.py ENG3_W=...): cell 8 stops\n"
             "# the run if this line no longer matches them.\n"
             f"ENG3_W = {float(w)!r}\n")]


TITLE0 = '# Variant M core + midx final layer + PF-syn + ISO isomer reranker\n\n'


def md0(text, w):
    assert text.startswith(TITLE0), text[:80]
    return (f"# Variant M core + midx final layer + ENG3: Engine 2 merged with our stage 1 (w {float(w)!r})\n\n"
            f"> **ENG3 (this build: scripts/make_pub_variant.py ENG3_W={float(w)!r}).** Engine 2's lists are the original\n"
            "> Engine 2 (frozen cell 32, unchanged) merged with our stage 1 (casmi.run_eng2 from bryantyu/casmi26-elig-assets\n"
            "> on bryantyu/casmi26-pool-v18c: stage 1 of our casmi26-own-s2v18cue, the Engine 2 of casmi26-pube1), which runs\n"
            "> right after it in its own process (CPU only; 45-minute cap from its start + 5 minutes grace, 15-minute stall\n"
            "> stop; any failure leaves the original lists). At the end of frozen cell 39, before the first reader, each\n"
            f"> molecule's Engine 2 list becomes the top 40 by 1 / (3 + Engine 2 rank) + {float(w)!r} / (3 + our rank) on the\n"
            "> pipeline's own structure identity (chem.score_key; ties: Engine 2 rank, then ours; Engine 2's SMILES for shared\n"
            "> structures; Engine 2's list unchanged where ours has none). Every reader of Engine 2 (ICEBERG / GLACIER input\n"
            "> union, two-engine fusion, safe tail, metric-identity fusion, fragment gap, window-40 overlay, rank-2\n"
            "> placement, evidence export, outside candidates) reads the merged lists. PF-syn re-weighting (OURS, PF_MID,\n"
            "> the PF blend) and ISO are off (OURS = False), but this is not a PF-free contrast to casmi26-pubm: the rankers\n"
            "> of our stage 1 take PeakFormer features from pf_tpu_f.pt (the PF-syn checkpoint), so PF-syn evidence enters\n"
            "> through the added list (a PF-free variant is not possible with these rankers; casmi.run_eng2 requires the\n"
            "> PeakFormer columns). Our two datasets are attached for this engine only; the ISO cell and its cell-1 knobs\n"
            "> ISO_COS_W / ISO_FREEZE_LEADER come from a newer generator version than casmi26-pubm's (off here, as there).\n>\n"
            + text[len(TITLE0):])


def apply(nb, meta, w, cfg):
    """Turn the generated pub variant (nb, meta) into the ENG3 build with weight w in place; returns a report."""
    w = float(w)
    assert math.isfinite(w) and w > 0, f'ENG3_W={w!r}: a finite weight > 0'
    assert 'ELIG' not in cfg, 'ENG3 builds are not ELIG builds'
    cells = nb['cells']
    i8 = next(i for i, c in enumerate(cells) if c.get('id') == '3e2e781f')
    c8 = cells[i8]['source']
    head = "FROZEN_SHA256 = '" + FROZEN_ORIGINAL_SHA256 + "'\nFROZEN_B64 = '"
    assert c8.count(head) == 1, 'cell 8 layout'
    start = c8.index(head) + len(head)
    end = c8.index("'\n", start)
    raw = base64.b64decode(c8[start:end])
    patched, info, changes = patch_frozen(raw, w)
    b64 = base64.b64encode(patched).decode()
    assert base64.b64decode(b64) == patched
    cells[i8]['source'] = (c8[:c8.index(head)] + "FROZEN_SHA256 = '" + info['patched_sha256'] +
                           "'   # eng3: frozen 0.465 notebook with the ENG3 patches of cells 32 and 39 (original " +
                           FROZEN_ORIGINAL_SHA256[:12] + "...)\nFROZEN_B64 = '" + b64 + c8[end:])
    cells[i8]['source'] = cells[i8]['source'].rstrip('\n') + '\n\n\n' + cell8_block(info, w)
    i9 = next(i for i, c in enumerate(cells) if c.get('id') == 'ac4866c0')
    cells[i9]['source'] = elig.splice(cells[i9]['source'], C9, f'cell {i9} (stage machinery, eng3)')
    cells[0]['source'] = md0(cells[0]['source'], w)
    nb['metadata']['casmi']['frozen_source_sha256'] = info['patched_sha256']
    record = dict(
        w=w, k=K_RRF, n=N_TOP, rule='top n by 1/(k + r_engine2) + w/(k + r_ours) on chem.score_key; ties Engine 2 rank, '
        'then ours; Engine 2 SMILES for shared identities; Engine 2 unchanged without a list of ours',
        frozen_original_sha256=FROZEN_ORIGINAL_SHA256, frozen_patched_sha256=info['patched_sha256'],
        frozen_patched_cells=info['patched_cells'], merge_point='end of frozen cell 39 (first reader: cell %d)'
        % info['first_reader'], modules={n: hashlib.sha256(s.encode()).hexdigest() for n, s in module_sources().items()},
        engine=dict(entry='python -m casmi.run_eng2 (bryantyu/casmi26-elig-assets)', pool='bryantyu/casmi26-pool-v18c',
                    config=ENG3_CFG, peakformer=f"{ENG3_CFG['pf']} (the PF-syn checkpoint): PeakFormer features inside "
                    'the stage-1 rankers; OURS / PF_MID / ISO (PF-syn re-weighting) are off, but the build is not PF-free'),
        keyless='an Engine 2 entry without a key stays its own identity at its own rank (SMILES and key as written)',
        datasets=DATASETS, notebook_cells_edited=sorted({0, 1, i8, i9}))
    assert not set(DATASETS) & set(meta['dataset_sources']), meta['dataset_sources']
    meta['dataset_sources'] = list(meta['dataset_sources']) + DATASETS
    return dict(frozen=info, changes=changes, frozen_patched=patched, cells_edited=sorted({0, 1, i8, i9}),
                record=record)          # the generator stores record as metadata casmi.ours.eng3
