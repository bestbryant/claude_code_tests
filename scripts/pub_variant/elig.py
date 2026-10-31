"""Eligibility builds of the public-lineage fork (ours): ELIG=0 = fixes F1 + F2a + F3 + F4 of docs/fork_eligibility.md.

Called by scripts/make_pub_variant.py (ELIG=0) after its own structural edits; nothing here is executed on public
code: the frozen 0.465 notebook is base64-decoded, its cell text patched by exact splices, re-encoded and re-pinned.
  F1  kernel-metadata dataset_sources: the original upstream datasets instead of the CC BY-NC-SA v4g container (and
      without the knowledge table, the panel benchmark and catalogue-experiment-saner); our bryantyu datasets only when
      the build runs PF-syn (OURS = True). Frozen cell 5 sets V4G / PANEL / TABLE to None, display cells 9 / 12 / 14 /
      16 / 18 are skipped, cell 54 hashes the new dataset list and tags its receipt; code appended to notebook cell 8
      (elig_cell.py) checks the mounted files against casmi26-pubm's v43_mounted_input_hashes.json (and the larger
      files of the re-mounted datasets in full against the container's MANIFEST sha256) and writes null_engine.py.
  F2a frozen cell 32 runs no Engine 2; cells 39 and 45 fill ENG with Engine 1's own lists (null_engine.py).
  F3  frozen cell 26 SRC_BITS 0 (pc_src_mask.npy no longer required); cell 1 SPEC_W 0.0 and the final layer (cell 16)
      does not load src_spectral_library (NP_W / NP_BONUS kept: NPAtlas is host-allowed).
  F4  frozen cell 36: the knowledge table's external reference spectra are not built.
  Markdown: cell 0 eligibility statement, section 6 / 8 (run it yourself, credits) name the attached datasets.
ELIG=1 (F2b: our stage 1 as Engine 2) is reserved.
"""
import ast
import base64
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_ORIGINAL_SHA256 = 'd1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88'
V43_FILE = 'v43_mounted_input_hashes_pubm.json'      # casmi26-pubm v1 commit run (Nov 1), 593 rows
V43_SHA256 = 'b07ab9e3b6ee3fc4deba37ce94c974450d38ee16ad332aafc7d02669af093347'

# F1: what the frozen pipeline reads, from the original publishers (listings checked 2026-10-31: every file of the
# matching v4g container folder is present with the same relative path and size; see the generator report).
UPSTREAM = ['ahmedberatozer/casmi26-v4b-models', 'ahmedberatozer/casmi26-v3-models',
            'ahmedberatozer/casmi26-fpnet-full1', 'ahmedberatozer/casmi26-v2-pool', 'ahmedberatozer/casmi26-iceberg',
            'ahmedberatozer/casmi26-glacier', 'dmitriigluzdov/casmi26-pubchem-popularity-prior',
            'dmitriigluzdov/casmi26-fold-safe-fpnet', 'sanpier/casmi-spectrum-model-v1', 'metric/rdkit-2026-3-3-wheel']
OURS_DATASETS = ['bryantyu/casmi26-code', 'bryantyu/casmi26-models']   # read only by the PF-syn cell 11: attached only
                                                                       # when the build has OURS = True (Kaggle licence
                                                                       # still "unknown", G19 / F8)
DETACHED = ['dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets',
            'dmitriigluzdov/casmi26-natural-product-knowledge-table',
            'dmitriigluzdov/casmi26-natural-product-panel-benchmark', 'sanpier/catalogue-experiment-saner']
# Upstream versions checked on 2026-10-31 (kernel-metadata cannot pin a version; Kaggle pins the version current at
# push time): lastUpdated of each dataset as `kaggle datasets list --user <owner>` showed it.
CHECKED_VERSIONS = {
    'ahmedberatozer/casmi26-v4b-models': '2026-09-26 12:21:47', 'ahmedberatozer/casmi26-v3-models': '2026-09-23 10:45:15',
    'ahmedberatozer/casmi26-fpnet-full1': '2026-09-29 10:10:49', 'ahmedberatozer/casmi26-v2-pool': '2026-09-21 08:18:58',
    'ahmedberatozer/casmi26-iceberg': '2026-09-26 09:48:06', 'ahmedberatozer/casmi26-glacier': '2026-09-28 09:32:41',
    'dmitriigluzdov/casmi26-pubchem-popularity-prior': '2026-10-27 17:07:06',
    'dmitriigluzdov/casmi26-fold-safe-fpnet': '2026-10-21 14:41:29',
    'sanpier/casmi-spectrum-model-v1': '2026-10-23 17:40:23', 'metric/rdkit-2026-3-3-wheel': '2026-09-08 22:23:23'}

# Full SHA-256 of the larger (> 32 MiB) files of the re-mounted datasets, for the commit-run hash check of cell 8 (the
# v43 rows hold only size + first / last 4 MiB for them). Source: the v4g container's MANIFEST.json (sha256 2593c1e95b98...,
# equal to its v43 row; scratchpad copy inv/dl_casmi26-attributed-v4g-reproduction-assets): files v2_pool/*,
# iceberg/wheels/...cp312...whl.ice, v3_models/fpnet_0|1.pt (the container lists those as the bytes of v4b fe_A / fe_B,
# whose sha256 the v4b MANIFEST.json (e64b96003776...) pins and frozen cell 26 verifies in full; v43 head/tail agree).
FULL_REFS = {
    'ahmedberatozer/casmi26-v2-pool/pool_fp.npy': (908986707, 'f02d91eeb4e2be5687ce678c199aa09a3acda1e821478372fc23fd1c7edf55e8'),
    'ahmedberatozer/casmi26-v2-pool/pool_frag_mass.npy': (357116708, '9afbdc02856b287e06e6c9f61a9f0032b2dd53b20557942c1d00542d9c490ffc'),
    'ahmedberatozer/casmi26-v2-pool/pool_meta.parquet': (36692358, '82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0'),
    'ahmedberatozer/casmi26-v2-pool/train_fp_sel.npy': (352761118, '0e6ddfbfb188b6cffb9f1b0ad9c7ba268ef709ce60b2711d05f98096cd772a8d'),
    'ahmedberatozer/casmi26-iceberg/wheels/rdkit-2025.3.6-cp312-cp312-manylinux_2_28_x86_64.whl.ice':
        (36119142, 'ed5565d4d6423def7c52eb20a6263afcd979383757086dec92f57e77d7c7f856'),
    'ahmedberatozer/casmi26-v3-models/fpnet_0.pt': (198790165, 'edccf479a2a2e53461a06bc5e1aee4341b22abb32302c4f6d2c45281000d742a'),
    'ahmedberatozer/casmi26-v3-models/fpnet_1.pt': (198790165, '6f60a22b3c3b21aa48d47328bb9523bdf07cc004bfb72966e0f9713507c24f2e')}
# Datasets whose larger files cell 8 hashes in full in the commit run (about 2.5 GB): those that replace container folders,
# except v4b (frozen cell 26 checks all 64 files of its MANIFEST.json in full). Files without a FULL_REFS entry
# (fpnet_full1.pt, glacier.pt, the metric wheels) get their full SHA-256 recorded as the reference for later runs.
FULL_DATASETS = ['ahmedberatozer/casmi26-v3-models', 'ahmedberatozer/casmi26-fpnet-full1', 'ahmedberatozer/casmi26-v2-pool',
                 'ahmedberatozer/casmi26-iceberg', 'ahmedberatozer/casmi26-glacier', 'metric/rdkit-2026-3-3-wheel']

SKIP = {9: "asset inventory of the v4g container", 12: "documentation-prior table (shows the NIST-flagged "
        "src_spectral_library column)", 14: "documentation-prior benchmark table (shows the NIST-flagged 'spectral "
        "libraries' signal)", 16: "natural-product panel benchmark", 18: "natural-product knowledge table"}
SKIP_FIRST = {9: 'ROLE = {', 12: 'try:\n    cols = [', 14: "try:\n    b = pd.read_csv(POP / 'BENCHMARK_SOURCES.csv')",
              16: 'try:\n    panel = pd.read_parquet(PANEL', 18: 'try:\n    nt = pd.read_parquet(TABLE'}

C5 = [("V4G = dataset_dir('casmi26-attributed-v4g-reproduction-assets', 'ATTRIBUTION.md')\n"
       "POP = dataset_dir('casmi26-pubchem-popularity-prior', 'pc_lsid.npy')\n"
       "PANEL = dataset_dir('casmi26-natural-product-panel-benchmark', 'panel.parquet')\n"
       "TABLE = dataset_dir('casmi26-natural-product-knowledge-table', 'natural_products.parquet')\n",
       "# elig0 (ours, fix F1 of docs/fork_eligibility.md): the CC BY-NC-SA v4g container, the panel benchmark and the\n"
       "# knowledge table are not mounted and never read: V4G / PANEL / TABLE stay None even if a copy is attached. The\n"
       "# display cells 9, 12, 14, 16 and 18 are skipped; cell 36 builds no external reference spectra (F4). The finders of\n"
       "# cells 26-41 are path globs (casmi26-v4b-models/**/..., pool_meta.parquet, ...) and resolve to the upstream datasets.\n"
       "V4G = None\n"
       "POP = dataset_dir('casmi26-pubchem-popularity-prior', 'pc_lsid.npy')\n"
       "PANEL = None\n"
       "TABLE = None\n"),
      ("assert FOLD is not None, 'Attach the updated casmi26-fold-safe-fpnet dataset'",
       "assert FOLD is not None, 'Attach the updated casmi26-fold-safe-fpnet dataset'\n"
       "print('elig0 | V4G', V4G, '| PANEL', PANEL, '| TABLE', TABLE, '| POP', POP, '| FOLD', FOLD, flush=True)")]

C26 = [("SRC_BONUS, SRC_BITS = 0.5, 1   # documentation prior + bonus for depositor-source groups (pc_src_mask.npy, dataset v5)\n",
        "SRC_BONUS, SRC_BITS = 0.5, 0   # documentation prior + bonus for depositor-source groups (pc_src_mask.npy, dataset v5)\n"
        "# elig0 (ours, fix F3): SRC_BITS 0 (was 1) -> no depositor-source bonus; bit 1 groups the PubChem deposits of the NIST\n"
        "# Mass Spectrometry Data Center with MassBank / MoNA. The PubChem channel (cell 34) then neither loads nor requires\n"
        "# the tier's pc_src_mask.npy (it is loaded only when bonus > 0 and bits); the delta / growth supplement stores still\n"
        "# memory-map their *_src_mask.npy and gather window slices that no score reads ('sm' is used only when loaded for\n"
        "# the tier, and & 0 would zero it). Left as is: that code runs in the Pool initializer of pc_runner.py.\n"),
       ("assert os.path.exists(os.path.join(os.environ['CASMI_POP_DIR'], 'pc_src_mask.npy')), 'pc_src_mask.npy missing'\n",
        "assert not SRC_BITS or os.path.exists(os.path.join(os.environ['CASMI_POP_DIR'], 'pc_src_mask.npy')), \\\n"
        "    'pc_src_mask.npy missing'   # elig0 (F3): required only when the depositor-source bits are used\n")]

C32_HEAD = "'''Our engine (two-ranker + BIO + AFIX) in a subprocess, then weighted reciprocal-rank fusion with the v4g lists.'''\n"
C32_NEW = (C32_HEAD +
           "# elig0 (ours, fix F2a of docs/fork_eligibility.md): Engine 2 is NOT run and its sources are not written. Its\n"
           "# inputs (prvsiyan FP models, ranker rows and COCONUT bank; megayak ranker rows; sergiomalv ChEBI/LMSD;\n"
           "# franciscoangulo PubChemLite) are not mounted. The null engine (null_engine.py, written by notebook cell 8)\n"
           "# fills ENG with Engine 1's own lists in Engine 2's schema {molecule_id: {smiles, keys}}:\n"
           "# step 1 at the end of cell 39 (Engine 1's top 40, read by the ICE union of cell 41), step 2 in cell 45 before\n"
           "# the fusion (the main list + Engine 1's tail; written to eng_lists.json as Engine 2 did).\n"
           "ENG = {}\n"
           "Path('/kaggle/working/eng_lists.json').unlink(missing_ok=True)\n"
           "print('ENGINE 2 OFF (elig0 F2a): no subprocess; ENG is filled from Engine 1 in cells 39 and 45', flush=True)\n")

C36 = [("_EXTERNAL_REFERENCE_STATE = _build_external_reference_cache('work', TABLE, _EXTERNAL_REFERENCE_PINS)\n",
        "# elig0 (ours, fix F4): the knowledge table's external reference spectra are not added to the library; cell 37\n"
        "# then logs 'external reference spectra disabled: off in elig0 ...' and builds the engine on Library('work').\n"
        "_EXTERNAL_REFERENCE_STATE = dict(path=None, added=0, keys=0,\n"
        "                                 error='off in elig0 (F4: external reference spectra not used)')\n")]

C39 = [("BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}\n",
        "BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}\n"
        "\n"
        "# elig0 (ours, fix F2a): null Engine 2, step 1 of 2. Until the fusion ENG holds Engine 1's own top 40 per molecule in\n"
        "# Engine 2's schema, with Engine 1's keys (chem.score_key, the identity Engine 2 wrote), so the ICE union of cell 41\n"
        "# adds no engine-only candidate and ICEBERG / GLACIER score the base list's same-formula groups as before.\n"
        "import null_engine as _null_engine\n"
        "ENG = _null_engine.null_engine_lists(te.molecule_id.astype(str).unique(), {str(m): v[0] for m, v in BASE.items()},\n"
        "                                     keys={str(m): v[1] for m, v in BASE.items()}, n=40)\n"
        "_null_engine.write_lists(ENG, '/kaggle/working/eng_lists_step1.json')\n"
        "print('NULL ENGINE 2 step 1 (Engine 1 top 40, read by the ICE union) |', _null_engine.summary(ENG), flush=True)\n")]

C45 = [("FUSION_OVERFLOW = {}\nif ENG:\n",
        "FUSION_OVERFLOW = {}\n"
        "# elig0 (ours, fix F2a): null Engine 2, step 2 of 2. ENG := the main list fused below (PREFUSION_LISTS: Engine 1 +\n"
        "# the gated PubChem merge, verbatim) followed by Engine 1's tail (BASE_TAIL_CANDIDATES, keys not listed yet), top 40,\n"
        "# keys chem.score_key(s) or s as fuse2 computes them for the main list. Fused with itself the main list keeps its\n"
        "# order (each entry scores (1 + ALPHA) times its own score), here and in cells 49 / 52, before their GLACIER\n"
        "# re-rank; the tail only fills ranks after it. Engine 1's top 40 alone would demote every PubChem insert and undo\n"
        "# every PubChem promotion of cell 43. Written to eng_lists.json, the file Engine 2 wrote.\n"
        "import null_engine as _null_engine\n"
        "ENG = _null_engine.null_engine_lists(te.molecule_id.astype(str).unique(), PREFUSION_LISTS, tails=BASE_TAIL_CANDIDATES,\n"
        "                                     key_of=lambda s: chem.score_key(s) or s, n=40)\n"
        "_null_engine.write_lists(ENG, '/kaggle/working/eng_lists.json')\n"
        "print('NULL ENGINE 2 step 2 (main list + Engine 1 tail, fused below) |', _null_engine.summary(ENG), flush=True)\n"
        "if ENG:\n")]

V43_OLD = ("V43_DATASET_SOURCES = ['dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets', "
           "'dmitriigluzdov/casmi26-pubchem-popularity-prior', 'dmitriigluzdov/casmi26-natural-product-panel-benchmark', "
           "'dmitriigluzdov/casmi26-natural-product-knowledge-table', 'dmitriigluzdov/casmi26-fold-safe-fpnet']\n")
V43_METHOD = ("        version=49, method='canqiang_E006_window40_ice_fallback_plus_v7c_fragment_gap_plus_e2_rank2_rel150_e2_bags_"
              "chebi_lmsd_pubchem_spectral_bonus_np_bonus_pubchemlite_lotus_refs_depth6_poplam015_fold0_n1a_syn050_pcdelta_"
              "extlib_n1challenger_jointicegl',\n")
V43_ELIG = dict(level=0, fixes=['F1', 'F2a', 'F3', 'F4'], doc='docs/fork_eligibility.md (bryantyu)',
                engine2='null: Engine 1 main list (+ gated PubChem merge) and Engine 1 tail in Engine 2 schema; '
                        'no Engine 2 subprocess, no prvsiyan / megayak / sergiomalv / franciscoangulo inputs',
                src_bits=0, spec_w=0.0, external_reference_spectra='off',
                mounts='upstream datasets (mounted_datasets); no v4g container, knowledge table or panel benchmark',
                method_note='method names the upstream E006 recipe. In elig0 its e2_* parts run on the null Engine 2 '
                            '(engine2_lists / engine2_nonempty count the null lists), chebi_lmsd and pubchemlite (Engine 2 '
                            'inputs) are not run, spectral_bonus is off (SRC_BITS 0) and extlib is off (F4)')
C54 = [(V43_OLD,
        "# elig0 (ours, fix F1): the receipt hashes the upstream datasets mounted instead of the v4g container (was: the five\n"
        "# dmitriigluzdov datasets); compare v43_mounted_input_hashes.json with casmi26-pubm's by path inside each dataset.\n"
        f"V43_DATASET_SOURCES = {UPSTREAM!r}\n"
        "# elig0 (ours): the receipt says what of its method string this build does not run\n"
        f"V43_ELIG = {V43_ELIG!r}\n"),
       (V43_METHOD, V43_METHOD + "        elig=V43_ELIG,\n")]


def splice(text, edits, what):
    for old, new in edits:
        assert text.count(old) == 1, f'{what}: {old[:70]!r} found {text.count(old)} times'
        text = text.replace(old, new)
    return text


def as_lines(text):
    """Back to the notebook's list-of-lines form ('\\n' kept on every line but the last)."""
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def patch_frozen(raw):
    """Patched frozen notebook bytes (same JSON serialisation as the original) and a record of what changed."""
    assert hashlib.sha256(raw).hexdigest() == FROZEN_ORIGINAL_SHA256, 'not the pinned frozen 0.465 notebook'
    nb = json.loads(raw)
    assert json.dumps(nb, separators=(',', ':'), ensure_ascii=True).encode() == raw, 'serialisation differs'
    cells = nb['cells']
    assert len(cells) == 59, len(cells)
    text = {i: ''.join(c['source']) for i, c in enumerate(cells)}
    new = {}
    for i, why in SKIP.items():
        assert cells[i]['cell_type'] == 'code' and text[i].startswith(SKIP_FIRST[i]), (i, text[i][:60])
        new[i] = (f"# elig0 (ours, fix F1 of docs/fork_eligibility.md): display-only cell skipped -- {why}.\n"
                  f"print('elig0: frozen cell {i} skipped (display only: {why.split(' (')[0]})', flush=True)\n")
    new[5] = splice(text[5], C5, 'frozen cell 5')
    new[26] = splice(text[26], C26, 'frozen cell 26')
    assert text[32].startswith(C32_HEAD) and text[32].count("ENG = {}\ntry:\n") == 1 and 'eng_runner.py' in text[32]
    new[32] = C32_NEW
    new[36] = splice(text[36], C36, 'frozen cell 36')
    new[39] = splice(text[39], C39, 'frozen cell 39')
    new[45] = splice(text[45], C45, 'frozen cell 45')
    new[54] = splice(text[54], C54, 'frozen cell 54')
    for i, t in new.items():
        assert cells[i]['cell_type'] == 'code', i
        compile(t, f'frozen cell {i}', 'exec', dont_inherit=True)    # syntax only (compile, never exec)
        cells[i]['source'] = as_lines(t)
    out = json.dumps(nb, separators=(',', ':'), ensure_ascii=True).encode()
    info = dict(original_sha256=FROZEN_ORIGINAL_SHA256, patched_sha256=hashlib.sha256(out).hexdigest(),
                patched_cells=sorted(new), skipped_cells=sorted(SKIP))
    return out, info, {i: (text[i], new[i]) for i in sorted(new)}


def cell1_edits():
    return [("POP_W, SPEC_W, NP_W = 0.10, 0.05, 0.02\n",
             "POP_W, SPEC_W, NP_W = 0.10, 0.0, 0.02   # elig0 (F3): SPEC_W 0.0 (was 0.05; spectral-library depositor flag\n"
             "                                        # includes the NIST MS Data Center); NP_W kept (NPAtlas is host-allowed)\n"),
            ("# ── Configuration ─────────────────────────────────────────────────────────────────────────────\n",
             "# ── Configuration ─────────────────────────────────────────────────────────────────────────────\n"
             "ELIG = 0                      # eligibility build (docs/fork_eligibility.md): 0 = F1 upstream mounts + F2a null\n"
             "                              # Engine 2 + F3 no NIST depositor flag + F4 no external spectra; 1 = F2b (reserved)\n")]


OPEN = ['F5a ICEBERG / GLACIER glue licence (owner question)', 'F6 host question (reconstructed retrains; NIST depositor flag)',
        'F7 saner retrain demonstration', 'F8 licence and attribution package (MIT LICENSE, NOTICE, fork ATTRIBUTION / '
        'reproduce, Kaggle licences of our datasets G19)']


def md0(ours):
    """Markdown cell 0: the eligibility statement (fix list, what differs from casmi26-pubm, what is open)."""
    if ours:
        mine = ("bryantyu/casmi26-code and casmi26-models are attached for PF-syn (OURS = True); their Kaggle licence is "
                "not set yet (G19, part of F8)")
    else:
        mine = ("our bryantyu/casmi26-code and casmi26-models are not attached either: with OURS = False no cell reads "
                "them (PF-syn, cell 11, is off, and so is ISO, cell 17)")
    return f"""# Eligible fork elig0 · Variant M core + midx final layer, without Engine 2, upstream datasets only

> **Eligibility build elig0** (scripts/make_pub_variant.py with ELIG=0; the audit and fixes F1-F8 are in
> docs/fork_eligibility.md of our repository). Changed against casmi26-pubm:
> - **F1, upstream datasets.** The CC BY-NC-SA container dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets, the
>   natural-product knowledge table, the panel benchmark and sanpier/catalogue-experiment-saner are no longer attached;
>   {mine}.
>   The pipeline reads the original datasets: ahmedberatozer casmi26-v4b-models, casmi26-v3-models, casmi26-fpnet-full1,
>   casmi26-v2-pool, casmi26-iceberg and casmi26-glacier; dmitriigluzdov casmi26-pubchem-popularity-prior and
>   casmi26-fold-safe-fpnet; sanpier/casmi-spectrum-model-v1; metric/rdkit-2026-3-3-wheel. The frozen pipeline's display
>   cells 9, 12, 14, 16 and 18 are skipped. Before the pipeline runs, cell 8 checks the mounted files against the hashes
>   of the casmi26-pubm run (files up to 32 MiB in full, larger ones by size and their first and last 4 MiB) and hashes
>   the larger files of the re-mounted datasets in full (log only). The pipeline's receipt carries an `elig` field.
> - **F2a, null Engine 2.** Engine 2 (prvsiyan, megayak, sergiomalv and franciscoangulo inputs) is not run. Every reader
>   of its lists gets the pipeline's own: Engine 1's top 40 until the PubChem merge (read by the ICEBERG / GLACIER
>   input union), then the main list (Engine 1 + the gated PubChem channel) followed by Engine 1's tail (frozen cells 32,
>   39 and 45; null_engine.py).
> - **F3.** No PubChem depositor-source bonus (SRC_BITS 0: that bit groups the NIST Mass Spectrometry Data Center's
>   deposits with MassBank / MoNA) and SPEC_W 0 in the final layer, which then does not load the flag column. The
>   PubChem channel's delta / growth supplement stores still memory-map their source-mask arrays, which no score reads.
>   NP_W / NP_BONUS stay: NPAtlas is on the host's list.
> - **F4.** The knowledge table's external reference spectra are not added to the library (frozen cell 36).
> - Sections 6 and 8 name the datasets attached here. Not part of the fixes: the ISO cell (cell 17, off with
>   ISO_ALPHA 0 in both builds) and two of its cell-1 knobs (ISO_COS_W, ISO_FREEZE_LEADER) come from a newer generator
>   version than casmi26-pubm's.
>
> **Still open (this build is not eligible until they are closed):** F5a, the ICEBERG / GLACIER runner glue
> (ice_runner.py, fuse.py, gl_runner.py, gl_fuse.py, shim/ in ahmedberatozer casmi26-iceberg / casmi26-glacier) carries
> no OSI licence yet: an owner question to its author. F6, a host question: do reconstructed, not bit-exact retrains with
> public code (Engine 1's ranker_v4b and fe_v4 tables, saner, our own G4 / G5) meet the reproducibility rule, and does a
> PubChem depositor flag naming the NIST MS Data Center count as using NIST? F7, a demonstrated retrain of saner's
> spectrum model. F8, the licence and attribution package: an MIT LICENSE for our code, a NOTICE for the Apache-2.0 and
> MIT sources, the fork's ATTRIBUTION and reproduce files, and the Kaggle licences of our datasets (G19).

"""


# Section 6 (run it yourself) and 8 (credits): name the datasets this build attaches; the forked notebook's table stays as
# upstream lineage, not attached.
RUN_EDIT = [("Copy & Edit this notebook (the competition data and the datasets in the credits stay\nattached).",
             "Copy & Edit this notebook (the competition data and the datasets\nattached to this build, listed in section 8, "
             "stay attached).")]
CREDITS_HEAD = "| dataset | author | used for | licence |\n|---|---|---|---|\n"
CREDITS_TAIL = ("The competition data is CC BY-NC 4.0, and the models above are trained on it: use them non-commercially, "
                "with\nattribution. We wrote the field record, the figure and the audit cells.")
KG = 'https://www.kaggle.com/datasets/'
ATTACHED = [
    ('ahmedberatozer/casmi26-v4b-models', 'Ahmed Berat Özer', "Engine 1: the casmi package, the fe_v4 feature families, "
     "FPNet A / B, fe_D and ranker_v4b", 'Kaggle "other"; code MIT'),
    ('ahmedberatozer/casmi26-v3-models', 'Ahmed Berat Özer', "PubChem channel: the v3 casmi code and FPNet A / B",
     'Kaggle "other"; code MIT'),
    ('ahmedberatozer/casmi26-fpnet-full1', 'Ahmed Berat Özer', "Engine 1's retrieval fingerprint model (fpnet_full1.pt)",
     'Kaggle "other"; training code MIT'),
    ('ahmedberatozer/casmi26-v2-pool', 'Ahmed Berat Özer', "the candidate pool (train + COCONUT, 710,701 structures) "
     "and its fingerprints", 'Kaggle "other"'),
    ('ahmedberatozer/casmi26-iceberg', 'Ahmed Berat Özer', "ICEBERG weights (ms-pred, Coley group), simulator wheels "
     "and runner glue", 'Kaggle "other"; weights MIT; wheels BSD-3 / Apache-2.0 / MIT; glue without an OSI licence (F5a)'),
    ('ahmedberatozer/casmi26-glacier', 'Ahmed Berat Özer', "GLACIER weights (ms-pred, Coley group) and runner glue",
     'Kaggle "other"; weights MIT; glue without an OSI licence (F5a)'),
    ('dmitriigluzdov/casmi26-pubchem-popularity-prior', 'Dmitrii Gluzdov', "PubChem tier and popularity prior (its "
     "depositor-source flags are not used here: SRC_BITS 0, SPEC_W 0)", "Other: PubChem courtesy of the U.S. NLM; LOTUS "
     "CC0; NPAtlas CC BY-NC 4.0 (NPAtlas is on the host's list)"),
    ('dmitriigluzdov/casmi26-fold-safe-fpnet', 'Dmitrii Gluzdov', "fold-safe FPNet checkpoint of the PubChem channel",
     'Other: checkpoint under competition data terms; scripts Apache 2.0'),
    ('sanpier/casmi-spectrum-model-v1', 'Saner Turhaner', 'the spectrum-candidate model', 'code Apache-2.0; trained on '
     'competition data (CC BY-NC 4.0)'),
    ('metric/rdkit-2026-3-3-wheel', 'Kaggle (metric)', "RDKit 2026.03.3, the metric's canonicalisation",
     'RDKit BSD-3 (Kaggle "unknown")')]
ATTACHED_OURS = [('bryantyu/casmi26-code', 'us', 'PF-syn: the casmi package (OURS = True)', 'ours; Kaggle "unknown" (G19)'),
                 ('bryantyu/casmi26-models', 'us', 'PF-syn PeakFormer pf_tpu_f.pt (OURS = True)', 'ours; Kaggle "unknown" (G19)')]


def credits_edit(text, ours):
    """Section 8: the attached datasets first; the forked notebook's table kept as upstream lineage (not attached)."""
    rows = ATTACHED + (ATTACHED_OURS if ours else [])
    assert sorted(r[0] for r in rows) == sorted(UPSTREAM + (OURS_DATASETS if ours else []))
    table = CREDITS_HEAD + ''.join(f'| [{r}]({KG}{r}) | {a} | {u} | {l} |\n' for r, a, u, l in rows)
    i, j = text.index(CREDITS_HEAD), text.index(CREDITS_TAIL)
    assert text.count(CREDITS_HEAD) == 1 and text.count(CREDITS_TAIL) == 1 and i < j
    old = [line for line in text[i + len(CREDITS_HEAD):j].split('\n') if line.startswith('| [')]
    lineage = [line for line in old if line.split('](', 1)[0][3:] in DETACHED]
    assert len(old) == 7 and len(lineage) == len(DETACHED), (len(old), len(lineage))
    gone = ("In this build the second engine does not run (fix F2a; its lists are the pipeline's own), hengck23's DreaMS "
            "resource and the\nknowledge table are not attached, and the external reference spectra are not added (F4).\n\n")
    return (text[:i] + gone + "**Datasets attached to this build** (kernel-metadata.json; licence as the Kaggle page shows "
            "it, with the component\nlicences of docs/fork_eligibility.md):\n\n" + table +
            "\n**Upstream lineage, not attached here.** The forked notebook was built on these datasets; elig0 neither mounts "
            "nor reads\nthem (fix F1). They are listed for the lineage's attribution only:\n\n" +
            CREDITS_HEAD.replace('| used for |', '| used for (forked notebook) |') +
            '\n'.join(lineage) + '\n\n' + CREDITS_TAIL + " In this build we also wrote the eligibility code of cell 8\n"
            "(mount and hash check, the null Engine 2 module)." + text[j + len(CREDITS_TAIL):])


# Final layer (notebook cell 16, F3): with SPEC_W 0 the NIST-including flag src_spectral_library is not loaded at all.
# The column is int64 0 / 1 without NaN in pool_popularity.csv (sha256 33955e2fb32d...), so the score term
# (SPEC_W if lib_hit else 0.0) * s_val was exactly 0.0 before and is 0.0 * 0.0 now: no score changes.
C16 = [("            cols = ['inchikey14', 'pop', 'src_spectral_library', 'np_coconut', 'np_lotus', 'src_natural_product']\n",
        "            # elig0 (ours, F3): the NIST-including depositor flag src_spectral_library is loaded only when SPEC_W > 0\n"
        "            cols = ['inchikey14', 'pop'] + (['src_spectral_library'] if SPEC_W else []) + \\\n"
        "                   ['np_coconut', 'np_lotus', 'src_natural_product']\n"),
       ("            spec_map = dict(zip(p_df['inchikey14'], p_df['src_spectral_library']))\n",
        "            spec_map = dict(zip(p_df['inchikey14'], p_df['src_spectral_library'])) if SPEC_W else {}\n"
        "            if not SPEC_W:\n"
        "                print('elig0 F3: src_spectral_library not loaded (SPEC_W 0)', flush=True)\n")]


def build_cell(frozen_info, null_engine_src, mounts):
    """Code appended to notebook cell 8: pinned data + scripts/pub_variant/elig_cell.py."""
    v43_text = open(os.path.join(HERE, V43_FILE), 'rb').read()
    assert hashlib.sha256(v43_text).hexdigest() == V43_SHA256, f'{V43_FILE} differs from its pin'
    rows = json.loads(v43_text)
    assert len(rows) == 593, len(rows)
    compact = json.dumps(rows, separators=(',', ':'))
    assert null_engine_src.isascii() and compact.isascii()
    head = ("# ── Eligibility (ours, elig0): mounted datasets, input hash check, null Engine 2 module ─────────────────────\n"
            "# Generated by scripts/make_pub_variant.py ELIG=0 (scripts/pub_variant/elig.py, elig_cell.py, null_engine.py).\n"
            f"ELIG_DATASETS = {mounts!r}\n"
            f"ELIG_UPSTREAM = {UPSTREAM!r}\n"
            f"ELIG_DETACHED = {DETACHED!r}\n"
            f"ELIG_CHECKED_VERSIONS = {CHECKED_VERSIONS!r}\n"
            "# full SHA-256 of larger files (v4g container MANIFEST.json 2593c1e95b98...; v3 fpnet_0 / 1 = v4b fe_A / fe_B)\n"
            f"ELIG_FULL_REFS = {FULL_REFS!r}\n"
            f"ELIG_FULL_DATASETS = {FULL_DATASETS!r}\n"
            f"ELIG_FROZEN = {frozen_info!r}\n"
            f"NULL_ENGINE_SHA256 = {hashlib.sha256(null_engine_src.encode()).hexdigest()!r}\n"
            f"NULL_ENGINE_SRC = {null_engine_src!r}\n"
            f"V43_HASHES_SHA256 = {hashlib.sha256(compact.encode()).hexdigest()!r}   # compact copy of casmi26-pubm's file "
            f"(original sha256 {V43_SHA256[:12]}...)\n"
            f"V43_HASHES_JSON = {compact!r}\n")
    return head + open(os.path.join(HERE, 'elig_cell.py'), encoding='utf-8').read()


def apply(nb, meta, level, cfg):
    """Turn the generated pub variant (nb, meta) into the eligibility build `level` in place; returns a report.
    cfg: the build's cell-1 configuration (OURS decides whether our bryantyu datasets are attached)."""
    assert level == 0, 'ELIG=1 (F2b: our stage 1 as Engine 2) is reserved and not built yet'
    ours = bool(cfg['OURS'])
    mounts = UPSTREAM + (OURS_DATASETS if ours else [])
    cells = nb['cells']
    weights = [n for n in ast.parse(cells[1]['source']).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
               and [t.id for t in n.targets[0].elts] == ['POP_W', 'SPEC_W', 'NP_W']]
    assert len(weights) == 1 and ast.literal_eval(weights[0].value)[1] == 0.0, 'ELIG=0 needs SPEC_W 0 in cell 1 (F3)'
    i8 = next(i for i, c in enumerate(cells) if c.get('id') == '3e2e781f')
    c8 = cells[i8]['source']
    head = "FROZEN_SHA256 = '" + FROZEN_ORIGINAL_SHA256 + "'\nFROZEN_B64 = '"
    assert c8.count(head) == 1, 'cell 8 layout'
    start = c8.index(head) + len(head)
    end = c8.index("'\n", start)
    raw = base64.b64decode(c8[start:end])
    patched, info, changes = patch_frozen(raw)
    b64 = base64.b64encode(patched).decode()
    cells[i8]['source'] = (c8[:c8.index(head)] + "FROZEN_SHA256 = '" + info['patched_sha256'] +
                           "'   # elig0: frozen 0.465 notebook with the elig0 patches (original " +
                           FROZEN_ORIGINAL_SHA256[:12] + "...)\nFROZEN_B64 = '" + b64 + c8[end:])
    assert base64.b64decode(b64) == patched
    src = open(os.path.join(HERE, 'null_engine.py'), encoding='utf-8').read()
    # appended to cell 8 (pinned sources), so every other cell keeps its number and the run order is unchanged
    cells[i8]['source'] = cells[i8]['source'].rstrip('\n') + '\n\n\n' + build_cell(info, src, mounts)
    # F3 in the final layer (the midx cell, our cell 16): src_spectral_library not loaded when SPEC_W is 0
    i16 = next(i for i, c in enumerate(cells) if c.get('id') == 'busyaprime-variant-m-merge')
    cells[i16]['source'] = splice(cells[i16]['source'], C16, f'cell {i16} (midx, F3)')
    # markdown: cell 0 statement; sections 6 and 8 name the attached datasets
    rest = cells[0]['source'].split('\n', 1)[1].lstrip('\n')
    rest = splice(rest, [('documentation prior 0.10 / 0.05 / 0.02 on', 'documentation prior 0.10 / 0.05 (0 here, F3) / 0.02 on')],
                  'cell 0')
    cells[0]['source'] = md0(ours) + rest
    i6 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and c['source'].startswith('## 6. Run it yourself'))
    cells[i6]['source'] = splice(cells[i6]['source'], RUN_EDIT, f'cell {i6} (run it yourself)')
    i23 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and c['source'].startswith('## 8. Credits'))
    cells[i23]['source'] = credits_edit(cells[i23]['source'], ours)
    casmi = nb['metadata']['casmi']
    casmi['frozen_source_sha256'] = info['patched_sha256']
    casmi['elig'] = dict(level=0, fixes=['F1', 'F2a', 'F3', 'F4'], doc='docs/fork_eligibility.md',
                         frozen_original_sha256=FROZEN_ORIGINAL_SHA256, frozen_patched_sha256=info['patched_sha256'],
                         frozen_patched_cells=info['patched_cells'], skipped_frozen_cells=info['skipped_cells'],
                         null_engine_sha256=hashlib.sha256(src.encode()).hexdigest(), cell=8,
                         notebook_cells_edited=sorted({0, 1, i8, i16, i6, i23}), datasets=mounts, detached=DETACHED,
                         not_attached_ours=[] if ours else OURS_DATASETS, open=OPEN)
    meta['dataset_sources'] = mounts
    return dict(frozen=info, changes=changes, frozen_patched=patched)
