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
ELIG=1 = elig0 with F2b instead of F2a (our stage 1 as Engine 2): the same F1 / F3 / F4 patches, plus
  F2b frozen cell 32 runs casmi.run_eng2 (our stage 1, the s2v18cue chain) from bryantyu/casmi26-elig-assets on
      bryantyu/casmi26-pool-v18c in its own process where the original Engine 2 ran (eng2_merge.py, written by cell 8);
      cells 39 and 45 re-check its keys with chem.score_key and merge its lists with elig0's null-engine lists (molecules
      it does not cover), so Engine 2's whole data flow (ICE union, fuse2, cells 48-53, export_evidence, outsiders)
      runs on our lists. dataset_sources: the upstream set + our two datasets (never casmi26-code / -models, also with
      OURS = True: PF-syn then reads the PeakFormer and its code from casmi26-elig-assets). Cell 9 labels stage 32.
      Cell 32 pins our 22 code and model files (ELIG_ASSETS_SHA256) besides the dataset's own MANIFEST.sha256; cell 8
      also flags the original Engine 2 inputs and casmi26-code / -models as detached, and any mount kernel-metadata.json
      does not list as EXTRA MOUNT (log only). Text only: frozen cell 31 (the original Engine 2's description) gets an
      elig1 comment line, cells 1 / 11 name casmi26-elig-assets for PF-syn, cell 20 no longer calls this kernel m1.
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
# ELIG=1 (F2b): our Engine 2. casmi26-elig-assets = scripts/pub_variant/make_elig_assets.py (the import closure of
# casmi.run_eng2 + fp_model_v1r.pt, pf_tpu_f.pt, ret_ranker_v18cu / v18cur1; MANIFEST.sha256 checked at run time);
# casmi26-pool-v18c = scripts/build_v18c.sh stage A (CC BY 4.0 on Kaggle), mounted as published.
ELIG1_DATASETS = ['bryantyu/casmi26-elig-assets', 'bryantyu/casmi26-pool-v18c']
# ELIG=1: also reported as "detached ones present" by cell 8 (with DETACHED): our G1-open model / code datasets and every
# input of the original Engine 2 (docs/fork_eligibility.md A29-A36, A45). None of them may be mounted in elig1.
DETACHED_ELIG1 = ['bryantyu/casmi26-code', 'bryantyu/casmi26-models', 'prvsiyan/casmi26-fp-models-v2',
                  'prvsiyan/casmi26-ranker-features', 'prvsiyan/coconut-casmi26-candidates', 'prvsiyan/chebi-lipidmaps-casmi26',
                  'megayak/casmi26-simulated-ranker-rows', 'sergiomalv/casmi-chebi-lmsd-structures',
                  'franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints', 'hengck23/hengck23-dreams-enveda-casmi26']
# SHA-256 of the 22 code and model files of casmi26-elig-assets as built (make_elig_assets.py) and tested locally: frozen
# cell 32 compares the files Engine 2 uses with these pins (log only; MANIFEST.sha256 alone shows only that the upload
# is self-consistent). make_elig_assets.py exits 1 when its output differs from them.
ELIG_ASSETS_SHA256 = {
    '__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'analog.py': 'd709dc82474ecdbf02728e6508de681d22987986a82208917b838f9e115868a7',
    'build.py': 'aa8b84d5985578fe8dd8482c9ca7d255a6626c5791616f9f3dc2475b60c78ea4',
    'entropy.py': '8ecc377071f6b8d7604d8420cd65f1f275dfa32b2c1d735eb26a1b56ff463faf',
    'fp_model_v1r.pt': '9b1112ecc16c93956daa18bbcf93971e250b4ff2d4f2ccd2720edc82df34df88',
    'fpmodel.py': '2930be30289b173669fff54ce0d9852a4fe117455f9f9cb1590b2900462a0760',
    'fpscore.py': '528195f3508b7cc4e3a74a05b522c399b275d08225efd1247b307852100ac598',
    'frag.py': '906d1462e12b20fa9291edd73158a7d57cac2dda28976bdaf42c01a4ea4172fb',
    'metric.py': '89919049e7689148b28245dfef455b39eb67797b85faaade26a56453081b415f',
    'own.py': 'fd73d4a73324e4d1c624649e45c34f3f07ef8776177f6e5c00b4f7c2e673cb28',
    'peakformer.py': '5e5bab5ddafc15db63951ec829bd61656de4d7ea649aceec407d91dfb53ecf62',
    'pf_tpu_f.pt': 'b95c8c7b2708152fc78469130cc2e2e252b9e266f7aea915c70dce20cb33d4dd',
    'prep_meta.py': '3e312a4062d4d9904a39ef018dc0e676aa30eb38a3a2c74d0b54b081ee7d3719',
    'ret_ranker_v18cu.txt': '9fd0eb428456107ef9d7cc0b93e3aae46de9a718a86359572d89bf4b009d56f8',
    'ret_ranker_v18cu_feats.txt': 'e8922e21257c907c7820c71603a8b93bb256f3ee8a35e73b1e79d4829f944224',
    'ret_ranker_v18cur1.txt': '5cceede93cdcc12682b613ef133ad93d654adba21d387888f31d62d02296f866',
    'ret_ranker_v18cur1_feats.txt': 'e8922e21257c907c7820c71603a8b93bb256f3ee8a35e73b1e79d4829f944224',
    'retrieval.py': 'e75fbc46e224e1c22315565c1f0574c71dc7ef4d6e9b8c1c351b8880251a5b86',
    'richfp.py': 'f7719ab099017dd5e5550ae2412d04ddd81c5c884d42cbbd3b8dfd4d0a702763',
    'run_eng2.py': '04d7280e29b8a81643aaea2a2868a13ef24dcb836fd84960c332fff9eea2f96b',
    'run_own.py': '68cd9f722c4198eab1073a20a6f53bc22c376a918ca565206310021f637d4958',
    'spectra.py': 'b7014fb302c9c05ee7db34d0619ee07a913edbfe5a016921dc995aafc1d9497f'}
POOL_V18C_SHA256 = {'coconut_prior.parquet': '3bf89c32f8f904e7c3b1167eae2271fab93d77c04fb9ff8f77095bcf3c6fe43c',
                    'pool.parquet': 'a3ae588bf270e6af642d88f2ffeb997c415e04147f2df8d27c0c8ab9fbb4fb6d',
                    'pool_fp.npy': 'f124a52b46808e50d2c4aacb60d63055df760e082a10fdecf2bdc1a770f6725b'}
# Engine 2 settings (frozen cell 32): the s2v18cue stage 1 with k 40 (+8 so that 40 remain after the score_key
# re-keying); cap 45 min from the subprocess start, then 5 min grace before the process group is killed; stall: no
# molecule finished for 15 min; procs 0 = min(4, CPUs). Expected rerun: ~22-27 min = s2v18c's 1,249 s all-in on 4 Kaggle
# CPUs (its 'stage 1 done' time counts from the start, the 260 s library build included) + ~1 min of score_key re-keying
# (in the workers; 15.5k keys x ~16 ms / 4), the stage-1 part x up to 1.24 for the hidden test's ~1,500 spectra (1,213
# visible); i.e. a 1.7-2.0x margin to the cap.
ENG2_CFG = dict(cap_s=2700.0, grace_s=300.0, stall_s=900.0, procs=0, k=40, margin=8,
                ranker='ret_ranker_v18cu,ret_ranker_v18cur1', fp='fp_model_v1r.pt', pf='pf_tpu_f.pt', lib_cap=10,
                work='/tmp/eng2_ours', omp_threads=None, pool_sha256=POOL_V18C_SHA256, asset_sha256=ELIG_ASSETS_SHA256)
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

# ---- ELIG=1 (F2b): our stage 1 as Engine 2 (cells 32, 39, 45; eng2_merge.py written by notebook cell 8) ----
C32_NEW1 = (C32_HEAD +
            "# elig1 (ours, fix F2b of docs/fork_eligibility.md): Engine 2 is OUR stage 1, casmi.run_eng2 from\n"
            "# bryantyu/casmi26-elig-assets (stage 1 of casmi26-own-s2v18cue: candidate pool bryantyu/casmi26-pool-v18c, FP\n"
            "# fp_model_v1r.pt, PeakFormer pf_tpu_f.pt, LightGBM seed ensemble ret_ranker_v18cu + v18cur1), run here in its own\n"
            "# process where the original Engine 2 ran: before the PubChem channel and Engine 1, CPU only, own process group,\n"
            "# time cap ENG2_CFG['cap_s'] (commit run: the 12 smoke molecules of COMP; rerun: all). It writes the top 40 per\n"
            "# molecule in Engine 2's schema, keys with chem.score_key's semantics (re-checked with chem.score_key in cell 39).\n"
            "# Cells 39 (step 1: ICE / GLACIER input union) and 45 (step 2: fusion, cells 48-53, evidence export, outsiders)\n"
            "# use these lists, and elig0's null-engine lists (null_engine.py) for molecules without a list of ours. The original\n"
            "# Engine 2's inputs (prvsiyan, megayak, sergiomalv, franciscoangulo) are not mounted. Nothing here can stop the\n"
            "# pipeline: any failure leaves ENG2_OURS empty, i.e. elig0.\n"
            "import eng2_merge as _eng2\n"
            "ENG = {}\n"
            "Path('/kaggle/working/eng_lists.json').unlink(missing_ok=True)\n"
            f"ENG2_CFG = {ENG2_CFG!r}\n"
            "ENG2_OURS, ENG2_REPORT = _eng2.run_engine2(os.path.join(COMP, 'test.parquet'), os.path.join(COMP, 'train.parquet'),\n"
            "                                          '/kaggle/working/eng2_ours.json', ENG2_CFG)\n"
            "_eng2.write_json(ENG2_REPORT, '/kaggle/working/eng2_ours_report.json')\n"
            "print(_eng2.report_line(ENG2_REPORT), flush=True)\n"
            "if ENG2_REPORT.get('log_tail'):\n"
            "    print(ENG2_REPORT['log_tail'], flush=True)\n")

C31_HEAD = "'''## 3. Engine 2: two-ranker engine (BIO + AFIX)\n"
C31_1 = ("# elig1 (ours, fix F2b of docs/fork_eligibility.md): the engine described below is NOT run in this build. Engine 2 is\n"
         "# our stage 1 (casmi.run_eng2 from bryantyu/casmi26-elig-assets on bryantyu/casmi26-pool-v18c), frozen cell 32.\n")

C39_1 = [("BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}\n",
          "BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}\n"
          "\n"
          "# elig1 (ours, fix F2b): Engine 2, step 1 of 2. Our keys are re-checked with the pipeline's own chem.score_key (exact\n"
          "# identity: a different key is replaced, a duplicate or unparseable entry dropped): every key in the commit run;\n"
          "# in the rerun the first 10 of each list when the subprocess ran this RDKit version (~20 ms per key), and every key\n"
          "# as soon as one of them differs. ENG := our list where we have one, else elig0's null-engine list (Engine 1's top\n"
          "# 40 with Engine 1's keys). The ICE / GLACIER input union of cell 41 then adds Engine 2's same-formula candidates to\n"
          "# the simulator input, as it did for the original Engine 2.\n"
          "import null_engine as _null_engine\n"
          "import eng2_merge as _eng2\n"
          "_t_e2 = time.time()\n"
          "_e2_head = 10 if IS_RERUN and ((ENG2_REPORT.get('meta') or {}).get('versions') or {}).get('rdkit') == rdkit.__version__ \\\n"
          "    else None\n"
          "ENG2_OURS, ENG2_KEYCHECK = _eng2.verify_keys(ENG2_OURS, chem.score_key, head=_e2_head)\n"
          "_e2_mids = te.molecule_id.astype(str).unique()\n"
          "_e2_null = _null_engine.null_engine_lists(_eng2.fallback_ids(_e2_mids, ENG2_OURS),\n"
          "                                          {str(m): v[0] for m, v in BASE.items()},\n"
          "                                          keys={str(m): v[1] for m, v in BASE.items()}, n=40)\n"
          "ENG, ENG2_STEP1 = _eng2.merge(_e2_mids, ENG2_OURS, _e2_null)\n"
          "_null_engine.write_lists(ENG, '/kaggle/working/eng_lists_step1.json')\n"
          "print(_eng2.step_line('step 1 (read by the ICE union)', ENG2_STEP1, ENG2_REPORT,\n"
          "                      f\"key check ({ENG2_KEYCHECK['mode']}): entries {ENG2_KEYCHECK['entries']}, checked \"\n"
          "                      f\"{ENG2_KEYCHECK['checked']}, mismatches {ENG2_KEYCHECK['mismatches']}, \"\n"
          "                      f\"dropped {ENG2_KEYCHECK['dropped']}, emptied {ENG2_KEYCHECK['emptied']} \"\n"
          "                      f\"({time.time() - _t_e2:.0f} s) | \" + _null_engine.summary(ENG)), flush=True)\n")]

C45_1 = [("FUSION_OVERFLOW = {}\nif ENG:\n",
          "FUSION_OVERFLOW = {}\n"
          "# elig1 (ours, fix F2b): Engine 2, step 2 of 2. ENG := our list (as in step 1) where we have one, else elig0's step-2\n"
          "# null list (PREFUSION_LISTS = Engine 1 + the gated PubChem merge, verbatim, then Engine 1's tail; keys\n"
          "# chem.score_key(s) or s as fuse2 computes them). Written to eng_lists.json, the file Engine 2 wrote; fuse2 below,\n"
          "# cells 48-53, export_evidence and outsiders read this ENG as they read the original Engine 2's.\n"
          "import null_engine as _null_engine\n"
          "import eng2_merge as _eng2\n"
          "_e2_mids = te.molecule_id.astype(str).unique()\n"
          "_e2_null = _null_engine.null_engine_lists(_eng2.fallback_ids(_e2_mids, ENG2_OURS), PREFUSION_LISTS,\n"
          "                                          tails=BASE_TAIL_CANDIDATES, key_of=lambda s: chem.score_key(s) or s, n=40)\n"
          "ENG, ENG2_STEP2 = _eng2.merge(_e2_mids, ENG2_OURS, _e2_null)\n"
          "_null_engine.write_lists(ENG, '/kaggle/working/eng_lists.json')\n"
          "ENG2_SUMMARY = dict(subprocess=ENG2_REPORT, key_check=ENG2_KEYCHECK, step1=ENG2_STEP1, step2=ENG2_STEP2)\n"
          "_eng2.write_json(ENG2_SUMMARY, '/kaggle/working/eng2_ours_report.json')\n"
          "print(_eng2.step_line('step 2 (fused below; eng_lists.json)', ENG2_STEP2, ENG2_REPORT, _null_engine.summary(ENG)),\n"
          "      flush=True)\n"
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
V43_ELIG1 = dict(level=1, fixes=['F1', 'F2b', 'F3', 'F4'], doc='docs/fork_eligibility.md (bryantyu)',
                 engine2='ours: casmi.run_eng2 = stage 1 of casmi26-own-s2v18cue (pool bryantyu/casmi26-pool-v18c, '
                         'fp_model_v1r.pt, pf_tpu_f.pt, ret_ranker_v18cu + v18cur1; bryantyu/casmi26-elig-assets), top 40 '
                         'per molecule, keys re-checked with chem.score_key; the elig0 null engine (F2a) for molecules '
                         'without a list of ours (engine2_run.step2.fallback); no prvsiyan / megayak / sergiomalv / '
                         'franciscoangulo inputs',
                 src_bits=0, spec_w=0.0, external_reference_spectra='off',
                 mounts='upstream datasets + bryantyu/casmi26-elig-assets + bryantyu/casmi26-pool-v18c (mounted_datasets); '
                        'no v4g container, knowledge table or panel benchmark',
                 method_note='method names the upstream E006 recipe. In elig1 its e2_* parts run on our Engine 2 '
                             '(engine2_lists / engine2_nonempty count ours plus the null-engine fallback), chebi_lmsd and '
                             'pubchemlite are those of our pool v18c (our ChEBI r255 + LMSD build, PubChemLite v3.3.0), '
                             'spectral_bonus is off (SRC_BITS 0) and extlib is off (F4)')
C54_1 = [(V43_OLD,
          "# elig1 (ours, fixes F1 + F2b): the receipt hashes the upstream datasets mounted instead of the v4g container (was:\n"
          "# the five dmitriigluzdov datasets) and our two Engine 2 datasets; compare v43_mounted_input_hashes.json with\n"
          "# casmi26-pubm's by path inside each dataset.\n"
          f"V43_DATASET_SOURCES = {UPSTREAM + ELIG1_DATASETS!r}\n"
          "# elig1 (ours): the receipt says what of its method string this build runs differently, and what Engine 2 did\n"
          f"V43_ELIG = {V43_ELIG1!r}\n"),
         (V43_METHOD, V43_METHOD + "        elig=dict(V43_ELIG, engine2_run=scope.get('ENG2_SUMMARY')),\n")]


def splice(text, edits, what):
    for old, new in edits:
        assert text.count(old) == 1, f'{what}: {old[:70]!r} found {text.count(old)} times'
        text = text.replace(old, new)
    return text


def as_lines(text):
    """Back to the notebook's list-of-lines form ('\\n' kept on every line but the last)."""
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def patch_frozen(raw, level=0):
    """Patched frozen notebook bytes (same JSON serialisation as the original) and a record of what changed.
    level 1 (F2b): cells 32 / 39 / 45 / 54 get the elig1 patches instead of elig0's, cell 31 (a description string)
    an elig1 comment line; every other cell is elig0's."""
    assert level in (0, 1), level
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
    new[32] = C32_NEW if level == 0 else C32_NEW1
    if level == 1:                              # text only: the original Engine 2's description, marked as not run
        assert cells[31]['cell_type'] == 'code' and text[31].startswith(C31_HEAD) and "prvsiyan's fingerprint" in text[31]
        new[31] = C31_1 + text[31]
    new[36] = splice(text[36], C36, 'frozen cell 36')
    new[39] = splice(text[39], C39 if level == 0 else C39_1, 'frozen cell 39')
    new[45] = splice(text[45], C45 if level == 0 else C45_1, 'frozen cell 45')
    new[54] = splice(text[54], C54 if level == 0 else C54_1, 'frozen cell 54')
    for i, t in new.items():
        assert cells[i]['cell_type'] == 'code', i
        compile(t, f'frozen cell {i}', 'exec', dont_inherit=True)    # syntax only (compile, never exec)
        cells[i]['source'] = as_lines(t)
    out = json.dumps(nb, separators=(',', ':'), ensure_ascii=True).encode()
    info = dict(original_sha256=FROZEN_ORIGINAL_SHA256, patched_sha256=hashlib.sha256(out).hexdigest(),
                patched_cells=sorted(new), skipped_cells=sorted(SKIP))
    return out, info, {i: (text[i], new[i]) for i in sorted(new)}


def cell1_edits(level=0):
    if level == 1:
        return [("POP_W, SPEC_W, NP_W = 0.10, 0.05, 0.02\n",
                 "POP_W, SPEC_W, NP_W = 0.10, 0.0, 0.02   # elig (F3): SPEC_W 0.0 (was 0.05; spectral-library depositor flag\n"
                 "                                        # includes the NIST MS Data Center); NP_W kept (NPAtlas is host-allowed)\n"),
                ("# ── Configuration ─────────────────────────────────────────────────────────────────────────────\n",
                 "# ── Configuration ─────────────────────────────────────────────────────────────────────────────\n"
                 "ELIG = 1                      # eligibility build (docs/fork_eligibility.md): 1 = F1 upstream mounts + F2b our\n"
                 "                              # stage 1 as Engine 2 (frozen cell 32; null engine where it has no list) + F3 no\n"
                 "                              # NIST depositor flag + F4 no external spectra; 0 = the same with F2a (null Engine 2)\n"),
                ("# Ours: PF-syn PeakFormer (bryantyu/casmi26-models pf_tpu_f.pt) joins the neural term before the fusion\n",
                 "# Ours: PF-syn PeakFormer (pf_tpu_f.pt of bryantyu/casmi26-elig-assets in elig1) joins the neural term before\n"
                 "# the fusion\n")]
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


def md0_elig1(ours):
    """Markdown cell 0 of ELIG=1: the eligibility statement (elig0 with F2b in place of F2a)."""
    if ours:
        pf = ("PF-syn (cell 11, OURS = True) reads its PeakFormer pf_tpu_f.pt and the casmi code from casmi26-elig-assets "
              "as well")
    else:
        pf = "with OURS = False, PF-syn (cell 11) and ISO (cell 17) are off"
    return f"""# Eligible fork elig1 · Variant M core + midx final layer, our stage 1 as Engine 2, upstream datasets only

> **Eligibility build elig1** (scripts/make_pub_variant.py with ELIG=1; the audit and fixes F1-F8 are in
> docs/fork_eligibility.md of our repository): elig0 (casmi26-pube0) with fix F2b in place of F2a. Changed against
> casmi26-pubm:
> - **F1, upstream datasets.** The CC BY-NC-SA container dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets, the
>   natural-product knowledge table, the panel benchmark and sanpier/catalogue-experiment-saner are no longer attached.
>   The pipeline reads the original datasets: ahmedberatozer casmi26-v4b-models, casmi26-v3-models, casmi26-fpnet-full1,
>   casmi26-v2-pool, casmi26-iceberg and casmi26-glacier; dmitriigluzdov casmi26-pubchem-popularity-prior and
>   casmi26-fold-safe-fpnet; sanpier/casmi-spectrum-model-v1; metric/rdkit-2026-3-3-wheel. Engine 2 reads our two
>   datasets: bryantyu/casmi26-elig-assets (its code and models, nothing else) and bryantyu/casmi26-pool-v18c (its
>   candidate pool). Our bryantyu/casmi26-code and casmi26-models, which also hold older models, are not attached;
>   {pf}. The frozen pipeline's display cells 9, 12, 14, 16 and 18 are skipped. Before the pipeline runs, cell 8 checks
>   the mounted upstream files against the hashes of the casmi26-pubm run (files up to 32 MiB in full, larger ones by
>   size and their first and last 4 MiB) and hashes the larger files of the re-mounted datasets in full (log only);
>   Engine 2 checks our files against the MANIFEST.sha256 of casmi26-elig-assets and the pool against pinned SHA-256
>   values, and the code and model files Engine 2 uses against SHA-256 pins in this notebook (log only). Cell 8 also
>   reports any mount that kernel-metadata.json does not list (EXTRA MOUNT) and whether casmi26-code / -models or an
>   input of the original Engine 2 is attached. The pipeline's receipt carries an `elig` field with Engine 2's counts.
> - **F2b, our stage 1 as Engine 2.** The original Engine 2 (prvsiyan fingerprint models, ranker rows and COCONUT bank,
>   megayak ranker rows, sergiomalv ChEBI/LMSD, franciscoangulo PubChemLite) is not run. In its place, frozen cell 32
>   runs stage 1 of our own pipeline (kernels/casmi26-own-s2v18cue: pool v18c = training structures, COCONUT 10-2026,
>   PubChemLite v3.3.0 and our ChEBI r255 / LIPID MAPS build; library, analog, fragment and documentation evidence;
>   fingerprint model fp_model_v1r; PF-syn PeakFormer pf_tpu_f; LightGBM seed ensemble ret_ranker_v18cu + v18cur1) in
>   its own process, as the original Engine 2 ran, on CPU and capped at 45 minutes. It writes its top 40 per molecule in
>   Engine 2's schema, keyed by the pipeline's own InChIKey14 identity (chem.score_key, re-checked in frozen cell 39).
>   Every reader of Engine 2 gets these lists, as it got the original's: the ICEBERG / GLACIER input union (cell 41),
>   the reciprocal-rank fusion (cell 45), the safe tail, metric-identity fusion, fragment-gap and window-40 cells
>   (48-52), the rank-2 placement of Engine 2's leader (cell 53), the evidence export and the outside candidates.
>   Molecules our stage 1 does not cover (no usable spectra, no candidate, an error or the time cap) get elig0's
>   null-engine lists (frozen cells 32, 39 and 45; eng2_merge.py, null_engine.py). The README of casmi26-elig-assets
>   gives each file's provenance: fp_model_v1r.pt and the pool are in docs/reproduce.md section 6; ret_ranker_v18cu /
>   v18cur1 come from scripts/build_pfswap_pair.sh (build_v18c.sh with BASE=v18c REUSE_C1=0, stages C-D; LightGBM
>   seed 0 / seed 1); pf_tpu_f.pt comes from the PF-syn chain (build_sim_jobs.py, 24 simw2 workers, syn-prep2 946badfb,
>   casmi26-pfs-tpu-f), which is still to be written up as docs/reproduce.md section 7 (open, see below). Frozen cell
>   31 still holds the original Engine 2's description, now marked as not run.
> - **F3.** No PubChem depositor-source bonus (SRC_BITS 0: that bit groups the NIST Mass Spectrometry Data Center's
>   deposits with MassBank / MoNA) and SPEC_W 0 in the final layer, which then does not load the flag column. The
>   PubChem channel's delta / growth supplement stores still memory-map their source-mask arrays, which no score reads.
>   NP_W / NP_BONUS stay: NPAtlas is on the host's list.
> - **F4.** The knowledge table's external reference spectra are not added to the library (frozen cell 36).
> - Sections 6 and 8 name the datasets attached here. Not part of the fixes: the ISO cell (cell 17, off with
>   ISO_ALPHA 0 in both builds) and two of its cell-1 knobs (ISO_COS_W, ISO_FREEZE_LEADER) come from a newer generator
>   version than casmi26-pubm's. Log lines of the inherited F1 / F3 / F4 patches still say "elig0". Sections 1-5
>   describe the forked notebook and its runs (m1, 0.477); casmi26-pube1 is a new kernel with a score of its own.
>
> **Still open (this build is not eligible until they are closed):** F5a, the ICEBERG / GLACIER runner glue
> (ice_runner.py, fuse.py, gl_runner.py, gl_fuse.py, shim/ in ahmedberatozer casmi26-iceberg / casmi26-glacier) carries
> no OSI licence yet: an owner question to its author. F6, a host question: do reconstructed, not bit-exact retrains with
> public code (Engine 1's ranker_v4b and fe_v4 tables, saner, our own G4 / G5: G5 is the 5-hour wall-clock stop of
> PF-b, the initialisation of pf_tpu_f; pf_tpu_f itself is a TPU run, not bit-exact, whose kernel metadata pins no
> image and whose checkpoint has no step or environment stamp) meet the reproducibility rule, and does a PubChem
> depositor flag naming the NIST MS Data Center count as using NIST? F7, a demonstrated retrain of saner's spectrum
> model. F8, the licence and attribution package: an MIT LICENSE for our code, a NOTICE for the Apache-2.0 and MIT
> sources, the fork's ATTRIBUTION and reproduce files (with the PF-syn chain of pf_tpu_f and the ranker pair, docs/
> reproduce.md section 7), and the Kaggle licences of our datasets (G19): casmi26-elig-assets still has to be created
> (prepared as "other": code MIT in LICENSE.txt, models CC BY-NC 4.0 in MODELS_LICENSE.txt), and casmi26-pool-v18c
> needs its attribution text (its Kaggle page has an empty description) and a licence that keeps its 275,810
> training-structure rows under the competition terms (a CC BY label conflicts with that).

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


ATTACHED_ELIG1 = [
    ('bryantyu/casmi26-elig-assets', 'us', "Engine 2 (F2b): our stage-1 code (the casmi modules casmi.run_eng2 imports) "
     "and models (fp_model_v1r.pt, PF-syn PeakFormer pf_tpu_f.pt, LightGBM rankers ret_ranker_v18cu / v18cur1); "
     "README and MANIFEST.sha256 give each file's provenance", 'ours, Kaggle "other" (prepared; the dataset is not '
     'created yet, G19 / F8): code MIT (LICENSE.txt); models CC BY-NC 4.0 (MODELS_LICENSE.txt: trained on train.parquet '
     'and the permitted data of docs/ATTRIBUTION.md)'),
    ('bryantyu/casmi26-pool-v18c', 'us', "Engine 2's candidate pool (1,160,133 structures: training structures, COCONUT "
     "10-2026, PubChemLite v3.3.0, our ChEBI r255 / LIPID MAPS build; PubChem documentation counts) with its "
     "fingerprints and COCONUT priors", 'CC BY 4.0 on Kaggle; its in_train rows (275,810 training structures) stay '
     'under the competition terms (CC BY-NC 4.0), and its attribution text is still open (G19). Sources: COCONUT CC0; '
     'ChEBI CC BY 4.0; PubChemLite CC BY 4.0; LIPID MAPS CC BY 4.0; PubChem: courtesy of the U.S. NLM')]


def credits_edit(text, ours, level=0):
    """Section 8: the attached datasets first; the forked notebook's table kept as upstream lineage (not attached)."""
    if level == 1:
        rows = ATTACHED + ATTACHED_ELIG1
        assert sorted(r[0] for r in rows) == sorted(UPSTREAM + ELIG1_DATASETS)
    else:
        rows = ATTACHED + (ATTACHED_OURS if ours else [])
        assert sorted(r[0] for r in rows) == sorted(UPSTREAM + (OURS_DATASETS if ours else []))
    table = CREDITS_HEAD + ''.join(f'| [{r}]({KG}{r}) | {a} | {u} | {l} |\n' for r, a, u, l in rows)
    i, j = text.index(CREDITS_HEAD), text.index(CREDITS_TAIL)
    assert text.count(CREDITS_HEAD) == 1 and text.count(CREDITS_TAIL) == 1 and i < j
    old = [line for line in text[i + len(CREDITS_HEAD):j].split('\n') if line.startswith('| [')]
    lineage = [line for line in old if line.split('](', 1)[0][3:] in DETACHED]
    assert len(old) == 7 and len(lineage) == len(DETACHED), (len(old), len(lineage))
    if level == 1:
        gone = ("In this build the second engine is our own stage 1 (fix F2b: bryantyu/casmi26-elig-assets on "
                "bryantyu/casmi26-pool-v18c;\nthe original's prvsiyan / megayak / sergiomalv / franciscoangulo inputs are not "
                "used), hengck23's DreaMS resource and the\nknowledge table are not attached, and the external reference "
                "spectra are not added (F4).\n\n")
        who, ours_code = 'elig1', ("(mount and hash check, the null Engine 2 module, the Engine 2 glue eng2_merge.py) and "
                                   "Engine 2 itself\n(our stage 1, casmi26-elig-assets).")
    else:
        gone = ("In this build the second engine does not run (fix F2a; its lists are the pipeline's own), hengck23's DreaMS "
                "resource and the\nknowledge table are not attached, and the external reference spectra are not added (F4).\n\n")
        who, ours_code = 'elig0', "(mount and hash check, the null Engine 2 module)."
    return (text[:i] + gone + "**Datasets attached to this build** (kernel-metadata.json; licence as the Kaggle page shows "
            "it, with the component\nlicences of docs/fork_eligibility.md):\n\n" + table +
            f"\n**Upstream lineage, not attached here.** The forked notebook was built on these datasets; {who} neither mounts "
            "nor reads\nthem (fix F1). They are listed for the lineage's attribution only:\n\n" +
            CREDITS_HEAD.replace('| used for |', '| used for (forked notebook) |') +
            '\n'.join(lineage) + '\n\n' + CREDITS_TAIL + " In this build we also wrote the eligibility code of cell 8\n" +
            ours_code + text[j + len(CREDITS_TAIL):])


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


# ELIG=1: splices of elig_cell.py's text (the ELIG=0 cell stays byte-identical): also write eng2_merge.py (pinned) and
# name F2b in the mount line and the report.
CELL8_ELIG1 = [
    ("#  2. records the mounts: every dataset of kernel-metadata.json present, none of the detached ones.\n",
     "#  2. records the mounts: every dataset of kernel-metadata.json present, none of the detached ones (elig1: also our\n"
     "#     casmi26-code / -models and the original Engine 2's inputs), and (elig1) any other mount as EXTRA MOUNT.\n"),
    ("#  1. writes null_engine.py next to the pipeline modules (SHA-256 pinned). The patched frozen cells 39 and 45 import it\n",
     "#  1. writes null_engine.py and (elig1) eng2_merge.py next to the pipeline modules (SHA-256 pinned): frozen cell 32 runs\n"
     "#     our stage 1 as Engine 2 through eng2_merge.py; cells 39 and 45 merge its lists with null_engine.py's.\n"
     "#     null_engine.py: the patched frozen cells 39 and 45 import it\n"),
    ("    path = _elig_write_null_engine()\n",
     "    path = _elig_write_null_engine()\n"
     "    eng2_path = _elig_write_eng2_merge()          # elig1 (F2b): Engine 2 glue of frozen cells 32 / 39 / 45\n"),
    ("    detached = _elig_detached_present()\n",
     "    detached = _elig_detached_present()\n"
     "    extra = [p for p in _elig_extra_mounts() if p not in detached]   # elig1: other unlisted mounts\n"),
    ("               absent=absent, detached_present=detached, spec_w=spec_w,",
     "               absent=absent, detached_present=detached, extra_mounts=extra, spec_w=spec_w,"),
    ("    rep = dict(elig=ELIG, null_engine=str(path), null_engine_sha256=NULL_ENGINE_SHA256, frozen=ELIG_FROZEN,\n",
     "    rep = dict(elig=ELIG, null_engine=str(path), null_engine_sha256=NULL_ENGINE_SHA256, frozen=ELIG_FROZEN,\n"
     "               eng2_merge=str(eng2_path), eng2_merge_sha256=ENG2_MERGE_SHA256, engine2_datasets=ELIG1_DATASETS,\n"),
    ("| F2a null Engine 2 (null_engine.py {NULL_ENGINE_SHA256[:12]}) | \"\n",
     "| extra mounts: {', '.join(extra) if extra else 'none'} | F2b our stage 1 as Engine 2 (eng2_merge.py \"\n"
     "          f\"{ENG2_MERGE_SHA256[:12]}; null engine where it has no list: null_engine.py {NULL_ENGINE_SHA256[:12]}) | \"\n"),
    ("(elig0_report.json)\", flush=True)", "(elig1_report.json)\", flush=True)"),
    ("    (ROOT / 'elig0_report.json').write_text(", "    (ROOT / 'elig1_report.json').write_text("),
    ("the metric wheels) are recorded in elig0_report.json.\n", "the metric wheels) are recorded in elig1_report.json.\n"),
    ("    if spec_w != 0 or FROZEN_SHA256 != ELIG_FROZEN['patched_sha256'] or detached:\n"
     "        print('ELIG WARNING: this run is not the elig0 configuration (SPEC_W, frozen pin or detached mounts)', flush=True)\n",
     "    if spec_w != 0 or FROZEN_SHA256 != ELIG_FROZEN['patched_sha256'] or detached or extra:\n"
     "        print('ELIG WARNING: this run is not the elig1 configuration (SPEC_W, frozen pin, detached or extra mounts)',\n"
     "              flush=True)\n"
     "    for _p in extra:\n"
     "        print(f'ELIG EXTRA MOUNT {_p} (not in kernel-metadata.json)', flush=True)\n")]
CELL8_ELIG1_HEAD = """ELIG1_DATASETS = {datasets!r}
ENG2_MERGE_SHA256 = {sha!r}
ENG2_MERGE_SRC = {src!r}


def _elig_write_eng2_merge():
    path = CODE / 'eng2_merge.py'
    path.write_text(ENG2_MERGE_SRC, encoding='utf-8')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ENG2_MERGE_SHA256, 'eng2_merge.py differs from its pin'
    return path


def _elig_extra_mounts():
    # inputs mounted besides kernel-metadata.json's datasets and the competition (a Copy & Edit fork or a dataset
    # attached by hand): every datasets/<owner>/<slug>, notebooks/<owner>/<slug>, models/<owner>/<slug> or other
    # top-level folder of /kaggle/input that is not one of ELIG_DATASETS or COMP
    # (log only: a failure of the check itself is reported as an entry, never raised)
    known = set(ELIG_DATASETS)
    slugs = {{ref.split('/', 1)[1].lower() for ref in ELIG_DATASETS}}
    comp = {{COMP.name.lower()}} if COMP is not None else set()
    extra = []
    try:
        for top in sorted(p for p in INPUT.iterdir() if p.is_dir() and not p.name.startswith('.')):
            if top.name in ('datasets', 'notebooks', 'models'):
                extra += [str(p) for p in sorted(top.glob('*/*')) if p.is_dir() and not p.name.startswith('.') and
                          not (top.name == 'datasets' and f'{{p.parent.name}}/{{p.name}}' in known)]
            elif top.name == 'competitions':
                extra += [str(p) for p in sorted(top.iterdir()) if p.is_dir() and p.name.lower() not in comp]
            elif top.name.lower() not in slugs | comp:
                extra.append(str(top))
    except Exception as exc:
        extra.append(f'(check failed: {{exc!r}})'[:200])
    return extra


"""


def build_cell(frozen_info, null_engine_src, mounts, level=0, eng2_src=None):
    """Code appended to notebook cell 8: pinned data + scripts/pub_variant/elig_cell.py (level 1: + eng2_merge.py)."""
    v43_text = open(os.path.join(HERE, V43_FILE), 'rb').read()
    assert hashlib.sha256(v43_text).hexdigest() == V43_SHA256, f'{V43_FILE} differs from its pin'
    rows = json.loads(v43_text)
    assert len(rows) == 593, len(rows)
    compact = json.dumps(rows, separators=(',', ':'))
    assert null_engine_src.isascii() and compact.isascii()
    if level == 1:
        head = ("# ── Eligibility (ours, elig1): mounted datasets, input hash check, null Engine 2 and Engine 2 glue modules ──────\n"
                "# Generated by scripts/make_pub_variant.py ELIG=1 (scripts/pub_variant/elig.py, elig_cell.py, null_engine.py,\n"
                "# eng2_merge.py).\n")
    else:
        head = ("# ── Eligibility (ours, elig0): mounted datasets, input hash check, null Engine 2 module ─────────────────────\n"
                "# Generated by scripts/make_pub_variant.py ELIG=0 (scripts/pub_variant/elig.py, elig_cell.py, null_engine.py).\n")
    head += (f"ELIG_DATASETS = {mounts!r}\n"
             f"ELIG_UPSTREAM = {UPSTREAM!r}\n"
             f"ELIG_DETACHED = {DETACHED + (DETACHED_ELIG1 if level == 1 else [])!r}\n"
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
    body = open(os.path.join(HERE, 'elig_cell.py'), encoding='utf-8').read()
    if level == 1:
        assert eng2_src is not None and eng2_src.isascii()
        head += CELL8_ELIG1_HEAD.format(datasets=ELIG1_DATASETS, sha=hashlib.sha256(eng2_src.encode()).hexdigest(),
                                        src=eng2_src)
        body = splice(body, CELL8_ELIG1, 'cell 8 (elig1)')
    return head + body


# notebook cell 9 (stage machinery): the compact trace's label of frozen cell 32 (ELIG=1 only)
C9_ELIG1 = [("32: 'Engine 2 · two rankers'", "32: 'Engine 2 · our stage 1 (elig1)'")]
# notebook cell 11 (PF-syn; runs only with OURS = True): in elig1 the PeakFormer and the casmi modules come from the flat
# casmi26-elig-assets (casmi26-code / -models are not attached), so the finders prefer it and the package copy skips the
# models, rankers and text files that sit beside the modules there.
C11_ELIG1 = [
    ("# ── PF-syn: our PeakFormer (bryantyu/casmi26-models pf_tpu_f.pt) joins the neural term ",
     "# ── PF-syn: our PeakFormer (pf_tpu_f.pt of bryantyu/casmi26-elig-assets in elig1) joins the neural term "),
    ('"""PF-syn PeakFormer scores for the neural-fusion candidates (bryantyu/casmi26-models pf_tpu_f.pt).\n',
     '"""PF-syn PeakFormer scores for the neural-fusion candidates (pf_tpu_f.pt of bryantyu/casmi26-elig-assets).\n'),
    ("<pkg_parent> holds casmi_pf/ (a copy of the casmi package of bryantyu/casmi26-code). Every candidate SMILES of the\n",
     "<pkg_parent> holds casmi_pf/ (a copy of the casmi modules of bryantyu/casmi26-elig-assets). Every candidate SMILES of\n"
     "the "),
    ("        src = _first_input('peakformer.py', 'casmi26-code', _is_pf_code).parent      # the casmi package\n"
     "        ckpt = _first_input('pf_tpu_f.pt', 'casmi26-models')\n",
     "        src = _first_input('peakformer.py', 'casmi26-elig-assets', _is_pf_code).parent   # the casmi modules (elig1)\n"
     "        ckpt = _first_input('pf_tpu_f.pt', 'casmi26-elig-assets')\n"),
    ("        shutil.copytree(src, pkg / 'casmi_pf', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))\n",
     "        shutil.copytree(src, pkg / 'casmi_pf', ignore=shutil.ignore_patterns(   # elig1: the modules only, not the\n"
     "            '__pycache__', '*.pyc', '*.pt', '*.txt', '*.md', '*.sha256', '*.json'))   # models beside them\n")]
# markdown cell 20 (section 5): casmi26-pube1 is a new kernel, not a new version of m1
MD20_ELIG1 = [("This notebook is a new\nversion of m1, so the score on this page is m1's.",
               "casmi26-pube1 is a new\nkernel (the eligible fork elig1): e1-m1 are the lineage's earlier runs, and the score "
               "on this page is its own."),
              ("This notebook's 0.477 is +0.005 against 0.472", "m1's 0.477 is +0.005 against 0.472")]
OPEN1 = OPEN[:1] + ['F6 host question (reconstructed retrains incl. our G4 / G5 = PF-b init of pf_tpu_f, and pf_tpu_f '
                    'itself: TPU, not bit-exact, no image pin, no checkpoint stamp; NIST depositor flag)',
                    OPEN[2], 'F8 licence and attribution package (MIT LICENSE, NOTICE, fork ATTRIBUTION / reproduce incl. '
                    'the PF-syn chain and the ranker pair (reproduce.md section 7), Kaggle licences of our datasets G19: '
                    'casmi26-elig-assets not created yet (prepared: "other", code MIT, models CC BY-NC 4.0); '
                    'casmi26-pool-v18c attribution text and a licence that keeps its in_train rows under the competition '
                    'terms)']


def apply(nb, meta, level, cfg):
    """Turn the generated pub variant (nb, meta) into the eligibility build `level` in place; returns a report.
    cfg: the build's cell-1 configuration (ELIG=0: OURS decides whether our bryantyu datasets are attached; ELIG=1
    attaches casmi26-elig-assets + casmi26-pool-v18c and never casmi26-code / -models)."""
    assert level in (0, 1), f'ELIG={level}: only 0 (F2a) and 1 (F2b) exist'
    ours = bool(cfg['OURS'])
    mounts = UPSTREAM + (OURS_DATASETS if ours else []) if level == 0 else UPSTREAM + ELIG1_DATASETS
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
    patched, info, changes = patch_frozen(raw, level)
    b64 = base64.b64encode(patched).decode()
    tag = f'elig{level}'
    cells[i8]['source'] = (c8[:c8.index(head)] + "FROZEN_SHA256 = '" + info['patched_sha256'] +
                           f"'   # {tag}: frozen 0.465 notebook with the {tag} patches (original " +
                           FROZEN_ORIGINAL_SHA256[:12] + "...)\nFROZEN_B64 = '" + b64 + c8[end:])
    assert base64.b64decode(b64) == patched
    src = open(os.path.join(HERE, 'null_engine.py'), encoding='utf-8').read()
    eng2_src = open(os.path.join(HERE, 'eng2_merge.py'), encoding='utf-8').read() if level == 1 else None
    # appended to cell 8 (pinned sources), so every other cell keeps its number and the run order is unchanged
    cells[i8]['source'] = cells[i8]['source'].rstrip('\n') + '\n\n\n' + build_cell(info, src, mounts, level, eng2_src)
    edited = {0, 1, i8}
    if level == 1:                              # the compact trace names frozen cell 32's new stage
        i9 = next(i for i, c in enumerate(cells) if c.get('id') == 'ac4866c0')
        cells[i9]['source'] = splice(cells[i9]['source'], C9_ELIG1, f'cell {i9} (stage labels)')
        i11 = next(i for i, c in enumerate(cells) if c.get('id') == 'pfsyn0001')       # PF-syn (runs only with OURS)
        cells[i11]['source'] = splice(cells[i11]['source'], C11_ELIG1, f'cell {i11} (PF-syn, elig1)')
        i20 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and
                   c['source'].startswith('## 5. Every submission of this pipeline'))
        cells[i20]['source'] = splice(cells[i20]['source'], MD20_ELIG1, f'cell {i20} (section 5, elig1)')
        edited |= {i9, i11, i20}
    # F3 in the final layer (the midx cell, our cell 16): src_spectral_library not loaded when SPEC_W is 0
    i16 = next(i for i, c in enumerate(cells) if c.get('id') == 'busyaprime-variant-m-merge')
    cells[i16]['source'] = splice(cells[i16]['source'], C16, f'cell {i16} (midx, F3)')
    # markdown: cell 0 statement; sections 6 and 8 name the attached datasets
    rest = cells[0]['source'].split('\n', 1)[1].lstrip('\n')
    rest = splice(rest, [('documentation prior 0.10 / 0.05 / 0.02 on', 'documentation prior 0.10 / 0.05 (0 here, F3) / 0.02 on')],
                  'cell 0')
    cells[0]['source'] = (md0(ours) if level == 0 else md0_elig1(ours)) + rest
    i6 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and c['source'].startswith('## 6. Run it yourself'))
    cells[i6]['source'] = splice(cells[i6]['source'], RUN_EDIT, f'cell {i6} (run it yourself)')
    i23 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and c['source'].startswith('## 8. Credits'))
    cells[i23]['source'] = credits_edit(cells[i23]['source'], ours, level)
    edited |= {i16, i6, i23}
    casmi = nb['metadata']['casmi']
    casmi['frozen_source_sha256'] = info['patched_sha256']
    if level == 0:
        casmi['elig'] = dict(level=0, fixes=['F1', 'F2a', 'F3', 'F4'], doc='docs/fork_eligibility.md',
                             frozen_original_sha256=FROZEN_ORIGINAL_SHA256, frozen_patched_sha256=info['patched_sha256'],
                             frozen_patched_cells=info['patched_cells'], skipped_frozen_cells=info['skipped_cells'],
                             null_engine_sha256=hashlib.sha256(src.encode()).hexdigest(), cell=8,
                             notebook_cells_edited=sorted(edited), datasets=mounts, detached=DETACHED,
                             not_attached_ours=[] if ours else OURS_DATASETS, open=OPEN)
    else:
        casmi['elig'] = dict(level=1, fixes=['F1', 'F2b', 'F3', 'F4'], fallback='F2a null engine for molecules without '
                             'a list of ours', doc='docs/fork_eligibility.md',
                             frozen_original_sha256=FROZEN_ORIGINAL_SHA256, frozen_patched_sha256=info['patched_sha256'],
                             frozen_patched_cells=info['patched_cells'], skipped_frozen_cells=info['skipped_cells'],
                             null_engine_sha256=hashlib.sha256(src.encode()).hexdigest(),
                             eng2_merge_sha256=hashlib.sha256(eng2_src.encode()).hexdigest(), cell=8,
                             engine2=dict(entry='python -m casmi.run_eng2 (bryantyu/casmi26-elig-assets)',
                                          pool='bryantyu/casmi26-pool-v18c', config=ENG2_CFG,
                                          assets_builder='scripts/pub_variant/make_elig_assets.py'),
                             notebook_cells_edited=sorted(edited), datasets=mounts, detached=DETACHED + DETACHED_ELIG1,
                             not_attached_ours=OURS_DATASETS, asset_pins=len(ELIG_ASSETS_SHA256), open=OPEN1)
    meta['dataset_sources'] = mounts
    return dict(frozen=info, changes=changes, frozen_patched=patched)
