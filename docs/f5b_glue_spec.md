# Behavioural specification of the ICEBERG / GLACIER runner glue (fix F5b)

Status: 2026-10-31. Read-only study, written for a clean-room re-implementation. No third-party code was executed. No code
from the unlicensed glue is reproduced here. Every rule is stated in prose or as a formula, with the location where it
is implemented.

**Purpose.** The fork (kernels/casmi26-pube0 / -pube1, frozen notebook FZ) calls Ahmed Berat Özer's runner glue, which
has no licence (docs/fork_eligibility.md A21 / A22, fix F5). This file specifies everything that glue does on the
fork's code path, so that our own modules in `casmi/` can be written from this spec, from ms-pred (MIT) and from our
own code. It covers:
- the two runner programs, `ice_runner.py` and `gl_runner.py`;
- the notebook-side helpers in `fuse.py` and `gl_fuse.py`;
- the `shim/` stand-ins;
- every place in the frozen pipeline that consumes their output.

Section 13 compares the spec with our `casmi/fwdrun.py`.

## 0. Sources, pins and citation keys

| key | file | sha256 | where read |
|---|---|---|---|
| IR | casmi26-iceberg `ice_runner.py` (31,551 B) | dc0d5d1e…0b2802 (= its MANIFEST) | f5b/dl/ice/ice_runner.py |
| F | casmi26-iceberg `fuse.py` (8,349 B) | 6d1bdf0c…c76bb1 | f5b/dl/ice/fuse.py |
| GR | casmi26-glacier `gl_runner.py` (39,880 B) | 5c4f7431…440fca (= FZ c41:165 pin, = MANIFEST) | f5b/dl/gl/gl_runner.py |
| GF | casmi26-glacier `gl_fuse.py` (7,533 B) | ea390f5f…a58e | f5b/dl/gl/gl_fuse.py |
| SH | `shim/dgl`, `shim/torch_scatter` (the same in both datasets); `shim/LinSATNet` (casmi26-glacier only) | dgl/__init__ 35dd2028…; LinSATNet a0ffdf64… | f5b/dl/{ice,gl}/shim/ |
| FZ cNN:L | elig0 frozen notebook `scratchpad/elig/elig0_frozen_patched.ipynb`, cell NN, line L of that cell | — | split copies in f5b/fz/cNN.txt |
| GC | `glb_core.py`, embedded in FZ c41:167-625 (Apache-2.0 notebook code) | f7a3b8cd…fdfa (= FZ c41:163 pin) | f5b/fz/glb_core.py (decoded) |
| GD | `glb_driver.py`, embedded in FZ c41:627-965 | e919a154…8502 (= FZ c41:164 pin) | f5b/fz/glb_driver.py (decoded) |
| OUT | pube1 outer notebook, cell 8 `MODULES[...]` | MODULE_SHA256 there | kernels/casmi26-pube1/notebook.ipynb |
| FR / EN / FF / RO | our `casmi/fwdrun.py` (sha 1bab3f2b…), `casmi/entropy.py` (8ecc3770…), `casmi/fwdfeat.py`, `casmi/run_own.py` | — | repository |
| MP | ms-pred @708148c2a8eb (MIT), `/home/user/ext/mspred/ms-pred/src/ms_pred/...` | git HEAD clean | local clone |

**Provenance of the copies.**
- The glue files were downloaded fresh on 2026-10-31 with `kaggle datasets download -f` into `scratchpad/f5b/dl/`.
  They are the current dataset versions of 2026-09-26 (casmi26-iceberg) and 2026-09-28 (casmi26-glacier).
- Their hashes match both datasets' MANIFEST.json and the pin in the frozen pipeline.
- The FZ cell numbers are those of the decoded elig0 notebook. elig1 changes only the Engine 2 cells (scratchpad/elig/elig1_frozen.diff). The glue call sites are the same in both.

**Runtime evidence.** Commit-run logs and the runners' own input and output files:
- `/home/user/work/kout/casmi26-pube1/submission_pipeline.log`, lines 137-250;
- `/home/user/work/kout/casmi26-pubm/{ice_work,gl_work,n1_work}/*.json` (the real in / out / meta files of the pubm
  commit: 12 visible molecules). Section 14 uses these as golden fixtures.

**Not glue.** The following parts are pipeline code (Apache-2.0 notebook code, eligible). This spec describes them only where they consume the glue:
- E004 / E006 (FZ c49 / c52);
- the N1 bench code (GC / GD);
- metric_identity / safe_extend (FZ c47);
- frag_rescore (FZ c51).

## 1. Where the frozen pipeline touches the glue

| # | site | glue entry point | what is used |
|---|---|---|---|
| 1 | FZ c26:90-106 | package layout of casmi26-iceberg (`wheels/rdkit-2025.3.6-*.whl*`) | `simulator_rdkit_wheel` picks the interpreter-compatible RDKit 2025.3.6 wheel from `<ICE_PKG>/wheels` or any `/kaggle/input/**/rdkit-2025.3.6-*.whl`. Not in a try: c26:105 `find('casmi26-iceberg/**/ice_runner.py')` raises if the dataset is not mounted. |
| 2 | FZ c41:26-55, 104 | `install_site` marker protocol (IR:60-86, GR:90-121) | The notebook pre-installs the wheels into `/kaggle/working/ice_site` and writes `.ice_site_ok` with the package's wheel list, so both runners skip their own pip. |
| 3 | FZ c41:59-61, 116-118 | module import `fuse`, `gl_fuse` from the dataset dirs | — |
| 4 | FZ c41:65, 82, 99 | `fuse.ice_candidates` | candidate selection (section 3.2) |
| 5 | FZ c41:102 | `fuse.build_ice_input(te, cands)` (covered_only=True) | runner input items |
| 6 | FZ c41:105-106 | `fuse.run_ice(...)` → `ice_runner.py` CLI | `ICE_SCORES` |
| 7 | FZ c41:120-124 | `gl_fuse.run_gl(...)` → `gl_runner.py` CLI | `GL_SCORES` (H1 = [M+H]+ only) |
| 8 | FZ c41:1173; GD:37-46 | `gl_runner.py` imported as a MODULE (25-name API, GD:24-26) | N1 driver and its self-test |
| 9 | FZ c41:1179, 1188, 1199 | `fuse.build_ice_input(..., covered_only=False)` | N1 input |
| 10 | FZ c41:1306, 1318 (off), 1330-1331 | `fuse.rerank`, `gl_fuse.rerank_multi` | PubChem-list re-order (base-list re-order is switched off) |
| 11 | FZ c45:55, 58 | `fuse.rerank` (with an empty dict), `gl_fuse.rerank_multi` | fused-list re-order |
| 12 | FZ c49:39-40 | `gl_fuse.rerank_multi` | E004 window-40 overlay |
| 13 | FZ c52:80-111 | `gl_fuse.rerank_multi`, `fuse.rerank` | E006 (GLACIER / joint N1 / frag gap / ICEBERG fallback) |
| 14 | FZ c43:103-104; FZ c50:29-41 | runner meta files `ice_work/ice_out.json.meta.json`, `gl_work/gl_out.json.meta.json` | run_manifest; commit acceptance asserts |
| 15 | OUT `export_evidence.py`:30-36 | `ICE_SCORES`, `gl_of` | `ice` / `gl` / `proxy_gl` columns of frozen_candidate_evidence.parquet (informational: `ranking.make_features`, the only reader, is never called) |

## 2. Execution environment (both runners)

- **E1 Process isolation.** Each runner is a separate Python process started by the notebook interpreter
  (`sys.executable`; F:76-77, GF:36-38). The notebook keeps RDKit 2026.03.3 (FZ c26:62-70) and never imports torch-DGL
  stand-ins or RDKit 2025.03.
- **E2 Import path.** The runner prepends three directories so that they come first, in this order (IR:137-144,
  GR:173-181):
  1. `<pkg>/shim`;
  2. `<pkg>/src` (the ms-pred subset);
  3. the site dir (unless `--site none`).

  It then evicts already-imported `rdkit`, `dgl`, `torch_scatter` and `ms_pred` (GL also `LinSATNet`) from
  `sys.modules` (IR:145-147, GR:182-184).
- **E3 Wheel site.** Wheels are shipped renamed `*.whl.ice`, so that the notebook's own `rdkit-*.whl` glob never picks
  them up.
  - Install: they are copied under `<site>/.wheels` with the `.ice` suffix stripped, then installed with pip offline,
    without dependencies, without bytecode, with `--upgrade --target <site>` (IR:60-86).
  - Idempotency: the marker `<site>/.ice_site_ok` holds the newline-joined sorted basenames of the package's wheel
    files. If the marker text equals that list, the install is skipped.
  - Failures: "no wheels" raises; a pip failure raises with the last 2,000 chars of its output.
  - GLACIER ships no wheels. It takes the wheel dir from `--wheels`, then `<pkg>/wheels`, then `$GL_WHEELS`, then
    the shortest `/kaggle/input/**/casmi26-iceberg/**/wheels` (GR:80-87). It reuses an existing marked site even when
    no wheel dir is found (GR:96-101).
  - The pipeline pre-seeds the site with the cp-compatible RDKit wheel and writes the marker itself (FZ c41:26-55),
    because the shipped RDKit wheel is cp312 only.
- **E4 RDKit pin.** The RDKit version must start with `2025.03` unless `--any-rdkit` is given (IR:169-172,
  GR:206-209).
  - The pipeline passes `--any-rdkit` to both runners (FZ c41:106, 124), but its pre-seed installs 2025.3.6 anyway.
    meta.rdkit is "2025.03.6" in every log, and FZ c50:36 asserts it in the commit run.
  - The N1 driver calls `setup_env` with any_rdkit False (GD:235), so the pin is enforced there.
- **E5 Module stubs.**
  - Three ms-pred modules are replaced by empty placeholder modules before import:
    `ms_pred.common.plot_utils` (whose three plot functions become no-ops), `ms_pred.nn_utils.tune_utils` and
    `ms_pred.nn_utils.base_hyperopt` (IR:153-157, GR:190-194).
  - `pygmtools`, `h5py`, `multiprocess` and `platformdirs` are first tried for real. Only those that fail to import
    are served by a permissive stub-module finder: any attribute is a placeholder, and capitalised names are
    subclassable classes (IR:89-128, 132, 158-168).
  - Logs show only `pygmtools` stubbed on Kaggle.
- **E6 Cache dir.** ms-pred's `chem_utils.user_cache_dir` is redirected to `<tempdir>/ice_ms_pred_cache` (IR:436-440,
  GR:501-505). This is the InChI tautomer-transform file used by ICEBERG's `predict_mol`; GLACIER does not use it.
- **E7 Tautomer mode.**
  - With RDKit 2025.03, `chem_utils` falls back to the `rdMolStandardize.TautomerEnumerator` ("v2") canonicaliser
    (MP common/chem_utils.py:23-29).
  - ICE logs the mode (IR:441), and every log says "v2". It matters only inside ICEBERG's `predict_mol` InChI round
    trip.
  - GLACIER canonicalisation does no tautomer step (rule G1).
- **E8 Determinism.**
  - Unless `--nondeterministic` is given: `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before torch is imported,
    deterministic algorithms are turned on with warn_only, and cudnn.benchmark is turned off (IR:641-642, 677-679;
    GR:775-776, 812-814; GD:200, 243-244).
  - The numba kernels compile with `cache=False`. `fastmath=True` is on for the weighting, query-prep and entropy
    kernels (IR:195-350).
  - The kernels fall back to pure Python without numba (IR:178-187).
- **E9 Device.**
  - `--device cuda` falls back to CPU when CUDA is unavailable (IR:669-672, GR:805-807, GD:239-241).
  - `--threads N>0` sets torch threads; `--memfrac f>0` sets the per-process CUDA memory fraction (IR:673-676).
  - Before ICE, the notebook frees its own GPU memory (FZ c41:62, 119).
  - Measured peaks: ICE 3.8-4.0 GB at bs 96; GL 0.33 GB.
- **E10 Shims.** These are behavioural stand-ins: same results as the real packages for the calls ms-pred inference
  makes.
  - `dgl` (SH dgl/__init__.py): homogeneous graphs; batch / unbatch; node subgraph keeping edges with both ends kept,
    in edge-id order; `reorder_graph` with a custom node permutation and stable edge sort by source; `update_all` with
    copy_u / copy_e / u_mul_e / u_add_e / u_mul_v messages and sum or mean reducers only; `apply_edges`; sum / mean
    readout.
  - `dgl.nn`: Avg / Sum / GlobalAttention pooling.
  - `dgl.backend.pytorch`: pad / pack.
  - `torch_scatter`: add / sum / mean / max / min, max and min with arg indices, scatter, scatter_softmax.
  - `LinSATNet` (GL only): the equality-constrained case `E x = f`, as Sinkhorn in log space. Its docstring argues it
    is a strictly monotone per-row transform, so top-k index sets match the real package except at exact float ties.
    GLACIER uses only the top-k indices at inference.
- **E11 ICE speed patches.** After loading, ICE swaps in two functions:
  - `pack_padded_tensor`, replaced by a mask-based version;
  - `GatedGraphConv.forward`, replaced by a per-edge-type linear + index_add + GRU version;
  
  both described as "same math" (IR:455-484).

## 3. Input building

### 3.1 `build_ice_input(test_df, cands_by_mid, covered_only=True)` (F:34-58)
- **Spectrum fields.** Spectra are grouped by `str(molecule_id)` (F:39-41). Each test row becomes one spectrum dict
  (F:47-51):
  - `mz` and `it`: the floats of `ms2_mzs` / `ms2_normalized_intensities`, or [] if missing or not iterable;
  - `prec` = float(precursor_mz);
  - `adduct`, `mode` (ionization_mode), `instrument` (instrument_type), `ce_orig` (collision_energy_orig) and
    `ce_units` (collision_energy_orig_units): strings, or None when missing / NaN;
  - `ce_ev`: a float list or None.
- **Malformed rows.** A row whose conversion raises is dropped silently (F:52-53).
- **covered_only.** Keeps only spectra with `mode == 'positive'` and adduct in {[M+H]+, [M+Na]+} (F:54-55).
- **Item order and shape.**
  - One item `{mid: str, spectra: [...], cands: [str]}` per key of `cands_by_mid`, in that dict's order (the
    priority order; F:44, 57).
  - Molecules with no surviving spectra are still emitted, with `spectra=[]`. Example: m_06e13d in the pubm
    `ice_in.json`.

### 3.2 Candidate selection (what gets predicted)
- **`ice_candidates(smis, formulas, top_n=40)`** (F:98-105). It takes the first m = min(top_n, len) entries of a
  ranked list and counts formulas among them. It returns, in rank order and de-duplicated by SMILES string, the
  entries whose formula occurs ≥ 2 times among those m. Only these entries can be moved by a re-rank.
- **Frozen pipeline union** (FZ c41:58-101):
  - **Molecule order (= runner priority):** ascending Engine 1 `lib_max`, so molecules without a library match go
    first (c41:64).
  - **Base list alone:** contributes nothing (`BASE_ICE = False`, c41:9, 65).
  - **ICE_UNION, run when ENG is non-empty:**
    - `bs` = the base list top 60 (TOPN, FZ c26:13) with its Engine-1 formula labels `bf`. Generated candidates
      carry the suffix `|gen`, so they never share a group with pool candidates (FZ c39:64).
    - `es` = the Engine 2 top 40 whose keys are not among the base top-60 keys, with formulas from the notebook's
      CalcMolFormula. If that raises, the formula is the SMILES itself (c41:66-86).
    - Selection: `ice_candidates(bs + es, bf + ef, len(bs) + len(es))`.
    - elig0's null engine makes `es` empty, so this is exactly the base list's same-formula groups within its top 60.
      elig1 adds our stage-1 candidates.
  - **ICE_PC:** for molecules with a PubChem list and `lib_max < LIB_TAU = 0.9` (FZ c26:12):
    - `pc_form` = CalcMolFormula of each PubChem SMILES, stored on the PC entry (c41:96-98, used later as a gate,
      rule P1);
    - selection: `ice_candidates(pc, pc_form, 25)`, appended and de-duplicated (c41:87-100).
  - **Final step:** molecules with an empty selection are dropped (c41:101). Then `items = build_ice_input(te, cands)`
    (c41:102).
- **GLACIER input** (FZ c41:120-121). The same items with spectra filtered to `adduct == '[M+H]+'` ("v29: +0.006 LB").
  Items left without spectra are dropped, so `gl_in` keeps the ICE order and candidate lists exactly; verified on the
  pubm fixtures.
- **Scale.** Commit runs: 12 molecules, 814 (pubm) / 832 (pube1) candidates, about 1,600 predictions per model.
  The README sizes the full test at about 786 distinct adduct / CE conditions × ≤ 60 candidates.

## 4. Runner CLI and file contracts

### 4.1 CLI (IR:621-636, GR:753-770)
`python <runner>.py <pkg_dir> <in.json> <out.json>`, with these options:

| option | default | notes |
|---|---|---|
| `--device` | `cuda` | falls back to CPU (rule E9) |
| `--budget S` | 0 = unlimited | wall clock for the whole run, measured from process start (so it includes the site install and model load; IR:645) |
| `--site DIR\|none` | env `ICE_SITE`, else `/tmp/ice_site` | |
| `--bs` | ICE 96, GL 32 | |
| `--chunk-jobs` | 3000 | |
| `--threads` | 0 | |
| `--memfrac` | 0 | |
| `--nondeterministic` | off | |
| `--any-rdkit` | off | |
| `--quiet` | off | |
| `--wheels DIR` | — | GL only |
| `--ckpt FILE` | `<pkg>/ckpt/glacier.pt` | GL only |


The pipeline's calls:
- ICE: `--device cuda --budget <ICE_BUDGET> --site /kaggle/working/ice_site --any-rdkit` (F:76-77, FZ c41:105-106).
- GL: the same, with budget 4000 and `--wheels <ICE_PKG>/wheels` (GF:36-38, FZ c41:123-124).
- Both run from workdirs `/kaggle/working/{ice,gl}_work` with files `{ice,gl}_in.json` and `{ice,gl}_out.json`
  (F:70, GF:30).

### 4.2 Input JSON
A list of items `{mid, cands: [smiles], spectra: [spectrum]}` (IR:9-13). Spectrum keys:
- `mz` and `it`: lists;
- `prec`;
- `adduct`;
- `mode`: "positive", +1 or -1;
- optional `instrument`;
- optional `instr`: a pre-resolved token, honoured if it is "Orbitrap" or "QTOF" (IR:396-399);
- either the raw `ce_ev` / `ce_orig` / `ce_units`, or an already-resolved `ce` (used only when `ce_ev` and
  `ce_units` are both absent; IR:374-380).

N1 items additionally carry `sid`, `tag` and `n1_src` (GD:14-15).

### 4.3 Output JSON
- **Schema.** `{mid: {smiles: score | null}}` for every input molecule and every input candidate string, duplicates
  included (IR:653-656).
- **Initial write.** The file is written all-null before anything heavy, then rewritten after every completed chunk
  and at the end (IR:657, 712, 661).
- **Encoding.** Atomic write: a temp file in the same dir, then os.replace (IR:41-45). JSON uses `allow_nan=False`.
- **null means:** no covered spectrum, SMILES rejected, prediction failed, or the molecule was not reached before the
  deadline.

### 4.4 Meta JSON: `<out>.meta.json` (IR:646, 659-663, 744-748; GR:780-781, 880-884)

| field | when set |
|---|---|
| `status` ∈ {init, bad_input, ok, budget, error} | always |
| `n_mols`, `n_mols_covered` (molecules with ≥ 1 covered spectrum), `n_mols_scored` (molecules whose chunk finished), `n_cands`, `n_scored` (non-null scores), `errors` (≤ 5 rejected-SMILES lines, plan / score errors, traceback tail) | always |
| `seconds`, `log_tail` (last 40 log lines) | always |
| `rdkit`, `torch`, `device`, `t_load` | after setup |
| `n_pred`, `n_pred_fail`, `t_pred`, `n_bad_smiles`, `pred_per_s` | at the end of a non-error run |
| `max_mem_gb` | on CUDA |
| `model: "glacier"`, `n_feat_fail`, `bs_final` | GL only |

**Consumers.**
- FZ c43:103-104 reads `n_mols_scored` from both meta files (−1 if either is missing).
- In the commit run, FZ c50:29-41 asserts for both stages:
  - `status == 'ok'`;
  - device starts with `cuda`;
  - `rdkit == '2025.03.6'`;
  - no errors;
  - `n_pred_fail == 0` and `n_feat_fail == 0`;
  - `n_mols_scored > 0` and `n_scored > 0` when any molecule is covered.

**Logs.** stderr lines prefixed `[ice   t s]` / `[gl   t s]` (IR:48-56). `run_ice` prints the subprocess output
tail (≤ 3,000 chars) and a meta summary (F:84-90).

## 5. Query side (identical in ICE and GL)

- **Q1 Coverage.** A spectrum is scored iff its mode is positive (the integer 1, or a string in {positive, pos, +, 1}
  case-insensitive) and its adduct is in {[M+H]+, [M+Na]+} (IR:34, 390-393). The pipeline sends GL only [M+H]+
  (section 3.2).
- **Q2 Instrument token.** "Orbitrap" iff `instr_family(instrument_type) == 1`, else "QTOF" (IR:396-399). The family
  test lower-cases the string and checks in this order (IR:354-364):
  1. contains "timstof" → 0;
  2. contains any of orbitrap / qft / ftms / "hybrid ft" / itft / exactive / ft-icr / fticr → 1 (Orbitrap);
  3. contains "tof" → 2;
  4. contains trap / qit / lit → 3;
  5. contains qq / quattro / triple → 4;
  6. otherwise → 5.
  
  None / NaN → 5 (QTOF). The visible test is 100 % timsTOF (1,213 / 1,213 spectra), so the token there is always QTOF.
- **Q3 Collision energy** (IR:367-380):
  - If the spectrum has `ce_units` or `ce_ev`:
    - units == "NCE": CE = the mean of all numbers found in the `ce_orig` string (regex for unsigned decimals);
    - otherwise: CE = mean(ce_ev), or NaN if it is empty / None.
  - Otherwise CE = float(`ce`) if finite, else NaN.
- **Q4 CE bucket** (IR:383-387). bucket = round(CE / 5) × 5, with numpy round-half-to-even, floored at 5. NaN stays
  NaN.
  - Stepped CE (20, 40, 60) therefore gives bucket 40, and a single CE of 2 gives 5.
  - The visible test is 100 % eV with means in {20, …, 60}.
- **Q5 Missing CE: "sum" prediction** (IR:35, 425-430, 583-592, 600-609). When the bucket is NaN:
  - predict at CE 20, 40 and 60, each max-normalised;
  - concatenate the three predictions, sum intensities at identical m/z rounded to 4 dp, and divide by the maximum;
  - no top-k after the merge;
  - the summed prediction exists only if all three parts exist.
- **Q6 Query peak cleaning** (IR:402-407, 195-237, 268-303). Applied in order:
  1. `clean_peaks` with floor 0.001, top 512 and max-over-precursor 2.0:
     - base = the maximum intensity among peaks with m/z ≤ prec + 2;
     - keep peaks with m/z ≤ prec + 2, m/z > 0 and intensity ≥ 0.001 × base;
     - if more than 512 remain, keep the 512 most intense;
     - divide intensities by base, sort by m/z, cast to float32;
     - empty input, or base ≤ 0, gives an empty spectrum.
  2. `clean_peaks` again with the same parameters, on the float32 result cast back to float64.
  3. `prep_query` with FLOOR 0.002, TOPK 256, POWER 1 and entropy weighting (IR:36):
     - keep peaks with intensity ≥ 0.002 × max;
     - if more than 256 remain, keep the 256 most intense, restoring m/z order;
     - weights per rule S2.
  
  The effective relative floor is therefore 0.002, and at most 256 peaks are kept. A query with zero peaks after
  cleaning is still a planned spectrum: its similarity is 0 for every candidate (S3).

## 6. Candidate side

- **C1 ICE acceptance** (IR:487-504). A candidate is accepted when all of the following hold:
  - RDKit parses it, with ≥ 1 atom;
  - every element is in ms-pred `VALID_ELEMENTS`;
  - the InChI round trip of the stereo-stripped SMILES gives a parseable SMILES.
  
  Rejected candidates stay null. Predictions are cached per input SMILES string, not per canonical form (IR:510).
- **G1 GL canonicalisation** (GR:520-565). It follows ms-pred `predict_smis_joint.prepare_entry` (MIT):
  1. parse the SMILES;
  2. reject it if any atom's atomic number is outside `VALID_ATOM_NUM`;
  3. remove explicit H atoms, and reject it if more than 100 heavy atoms remain;
  4. canonical SMILES = RDKit's default canonical output (stereo kept);
  5. the canonical form must re-parse with ≥ 1 atom.
  
  The heavy-atom count of the canonical form is kept for batching. The result is cached per input string in a
  module-level table (`_CANON`) as (canonical or None, heavy atoms, reason). Jobs are de-duplicated on the canonical
  form, so identical canonical forms are predicted once (GR:620, 662-675).
- **C2 Per-molecule handling** (IR:719-737, GR:855-873):
  - candidates are de-duplicated preserving order;
  - each new string is checked once per run;
  - the first 5 rejections are recorded in meta.errors.
  - Environment guard: if ≥ 50 strings were checked and all were rejected, the run raises, giving status error
    (IR:731-732).
  - Molecules with no covered spectrum, or no accepted candidate, are skipped and stay all-null.

## 7. Prediction

- **P-ICE** (IR:513-520, 411-422; MP iceberg/joint_model.py:65). One batched call to ICEBERG `JointModel.predict_mol`
  per (adduct, instrument token, CE) condition, with:
  - collision energy = the bucket, precursor m/z 0, the adduct as given, threshold 0, max_nodes 100;
  - instrument = the token, binned output off.
  
  Each returned spectrum is post-processed:
  1. drop rows with intensity ≤ 0;
  2. merge identical m/z (rounded to 4 dp) by SUMMING intensities;
  3. keep the 100 most intense;
  4. sort by m/z and divide by the maximum (float32).
  
  An empty result is treated as a failure (None).
- **P-GL** (GR:568-638, 448-461; MP glacier/joint_model.py:953-981).
  1. Per job, build the item:
     - the GLACIER `TreeProcessor`, with the model's `pe_embed_k`, `embed_elem_group` and `multi_hop_max_dist`,
       root encoding "graphormer", featurises the tree {canonical SMILES, adduct, CE} without targets;
     - the featurisation is cached LRU (512 entries) per (canonical, adduct), and the item's `collision_energy` is
       overwritten with the job's CE;
     - item fields: adduct = the one-hot position of the adduct; precursor 0.0; instrument = the one-hot position of
       the token, falling back to Orbitrap's position.
  2. Collate with `IntenDataset.collate_fn`, move to the device, and run `predict_inten_frag_batch` (dense: all
     predicted rows).
  3. Post-process each molecule's [rows, 2] (m/z, intensity):
     - drop rows with intensity ≤ 0 or m/z ≤ 0;
     - merge identical m/z (4 dp) by MAXIMUM;
     - keep the 100 most intense (stable sort);
     - sort by m/z and divide by the maximum (float32).
  
  A featurisation failure fails only that job (counted in `n_feat_fail`).
- **P-load.**
  - ICE loads `gen.pt` and `inten.pt` as {hyper_parameters, state_dict}: weights-only, strict, into FragGNN /
    IntenGNN wrapped in JointModel, eval mode (IR:433-452).
  - GL loads `glacier.pt` (GR:499-517). It drops hyper-parameters whose names start with `_` and
    aligns the instrument / adduct one-hot tables with the checkpoint's embedder sizes. A 2-slot instrument embedder
    trims the table to {Orbitrap, QTOF}; any other mismatch raises (GR:477-496). It then loads strict, freezes, and
    marks the model prepared.
  - In the logs the notes are empty: the MassSpecGym GLACIER checkpoint matches ms-pred's 4-slot table
    (MP common/chem_utils.py:277-283). `glacier.pt` is a re-save of best.ckpt (sha 5a47cecc…).
- **P-batch.**
  - ICE (IR:545-568) loops over conditions in sorted order. Within a condition, SMILES are sorted by (length,
    string). The batch size depends on the length L of the last SMILES in the next window:
    - L ≤ 60: bs;
    - L ≤ 90: bs / 2;
    - L ≤ 140: bs / 4;
    - otherwise: bs / 8, at least 1.
  - GL (GR:662-700, 472-474) sorts its unique canonical jobs by (heavy atoms, canonical, adduct, instrument, CE).
    Batch = min(bs, ⌊bs·1600 / N²⌋), at least 1, where N is the maximum heavy-atom count among the next bs jobs.
- **P-fail.** A failed batch is retried in quarters, recursively; a single failing job gives None.
  - On a CUDA out-of-memory error with more than 4 jobs, the run-wide max batch drops to max(4, ⌊failed size / 2⌋)
    for the rest of the run (IR:522-543, GR:640-660).
  - GL fail-fast: once ≥ 64 predictions have been made and all of them failed, the run raises, giving status error
    (GR:57, 695-698).
- **P-chunk** (IR:694-743). Molecules are taken in input order. Their (candidate, adduct, token, CE) jobs, including
  the 20 / 40 / 60 jobs of NaN buckets, accumulate into a chunk until the chunk holds ≥ `--chunk-jobs` (3000) jobs.
  The chunk is then predicted, and only if prediction finished before the deadline:
  - all its molecules are scored;
  - `n_mols_scored` is counted;
  - out.json is rewritten.
  
  The deadline is checked before every batch (IR:556-558). A chunk interrupted by the deadline leaves all its
  molecules null, so coverage is all-or-nothing per molecule, in priority order.

## 8. Score definition written to out.json

- **S1 Score.** `ice_ex_mean` / `gl_ex_mean`(candidate) = the arithmetic mean, over the molecule's covered spectra
  that have a prediction for this candidate, of entropy_sim(query, prediction) (IR:595-617, GR:727-749). It is null
  when no spectrum contributes, and NaN similarities are skipped. Per spectrum:
  - **finite bucket:** the prediction at (candidate, adduct, token, bucket);
  - **NaN bucket:** the summed prediction of Q5.
  
  The prediction goes through the same `prep_query` as the query: floor 0.002, top 256, weights S2 (IR:612).
- **S2 Peak weights** (IR:240-265). p_i = I_i / ΣI (POWER 1, i.e. linear). Then S = −Σ p ln p, in nats. If S < 3:
  - p_i ← p_i^w / Σ p^w with w = 0.25 + 0.25·S.
  
  This is the weighted spectral entropy of Li et al. 2021.
- **S3 Similarity** (IR:306-350). Both spectra are sorted by m/z, and two cursors walk them together:
  - at each step the tolerance is t = max(0.01 Da, 20 ppm × the current query m/z);
  - if the query peak lies below the prediction peak by more than t, it is emitted alone as p_q / 2;
  - if it lies above by more than t, the prediction peak is emitted alone as p_c / 2;
  - otherwise both are emitted together as (p_q + p_c) / 2 and both cursors advance (no nearest-neighbour search);
  - leftover peaks are emitted alone.
  
  Then S_AB = −Σ v ln v over the emitted values, and sim = 1 − (2·S_AB − S_A − S_B) / ln 4. It is clipped to
  [0, 1], with values below 1e-9 set to 0. If either spectrum is empty, sim = 0.
- **S4 Exclusions.** There is no cosine variant and no m/z binning. The only m/z rounding is the 4-dp
  identical-m/z merge of predictions. There is no intensity transform other than linear + entropy weighting.
  Precursor-region peaks are not removed beyond the prec + 2 Da cut.

## 9. Failure semantics and budgets

### 9.1 Runner level
- **Unreadable input:** `{}` and status `bad_input` are written, exit 0 (IR:647-652).
- **Any later exception:** the traceback goes to meta, scores so far are kept, status `error`, exit 0. Exit code 2
  only if even the final write fails (IR:750-756).
- **Budget stop:** status `budget`, exit 0.

### 9.2 Wrapper level
`run_ice` / `run_gl` (F:62-95, GF:22-56) never raise:
- they delete stale out / meta files first;
- the subprocess gets a timeout of budget + `grace_s` (300 s), or none when budget = 0;
- on a timeout, the out.json the runner checkpointed is still read;
- a missing out.json gives `{}`.

### 9.3 Pipeline level (FZ c41)
- **Exceptions around ICE:** an exception while building the ICE input, pre-seeding or calling ICE sets
  `ICE_SCORES = {}` and the ICE health string (c41:110-112). The health strings are only written on failure, so the
  logs say "ICE NOT RUN" after a successful run (pube1 log line 188).
- **GLACIER failure:** `GL_SCORES = {}`, so only ICE remains (c41:127-130).
- **Coupling (important).**
  - Every later GLACIER use requires a truthy per-molecule ICE dict: c41:1328 (PubChem list), FZ c45:53,
    FZ c49:38, FZ c52:77.
  - Because the ICE runner writes an all-null dict for every input molecule, a molecule that was in the ICE input
    passes this gate even when ICEBERG scored nothing.
  - If `ICE_SCORES` is `{}` (total ICE failure), GLACIER is unused in every final list.
  - Any replacement, including "drop ICEBERG", must keep supplying a truthy dict per molecule, or the gates must be
    patched.
- **Budgets** (FZ c26:13, c30:16, c41:114, c41:150, c41:1021-1022):

  | stage | rerun | commit |
  |---|---|---|
  | ICE | 5,400 s | 300 s |
  | GL | 4,000 s | 4,000 s |
  | N1 driver | 1,200 s, subprocess timeout + 600 s | 600 s |

- **Measured speed:**
  - commit ICE: 1,627 predictions in 39 s, i.e. 42 / s, plus 32-57 s of load;
  - GL: 87 / s, plus 21-35 s of load;
  - N1: 93-98 / s.

## 10. N1 proxy path (GC / GD, run in FZ c41:1159-1296)

- **N1-0 Gate.** N1 runs only if H1 GLACIER produced at least one non-null score (c41:1161-1164).
- **N1-1 Input.** `build_ice_input(te, cands, covered_only=False)` (c41:1179), then `build_variants` (GC:97-124):
  - only positive spectra are considered, and each gets `sid = "<index of the spectrum in the molecule>:<tag>"`;
  - [M+H]+ → tag H, plus a diagnostic "Hcut" copy;
  - [M+Na]+ → Na;
  - [M+NH4]+ → NH4p proxy, and [M−H2O+H]+ → WLp proxy (proxies per N1-2).
  
  `n1_select` (c41:977-1004) keeps the molecules with no real positive [M+H]+ that have ≥ 1 NH4p / WLp proxy and
  non-empty candidates. It keeps only their proxy spectra, in priority order.
- **N1-2 Proxy construction** (GC:25-30, 52-84). Masses: PROTON 1.007276, NH4 adduct 18.033823, H2O 18.010565,
  cut 1.5 Da.
  - **[M+NH4]+:** M = prec − 18.033823. Keep peaks with m/z < M + PROTON − 1.5.
  - **[M−H2O+H]+:** M = prec + H2O − PROTON. Keep peaks with m/z < prec − 1.5, i.e. the original precursor.
  - **Both:** a proxy needs ≥ 1 kept peak with intensity > 0. It is relabelled [M+H]+ / positive with
    prec = M + PROTON, and records `n1_src`.
  - Never proxied: [M−2H2O+H]+, [M+Na]+, [M+K]+.
- **N1-3 Driver** (GD, one subprocess: `glb_driver.py <GL_PKG> n1_in.json n1_out.json --device cuda --budget B
  --site /kaggle/working/ice_site --wheels <ICE_PKG>/wheels`; c41:1017-1018).
  - It loads `<GL_PKG>/gl_runner.py` as a module and requires 25 names: `setup_env`, `load_model`, `Predictor`,
    `smiles_ok`, `_CANON`, `molecule_jobs`, `score_molecule`, `prep_spectrum`, `prep_query`, `entropy_sim`, `merge`,
    `post`, `ce_bucket`, `spectrum_ce`, `is_covered`, `instr_token`, `SUM_CE`, `FLOOR`, `TOPK`, `POWER`, `ENTW`,
    `TOL`, `PPM`, `write_json`, `Log` (GD:24-26, 37-46). It also uses `HAVE_NUMBA` (GD:247).
  - The calling conventions these imply, which a replacement `gl_runner` module must honour:
    - `setup_env(namespace with pkg, site, wheels, any_rdkit, log)` returns the RDKit version string;
    - `load_model(abs pkg, device, log)` returns the model;
    - `Predictor(model, device, bs, log)` has `.predict(set of (smiles, adduct, token, ce), deadline)` returning a
      bool, plus `.cache` (dict keyed by the job tuple, value (m/z f64, intensity f64) or None) and the counters
      `n_pred`, `n_fail`, `n_feat_fail`, `t_pred`, `bs`;
    - `smiles_ok(smi)` returns (ok, reason) and fills `_CANON[smi] = (canonical | None, heavy atoms, reason)`;
    - `molecule_plan` entries are (qm, qp, adduct, token, bucket).
  - **Per spectrum** (GD:50-101):
    - **mean mode** = the production `gl_ex_mean` on that spectrum alone.
    - **merged mode:** if the units are not NCE and the `ce_ev` values give ≥ 2 distinct finite buckets (rule Q4,
      including the floor at 5), predict at each bucket, merge them as in Q5, then prep and score. It is null if any
      bucket's prediction is missing; otherwise merged = mean.
  - **Run behaviour** (GD:197-333):
    - a molecule with nothing to score is "done" with empty spec;
    - checkpoints are atomic after every chunk;
    - `meta.done` lists the finished molecules.
  - **Output:** `{mid: {cands, spec: {sid: {tag, nb, mean: [...], merged: [...]}}}}`, lists aligned with `cands`.
- **N1-4 Scores** (c41:1043-1063, GC:192-209). Arm N1A_D gives tags (NH4p, WLp) in merged mode. A candidate's score
  is the mean over that molecule's proxy spectra of the non-null per-spectrum values.
  - Only molecules listed in `done` are used, and only if their candidate order matches the input.
  - Values that are non-finite or outside [0, 1] ± 1e-9 become null.
  - Molecules whose scores are all null are dropped.
- **N1-5 Merge into GL.**
  - In the rerun, molecules that already have H1 GL scores are dropped as clashes; the commit run asserts there are
    none (c41:1213-1215, 1251).
  - `N1_COMPLEMENT = False`, so no ICE-based filter is applied (c41:160).
  - `GL_SCORES.update(N1 scores)` and `N1_GL_MIDS` = the scored molecules (c41:1276-1277).
  - Any exception removes the N1 entries, giving H1 behaviour; the commit run re-raises (c41:1279-1286).
- **N1-6 Commit exercise** (c41:1185-1249). The 12 smoke molecules have no N1 case. The commit therefore builds
  synthetic `n1x:<mid>` copies:
  - each [M+H]+ row becomes an [M+NH4]+ row with an injected precursor peak;
  - the first such row also becomes an [M−H2O+H]+ row.
  
  These go through the same code, and the commit asserts:
  - the proxy arithmetic (< 1e-6 Da) and the cut;
  - an arm-B guard;
  - merged-CE paths exercised and merged == mean for single-CE spectra;
  - Spearman correlation of proxy vs H1 GL scores ≥ 0.3 (median).
  
  pube1 commit: 11 synthetic molecules, 39 spectra (8 merged), 1,971 predictions, status ok.
- **N1-7 Hash check.** The gl_runner sha is compared with its pin (c41:1175-1177) but only printed, not asserted. A
  replacement module prints False and still runs.

## 11. From scores to re-orderings

### 11.1 z definition (F:109-120 = GF:60-71)
- A value is usable if it is an int or float that is finite. numpy float32 does not count: the type test is against
  Python int / float.
- With n usable values: mean μ, sample SD with ddof = 1.
- n < 2, or SD not > 0: all z = 0.
- z = (x − μ) / SD for usable values; unusable values get z = 0.

### 11.2 `rerank(smis, keys, scores, formulas, ice_scores, lam=0.5, top_n=60, min_covered=2)` (F:123-160)
- **Inputs:** aligned lists in the current ranked order. `scores` are the upstream ranker's scores, higher is better.
- **Empty case:** an empty list or a falsy `ice_scores` returns the identity order.
- **Lookup:** by exact SMILES string, then by the key. A list aligned with `smis` is also accepted (F:136-144).
  Note: the pipeline's keys are score keys, which never equal the runner's SMILES keys, so in practice the lookup is
  by exact SMILES string. A candidate predicted under a different SMILES spelling is uncovered.
- **Groups:** same formula string among the first min(top_n, n) positions (F:145-147). Groups of size 1 are skipped.
- **Coverage gate:** a group needs ≥ max(2, min_covered) members with finite re-scorer values (F:151-154).
- **Fused score:** fused = z(scores) + lam · z(re-scorer), both within the group (F:155-156).
- **Order:** members are re-sorted by (−fused, current position), i.e. stable on ties, and written back into the
  group's own ascending slots. Everything else stays put (F:157-159). The result is a permutation of indices.

### 11.3 `rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9)` (GF:94-139)
- **Weights:** one lam per score dict, or ValueError (GF:109-110).
- **Absent re-scorers:** an empty or None dict counts as absent (GF:111). If none is present, the identity order is
  returned.
- **Values:** cleaned to finite float or null; non-numeric values become null (GF:87-91, 121-124).
- **Group gate:** a group is processed if at least one re-scorer covers ≥ max(2, min_covered) of its members
  (GF:125-126).
- **Fused score:** fused = z(scores) + Σ lam_k · z(score_k), where only re-scorers with ≥ 2 covered members are added
  (GF:127-133).
- **Ties:** if ≥ 2 re-scorers contributed, fused values are mapped to integers round(x / 1e-9) before sorting, so
  float-noise near-ties keep the current order (GF:134-135). Example: a 2-member group where ICE and GL both oppose
  the ranker at λ = 0.5 each.
- **Single re-scorer:** with one re-scorer it is arithmetically identical to `rerank`.
- **Order and slots:** the same stable sort and slot write-back as `rerank` (GF:136-138).

### 11.4 Consumers: windows, weights, formulas

Every consumer uses ICE_LAM = GL_LAM = 1.0 (FZ c26:13, c41:114), not the 0.5 in the docstrings.

**P1 PubChem list** (FZ c41:1327-1337).
- **Condition:** the molecule has a PC entry with `pc_form` (ICE_PC, lib_max < 0.9) and a truthy ICE dict.
- **Steps:**
  1. `rerank(pc, pc_keys, pc_fz, pc_form, ice, 1.0, top_n=25)`;
  2. if `gl_of(m)` is non-empty, `rerank_multi` with [ice, gl] and [1, 1], top_n 25, recomputed from the ORIGINAL
     PubChem order, which replaces step 1's order (c41:1301-1311);
  3. `pc` and `pc_keys` are permuted; `pc_fz` keeps its original first value and the permuted remainder.
- **Effect downstream:** the gated merge (FZ c43:48-87) interleaves the re-ordered PubChem list. It looks up
  promotion evidence for the new leader (`pc_keys[0]`) and asserts that the evidence exists. ICE / GL can therefore
  change which PubChem candidate is promoted.

**P2 Base list.** Not re-ordered (BASE_ICE False, FZ c41:1316). It is truncated to 25 (c41:1326).

**P3 Fused list** (FZ c45).
- `fuse2`: reciprocal-rank fusion over score keys, 1 / (3 + r) for the main list plus 0.6 / (3 + r) for Engine 2,
  top 40 (c45:5, 19-29).
- If the ICE dict is truthy: `rerank` with an empty dict (a no-op, CELL19_ICE False, c45:6).
- If `gl_of` is non-empty: `rerank_multi` with [{}, gl], top_n = list length (≤ 40). Formulas come from the
  notebook's CalcMolFormula on RDKit 2026.03.3 (`_form2`).
- The top 25 go to submission.csv, and the remainder is kept as FUSION_OVERFLOW (c45:44-70).

**P4 E004 window-40 overlay** (FZ c49:26-44).
- `metric_fusion` (c49:4-20): RRF with one vote per metric identity per list. The identity is the InChIKey-14
  after stereo removal and tautomer canonicalisation, paired with the heavy-atom composition (FZ c47:6-61).
  - Main list (PREFUSION_LISTS without "CCO"): weight 1.0.
  - Engine 2: weight 0.6.
  - Score = weight / (3 + unique rank); the first-seen SMILES represents each identity; limit 40; ties stay in
    insertion order.
- If the ICE dict and `gl_of` are both truthy: GL-only `rerank_multi` ([{}, gl]) over the whole window.
- Result: `safe_extend`(top 25, rest of the window + the previous final list). The first 25 unique metric
  identities are kept (FZ c47:63-74).

**P5 Fragment gap** (FZ c51:16-31).
- Applies to molecules with no GL scores and no finite ICE value.
- frag_rescore (Apache-2.0) scores the PREFUSION_LISTS ∪ Engine 2 SMILES on up to 16 spectra, giving FRAG_GAP.

**P6 E006** (FZ c52:27-125, called with window 40 and ice_fallback True, c52:219).
- **Control reconstruction:** the same window-40 metric_fusion as P4. The run first rebuilds the E004 output with
  ice_fallback False and asserts that it equals the E004 file byte for byte (c52:154-157).
- **Branches, in order:**
  1. **"glacier":** the ICE dict and `gl_of` are both truthy. `rerank_multi` with [ICE dict if the molecule is in
     N1_GL_MIDS else {}, gl] and [1, 1] (c52:77-91). This is the joint ICEBERG + GLACIER re-ordering: it applies
     only to N1 proxy molecules and uses the tie grid of 11.3.
  2. **"frag_gap":** no `gl`, a non-empty FRAG_GAP entry, and no non-null ICE value. `rerank_multi` with
     [FRAG_GAP] and [0.5] (c52:92-99).
  3. **"ice_fallback":** a truthy ICE dict and no `gl`. `rerank` with ICE, lam 1.0, window-wide (c52:100-113). An
     all-null ICE dict makes this a no-op; it is counted in the audit.
  4. **"unrescored":** otherwise.
- **Output:** `safe_extend`(top 25, rest + original) (c52:114-118).
- **Audit counters** (c52:41-49, 174-201): finite ICE / GL coverage, fallback rows, joint rows. Visible commit:
  1 changed row, 0 fallback rows.

**P7 Rank-2 insert** (FZ c53). No glue input.

**P8 Evidence export** (OUT export_evidence.py:30-36). Exports ICE / GL per final candidate (lookup by SMILES, then
by score key) and `proxy_gl`. No ranking reads them in pube0 / pube1.

**Formulas.** The selection in 3.2 groups by Engine-1 labels (with `|gen`) or by CalcMolFormula. Every fusion groups
by CalcMolFormula on the fused SMILES (`_form2`, FZ c45:12-18), with the SMILES string as fallback. So a group at
fusion time can contain members that were never sent to the runner; they count as uncovered (z = 0).

## 12. If ICEBERG is dropped (F5b option)

The behaviour changes as follows. Each point needs either an explicit decision or a patch.
1. **Truthy placeholder.** Keep supplying `ICE_SCORES[mid] = {c: None, ...}` for every input molecule, or GLACIER
   disappears from P1, P3, P4 and P6 (section 9.3).
2. **P1:** becomes GL-only (single-scorer arithmetic, no tie grid).
3. **P6 "glacier":** becomes GL-only for N1 molecules too.
4. **P6 "ice_fallback":** never fires. In the visible test, 5 of 400 molecules have a positive [M+Na]+ but no
   [M+H]+ spectrum (2 more have only [M+NH4]+ / water-loss spectra). Such molecules move to "frag_gap" whenever
   FRAG_GAP has them; before, they did not, because frag_gap requires no finite ICE value. This is a policy change:
   the lineage credited the ICE fallback with +0.005 (docs/fork_eligibility.md:112).

## 13. Comparison with our `casmi/fwdrun.py` (+ `casmi/entropy.py`)

### 13.1 Identical (behaviourally)

| aspect | glue | ours |
|---|---|---|
| own-process runner, offline wheel site, shim → src → site path order | E1-E3 | FR:60-79, 177 |
| GLACIER model, TreeProcessor parameters, featurise without targets, collate_fn, adduct / instrument one-hot, precursor 0, CE overwritten per job | P-GL | FR:190-202, 262-280 |
| candidate canonicalisation: valid atoms, RemoveHs, ≤ 100 atoms, MolToSmiles; jobs de-duplicated on the canonical form | G1 | FR:207-212, 225-230 |
| coverage set {positive [M+H]+, [M+Na]+} (inputs from `build_ice_input` are the string "positive") | Q1 | FR:16, 219 |
| prediction post-processing rule: positive m/z and intensity, 4-dp merge by MAX, top 100 stable, sort, max-normalise (applied to different raw rows, see 13.2) | P-GL | FR:103-116 |
| entropy weighting S < 3 → w = 0.25 + 0.25·S on sum-normalised intensities; similarity formula 1 − (2 S_AB − S_A − S_B) / ln 4 | S2, S3 | EN:6-21, 64 |
| molecule score = mean over covered spectra with a prediction, null if none; all-null output first; same out.json schema | S1, 4.3 | FR:204-205, 303-316 |
| stepped CE → mean → round(/5)·5 (half-to-even) for eV spectra with CE present (all of the visible test) | Q3, Q4 | FR:119-122 |
| DGL / torch_scatter stand-ins: own code (fwdshim, MIT), reported score-identical to the glue's shim (max diff 0.0, docs/plan.md:2294-2295) | E10 | fwdshim/ |
| weights: casmi26-glacier `ckpt/best.ckpt` = the upstream file (sha 5a47cecc…), of which Ahmed's glacier.pt is a re-save | P-load | /home/user/ds/casmi26-glacier |

### 13.2 Different

| aspect | glue | ours | effect on timsTOF / eV data |
|---|---|---|---|
| query floor / top-k | effective floor 0.002, top 512 then 256, float32 round trip (Q6) | floor 0.001, top 256, float64 (FR:87-100) | more low peaks kept: changes every similarity slightly |
| short queries | kept; an empty query contributes sim 0 | spectra with < 2 peaks dropped (FR:222) | rare |
| prediction prep before scoring | `prep_query` on the prediction: floor 0.002, top 256 (IR:612, GR:744) | none: weights over all ≤ 100 peaks | systematic |
| raw prediction | dense `predict_inten_frag_batch`: all rows → MAX merge → top 100 | `predict_inten_frag_batch_sparse(col, 100)`: top 100 ROWS first, then MAX merge (MP glacier/joint_model.py:983-1047; FR:283-287) | can leave < 100 distinct peaks when rows share m/z; small |
| peak matching / tolerance | two-pointer walk, first peak within tolerance, t = max(0.01 Da, 20 ppm·m/z) (S3) | greedy nearest unused peak within an absolute 0.01 Da (EN:42-56) | differs in dense regions; 20 ppm matters only above m/z 500 (visible test max 442) |
| clipping | to [0, 1], < 1e-9 → 0 | none (EN:64) | cosmetic |
| numba | fastmath kernels | no fastmath, cache=True | float noise |
| LinSATNet | monotone shim | real LinSATNet 0.1.3 wheel (MIT; casmi26-glacier/wheels) | max score diff 4e-4 (docs/plan.md:2293-2294) |
| model load | state dict of the re-saved glacier.pt + one-hot alignment (GR:499-517) | Lightning `load_from_checkpoint(best.ckpt, weights_only=False)` (FR:190) | same tensors; alignment is a no-op for this checkpoint |
| instrument token | `instr_family == 1` list, timsTOF first, honours `sp['instr']` (Q2) | contains orbitrap / ft / "q exactive" (FR:82-84) | none on timsTOF; differs for e.g. "Exactive" |
| CE edge cases | NCE units → numbers in `ce_orig`; floor at 5; NaN → summed 20 / 40 / 60 prediction (Q3-Q5) | NCE ignored; no floor (bucket 0 possible); missing → 30 eV (FR:119-122) | none on visible (all eV, 20-60) |
| batching | heavy-atom-adaptive B·N² ≤ bs·1600; deterministic algorithms | fixed bs 32 sorted by canonical length; no determinism flags (FR:231, 252) | float noise only |
| budget / priority | input order = priority; chunk all-or-nothing; per-batch deadline (P-chunk) | one global job list sorted by SMILES length, scored at the end; a budget stop leaves random molecules PARTIALLY scored (means over a subset of spectra) (FR:231, 255-257, 303-315) | only under budget pressure, but then wrong |
| failure handling | quarter-split retry, OOM batch reduction, fail-fast after 64, exit 0 with a status | a failed batch is wholly lost (FR:288-290); 300 s per-batch SIGALRM; crash supervisor restarts up to 30 times, exit 1 (FR:147-156) | — |
| stubs | try-import first; three ms-pred placeholder modules; platformdirs stub + cache redirect | unconditional stubs for 8 names, no platformdirs (FR:17, 45-57) | none if platformdirs is installed |
| site marker | `.ice_site_ok` = package wheel list, `--upgrade` | `.casmi_site_ok`; filters wheels to the interpreter's cp tag (FR:63-66) | equals the effect of the FZ c41 preseed |
| CLI | `<pkg> in out`, device default cuda (4.1) | `in out --src --ckpt --site [--wheels] --shim --device cpu ...` (FR:162-172) | wrapper must adapt |
| input builder | F `build_ice_input`: float lists, `ce_orig` / `ce_units`, malformed-row guard, covered_only | `build_items` (FR:125-144): no `ce_orig` / `ce_units`, no row guard | — |
| z helper | `_z`: n < 2 → 0 | `fwdfeat._gz` (FF:22-26): n < 2 → NaN (features, not fusion) | — |
| out.json writes | atomic, after every chunk, `allow_nan=False` | non-atomic, at start and end only (FR:205, 316) | — |

### 13.3 Missing in ours
1. **meta JSON.** Required by FZ c43:103-104 and by the commit asserts at FZ c50:29-41: status ok, device cuda,
   rdkit 2025.03.6, no errors, `n_pred_fail` 0, `n_feat_fail` 0, scored > 0.
2. **RDKit version check** and recording; device fallback to CPU; determinism flags.
3. **The gl_runner MODULE API for glb_driver** (section 10, N1-3), or a patched N1 cell.
4. **The whole ICEBERG runner** (P-ICE, C1, E11; SUM merge, length-based batching). We have no ICEBERG code path.
5. **The notebook helpers:**
   - `ice_candidates` (3.2);
   - the never-raising run wrappers with budget + grace timeout;
   - `rerank` and `rerank_multi` (11.2-11.3);
   - the `gl_of` "any non-null" rule.
6. **Priority-ordered chunking with all-or-nothing molecules;** the environment guard (≥ 50 rejected).
7. **Pipeline patch points** needed once Ahmed's datasets are unmounted: FZ c26:105-106 (unguarded `find`), FZ c41:59,
   116 (`find` + import of `fuse` / `gl_fuse`), c41:1017-1018 (driver args use ICE_PKG / GL_PKG), and
   c41:26-55 / c26:90-103 (wheel dir).

### 13.4 Our datasets (the "check" items)
- **bryantyu/casmi26-glacier** (local /home/user/ds/casmi26-glacier) is clean:
  - ms-pred @708148c2, unmodified: `diff -r` against the clean git checkout shows only `__pycache__`;
  - upstream best.ckpt (5a47cecc…);
  - PyPI wheels: rdkit 2025.3.6 cp312 + cp313, LinSATNet 0.1.3, etc.;
  - our own fwdshim (identical to repository fwdshim/; difflib similarity 0.26 to the glue's shim).
- **bryantyu/casmi26-fwd** (published 2026-10-18; listing checked with the Kaggle CLI) is NOT clean. It ships:
  - Ahmed's `glacier/gl_runner.py` (byte-identical, 5c4f7431…), `glacier/gl_fuse.py` (identical) and
    `iceberg/fuse.py` (identical);
  - a lightly modified `iceberg/ice_runner.py` (adds a cp-tag wheel filter);
  - his README / LICENSE-NOTICE text;
  - his re-saved weights `iceberg/ckpt/gen.pt` (deef0451…), `inten.pt` (c49e330e…) and `glacier/ckpt/glacier.pt`
    (9192e4c2…), byte-identical to his MANIFEST.
  
  Its shim/ dirs are ours. docs/ATTRIBUTION.md §1.11 already notes the glue.
- **Consequences:**
  - The eligible fork must not mount casmi26-fwd.
  - Our ICEBERG weights are currently only Ahmed's re-saves. A clean ICEBERG needs the upstream msg_all checkpoints
    from the ms-pred README Dropbox link (MP README:94, 281), checked tensor by tensor against gen.pt / inten.pt, and
    published in a new dataset without the glue files.
- **Old harness numbers.** "isomer top-1 0.838 ours vs 0.743 public runner" (docs/plan.md:2289, 2359) came from
  different harness runs. It cannot be attributed to the differences above until both are run on identical
  candidates (section 14).

## 14. Acceptance tests for the clean-room modules (to live under scratchpad/f5b/)

**T1 Unit rules.** Each Q, S, C and G rule above as table-driven cases, including:
- the glb_driver self-test expectations (GD:144-180): timsTOF → QTOF; 20/40/60 → bucket 40; ce_ev [0, 20] → merged
  buckets [5, 20]; NCE → no merged buckets; NH4 or negative spectra not covered; no CE → 20/40/60 jobs.

**T2 Fusion.**
- Singletons; coverage < 2; SD 0; missing values → z 0; stable ties; slot preservation; `rerank_multi` equal to
  `rerank` for one dict.
- The tie-grid case: a 2-member group where two re-scorers both oppose the ranker at λ = 0.5 each.

**T3 Golden GL** on `/home/user/work/kout/casmi26-pubm/gl_work/gl_in.json` → `gl_out.json`:
- 11 molecules, 793 scores, in [0, 0.705].
- Require the same null pattern, |Δ| ≤ 1e-3 (target 1e-4), and identical within-formula orders after
  `rerank_multi`.
- Report the effect of each 13.2 difference by toggling it.

**T4 Golden N1.** `n1_work/n1_in.json` → `n1_out.json`: per-spectrum mean and merged values, 8 merged spectra.

**T5 Golden ICE** (only if an ICEBERG runner is built). `ice_work/ice_in.json` → `ice_out.json`: 12 molecules,
793 / 814 scored.

**T6 Pipeline.** In a commit run, the asserts at FZ c50:29-41 and c52:154-157 pass, and the final submission is
diffed against the pubm commit's submission.csv.
