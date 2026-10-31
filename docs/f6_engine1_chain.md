# F6 map: Ahmed Berat Özer's training chain behind Engine 1 of the frozen 0.465 pipeline

Status 2026-10-31. This pass was read-only: I downloaded, read and statically parsed files. Nothing third-party was
executed, nothing was pushed to Kaggle and nothing was committed.

**Sources** (each in a fresh directory under `scratchpad/f6/`):
- `tc/` + `tcx/`: `ahmedberatozer/casmi26-training-code`, the full zip (926,128 B, sha256 41bf3f2d...; dataset updated
  2026-10-19), 363 files: README, LICENSE, scripts, logs, run records.
- `v4b/`: from `casmi26-v4b-models`: `MANIFEST.json`, `LICENSE-NOTICE.txt`, `ranker_v4b.pkl` (14.3 MB),
  `fragnet/stageA.npz`, `fragnet/stageB.txt` and `frag2/caps.txt`; also `fragnet/sets.pkl` (`dl_sets/`) and
  `fragnet/countsA.npz` (`dl_counts/`). Every downloaded file that MANIFEST lists matches its sha256. MANIFEST pins 64
  files; the dataset holds 66 (`MANIFEST.json` itself and `LICENSE-NOTICE.txt` are not listed).
  - `caps.txt` has CRLF line ends (his Windows text-mode write), and `stageB.txt` ends with two CRLF lines (lightgbm's
    Python text-mode append of `pandas_categorical`). Neither file can be sha-identical when regenerated on Linux; they are
    compared as text / tree section.
- `pool/fp_bits.npy` from `casmi26-v2-pool` (sha256 c3869e86..., the same value the pubm run mounted).
- `spec/README.md` from `casmi26-spec-cache`.
- `meta/*/dataset-metadata.json`: the cards of all six datasets. `lists/`: their full file listings.
- **Not downloaded** (over 20 MB): fe_A/B/D, fpnet_0/1, fpnet_full1, prior_tables.pkl, featB.npz, the pool arrays and
  the spec cache. The plan reads them inside Kaggle kernels.
- The README of fpnet-full1, v3 and v2-pool exists only as the Kaggle card text (in `meta/`). Only v4b ships a
  MANIFEST.

**Legend.**
- **[verified]**: checked in this pass from the files themselves.
- **[Ahmed]**: stated by his README, card or logs, and not independently checked.

## 0. Bottom line

1. **The chain is complete on paper.** Every trained artefact the fork loads has a producing script in the MIT
   training-code dataset, together with run logs or a JSON record.
2. **The inputs are public and permitted:**
   - train.parquet;
   - its derived caches `casmi26-spec-cache` and `casmi26-v2-pool`;
   - COCONUT 09-2026;
   - the DreaMS SSL checkpoint;
   - PubChemLite 2026-08-28. It is undeclared on the v2-pool card, and it feeds only annotation columns that Engine 1
     never reads.
   - There is **no NIST input and no non-commercial input** beyond train.parquet.
3. **The reconstructed ranker command matches the pickle exactly [verified].** The pickle is plain data (no globals) and
   holds 4 LightGBM boosters. Each has `objective=lambdarank`, lr 0.03, 63 leaves, min_data_in_leaf 40,
   feature_fraction 0.7, bagging 0.8/1, lambda_l2 1, truncation 30, `num_threads 12`, 500 trees, `seed` 0/1/2/3,
   `deterministic 0`, `force_col_wise 0` and `force_row_wise 0`. The 160 feature names equal `fe_v4/ranker_v4b.json`.
   The `only` list equals `fe_v4/v4b_only.txt` (66 family features). README step 4 is therefore right about the
   arguments. What is missing is a log and the training rows.
4. **The fe_v4 tables are the cheapest decisive target.**
   - Every table is CPU-only and seconds to minutes of compute: fragnet ~10 min, priors ~8 min, analog_struct and frag2
     ~15 min on his laptop.
   - Most of them should reproduce **bit-exactly**:
     - fragnet stage B is LightGBM `deterministic 1`, `force_row_wise 1`, 4 threads, seed 0 [verified in
       stageB.txt];
     - stage A is closed-form numpy;
     - the record tables are deterministic RDKit.
   - Every intermediate (rec_*, sets.pkl, countsA.npz, featB.npz) is published with a MANIFEST sha256. Each step can
     therefore be checked on its own.
   - That directly tests his "scripts edited after the runs" and "no log for sets / fitA" gaps.
5. **The ranker cannot be refit from published data.** The sim1x rows and features were never published. Reproducing it
   means regenerating 15,701 simulated queries on a GPU (6-10 T4-h), then the family features on CPU (1-2 h), then the
   fit (~10 min).
   - Bit-exact trees are possible only if the fp16 GPU features also come out bit-identical. That is unlikely on a T4
     against his RTX 5090 Laptop.
   - The realistic outcome is identical draws and identical CPU-derived features, then a ranker within the seed spread
     of its own 4-booster bag.
   - **The pickle itself provides strong comparison targets [verified].** LightGBM's per-booster `feature_infos` (the
     [min:max] over a seeded 200k-row bin sample) differ across the 4 boosters. They are therefore an exact fingerprint
     of the training matrix and its row order.
6. **The weights are GPU work and only statistically reproducible.**
   - The FPNet trainer and the DreaMS trainer seed torch and numpy. However, they use bf16 autocast, CUDA atomics, a
     different GPU architecture and torch 2.14 against Kaggle's 2.10, and the DreaMS resume does not restore RNG
     state.
   - **Cheap, decisive checks exist:**
     - recompute the logged validation numbers of each published file (a frozen evaluator table is in
       `dreams_ft/s2_eval/D20k_report.txt`);
     - reproduce FPNet A's printed step-0 batch line with seed 7;
     - scan seeds to *recover* the unrecorded seeds of B and full1 from their own step-0 lines.
   - Full retrains cost 6-12 T4-h each, ~25-45 T4-h for all four: one to one and a half weekly quotas.
7. **Recommended order (section 6):**
   - **K1**: a CPU tables kernel (~2 h CPU, ~1 day of work).
   - **K2**: a T4 file-against-log and seed-recovery kernel (~1.5-2 T4-h).
   - Together they settle A10 and the seed gaps of A7/A8 within a day.
   - **K4** (sim1x to ranker, 6-10 T4-h) is the one expensive step worth funding if the host wants a demonstration for
     A11.
   - Full weight retrains (K5) only if the host explicitly asks.

## 1. What the fork loads from Ahmed (inference side, from the decoded frozen pipeline FZ c26 / c34 / c37)

| file (dataset) | loaded in | role |
|---|---|---|
| all 64 files of `casmi26-v4b-models` | FZ c26 sha256 check against MANIFEST.json | integrity gate (assert) |
| `code/casmi` (v1 engine), `code/fe_v4` (families, v1engine, hooks_v1, ctxlib, ...) | FZ c37 | Engine 1 code (MIT) |
| `fe_A.pt`, `fe_B.pt`, `fe_D.pt` (`MAN['fe_models']`) | FZ c37 `V1FE(...)` | model_views family (fzA/B/D ranks and gaps; model_views is 47.7% of the ranker's top-30 gain) |
| `ranker_v4b.pkl` | FZ c37 `RK['boosters']`, the mean of 4 `lgb.Booster(model_str=...)` | Engine 1 ranking (160 features) |
| `code/fe_v4/res/derivation/prior_tables.pkl` + `parity_fixture.pkl` | V1FE derivation family; `self_test()` must return 0 mismatches (FZ c37) | derivation features (19) |
| `code/fe_v4/res/frag2/*` | frag2 family | 13 features |
| `code/fe_v4/res/analog_struct/*` | analog_struct family | 13 features |
| `code/fe_v4/res/fragnet/rec_*`, `stageA.npz`, `stageB.txt` | fragnet family | 14 features |
| `code/fe_v4/res/fragnet/sets.pkl`, `countsA.npz`, `featB.npz` | hashed only | training intermediates |
| `casmi26-fpnet-full1/fpnet_full1.pt` | FZ c37 engine FP bank (falls back to `fpnet_0` = A) | base features fz_*, fp_*, el_*, gen_rank_fz |
| `casmi26-v3-models/fpnet_0.pt`, `fpnet_1.pt` (= A, B) | FZ c34 pc_runner `ModelBank(A, B, fold0)` | PubChem channel logits |
| `casmi26-v3-models/ranker_0.pkl` | folder locator only (never unpickled) | none |
| `casmi26-v2-pool`: pool_meta, pool_fp, fp_bits, pool_frag_off/mass, train_fp_sel, train_structs | Engine 1 Pool / Library / engine; PubChem channel filters | candidate pool |

**Train/serve mismatch, not a reproducibility issue.** ranker_v4b was trained on sim1x with **FPNet A** in the engine
bank, but the fork serves Engine 1 with **fpnet_full1** in that bank (v4m / v4n). A ranker reproduction must therefore be
judged in the A-bank setting it was trained in.

**File identities [verified]:**
- The fe_A = fpnet_0 (v4b) and fe_B sha256 values in MANIFEST equal Ahmed's README list.
- The pubm mount's head-tail hashes show v3 `fpnet_0` = v4b `fe_A` = v4b `fpnet_0` (9b19b196...) and v3 `fpnet_1` =
  `fe_B` (77bd2958...).

## 2. The chain (data flow)

```
train.parquet ──03_build_spec_cache──> spec_{mz,it,off}.npy, spec_meta.parquet ─┐  = casmi26-spec-cache
train.parquet ──01_build_train_structs──> train_structs.parquet, train_fp_raw ──04_make_split (rng 2026)──> split.parquet, fp_bits.npy (select_bits seed 0)
COCONUT csv lite 09-2026 ──02_build_coconut──> coconut_structs ─┐
PubChemLite_CCSbase_20260828 ──02b──> pcl_structs ──────────────┼─05_build_pool──> pool_meta.parquet, pool_fp.npy ──05b (depth 2)──> pool_frag_{off,mass}
                                                                 └─ 09_predict side effect ──> train_fp_sel.npy          = casmi26-v2-pool
spec-cache + v2-pool ──06_train_fpnet seed 7──> FPNet A (fe_A = fpnet_0)   [log_train_a]
                     ──06_train_fpnet seed ?──> FPNet B (fe_B = fpnet_1)   [log_train_b]
                     ──06_train_fpnet --holdout "" seed ?──> fpnet_full1   [log_train_full1]
spec-cache + v2-pool + fpnet_v2/{np_split,evalq} + DreaMS ssl_model.ckpt
                     ──review_fixes/train_dreams_fp seed 11──> ckpt/pilot_d1_last.pt ──extract_step20k_ema (interactive)──> fe_D
v2-pool (+split) ──priors_build build 30000 / tables (random_state 2026, 6 procs)──> prior_tables.pkl ──res_derivation──> + parity_fixture.pkl
v2-pool ──res_fragnet rec──> rec_*;  spec-cache+v2-pool+split ──sets (seed 0, nspec 32000)──> sets.pkl ──countA──> countsA.npz
        ──fitA──> stageA.npz ──featB──> featB.npz ──fitB──> stageB.txt
v2-pool ──res_analog_struct rec/str──> analog_struct/*;  ──res_frag2 caps/f1──> frag2/*
spec-cache + v2-pool + train_fp_sel + FPNet A (engine bank) + A/B/D (views)
        ──simulate_v1ctx seed 21, 6 shards──> sim1x rows (94 base feats) + ctx   [v1fe_sim1x/log_*]
        ──build_feats (ids, derivation, frag2, analog_struct, model_views, fragnet; uses ALL res tables)──> feats
        ──evalfe (go decision)──> v1_all5_b3.json   ──train_v4_ranker seeds 0-3──> ranker_v4b.pkl (+ ranker_v4b.json)
```

**Environment then [Ahmed]:**
- Windows laptop: `C:/Users/omen/...`; the build loop polls `powershell`.
- RTX 5090 Laptop GPU.
- python 3.14.5, torch 2.14.0+cu130, lightgbm 4.7.0, rdkit 2026.03.3.

**Kaggle now:** the pinned image is Linux, python 3.12 and torch 2.10 (fork_eligibility A2). The LightGBM version in it
must be checked and pinned.

## 3. Per-artefact map

### 3.1 ranker_v4b.pkl (A11; MANIFEST sha256 6172e926...)

**Producers.**
- `fe_v4/simulate_v1ctx.py` (sha256 11948e11...), launched by `launch_sim1x.sh` (6 shards).
- `fe_v4/build_feats.py`, via `build_sim1x_loop.sh`.
- `fe_v4/evalfe.py`: the go decision only.
- `fe_v4/train_v4_ranker.py` (124c4bc8...), which calls `RankerBag` in `engine_v6_casmi/ranker.py`. That class shuffles
  rows within groups with seed 0, then sets `seed` = `bagging_seed` = `feature_fraction_seed` = s for each booster.
- Helpers: `ctxlib.py`, `v1engine.py`, `hooks_v1.py`, `fams/*.py`.
- **Code identity [verified]:** `v1fe_code_casmi/` (17 files) and the shipped `fe_v4` files are sha-identical to the v4b
  MANIFEST entries.

**Inputs.**
- spec-cache: Library, 2,539,608 spectra.
- v2-pool: 710,701 structures, plus `train_fp_sel`.
- `split.parquet`.
- FPNet A as the engine bank; A, B and D as model views.
- all fe_v4 res tables (section 3.2).
- All are public and permitted.

**Command.**
- The steps 1-3 launch scripts are recorded, but they are Windows-pathed and use powershell polling.
- Step 4 is **reconstructed**. It matches the pickle parameters, `ranker_v4b.json` and the identical v4d line in
  `after_sim1x_d2.sh` [verified, section 0.3].

**Seeds.**
- Simulation: `default_rng([21, sid])` per structure, so the draws do not depend on sharding.
- Fit: shuffle seed 0; boosters seeds 0-3. The derived `data_random_seed` values are 38/41/45/48 [verified].

**Logs.**
- `v1fe_sim1x/log_0..5.txt`. Shards 0, 1, 2 and 5 were interrupted and resumed on Sep 25, which is harmless because
  parts are committed every 40 structures and the draws are per structure.
- Per-shard query totals reconstructed from the logs [verified]: 2,652 / 2,624 / 2,586 / 2,607 / 2,595 / 2,637 =
  **15,701**. That is 35,381 process-seconds, ~2.25 s per query.
- `build_sim1x_loop*.log`.
- `eval_sim1x.log` and `v1fe_sim1x/eval/v1_all5_b3.json`. The 5-fold structure CV with bag 3 and 300 rounds gives, for
  v1_all5_b3 (the 66 v4b family features), ALL base 0.9051 / new 0.9171, d +0.0120 [+0.0096, +0.0143] (= README step 3).
  The same log first holds v1_all4_b3 (52 features, no fragnet): new 0.9157, d +0.0106; do not confuse the two. The
  per-fold and per-regime table is printed to 4 decimals.
- `after_sim1x_d2.log`: the edited simulator reproduced all 15,701 draws (sid / regime / n_cand / has_truth).
- **There is no training log for the final fit.** `ranker_v4b.json` (rows 1,204,705, groups 11,086, top gains) is its
  only record. The v4d refit with the same command took 193 s.

**Gaps [Ahmed].** The launch command is reconstructed; `simulate_v1ctx.py` gained `--engine_models` / `--no_ctx` on
09-26; `ctxlib.py` was edited 5 minutes after the shards started; there is no fit log.

**Determinism.**
- The simulation forces **fp16 autocast** (written "for Kaggle T4 precision"). Its features are bit-stable only on the
  same GPU and kernels: fz*, fp_*, el_*, gen_rank_fz, model_views__* and the candidate truncation by model score
  (`frag_max_cand 600`).
- The families are CPU and deterministic.
- LightGBM has `deterministic 0` and an automatic row-/col-wise choice made by timing. In our own G3 experience refits
  were byte-identical only with a forced row-wise mode. So bit-exact trees need identical rows, the same histogram mode
  (try both with `num_threads=12`) and the same LightGBM build. Windows MSVC against Linux GCC is a residual risk.

**Kaggle compute.**
- Simulation: **6-10 T4-h**, as 3 kernels of 2 shards each, 2 processes per T4 at ~3-4 s per query on 4 vCPUs. The
  simulator asserts a CUDA fp16 bank, so it is GPU-only unless patched.
- Features: **1-2 CPU-h**. Logged per query: derivation 101 ms, fragnet ~220 ms, analog_struct 67 ms, frag2 18 ms.
- CV check: ~0.5 CPU-h. Fit: ~10-15 min.

**What success looks like (levels):**

| level | criterion | outcome |
|---|---|---|
| (a) draws | 15,701 queries with the shard totals above | exact |
| (b) rows | 1,204,705 rows / 11,086 groups with the truth | exact |
| (c) features | per-booster `feature_infos` [min:max] equal for all CPU-derived features; fp16 GPU features within ~1e-3 relative | CPU features exact |
| (d) CV | the `v1_all5_b3` table to within ±0.002 | exact if (c) is exact |
| (e) model | *either* the tree sections of all 4 boosters byte-identical (only possible if (c) is fully exact) *or* statistical equivalence (below) | identical or equivalent |

The statistical test at level (e) runs on the regenerated groups and on the 400-row dry-run feature set. For each
query, the retrained-against-published difference in MRR and top-1 must stay inside the published bag's own seed spread
(booster k against the mean of the other three). This spread is computable from the pickle. The top-gain list should
also match `ranker_v4b.json` (model_views__fzABD_gap first, then fz_softmax, derivation__dg_rank_site, ...).

### 3.2 fe_v4 tables (A10)

#### 3.2a fragnet `stageA.npz` (3a501ecb...) / `stageB.txt` (6916c25c...) and the intermediates

**Producer:** `fe_v4/res_fragnet.py` (02c98941...), steps rec, sets, countA, featB, fitA, fitB. It uses the model and
constants in `fams/fragnet.py`.

**Inputs:** v2-pool (`pool_meta` smiles / formula / mass / src), `split.parquet`, and the spec-cache Library (mmap) for
the training spectra. All are public.

**Command.**
- Recorded in the README with `--procs 4`.
- `--nspec 32000` is inferred (the default is 30,000) from the decision log in `fams/fragnet.py`: A 16,144 / B 12,751 /
  V 3,105.
- The published `sets.pkl` confirms it [verified, numpy-only unpickler]: 32,000 sets, parts 16,144 / 12,751 / 3,105,
  23.3 decoys per set. It was pickled with protocol 5 (python 3.14), so a Linux rerun differs in bytes; compare it by
  content.

**Seeds and settings.**
- sets: `default_rng(seed=0)`.
- fitA: closed form; its self-check uses `default_rng(0)`.
- fitB: `STAGEB_PARAMS` with seed 0, `deterministic True`, `force_row_wise True`, 300 rounds, `num_threads 4`.
  stageB.txt records exactly this [verified].

**Logs:**
- `res_fragnet_rec.log`: 710,701 records, ok 0.9987, 22,234,636 atoms, 23,867,499 bonds, 198 s.
- `countA.log`: A sets 16,144; decoys train-derived 185,308 / COCONUT 115,959; 1,987,521 truth fragment rows; 60 s.
- `featB.log`: 383,923 rows, 15,846 sets, 112 s.
- `fitB.log`: V (3,103 sets) MRR fr_int 0.6465 / frag2_mix 0.7022 / stageA_pk_fcu 0.7549 / **fragnet 0.7712**.
- **sets and fitA have no logs.**

**Gaps [Ahmed]:** `res_fragnet.py` and `fams/fragnet.py` were edited after the fits; `rec_bond.npy` was re-saved as uint8
without a recorded command. The current code casts to uint8, so a rerun tests this directly.

**Determinism:** deterministic, with one exception. countA merges its worker results with `imap_unordered`, so the
float64 decoy weights (D counts) depend on completion order in the last bits, even on his machine. Keys and truth counts
(weight 1.0) are exact. Downstream, featB / stageA can differ by float32 ulps, and stageB by bin / tie effects.

**Kaggle compute:** ~20-40 min on a 4-vCPU CPU kernel. Ahmed quotes ~10 min with 4 processes.

**Success:**
- `rec_*.npy` sha256 equal to MANIFEST.
- sets.pkl content equal.
- countsA: keys and truth counts equal, decoy counts within ~1e-12 relative (merge order, above). featB: row keys equal,
  float aggregates equal or within float tolerance. Compare arrays, not files: his numpy writes fixed 1980 zip
  timestamps, but the deflate stream need not match (Python 3.14 on Windows ships zlib-ng).
- stageA arrays `np.array_equal` (from the published countsA), or within 2 float32 ulps (from regenerated counts).
- stageB.txt tree section byte-identical, plus the four V MRR lines to 4 decimals. The whole file cannot match on Linux
  (CRLF trailer, see Sources).
- **Two-stage check:** first refit fitA and fitB from the *published* countsA.npz / featB.npz (seconds). This isolates
  the post-fit edits of the fit code from any data-step drift.

#### 3.2b derivation `prior_tables.pkl` (217905818...) and `parity_fixture.pkl`

**Producers:** `class3_v2/priors_build.py` (1c9a4188...; `build 30000`, then `tables`), which uses `class3_v2/priors.py`
and `casmi/derive.py` (71 transforms). `fe_v4/res_derivation.py` then copies the table (with an md5 check) and writes
parity_fixture.pkl.

**Inputs.**
- Parents: COCONUT entries taken from v2-pool `pool_meta` (src 2/3, 5-80 heavy atoms; holdout keys dropped).
- `split.parquet`.
- No separate COCONUT download is needed.

**Command:** recorded in the README. The script docstring says 24,000 parents; the log shows 30,000. The ROOT
(`C:/Users/omen/...`) is hard-coded in `priors.py` and `priors_build.py`, which needs a 1-line patch each.

**Seeds:** `coco.sample(30000, random_state=2026)`; a 6-process pool.

**Logs:**
- `priors_build.log`: 3,378,874 rows, 36,145 found, 372 holdout hits, 457 s.
- `priors_tables.log`: the full per-transform table, 3,378,502 rows, 29,994 parents.
- `res_derivation.log`: file md5 c7ab5cbe...; parity on 1,500 products, 71 not reproduced by either implementation,
  26,369 siblings, **0 mismatches**.

**Gaps [Ahmed]:** `priors.py` was edited after the 14:10 build. The pickle embeds `built=` (a timestamp), so its file
hash cannot reproduce.

**Determinism:** deterministic apart from the timestamp. The RDKit build must be 2026.03.3; the fork already asserts this
through `self_test`.

**Kaggle compute:** ~15-30 min on 4 vCPUs.

**Success:**
- Every table equal except `built`.
- `priors_tables.log` reproduced line for line.
- The parity fixture rebuilt with 0 mismatches.
- The fork's `self_test()` passes against the regenerated table.

#### 3.2c analog_struct `{pool,train}_{hash,num,fvec,elsmi_*,coresmi_*}.npy` and 3.2d frag2 `pool_cap.npy`, `caps.txt`, `pool_f1_{off,mass}.npy`

**Producers:** `res_analog_struct.py --step rec` and `--step str`; `res_frag2.py --step caps` and `--step f1 --procs 4`.

**Inputs:** v2-pool, `train_structs`.

**Logs:**
- `res_analog_struct.log`: train (275,810, 8) empty-core 0.050; pool (710,701, 8) 0.089.
- `res_frag2.log`: caps shares.
- `res_frag2_f1.log`: 15,632,964 masses, 266 s.

**Gap [Ahmed]:** `res_analog_struct.py` was edited after its rec outputs.

**Determinism:** deterministic RDKit per structure.

**Kaggle compute:** ~20-40 min.

**Success:** sha256 equal to MANIFEST for every `.npy` (`np.save` is byte-deterministic), plus the log lines.
`caps.txt` is CRLF in the published copy, so it is compared as text with line ends normalised.

### 3.3 FPNet A = fe_A.pt = v4b fpnet_0.pt = v3 fpnet_0.pt (A8; edccf479...)

**Producer:** `scripts/06_train_fpnet.py` (20ae6ca4...), with `casmi/traindata.py` and `casmi/fpnet.py`. These are
sha-identical to the v4b copies [verified].

**Inputs:** spec-cache (`spec_*`, `spec_meta`, `split`) and v2-pool (`pool_meta`, `pool_fp`, `fp_bits`,
`train_structs`). Ahmed says they are byte-identical to the training-time files.

**Command:** `--steps 40000 --out work/fpnet_a.pt`, taken from the script docstring; all other arguments default.

**Recipe:**
- 40,000 x 256 steps, AdamW 3e-4, wd 0.05, warm-up 2,000, cosine to 2%.
- bf16 autocast, clip 1.0, EMA 0.999.
- d512 x 8 layers, 49.7M parameters, K 63, merge_p 0.5.
- Holdout np, nplib, syn, plusk, twin.

**Seed:** 7 (the default). It seeds `torch.manual_seed`, the Data sampler `default_rng` and the dropout RNG. The sampler
is single-threaded numpy, so the data stream is deterministic.

**Log:** `logs/log_train_a.txt`.
- 2,409,534 training spectra / 364,451 groups.
- Step-0 line: `bce 0.6975 ctr 4.1568 top1 0.051 el 2.264`.
- VAL every 2,000 steps. Best at **38,000**: np 0.5780 / np-merged 0.6144 / nplib 0.6156 / syn 0.7114 (score 0.5962).
- 13,179 s for the run (~775-930 spectra/s).

**Gaps:** none beyond what is listed under Determinism. The command is in the docstring and the seed is the default.

**Determinism:**
- bf16, CUDA atomics in backward, no deterministic flags.
- RTX 5090 (sm_120) against T4 (sm_75); torch 2.14 against 2.10.
- So not bit-exact.
- The step-0 forward is reproducible to about 1e-3: torch CPU init is seeded, and the batch is pure numpy.
- **T4 caveat:** the trainer hard-codes `torch.autocast(..., bfloat16)`. A T4 has no bf16 tensor cores: it emulates them
  (slow) or refuses. The visible fallback patch is fp16 with GradScaler; it changes numerics, which is acceptable for a
  statistical target.

**Kaggle compute:**
- Step-0 check: ~10 min (data load plus one batch).
- Full retrain: **6-10 T4-h**. The data loader is CPU-bound and Kaggle's vCPUs are slower than his laptop; measure in
  K3 first. The run saves `_last.pt` every 2,000 steps, but `--resume` does not restore RNG state.

**Success:**
- **(i) cheap.** The published fe_A re-evaluated with the script's own `window_mrr` gives the logged step-38,000 VAL to
  within ±0.002, and the frozen Evaluator column A in `D20k_report.txt` (np-dev merged 0.6240, nplib merged 0.7246). The
  step-0 line is reproduced (top1 = 13/256 exactly; the losses to ±2e-3).
- **(ii) retrain.** Every logged VAL point within ±0.01. Best score within ±0.01 of 0.5962 (the A-B gap is 0.0035;
  np-merged differs by up to 0.014 at the end). Frozen-Evaluator metrics within the A-B gap (0.012 on np-dev merged).
  Per-query logit agreement with fe_A at least as high as fe_A against fe_B (the F7-style seed reference).

### 3.4 FPNet B = fe_B.pt = v3 fpnet_1.pt (A8; 6f60a22b...)

**Same as 3.3, except:**
- **The seed was not recorded.** B's step-0 line is `bce 0.6973 ctr 4.5678 top1 0.035 (9/256) el 2.226`, so the seed is
  not 7.
- Best is at step 40,000: 0.5749 / 0.6245 / 0.6179 / 0.7151 (score 0.5997). The run took 12,392 s.

**Cheap closure: seed recovery.**
- Build the Data object once.
- For each seed s: `torch.manual_seed(s)`, `D.rng = default_rng(s)`, init the model, build one batch and run the forward.
  Data.__init__ draws no random numbers, so this is valid.
- Match B's step-0 line. About 3 s per seed on a T4, so seeds 0-999 plus common values (42, 123, 2026, ...) take ~1 h.
- A match turns "seed not recorded" into "seed recovered and verified".

**Retrain success:** as in 3.3, against B's log.

### 3.5 fpnet_full1.pt (A7; d13a3611...)

**Same trainer and recipe as A.**

**Command:** reconstructed as `--holdout "" --val_every 40000`.
- The log shows 2,539,608 training spectra / 371,921 groups, which any holdout naming no fold produces.
- It has one VAL line, at step 40,000: 0.9181 / 0.9256 / 0.9138 / 0.8092. These are training-fold structures, so they
  are not a generalisation measure.
- The run took 11,678 s.

**Seed not recorded.** The step-0 line is `bce 0.6984 ctr 4.1634 top1 0.012 (3/256) el 2.160`. The same seed scan
recovers it, run under `--holdout ""`.

**Success:**
- The Data line count reproduces.
- The seed is recovered from the step-0 line.
- The published file's VAL is reproduced.
- A retrain matches its VAL within ±0.01, and its logit agreement with the published file is at least the A-against-B
  level.

### 3.6 fe_D.pt (A9; DreamsFP, a DreaMS fine-tune; 8ae5bee2...)

**Producers:**
- `dreams_ft/review_fixes/train_dreams_fp.py` (7b4c8573...). It differs from the top-level copy only in the kill-rule
  check [verified by diff], and that rule did not fire.
- `dreams_model.py` (the training copy; 36a7f345...). The v4b inference copy differs only in its docstring and a no-op
  patch guard [verified by diff].
- `fpnet_v2/v2lib.py` (`NegSampler`, `Evaluator`), `np_split.parquet` and `evalq.parquet` (all shipped).
- `extract_step20k_ema.py`, which records the interactive re-wrap after the fact.

**Inputs:**
- spec-cache and v2-pool.
- DreaMS `ssl_model.ckpt`: Zenodo 10997887, CC BY 4.0. Today's listing shows 1,392,710,212 B and md5 4deee91d...
  [verified], which matches the size in the README. It is also on Hugging Face (MIT). Host-allowed (749217).

**Command:** reconstructed from `runs.jsonl` line 10 and the args saved in the summary JSON and inside fe_D.pt: 20,000
steps, bs 128, n_peaks 60, grad_ckpt, attn_bf16, save_every 4000, final_eval, mem_cap 9.

**Seeds:** 11. The negatives seed is 10,011 and the validation rng is 123.

**Logs:**
- `logs/pilot_d1.txt` and its summary JSON: 6,311 s, 405.6 spectra/s, 4.55 GB peak.
- VAL step 18,000 (the trainer's "best"): 0.6158 / 0.6679 / 0.6210.
- **VAL step 20,000 (= fe_D): np-dev 0.6096 / np-dev-merged 0.6633 / nplib 0.6213.**
- Final Evaluator line.
- `s2_eval/D20k_report.txt`: a frozen Evaluator table for A, B, D20k and their ensembles. D20k has np-dev merged 0.6729
  and nplib merged 0.7197.

**Gaps [Ahmed]:**
- The command was not logged.
- The EMA re-wrap was interactive. `extract_step20k_ema.py` has never been run, and its `--check` needs
  `ckpt/pilot_d1_last.pt`, which is unpublished, so it cannot be run by anyone else.
- The DreaMS download command was not recorded.
- Host context: "third-party DreaMS fine-tunes without a published recipe are NOT allowed" (our research_oct25 notes on
  749217 / 748882). Here the recipe is published but reconstructed.

**Determinism:** bf16 attention, grad checkpointing, fused AdamW; resume does not restore numpy or torch RNG. Not
bit-exact.

**Kaggle compute:**
- File-against-log evaluation: ~15 min on a T4.
- 2,000-step short run: ~0.7-1.3 T4-h.
- Full 20,000 steps: **6-12 T4-h**, two sessions with `--max_minutes` / `--resume`.

**Success:**
- **(i) cheap.** The published fe_D gives the **step-20,000** VAL numbers, not the 18,000 ones. That proves the published
  file is the step-20k EMA of the logged run. It also reproduces the D20k column of the frozen Evaluator table. Read
  `step`, `args` and `val=None` from its pickle header with pickletools, without unpickling.
- **(ii) retrain.** The step-200 to step-2,000 lines track the log (step 200: bce 0.5486, ctr 3.9283, top1 0.081). VAL
  at 20k is within ±0.015; there is no same-recipe second seed, so the FPNet A-B gap is the reference. The Evaluator D
  column is within the same band. Our run is passed through `extract_step20k_ema.py` (without `--check`).

### 3.7 Data inputs: v2-pool and spec-cache (A13; built, not trained)

**v2-pool producers:** scripts `01_build_train_structs` (20 workers), `02_build_coconut` (22 workers), `02b`,
`04_make_split`, `05_build_pool`, `05b_build_frags` (depth 2, 20 workers), and `09_predict` (writes `train_fp_sel.npy` as
a side effect).

**v2-pool inputs.**
- train.parquet. It reads `normalized_smiles`, which is the current column name; the saner F7 patch was the reverse
  case.
- COCONUT 09-2026 CSV lite: `coconut.s3.uni-jena.de/prod/downloads/2026-09/coconut_csv_lite-09-2026.zip`. The URL
  returned HTTP 200 today, 199,089,109 B [verified].
- `PubChemLite_CCSbase_20260828.csv`. It is **not declared** on the card and I did not find it in a quick Zenodo search
  (the 2026-09-25 CCSbase release is record 22953182). It only fills the `in_pcl`, `pubmed`, `patent` and `anno` columns
  of pool_meta; no Engine 1 or family code reads them [verified by grep].

**v2-pool logs:** none for scripts 01-05.

**v2-pool determinism:**
- Mostly deterministic: mergesort by mass; `select_bits` with seed 0; `04_make_split` with rng 2026.
- Residual risk: RDKit build / CPU-dependent float masses changing tie order (our G12).

**v2-pool compute:**
- 1-2 CPU-h without the fragments.
- The depth-2 fragments of 710k structures are unlogged; I estimate 3-8 CPU-h.

**v2-pool success:** sha256 equal for `fp_bits.npy` (c3869e86...), `pool_fp.npy`, `pool_frag_*` and `train_fp_sel.npy`;
parquet contents equal except the PubChemLite columns, if the 20260828 file cannot be found.

**spec-cache** (`03_build_spec_cache`, `04_make_split`; published sha256 values in `spec/README.md`):
- ~0.5-1 CPU-h.
- Success: `spec_*.npy` sha256 equal and `split.parquet` content equal.
- This is the step that ties every model input to train.parquet.

## 4. Inputs: public? permitted? NIST / NC?

| input | used by | licence / ruling | NIST | NC beyond train | public at |
|---|---|---|---|---|---|
| competition train.parquet | everything | CC BY-NC 4.0; host 745072: treat as open, NC-via-train tolerated | no (libraries: enveda-180, pluskal, riken, gnps, massbank, mona, spectraverse, msdial, drug_plus, enveda-np-examples, masaryk) | tolerated | competition |
| casmi26-spec-cache, casmi26-v2-pool | all trainers, sim, families | "other"; derived from train + COCONUT (+ PCL) | no | no | Kaggle |
| COCONUT csv lite 09-2026 | pool, decoys, prior parents, analog / frag records | CC BY 4.0 per Ahmed (COCONUT is host-allowed) | no | no | COCONUT S3 (verified reachable) |
| PubChemLite_CCSbase_20260828 | pool_meta annotation columns only (unused by Engine 1) | CC BY 4.0 (PubChem host-allowed); undeclared | no | no | release not located |
| DreaMS ssl_model.ckpt | fe_D init | CC BY 4.0 / MIT; host 749217 "allowed" | no (GeMS / MassIVE) | no | Zenodo 10997887 (verified) |
| fpnet_v2/np_split, evalq | DreaMS val / eval | derived from train | no | no | in training-code |
| RDKit 2026.03.3, LightGBM 4.7.0, torch 2.14 | runtime | BSD / MIT / BSD | - | - | PyPI |

**Verdict:** no NIST and no extra NC anywhere in the chain. The only paperwork gap is the undeclared PubChemLite input of
v2-pool, which is functionally inert for Engine 1.

## 5. Determinism summary

| step | seeded by | non-determinism | expected on Kaggle |
|---|---|---|---|
| spec-cache, v2-pool, rec / analog / frag2 tables | pure functions, rng 2026, select_bits seed 0 | RDKit / CPU float (G12) | sha256-identical (likely) |
| priors_build | `random_state=2026` | `built` timestamp in the pickle | content-identical |
| fragnet sets / countA / featB / fitA | `default_rng(0)` / closed form | countA worker merge order (`imap_unordered`): last-bit float64 decoy weights | sets identical; countA keys / truth counts identical, decoy counts ~1e-14 relative; featB / stageA equal or within float32 ulps |
| fragnet fitB | seed 0, deterministic 1, force_row_wise, 4 threads | LightGBM build (MSVC against GCC) | identical trees (likely) |
| simulate_v1ctx | `default_rng([21, sid])` | fp16 GPU inference | identical draws; GPU features ~1e-3 |
| build_feats | none needed | none | identical given identical ctx |
| train_v4_ranker | shuffle 0; seeds 0-3 | deterministic 0, automatic histogram mode, rows from fp16 GPU | trees identical only if rows identical; otherwise equivalent |
| 06_train_fpnet | seed 7 / ? / ? (torch + numpy) | bf16, CUDA atomics, GPU arch, torch version | step-0 reproducible; run statistically equivalent |
| train_dreams_fp | seed 11 (+10,011, 123) | bf16 attention, fused AdamW, resume RNG not restored | statistically equivalent |
| extract_step20k_ema | none | none (the interactive original is unverifiable) | n/a |

## 6. Reproduction steps ranked by decisiveness / cost

**Decisiveness D (0-5)** measures how much a positive result moves the host's question: whether reconstructed commands,
unrecorded seeds and post-run edits still allow reproduction from public code and permitted data. It is weighted toward
the U-rated assets A10 / A11 and toward bit-exact outcomes.

**Cost C** is in effort-hours: engineering h + T4-h + 0.25 x CPU-h. T4 hours count fully because the weekly quota of
about 30 h is shared with our own work.

| rank | step | closes | D | C (eng + T4 + CPU) | D/C | success criterion | kernel |
|---:|---|---|---:|---|---:|---|---|
| 0 | Static record checks (**done in this pass**) | A11 "reconstructed command"; code identity | 2 | 0 | ∞ | pkl params, features, `only` list = json / txt; v4b code = training-code copies; stageB params as claimed; shard totals sum to 15,701 | - |
| 1 | analog_struct + frag2 rebuild | A10 "res_analog_struct edited after rec" | 3 | 0.5 + 0 + 0.7 → **0.7** | 4.3 | sha256 = MANIFEST for all 17 `.npy` | K1 |
| 2 | fragnet full chain rec → sets → countA → featB → fitA → fitB | A10 "sets no log", nspec 32000 inferred, rec_bond re-save | 4 | 1 + 0 + 0.7 → **1.2** | 3.3 | rec sha256; sets / counts / featB arrays equal; same stageA / B | K1 |
| 3 | fragnet fitA / fitB from the *published* countsA / featB (runs first inside step 2; it isolates the fit code from data-step drift) | A10 "fragnet edited after fits", "fitA no log" | 3 | 1 + 0 + 0.1 → **1.0** | 3.0 | stageA arrays equal; stageB tree text byte-identical; V MRR 0.7712 | K1 |
| 4 | priors build 30000 + tables + res_derivation parity | A10 "priors.py edited after build", 24000 vs 30000 | 4 | 1.5 + 0 + 0.5 → **1.6** | 2.5 | tables equal except `built`; `priors_tables.log` reproduced; parity 0 mismatches | K1 |
| 5 | sim1x draws only (CPU `Ctx.draw` over 5,350 sids) | A11 "simulate_v1ctx / ctxlib edited" (draw part) | 1.5 | 1 + 0 + 0.2 → **1.05** | 1.4 | draws per shard ≥ 2,652 / 2,624 / 2,586 / 2,607 / 2,595 / 2,637; exact match after empty-candidate skips | K1 |
| 6 | File-against-log evaluation: frozen v2lib Evaluator (A, B, D20k columns) + `window_mrr` VAL for A, B, full1 + fe_D header via pickletools | A8 / A9 / A7 provenance; fe_D "interactive re-wrap" | 3 | 2 + 0.5 + 0 → **2.5** | 1.2 | D20k_report columns ±0.002; fe_D = the step-20k VAL (not 18k); A = 38k, B = 40k, full1 = 40k VAL | K2 |
| 7 | spec-cache + train_structs / split / fp_bits from train.parquet | "inputs byte-identical" claim (all trainers) | 2 | 1.5 + 0 + 1 → **1.75** | 1.1 | spec_* sha256 = spec README; fp_bits c3869e86 | K0 |
| 8 | Step-0 checks: A with seed 7 (exact) + seed scan for B and full1 | A8 "B seed unrecorded"; A7 seed and `--holdout` | 3 | 2 + 1.3 + 0 → **3.3** | 0.9 | A line reproduced; B and full1 seeds found (unique match on top1 count + losses ±2e-3) | K2 |
| 9 | Short-horizon retrains: FPNet A 2,000 steps; DreaMS pilot_d1 2,000 steps (also T4 timing calibration) | recipe and arguments correct (A7-A9) | 2.5 | 2 + 1.8 + 0 → **3.8** | 0.66 | step 200-2,000 lines track the logs; VAL step 2,000 (A 0.3728 / 0.4192 / 0.4903 / 0.4552; D 0.4460 / 0.4794 / 0.5201) ±0.01 | K3 |
| 10 | v2-pool rebuild (COCONUT 09-2026, train; PCL if found) | A13 (already E) | 2 | 3 + 0 + 6 → **4.5** | 0.44 | pool_fp / fp_bits / pool_frag / train_fp_sel sha256 equal | K6 |
| 11 | FPNet A full retrain, seed 7 | A8 (statistical) | 3 | 1 + 8 + 0 → **9** | 0.33 | VAL within ±0.01 at every point; Evaluator within the A-B gap; agreement with fe_A ≥ A-against-B | K5 |
| 12 | fe_D full retrain + EMA extract | A9 (statistical; the DreaMS recipe ruling) | 3.5 | 2 + 9 + 0 → **11** | 0.32 | VAL 20k ±0.015; D20k Evaluator column ±0.015 | K5 |
| 13 | **sim1x → features → CV → ranker refit** | **A11 in full** (and A10 in use) | **5** | 7 + 8 + 3 → **16** | 0.31 | levels (a)-(e) of 3.1: draws, rows, feature_infos, CV table ±0.002, trees identical or within the bag's seed spread | K4 |
| 14 | FPNet B and full1 full retrains (after seed recovery) | A8 / A7 (statistical) | 2 each | 9 each | 0.22 | as 11, against their own logs | K5 |

**How to read the ranking.**
- Steps 1-5 together form **one CPU kernel**, K1: ~2 CPU-h, no GPU, ~5 engineering hours. If it comes out exact, A10
  ("doubtful") becomes "reproduced bit-exactly from the published, post-edit scripts". That is the strongest evidence
  available for the host question, at almost no cost.
- Steps 6 and 8 are **one T4 kernel**, K2 (~1.5-2 T4-h). It ties every weight file to its log and either recovers or
  rules out the B and full1 seeds.
- Step 13 has the lowest ratio but the **highest decisiveness**. It is the only step that tests A11 itself. If one
  expensive step is funded, fund K4, not the FP retrains: the ranker is a U asset; the weights are E with caveats.
- Steps 11, 12 and 14 (K5, ~25-45 T4-h) only if the host explicitly wants retrained weights. That exceeds one weekly
  quota, and K3 should measure the T4 speed first because bf16 emulation could double the estimate.

## 7. Repro kernels on the F7 template (`kernels/casmi26-saner-repro`)

Each kernel follows the F7 template:
- **Published-sha check:** v4b MANIFEST (64 files), the README sha list, the training-code script sha256 values in this
  map, and pins of the dataset versions.
- **Minimal visible patch,** printed as a diff:
  - the hard-coded ROOT in `priors.py`, `priors_build.py` and `train_dreams_fp.py`;
  - the repo layout restored by symlinks (`work/` = spec-cache + v2-pool; `work/fpnet_a.pt` → fe_A, `work/fpnet_b.pt` →
    fe_B, `work/analysis/dreams_ft/s2/pilot_d1_step20k_ema.pt` → fe_D; `work/analysis/{fe_v4,v1fe/code,engine_v6/code,
    class3_v2,fpnet_v2,dreams_ft}`; and `src/casmi` ← the package's `casmi/`, which `priors.py`, `priors_build.py`,
    `06_train_fpnet.py` and `train_dreams_fp.py` import from `src/`. It is NOT the same code as `engine_v6_casmi/`, which
    the fe_v4 `res_*.py` scripts import from `work/analysis/engine_v6/code`);
  - bf16 → fp16 only if the T4 refuses bf16, and stated as such.
- Run, compare, and print `F6 CHECK <name>: OK | FAIL | INFO {...}` lines.
- Write `/kaggle/working/repro_report.json`.
- RDKit 2026.03.3 from `metric/rdkit-2026-3-3-wheel` (asserted). Record `lightgbm.__version__` (his 4.7.0). Internet
  is off, so a 4.7.0 wheel can be installed only if one is mounted. Otherwise the LightGBM comparisons fall back from
  bytes to structure / agreement, and the log says so.

| kernel | machine / internet | mounts | steps |
|---|---|---|---|
| K1 `casmi26-ahmed-tables` | CPU, off | training-code, v4b-models, v2-pool, spec-cache, rdkit wheel | 1-5. The regenerated res go to a separate dir and are compared against the mounted v4b files. Ranks 1-3 are written as `kernels/casmi26-f6-repro1` (CPU image 81b1e7c4, no source edits). Ranks 4-5 (priors with the ROOT patch, sim1x draws) follow in `casmi26-f6-repro2`. |
| K2 `casmi26-ahmed-fpcheck` | T4, off | + fpnet-full1, v3-models | 6, 8 |
| K3 `casmi26-ahmed-short` | T4, on (Zenodo ckpt) | as K2 | 9 (+ timing calibration for K5) |
| K4 `casmi26-ahmed-sim1x-{0,1,2}` + `casmi26-ahmed-v4b-refit` | 3 x T4 (2 shards each, 2 processes), then CPU with the sim kernels as `kernel_sources` | as K1 + fe models | 13: `simulate_v1ctx --shard s --nshard 6` with the default seed and engine; `build_feats --fams ids,derivation,frag2,analog_struct,model_views,fragnet`; `evalfe --bag 3 ...` (v1_all5_b3); `train_v4_ranker` with the README step-4 arguments, run as recorded plus `force_row_wise` and `force_col_wise` variants for the tree comparison |
| K5 `casmi26-ahmed-{fpneta,fpnetb,full1,dfp}` | T4 (DreaMS in 2 sessions) | as K2 / K3 | 11, 12, 14 |
| K0 / K6 `casmi26-ahmed-inputs` | CPU, on (COCONUT; PCL if found) | competition, training-code | 7, 10 |

**Disk and RAM.**
- Inputs are mounted read-only; outputs go to `/kaggle/working`, limited to 20 GB.
- The sim1x ctx pickles are unpublished; I expect a few GB.
- Each sim process holds Library + Pool + train_fp_sel, ~3-4 GB, so 2 processes fit a T4 kernel.

## 8. Open points and risks

- **Cross-platform bit-exactness** (Windows MSVC LightGBM / RDKit against Linux) is the main residual risk for steps
  1-4. If the stageB tree section is not byte-identical, there are two fallbacks, as implemented in casmi26-f6-repro1:
  - NUMERIC-EQUAL: identical structure, equal thresholds and leaf values within 1e-9;
  - otherwise EQUIVALENT: V MRR within 0.002 of 0.7712, and per-group Spearman against the published model at least that
    of a seed+1000 refit on the same rows.
- **T4 bf16** support and speed for the two trainers is unmeasured. All T4 hour estimates carry ±2x until K3 runs.
- **The original resume of sim1x shards** had no effect on the draws, because they are seeded per sid.
- **`extract_step20k_ema.py --check` cannot be run by anyone:** `pilot_d1_last.pt` is unpublished. Step 6 replaces it
  with a functional check: the step-20k VAL against the step-18k VAL.
- **The PubChemLite 20260828 file** may be unobtainable. That only blocks byte-rebuilding four unused pool_meta columns.
- **Context, not a repro gap:** fpnet_full1 trained on all folds, and the ranker was trained against the FPNet A bank
  while the fork serves full1.
- **Kaggle versions:** the image's LightGBM version is unverified, and the Kaggle `casmi26-training-code` dataset
  version should be pinned. My download is sha256 41bf3f2d... (2026-10-19).
  - casmi26-f6-repro1 checks the dataset by 40 per-file pins and a whole-tree hash (85e254f6..., 363 files).
  - It runs on the CPU image 81b1e7c4, which our CPU kernels already use with the RDKit wheel. The pubm T4 image
    37c64f7d is unproven in a CPU session.
  - His published pickles come from numpy 2 (`numpy._core`). A numpy-1 image would also change NEP 50 scalar
    promotion inside his code. The repro1 environment line records the numpy version: read it before judging any
    NUMERIC-EQUAL result.
