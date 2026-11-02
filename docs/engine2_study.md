# Engine 2 study: what it is, why it beats our stage 1 inside the fork, and the cheapest eligible rebuild

Status 2026-11-02. Research only. All third-party code (pubm notebook, frozen 0.465 notebook, ENG_FILES, megayak / prvsiyan
notebooks, cubres GitHub sources) was read as text. Nothing from it was executed, nothing was pushed, submitted or
committed. Three small analysis scripts of our own (scratchpad `eng2_tools/overlap12.py`, `comp12.py`, `npx_channels.py`)
ran on our own files only: kernel outputs, the pool and the val rows.

## 0. Key findings

1. **Engine 2 is not stronger than our stage 1 on its own; it is more useful inside the fork.** Engine 2-family kernels
   scored 0.333 (prvsiyan ranker alone) and 0.337 (0.65 / 0.35 blend) on the public LB, with a seed spread of 0.017
   (megayak "One Engine, Nine Scores", §1). Our own line scores 0.339 (v18c) and 0.348 (v18cte), so the two engines are
   on par alone. Even so, Engine 2 is worth +0.085 in the fork (pubm 0.479 vs the null engine 0.394), and ours only
   +0.041 (pube1 0.435). The gap is about what the fork consumes, not about standalone accuracy.
2. **The fork consumes the top of Engine 2's list.** The frozen pipeline uses Engine 2's ranks in four ways:
   - an RRF that adds 0.6/(3+r) to Engine 1's 1/(3+r) over 40 candidates, so Engine 2 picks the leader among Engine 1's
     top ~6;
   - an unconditional rank-2 insert of Engine 2's #1;
   - fragment-check support votes for candidates at Engine 2 rank ≤ 5;
   - an ICE / GLACIER candidate union.

   Deep recall matters much less. eng3a supports this: RRF-merging our list into Engine 2's displaced about half of
   Engine 2's top picks and cost 0.034.
3. **Measured on the hidden-like enveda-np-examples (npx) queries, Engine 2's library and analog channels beat ours, and
   our PeakFormer beats its FPNet.** All numbers are channel-only MRR@25 in a comparable train ∪ COCONUT ±10 ppm window
   (§3.2).

   | channel (query class) | ours | Engine 2 (megayak) |
   |---|---|---|
   | library (class 1) | 0.861 | 0.930-0.938 |
   | analog (class 2) | 0.464 | 0.524-0.532 |
   | spectrum → fingerprint, f·z (class 2) | pf_bern 0.529 | FPNet 0.493; leak-free retrain 0.452 |
   | MetFrag-lite (class 2) | 0.261 | 0.243 |

   So the pv_fp FPNet is not what we lack. That removes the GPU part of the audit's F2c estimate (40-60 h).
4. **Engine 2 is provenance-free; our ranker is not.** Engine 2 uses no documentation or source features. Our ranker
   carries 20 of them (in_train, in_coco, in_bio, PubChemLite / PubChem counts, COCONUT priors). That duplicates what
   Engine 1's popularity re-rank and the midx DocPrior already apply. On the 12 visible molecules:
   - top 10 of the lists: COCONUT 41% for ours vs 8% for Engine 2; training structures 39% vs 81%;
   - top 40 of the two lists: they share only 15.5 of 40 entries.
5. **The training code is public and Apache-2.0.** The FPNet trainers (single-spectrum and merged-spectrum), the
   preparation scripts and the 31-column row builder are in cubres/enveda-casmi-2026 `reproduction/legacy-v2`
   (NOTICE: 29 files). That source also defines the 6,930-bit list: bits with frequency between 0.005 and 0.995 over the
   training structures, out of a 10,407-column Morgan / RDKit-path / MACCS descriptor. Only the shipped weights, rows and
   bank are ineligible, and we can rebuild each of them.
6. **Recommendation (option R, §4-5): an Engine 2 clone on eligible inputs.**
   - It keeps:
     - Engine 2's code: the library, analog and MetFrag channels, the 31 features plus megayak's blocks, the in-kernel
       HistGB fits, the 0.88 blend and the top-40 list;
     - its pool contents: COCONUT + ChEBI / LMSD + PubChemLite + training structures, which pool v18c already covers.
   - It substitutes, all eligible:
     - the bank: pool v18c with our own frequency-selected bits;
     - the FP channel: our PeakFormer f·z, with no FPNet training;
     - the rows: replays of our own val / rank queries, which are held out of PeakFormer training.
   - Cost: 25-35 h, about 10 CPU-hours, no GPU.
   - Expected recovery: 50-80% of the 0.044 public gap. Roughly 0.01 of that gap is selection optimism that no
     replacement recovers on the public board.
   - Step 0 runs in parallel: 2-3 h and one LB slot for a source-blind version of our current Engine 2, the cheapest
     test of finding 4.

## Citation key

| tag | source |
|---|---|
| FZ cNN Lx | Frozen 0.465 notebook decoded from pubm (sha256 d1e757c2..., same numbering as docs/fork_eligibility.md): cell NN (0-based), line x of the cell source. Text copy: scratchpad `eng2_study/fz/`. |
| E2:file Lx | ENG_FILES written by FZ c32 L3 (`pv.py`, `pv_fp.py`, `casmi_engine.py`). `eng_runner.py` is the string written at FZ c32 L6. Text copies: scratchpad `eng2_study/eng/`. |
| NB cNN | `kernels/casmi26-pubm/notebook.ipynb` cell NN. NB c10 `_post` Lx and NB c08 `MODULES['x.py']` Lx are the embedded sources (scratchpad `eng2_study/pubm_src/`). |
| cubres:path | github.com/cubres/enveda-casmi-2026, `reproduction/legacy-v2/...` (main branch, raw text fetched 2026-11-02; copies in scratchpad `eng2_study/gh/`). |
| MY1 / MY9 | megayak `casmi26-two-rankers-one-engine` (scratchpad `elig/a1_megayak`) and `casmi26-one-engine-nine-scores` (scratchpad `eng2_study/dl_nine`), markdown cells. |
| LOG | `/home/user/work/kout/casmi26-pubm/casmi26-pubm.log`, the Engine 2 subprocess output of the pubm commit run. |

## 1. Engine 2 end to end

### 1.1 Wiring
- **Launch (FZ c32).** Engine 2 runs first, in its own subprocess, while GPU memory is still empty.
  - L2-6: write the engine files.
  - L8-14: `subprocess.run(eng_runner.py test train sample eng_lists.json)`, timeout 4 h. On failure the v4g lists are
    kept, but FZ c48 L11 asserts coverage, so a failure is fatal in the frozen pipeline.
- **Runner settings (E2:eng_runner.py).**
  - L9-12: `fp_bits.npy` is forced to prvsiyan's 6,930-bit file.
  - L14: `RANK.W_A = (0.35, 0.55)`, `RANK.SEEDS = (0, 1)`, `CFG.N_ANALOG = 200`.
  - L19-20: `E.main(..., sim_rank_rows_nofp.npz, rank_train.npz, fp_models, w_pv=0.88)`.
  - L21-36: blend, take the top 80, de-duplicate by tautomer key14 (`canon_key`, E2:casmi_engine.py L133-142, stereo
    kept) and keep 40. Output `{mid: {smiles, keys}}`.
- **Commit-run log (LOG).**

  | time | event |
  |---|---|
  | 119 s | library 2,539,608 spectra; formula mass found for 100% of rows; 45.5% of rows are timsTOF |
  | 387 s | pool of 1,137,233 structures built |
  | 406 s | 275,810 analog representatives |
  | 418 s | FP models loaded ("1 single + 1 merged on cuda") |
  | 487 s | MetFrag-lite on 2,109 candidates (12 molecules) |
  | 630 s | "our ranker": 4 HistGB on 399,790 rows × 51 features |
  | 687 s | "prvsiyan ranker": 8 HistGB on 142,762 rows × 31 features |
  | 715 s | done |

### 1.2 Input spectra
- **Test spectra (E2:casmi_engine.py L351-357).** All spectra of a molecule are used. Peaks are kept up to the precursor
  m/z + 2 Da and intensities are divided by their maximum.
- **Neutral mass.** Computed per spectrum from a 29-adduct table that accounts for the electron mass (E2:pv.py
  L163-190), so it is correct for the hidden test's corrected m/z. The target mass is the median over the molecule's
  spectra (L361).
- **Library neutral masses** come from `molecular_formula` arithmetic (E2:casmi_engine.py L18-26, L89-92), not from
  the precursor m/z. LOG: formula found for 100% of rows.
- **Cleaning for similarity (`_clean`, E2:pv.py L30-66):**
  - drop peaks below 0.002 × the maximum and keep the top 256;
  - normalise intensities to sum 1;
  - reweight by entropy when S < 3, with w = 0.25 + 0.25·S.

  The similarity is unweighted-sum spectral entropy similarity, mz tolerance 0.01 (L69-92).
- **FP model input (`prep_peaks`, E2:pv_fp.py L28-56):**
  - keep peaks up to the precursor m/z + 1.5;
  - floor at 1e-3 of the maximum;
  - keep a window-diversified top 128 (at most 8 peaks per 50 Da bucket);
  - feed sqrt(intensity / maximum).

  There is no binning: exact m/z goes in as sinusoidal embeddings.
- **Metadata.** Adduct (26-entry vocabulary), instrument family (timsTOF / Orbitrap / QTOF / IT / other; the test is
  timsTOF = 0), collision energy (mean; 25 if missing) and polarity (E2:pv_fp.py L10-25, E2:casmi_engine.py L286-291).

### 1.3 Candidate bank and retrieval (E2:casmi_engine.py L145-200, L364-365)
- **Bank contents (LOG):**

  | source | structures |
  |---|---|
  | prvsiyan COCONUT 2.0 | 436,389 |
  | ChEBI / LIPID MAPS (new, key-filtered against the COCONUT keys) | 98,433 |
  | PubChemLite CC0 copy (new) | 356,820 |
  | training structures, fingerprinted in-kernel (RDKit, L176-184) | 245,591 |
  | **total** (mass-sorted, ExactMolWt) | **1,137,233** |

- **Candidates:** every bank structure within ±10 ppm of the target neutral mass, widened to ±30 ppm if the window is
  empty (L364-365).
  - No formula filter.
  - No top-k pre-filter: every window candidate is scored.

### 1.4 Evidence channels (per molecule, E2:casmi_engine.py L254-283, L384-404)
- **Library ("exact-match" component).**
  - Every library row whose neutral mass is within ±10 ppm, i.e. all 2.54M training spectra, both polarities, no
    per-structure cap.
  - For each query spectrum: entropy similarity (`pv.search`, L265-267) and an adduct-shifted variant in which the
    reference is shifted by (query precursor − reference precursor) and the maximum of the shifted and unshifted scores
    is kept (`search_shift_rows`, E2:pv.py L128-139; L268-271).
  - Aggregated per structure: maximum (lv), mean (lib_mean), shifted maximum (libsh) and row count (L272-274).
  - There is no hard library-hit override: library evidence enters only through features.
- **Analog.**
  - One representative spectrum per structure, chosen to prefer timsTOF, then the most peaks (`_reps`, L206-214; ADDED
    L209-210), pre-cleaned (`clean_store`).
  - Search window ±200 Da of neutral mass. Similarity: maximum of the entropy similarity and its mass-shifted version
    (shift = target − representative mass; `search_shift_pre`, E2:pv.py L117-125; L237-245).
  - Keep the top `N_ANALOG` structures (80 in the rows, 200 in pubm; L248-251).
  - Searched three ways: all representatives, positive only and negative only (L275-282).
- **FP model.** Logits z ∈ R^6930 averaged over two models (`molecule_logits`, E2:pv_fp.py L167-182):
  - the single-spectrum model, averaged over the spectra;
  - the merged model, on one pseudo-spectrum of all peaks (0.005 Da de-duplication, first adduct, CE 25, mean mode).

  Candidate score: raw = candidate_bits · z; nrm = raw / sqrt(popcount) (E2:pv.py L227-231).
- **MetFrag-lite** (E2:pv.py L267-357):
  - fragments from all 1-bond and 2-bond cuts of the heavy-atom graph (up to 34 bonds), each with H shifts of −2 to +2;
  - intensity explained is sqrt-weighted;
  - three variants: with the polarity's proton, with adduct-aware charge carriers (Na, K, NH4, Cl, formate; ADDED), and
    a per-spectrum mean (E2:casmi_engine.py L384-404).

### 1.5 FP model architecture (E2:pv_fp.py L59-117; identical to cubres `checkpoint_lineage/original_sources/fpmodel.py`)
- **Peak tokens.**
  - m/z and neutral loss (precursor − m/z, clamped at 0) each go through `SinEmb(512)`: wavelengths 10^-2 to 10^3.2,
    sin and cos.
  - Token = Linear(2·512 + 1 → 512) over [m/z embedding, loss embedding, sqrt intensity].
- **Global token.**
  - Linear(512 + 3 → 512) over [precursor SinEmb, CE/100, mode, log1p(precursor)/10].
  - Plus an adduct embedding (26) and an instrument embedding (5).
- **Encoder.**
  - Six pre-LayerNorm transformer blocks: d 512, 8 heads, FF 4d GELU, dropout 0.1, padding-masked SDPA.
  - A final LayerNorm.
- **Head.**
  - Input: [global token, masked mean of the peak tokens] (1024).
  - Linear(1024 → 2048), GELU, Linear(2048 → 6930).
  - 6,930 output logits; no sigmoid at use.
- **Checkpoints.** `fp_single_s2.pt` (step 20,000) and `fp_merged_m1.pt` (step 24,000), 144,110,255 bytes each
  (~36M fp32 parameters; prvsiyan/casmi26-fp-models-v2 card).
- **Training (cubres `.../original_sources/train_fp3.py`; train_fp4.py is the merged variant).**

  | item | setting (train_fp3.py line) |
  |---|---|
  | loss | BCE on the 6,930 bits (L118) + λ·softmax CE over f·z of the positive and K = 63 negatives (L120-126) |
  | negatives | drawn uniformly from the ±10 ppm pool window; random decoys when the window has no isomers (`sample_negs` L34) |
  | score scaling | centred within the example, divided by sqrt(nbits) |
  | optimiser | AdamW, lr 2e-4, weight decay 0.01, betas (0.9, 0.98) (L92); warmup 2,000 then cosine down to 2% (L100) |
  | batch | 256, fp16 autocast, gradient clip 1.0 |
  | augmentation | peak dropout U(0, 0.3), log-normal intensity jitter σ 0.25, m/z jitter 5 ppm (L58-61) |
  | checkpoint selection | best "hard-negative top-1" on 2,048 val spectra, checked every 2,000 steps |
  | merged model (train_fp4.py) | merge 2-4 same-structure spectra with p = 0.6 |
  | known gap | the seed is set after model construction (L91), so the run is not bit-reproducible |

- **Training data (cubres `public_source/build_dataset.py`).**
  - All train.parquet spectra with |precursor_error_ppm| ≤ 20 and ≥ 3 peaks (L21-28): 2,334,531 spectra and 273,983
    structures (public_source README).
  - Validation structures (L45-50): every enveda-np-examples structure, plus 2,500 random structures that come from NP
    libraries and are also in COCONUT (rng 0).
  - Everything else is used for training. This is why the public weights "have seen most library structures" (MY9 §2):
    their FP-only MRR is 0.76-0.82 on gnps / mona / msdial / massbank, against 0.49 on the held-out npx.

### 1.6 The 6,930-bit list
- **Descriptor:** 10,407 columns = Morgan radius 2 (4,096) + Morgan radius 3 (4,096) + RDKit path fingerprint
  (2,048, maxPath 6) + MACCS (167) (E2:casmi_engine.py L106-130; cubres `public_source/build_pool.py`).
- **Selection:** keep the bits with 0.005 ≤ frequency ≤ 0.995 over all legacy training structures (cubres
  `public_source/train_fp.py` L10-19; public_source README "Schema and split"). This gives 6,930 bits.
- **Reproducibility:** fully reproducible from training structures and RDKit, with no learning involved. The same bits
  serve as FP targets, as candidate fingerprints for f·z, and as the analog Tanimoto fingerprint.

### 1.7 Features
- **prvsiyan's 31 columns** (`rank_features`, E2:pv.py L210-255; names in cubres
  `ranker_lineage/FEATURE_COLUMNS.json`):

  | columns | group | contents |
  |---|---|---|
  | 0-4 | library | lv, rank, max, lv − max, lv > 0 |
  | 5-8 | analog, cubic | ap = max over analogs of Tanimoto × sim³; rank; max; gap |
  | 9-14 | analog context | linear analog maximum, best Tanimoto, Tanimoto to the top analog, sim³-weighted mean Tanimoto, top analog similarity, log(number of candidates) |
  | 15-20 | FP model | z(raw), rank(raw), raw − max, z(nrm), rank(nrm), is-max |
  | 21-24 | MetFrag | frag, rank, gap, z |
  | 25-30 | cross-channel agreement | lv·(1 − model rank), ap·(1 − model rank), model-rank agreement at the library best, model-rank agreement at the analog best, library agreement × lv max, corr(lv, −model rank) |

- **megayak blocks** (E2:casmi_engine.py L316-341). The shipped rows have 51 columns: `base` (31, with the FP columns
  0) + `libx` (8: lib_mean, libsh, log count, each with rank and gap) + `anapol` (5: same-polarity analogs) + `fragx`
  (7: adduct-aware and mean MetFrag). LOG: 51 features. The `mass` block is unused.
- **What is absent:** documentation, source membership, formula and in_train flags. The only implicit membership signal
  is lv > 0.

### 1.8 Rankers, rows and labels
- **prvsiyan ranker (s_pv).**
  - Rows: `rank_train.npz` with X (142,762 × 31) float32, Y, M and G (card prvsiyan/casmi26-ranker-features).
  - Builder: cubres `ranker_lineage/original_sources/make_ranker3.py`.
    - Dataset A: 250 enveda-np-examples (timsTOF natural products, the hidden domain).
    - Dataset B: 569 "obscure" held-out structures.
    - Each query is simulated twice: class 1 ('c1': the truth keeps its other library spectra) and class 2 ('c2': all
      spectra of the truth removed from library and analog search).
    - About 87 candidates per query; label Y = 1 for the truth candidate (L43); M = 0 for c1, 1 for c2 (L88);
      G = query index.
    - The FP columns come from logits of models for which these structures were validation structures, so they are
      honest. The exact cache producers are unresolved; that is why the audit rates A30 not reproducible.
    - W1 was chosen by 5-fold CV on A, objective 0.162·c1 + 0.271·c2 (L9, L80-81), with `N_ANALOG` = 80 (L36).
  - Fitted in-kernel on every run (E2:casmi_engine.py L422-427): 8 `HistGradientBoostingClassifier` models (depth 6,
    500 iterations, lr 0.03, min leaf 80, l2 1.0), priors {0.30, 0.60} × seeds {0..3}, sample weight w on c1 rows and
    1 − w on c2 rows. Score: mean predict_proba. Uses the base block with FP.
- **megayak ranker (s_ours).**
  - Rows: `sim_rank_rows_nofp.npz` with X (399,790 × 51), Y, S (0 = c1, 1 = c2) and blocks.
  - Built from 2,250 molecules from 7 libraries, each simulated as c1 and c2 (MY1 TL;DR). The simulation driver is
    unpublished.
  - Fitted in-kernel (L413-419): 4 HistGB models, W_A {0.35, 0.55} × seeds {0, 1}, without the FP channel
    (`use_fp=False`, L503).
- **Train / serve skew in pubm.** `N_ANALOG` is 200 at serving (eng_runner L14) but 80 in prvsiyan's rows. The tree
  depth was tuned on the public board: depth 7 → 6 moved the frozen notebook from 0.448 to 0.455 (FZ c58 rows 7-9).

### 1.9 Blend and list
- **Blend** (`blend_scores`, E2:casmi_engine.py L441-447): 0.88·(1 − rank_norm(s_pv)) + 0.12·(1 − rank_norm(s_ours))
  + 1e-6·s_pv, computed within each molecule. megayak shipped 0.65; pubm uses 0.88.
- **List:** top 80 by blend, de-duplicated by key, cut to 40 (eng_runner L27-35).
- **What is absent:** a PubChem channel and generation. ChEBI / LMSD and PubChemLite enter only as bank members.

### 1.10 Licences
- **Training and model code.** cubres `reproduction/legacy-v2/NOTICE` puts 29 listed files under Apache-2.0, among
  them:
  - fpmodel.py (the FPNet in E2:pv_fp.py), train_fp3.py, train_fp4.py;
  - build_dataset.py, build_fps.py, build_pool.py, build_coconut.py;
  - make_ranker3.py, rankfeat.py, casmi_core.py, engine.py, speclib.py.

  This resolves A28's "pv_fp.py licence on cubres github unverified".
- **Engine code.** pv.py and casmi_engine.py are public-notebook code (prvsiyan / megayak), Apache-2.0 by default.
- **Weights, rows and bank** stay ineligible: A29-A36. CC BY-NC-SA, not byte-reproducible, unpublished drivers.

## 2. Where pubm consumes Engine 2

| # | where | what | weight / threshold | property of the Engine 2 list that matters |
|---|---|---|---|---|
| 1 | FZ c41 L66-85 (ICE_UNION); L118-123 | Engine 2 top-40 candidates absent from Engine 1's top `TOPN` = 60 (FZ c26 L13) are added to the ICEBERG input; GLACIER scores the same items ([M+H]+ only). | none | new candidates (recall) at the ranks that later survive fusion |
| 2 | FZ c45 L1, L15-25, L29-50 | `fuse2` (superseded by #4 / #6) | Engine 1 adds 1/(3+r), Engine 2 adds 0.6/(3+r), top 40; then GLACIER `rerank_multi` within formula (ICE off, L2) | head agreement |
| 3 | FZ c48 L11, L36-38 | assert that Engine 2 covers every molecule; Engine 2 SMILES appended after fusion overflow and base tail in the metric-safe 25-fill | none | tail only |
| 4 | FZ c49 L4-19, L30-41 | `metric_fusion` with one vote per metric identity, alpha = ALPHA 0.6, offset = KRR 3, limit 40, then GLACIER rerank | as #2 | head agreement |
| 5 | FZ c51 L20-21 | candidate set for the MetFrag gap-fill (molecules with no GLACIER or ICEBERG) = prefusion ∪ Engine 2 SMILES | none | recall when no forward model is available |
| 6 | FZ c52 L53-99, L219 | **the active frozen order (E006).** `metric_fusion`(prefusion, Engine 2; 0.6, 3, window 40), then GLACIER rerank (joint ICEBERG for N1), else ICEBERG fallback, else MetFrag gap λ 0.5 | as #2 | head agreement + same-formula candidates for GLACIER slot-keeping |
| 7 | FZ c53 L13-26 | rank-2 insert: Engine 2's #1 goes to rank 2 unless it is already rank 1 | unconditional | **rank-1 accuracy of Engine 2 where Engine 1's #1 is wrong** |
| 8 | NB c08 MODULES['export_evidence.py'] L28 | Engine 2 rank map → `engine2_rank` column of the neural evidence | none | ranks ≤ 5 |
| 9 | NB c10 `_post` L50-56, L85-92, L120-132, L160-171 (fragment check, PHYSICAL True) | Outsider tail candidates need ≥ 2 of {Engine 2, PubChem, main} or fusion rank ≤ 3 (L89). A rank-1 move needs support (Engine 2 / PubChem / main rank ≤ 5) or a simulator vote ≥ 1 (L123-130). | contender score = margin + 0.05·neural margin + 0.025·simulator vote + 0.01·supports (L132); tail key + 0.015·supports (L171) | Engine 2 top 5 |
| 10 | NB c08 MODULES['outsiders.py'] L32-38 → NB c16 L158-163 | round-robin outsider pool (fusion overflow, Engine 2, main tail, PubChem); midx fills only empty slots | none | negligible |
| 11 | NB c14 `source_votes` L64-75 | +1 if the candidate is at Engine 2 rank ≤ 6; −1 if the leader is at Engine 2 rank ≤ 4 and the candidate is absent or ranked > 12 | **inactive** (PROVENANCE False, NB c01 L22; NB c15) | n/a |

### How much the head matters (rows 4, 6 and 7)
- Engine 1's ranks 1 / 2 / 3 / 6 earn 0.250 / 0.200 / 0.167 / 0.111. Engine 2's #1 earns 0.150 and its #3 earns 0.100.
- So if Engine 2's #1 is Engine 1's #2-#6, it overtakes Engine 1's #1, unless Engine 1's #1 sits in Engine 2's top 3.
- An Engine 2-only #1 lands between Engine 1's #3 and #4 before GLACIER slot-keeping. Engine 2 ranks beyond ~15 never
  reach the top 25.
- **What matters, then:** (a) Engine 2's top-1 to top-5 accuracy conditional on Engine 1's head, and (b) a few new
  same-formula candidates near the top for GLACIER.
- **Evidence from eng3a** (RRF of the two lists inside Engine 2, NB eng3 note in plan.md):
  - 143 of our candidates entered the 40-lists of the 12 visible molecules, and about half of Engine 2's #1 picks were
    contested;
  - result: −0.034 LB, rated "real harm" by docs/perturbation_analysis_nov2.md.

  Diluting Engine 2's top picks costs; adding recall does not pay. The null engine (Engine 1's top 40) removes (a) and
  (b) together: −0.085.

## 3. Engine 2 vs our stage 1 (pube1's Engine 2 = casmi/run_eng2.py → casmi/own.py stage1)

### 3.1 Side by side

| aspect | Engine 2 | ours (pube1) |
|---|---|---|
| bank / pool | 1,137,233: COCONUT 436k + ChEBI / LMSD 98k + PubChemLite 357k + train 246k | pool v18c, 1,160,133 (COCONUT 10-2026, own ChEBI r255 / LMSD, PubChemLite v3.3.0, train) |
| window | ±10 ppm (±30 ppm fallback), all candidates scored | ±10 ppm, minimum 0.002 Da (retrieval.py L77), all scored |
| formula handling | none | none in retrieval; within-formula z / rank / gap features for pf_* (own.py formula_rel) |
| library channel | all 2.54M spectra, both polarities, plus adduct-shifted match; max, mean and count | lib10 (≤ 10 spectra per structure), same polarity only (retrieval.py L116), no shift; max and count |
| analog channel | 1 representative per structure (timsTOF first), entropy similarity max(plain, shifted), ±200 Da, top 80 / 200 structures, 8 analog features | analog_lib (3 representatives per structure), modified cosine with ≥ 3 matches, ±350 Da (analog.py L33), top 40 per spectrum, 20 analogs, 4 features (retrieval.py L125-140) |
| FP / spectral model | FPNet single + merged, 6,930 bits, f·z; leaky on training structures; timsTOF token | PeakFormer pf_tpu_f (f·z + cosine, PF-syn, instrument-free) + binned dual encoder fp_model_v1r (cos / Bernoulli, timsTOF token 1) |
| fragments | MetFrag-lite, 3 variants (proton, adduct carriers, mean) | frag / frag_n (1-2 cuts, H shifts −2..+1) |
| documentation / source | none | 20 features: in_train, in_coco, in_bio, in_pcl, pcl_*×5, pc_sid / pmid / patent / ncid, n_org / n_coll / n_doi / n_syn / has_cas / has_name / ann / npl (`_feats.txt` of v18cu) |
| cross-channel features | 6 agreement features (§1.7) | none (ranks and gaps per channel only) |
| ranker | pointwise HistGB classifiers, 8 + 4, fitted in-kernel; c1 / c2 regime weights; rank blend 0.88 / 0.12 | LightGBM LambdaRank, 31 leaves, 500 rounds (train_ret_ranker.py L93-105), seed pair v18cu + v18cur1, C1_DUP 2 (L66-68), DOC_W weights (L101-103) |
| rows | 819 queries × 2 (250 npx + 569 held out), honest FP; plus 2,250 × 2 from 7 libraries (no FP) | 7,788 train queries (rank_keys: 5,000 COCONUT & train + 3,000 train-only; make_split_keys.py L40); val 1,463 (val_keys: all 250 npx + 750 + 500, L38). PeakFormer and FP held out on both. Reach filter for class 2. |
| list | top 80 → key de-duplication → 40 | top 48 → key de-duplication → 40 (run_eng2.py) |
| standalone public LB | 0.333 (prvsiyan ranker) / 0.337 (blend) (MY9 §1) | 0.339 (v18c) / 0.348 (v18cte, with GLACIER stage 2) |
| value inside the fork | +0.085 over null | +0.041 over null |

### 3.2 Measured channel strength on the hidden-like npx queries
- **Setup.** Our rows `ret18cu_val_fp.parquet`; the 250 enveda-np-examples val queries split 80 class 1 / 169 class 2;
  channel-only MRR@25 (scratchpad `eng2_tools/npx_channels.py`).
- **Reference.** megayak's channel table on the same 250 structures in a train ∪ COCONUT ±10 ppm window (MY1 §5,
  MY9 §2).
- **Cross-harness caveat.** Query construction differs: ours takes 1-5 spectra per query, timsTOF preferred
  (dump_ret_rows.py L67-69).

| channel | npx class 1, full v18c window (median 90) | class 1, train ∪ COCONUT (59) | npx class 2, full window (99) | class 2, train ∪ COCONUT (59) | Engine 2 / megayak (train ∪ COCONUT) |
|---|---|---|---|---|---|
| library | 0.861 | 0.861 | 0.043 | 0.057 | class 1: 0.930; adduct-shifted 0.938 |
| analog | 0.595 | 0.620 | 0.439 | 0.464 | class 2: 0.524; same polarity 0.532 |
| PeakFormer f·z (pf_bern) | 0.603 | 0.610 | 0.501 | 0.529 | class 2, public FPNet: 0.493; leak-free retrain 0.452 |
| our FP cosine (fp_cos) | 0.309 | 0.375 | 0.237 | 0.296 | |
| MetFrag | 0.246 | 0.286 | 0.207 | 0.261 | class 2: 0.243 |

For reference, the whole Engine 2 ranker on npx scored class 1 0.872 / class 2 0.655 without FP and class 2 0.689 with
the leak-free FP (megayak grouped CV). None of the 80 npx class-1 queries is an artifact query (truth with no
remaining library spectrum).

### 3.3 The 12 visible molecules
- **Inputs.** pubm `e004_candidate_inputs.json` + `eng_lists.json` and our local `e1/runs/eng2_s12.json`. All 12 are
  library hits (≥ 0.87).
- **Agreement at rank 1.** Engine 1 #1 = Engine 2 #1 = our #1 for 12 / 12 molecules.
- **Overlap.**
  - Engine 2's top 40 and ours share 15.5 entries.
  - New entries versus Engine 1's prefusion list: 28.6 for Engine 2 and 27.4 for ours.
- **Composition** (our pool flags):

  | list | train | COCONUT-not-train | PubChemLite | bio |
  |---|---|---|---|---|
  | Engine 2, top 10 | 81% | 8% | 10% | 1% |
  | ours, top 10 | 39% | 41% | 14% | 3% |
  | Engine 2, top 40 | 64% | 14% | 17% | 3% |
  | ours, top 40 | 31% | 39% | 22% | 4% |

- **Reading.** Same candidate count, very different contents: our lists are led by documentation and COCONUT priors,
  Engine 2's by library, analog and FP evidence, which leans to training structures.

### 3.4 Ranked explanations of the 0.044
1. **Weaker library and analog channels at the top of the list (high).**
   - These two channels decide Engine 2's top ranks, and the fork consumes exactly those (§2).
   - On npx, Engine 2 leads by about +0.07 on class-1 library matching and about +0.06 on class-2 analog matching (§3.2).
   - Mechanisms seen in the code:
     - Library: all spectra, no 10-spectrum cap; adduct-shifted matching; mean and count; cross-polarity rows; formula
       masses.
     - Analog: entropy similarity with mass shift instead of modified cosine; timsTOF-preferred representatives;
       ±200 Da; 80-200 analogs instead of 20.
   - megayak measured +0.02 class 1 / +0.011 class 2 for its libx block alone (MY1 table).
   - Plausible share: 0.01-0.02.
2. **Provenance features make our engine echo Engine 1 and midx (medium-high).**
   - The fork already applies documentation twice: Engine 1's z + 0.15·pop re-rank of its top 60 (LOG "v19 pool
     popularity re-rank, mu 0.15") and the midx DocPrior.
   - The fusion constants, the rank-2 rule and the support thresholds were tuned with a provenance-free Engine 2.
   - Our ranker's top-1 follows documentation, so it fails where Engine 1's documentation-led head fails. Hidden
     class-2 answers are documentation-poor: our class-2 MRR fell from 0.919 to 0.551-0.652 under donor documentation
     (plan.md "COCONUT gap study").
   - Our own interleaving study found that the public chain (provenance-free rankers, popularity after, RRF) merges
     sources about 3× better (plan.md "Interleaving loss").
   - Not yet measured inside the fork: Step 0 measures it.
   - Plausible share: 0.005-0.015.
3. **Selection optimism (medium, certain sign).**
   - About 0.015 (0.009-0.019) of pubm's public 0.479 is optimism from tuning to the 124 public molecules. A structural
     swap of Engine 2 gives back up to about 0.01 of it on the public board, whatever its private value
     (docs/perturbation_analysis_nov2.md).
   - Consequence: the recoverable private gap is about 0.03-0.035.
4. **Ranker recipe (low-medium).**
   - Engine 2: pointwise classifiers on honest-FP rows from the hidden domain (npx, held-out NPs), c1 / c2 regime
     weights, and six agreement features that reward channel consensus.
   - Ours: LambdaRank on broader rows, documentation-weighted, with no consensus features.
   - Plausible share: ≤ 0.005-0.01, uncertain.
5. **FP model (not a cause).** On npx class 2, our pf_bern is 0.529 against the public FPNet's 0.493 in a comparable
   window. Our fp_model_v1r is weak (0.296), but the PeakFormer carries the channel. Training a pv_fp FPNet should not
   be expected to close the gap.
6. **Pool, window, list length, keys, MetFrag (≈ 0).** Same sources, size within 2%, same ±10 ppm, 40 entries,
   tautomer key14; MetFrag 0.261 vs 0.243. In the frozen notebook, adding PubChemLite to Engine 2 scored −0.003
   (FZ c58 row 7).

## 4. Eligible rebuild options, cheapest first

The shares below are judgement estimates on the public board, against the 0.044 gap, with about 0.01 not recoverable
(§3.4 point 3).

| # | option | work | compute | expected share of the gap | blockers / risks |
|---|---|---|---|---|---|
| 0 | **Source-blind Engine 2 OURS.** Retrain the v18cu pair on the existing ret18cu rows with `--source-blind` (drops in_coco, in_bio, PubChemLite, PubChem, LOTUS, priors and vendors; keeps in_train and lib_n; train_ret_ranker.py L74); swap with `run_eng2 --ranker`. | 2-3 h | local LightGBM 2 × ~15 min; 1 LB slot | 0-25% (0 to +0.01) | None. It drops in_bio / PubChemLite, which carried some "other" answers. The Oct 21 offline test inside our own pipeline (plan.md "Second-ranker rank-2 insertion") does not predict the fork. |
| 1 | **Port Engine 2's library and analog channels and the agreement features into our stage 1** (option b). New feature code in casmi/retrieval.py; replay the ret18cu q_rows; retrain the v18cu pair. | 12-18 h | ~6-8 CPU-h for rows (Kaggle CPU kernels) | 20-45% in the fork; also lifts our own line (pick #2) | Still documentation-aware (keep option 0's flag as a variant); full row rebuild. |
| **R** | **Recommended: Engine 2 clone on eligible inputs** (option a without FPNet training). Keep the Engine 2 code (pv.py, casmi_engine.py; FPNet source Apache-2.0). Bank = pool v18c with our own 0.005-0.995 bit selection; FP channel = PeakFormer f·z (pf_tpu_f now, PF-x later) fed as raw / nrm; rows = replay of our val / rank queries through Engine 2's own channel code, PeakFormer columns joined from ret18cu rows; in-kernel HistGB unchanged. | 25-35 h | bank ~2 CPU-h; rows ~6-10 CPU-h (Kaggle CPU); no GPU; kernel run ≈ Engine 2's own (~25-35 min hidden) + PeakFormer ~4 min | 50-80% (+0.02 to +0.035 public) | (i) Engine code licence: public-notebook Apache default (A28); keep the NOTICE. (ii) Row simulation must mask library rows and representatives exactly (gate in step 2). (iii) The fork's tuned constants expect Engine 2-like lists: the clone is the closest eligible match. |
| 3 | **Faithful F2c:** option R plus our own FPNet S2 / M1 trained with cubres train_fp3 / train_fp4 (Apache-2.0) on our spec export and pool negatives, with our bit list. | 40-55 h | + 2 × ~5 h T4 (megayak's leak-free retrain: 20k steps on a T4, best at 16k) or ~2 × 3 h TPU | R ± 0.005 | GPU quota. Value doubtful: our PeakFormer already matches or beats the public FPNet on npx. Only if R's FP channel underperforms in CV, or as a second FP block. |
| 4 | Rejected: re-tune the fork's alpha / rank-2 / support constants for our engine | 2 h | slots | n/a | Re-weightings cannot be read on this board (public / private correlation 0.04-0.7). |
| 5 | Rejected: megayak's leak-free FPNet or fp16k rows | — | — | — | CC0 label, but built on the NC-SA bank with an unpublished driver (A32 class). |
| 6 | Rejected: merging our lists into Engine 2 | — | — | — | eng3a: −0.034. |

**Why R beats F2c's 40-60 h:**
- no FPNet training (our PeakFormer is already the stronger f·z model);
- the PeakFormer features of the ranker rows are already computed for the same queries and candidates (ret18cu);
- the channel code is reused as is.

**Training code status (asked in task 4a).**
- The FPNet trainers, preparation, bit selection and the 31-column builder are all public and Apache-2.0 (§1.5-1.8,
  §1.10).
- The implied loss is BCE over the bits plus a window-contrastive f·z cross-entropy. The data is train.parquet
  structures with a ±10 ppm pool window of train ∪ COCONUT.
- Only the row caches' producers and megayak's simulation driver are missing. We replace them with our own replay
  driver.

## 5. Plan for option R

**Order of work.**
- Step 0 starts at once, in parallel with steps 1-3.
- Steps 1-3 run Nov 3-7; step 4 (kernel) on Nov 8; step 5 (LB pair) on Nov 9-10.
- That leaves margin before the Nov 19-20 deadline for a scored final eligible version.
- Notation: W = /home/user/work, R = repo. All new code is ours. Copies of pv.py / casmi_engine.py keep their origin
  header and an Apache-2.0 NOTICE line; every modification is marked `# OURS`.

**Step 0 (cheap test of explanation 2): source-blind Engine 2 OURS.**
- Train:
  - `RANKER_PARAMS='{"num_threads": 4}' python3 scripts/train_ret_ranker.py $W/ret18cu_train_fp.parquet $W/ret18cu_val_fp.parquet $W/own_models/ret_ranker_v18cusb.txt $W/pool_v18c --drop-artifact --source-blind`
  - The same with `'{"num_threads": 4, "seed": 1}'` gives `ret_ranker_v18cusbr1`.
  - Without DOC_W, since there are no documentation columns.
- Offline gate:
  - val MRR by subset (printed by the script) and HL-val class 2 (scripts/hlval.py, documentation transplant);
  - require HL-val class 2 ≥ v18cu's, and npx class 1 within −0.01 of v18cu's;
  - HL-val is used here only for the documentation dimension it was built for.
- Deploy:
  - add both models to `bryantyu/casmi26-elig-assets`;
  - build `casmi26-pube1-sb` with scripts/make_pub_variant.py (Engine 2 OURS with `--ranker ret_ranker_v18cusb,ret_ranker_v18cusbr1`);
  - check that the verified run logs "ENGINE 2 OURS 12 / 12".
- LB rule:
  - pube1-sb − pube1 ≥ +0.008: provenance is a real part of the gap; option R stays doc-blind, and option 1 should use
    the source-blind feature set;
  - ≤ −0.008: our documentation features carry value in this slot; option R adds a documentation-aware second ranker
    as the 0.12 partner;
  - in between: no information; continue R.

**Step 1: the bank** (`scripts/build_e2_bank.py <pool_dir> <train_meta.parquet> <out_dir> [procs]`).
- Inputs:
  - `$W/pool_v18c/pool.parquet` (key, smiles, mass, in_train; mass-ascending: assert it);
  - `/home/user/data/train_meta.parquet` (inchikey14 of the training structures);
  - RDKit 2026.03.3.
- Processing:
  - our own implementation of the 10,407-column descriptor: Morgan r2 4096 + r3 4096 (rdFingerprintGenerator), RDKit
    path fingerprint 2048 / maxPath 6, MACCS 167;
  - bit frequencies over the pool rows with in_train == 1; keep 0.005 ≤ f ≤ 0.995;
  - pack the selected bits in pool order.
- Outputs (`$W/e2_bank/`):
  - `fp_bits.npy` (int64, sorted);
  - `bank_fp.npy` (uint8, N × ceil(nbits/8));
  - `bank_mass.npy` (float64);
  - `bank_meta.parquet` (key, smiles, pool_row; rows aligned with the pool);
  - `MANIFEST.json` (md5s, nbits, thresholds, RDKit version, failure count).
  - Publish as dataset `bryantyu/casmi26-e2-bank`, with the build script in casmi26-code.
- Gates:
  1. RDKit failures ≤ 0.5%. Failed rows keep zero bits plus a flag; no row is dropped, so the arrays stay aligned with
     the pool.
  2. nbits in [6,000, 8,000] (legacy: 6,930).
  3. Masses finite and ascending.
  4. 1,000 random rows recomputed from SMILES give identical bits.
  5. The ±10 ppm windows of the 1,463 val queries match the ret18cu candidate pid sets for ≥ 99% of queries.
  6. Wall time and RSS logged. Expected ~1.5-2 h on 4 processes (Engine 2 did 245k structures in 232 s, LOG).

**Step 2: clone package, replay simulator, channel-parity gate.**
- Files:
  - `casmi/e2/pv.py`: verbatim copy of E2:pv.py.
  - `casmi/e2/engine.py`: copy of E2:casmi_engine.py with these `# OURS` changes.
    - (a) `build_pool` loads `e2_bank` instead of the COCONUT / bio / PubChemLite files and the in-kernel train
      fingerprinting.
    - (b) FP adapter. `blk_base` and `rank_features` take `model_raw` (per-candidate PeakFormer f·z from
      `casmi.peakformer` PFEnsemble `.score`). nrm = raw / sqrt(max(popcount of the bank bits, 1)). All 12 FP and
      agreement columns keep their formulas.
    - (c) `compute_channels(..., hide=None)`.
      - Class 1 (`hide={'rows': q_rows}`): those library rows are dropped from the ±10 ppm match. If a hidden row was
        its structure's analog representative, the representative is re-picked from the remaining rows by the same
        rule (timsTOF, then most peaks).
      - Class 2 (`hide={'key': truth}`): all rows of the truth are dropped from the library match, and the truth's
        representative is excluded from the analog results (similarity set to 0).
    - (d) The query electron-mass correction for enveda-180 positive rows, as in dump_ret_rows.py.
    - (e) The `N_ANALOG` setting is shared by rows and serving (80 and 200 both built; CV picks one).
  - `scripts/e2_replay.py <ret18cu_{train|val}_pf.parquet> <out.parquet> [--limit N] [--procs 4]`.
    - For every qid it reads (q_rows, cls, truth), builds the query from those train rows and runs the channels and
      MetFrag-lite with hiding.
    - It joins pf_bern by (qid, pid). Candidates missing from the ret18cu window are scored on the fly with the same
      PFEnsemble.
    - Output: one row per (qid, candidate) with base (31), libx, anapol, fragx, label, cls, qid and truth.
- Gate on all 1,463 val queries (report npx / np / other × class):
  1. Leakage: class-2 truth lv = 0 in 100% of queries and the truth never among the analogs; class-1 hidden rows never
     matched.
  2. Window identity with ret18cu ≥ 99%.
  3. **Channel parity** (the direct test of explanation 1):
     - Engine 2 library channel (column 0) on npx class 1 ≥ 0.90 (ours: 0.861);
     - Engine 2 analog channel (column 5, ap) on npx class 2 ≥ 0.48 in the full window, or ≥ 0.50 in train ∪ COCONUT
       (ours: 0.439 / 0.464);
     - MetFrag column 21 ≥ 0.20.
  4. Throughput ≤ 3 s per query and CPU, excluding MetFrag.
- If check 3 fails, Engine 2's channel advantage does not reproduce in our window and queries. Stop and diagnose
  (window size, query construction) before step 3; options R and 1 then lose most of their case.

**Step 3: rows, ranker recipe, offline CV gate.**
- Rows:
  - Run e2_replay on ret18cu_val (1,463) and ret18cu_train: first a 3,000-query stratified subset, then all 7,788
    queries if CV still improves. Use Kaggle CPU kernels (long local jobs die with container recycles, plan.md
    Oct 24).
  - `scripts/e2_pack_rows.py` writes the formats the engine already reads:
    - `$W/e2_rows/e2_rank_train.npz` with X (n × 31) float32, Y, M = cls − 1, G = qid (for `fit_pv_rankers`);
    - `$W/e2_rows/e2_sim_rows_nofp.npz` with X (n × 51), Y, S, blocks = [base, libx, anapol, fragx] (for
      `fit_rankers`).
- CV (`scripts/cv_e2.py`): 5-fold grouped by truth key on train rows, plus held-out scoring of the val rows.
  - (i) prvsiyan recipe: HistGB depth 6 / 500 iterations / lr 0.03 / min leaf 80 / l2 1; priors {0.30, 0.60} × seeds
    0-3; weight w on c1.
  - (ii) megayak recipe: 51 columns, no FP; {0.35, 0.55} × seeds {0, 1}.
  - (iii) rank blend at 0.88, plus a 0.65-0.95 sweep for information only. Keep 0.88 unless CV prefers another value
    by more than 0.01.
  - Report MRR@25 by subset × class. Score the same val queries with the v18cu pair from `ret18cu_val_fp`, documentation
    kept and HL-transplanted.
- Pre-registered gate:
  - (G-a) clone on npx: class 1 ≥ 0.87 and class 2 ≥ max(0.655, v18cu-HL npx class 2);
  - (G-b) clone over all val ≥ v18cu-HL − 0.01;
  - (G-c) complementarity proxy on val (HL documentation):
    - MRR(RRF(v18cu, X; α 0.6, K 3)) − MRR(v18cu) ≥ +0.005 for X = the clone;
    - and higher than for X = step 0's source-blind ranker;
    - report the top-1 disagreement rate and P(X's #1 correct | v18cu's #1 wrong).
- If G-a, G-b and G-c pass, go to step 4:
  - kernel `casmi26-pube2` = pube1 with the `casmi.e2` subprocess writing `eng_lists.json` (the run_eng2 schema, k 40);
  - commit-run checks: 12 / 12 lists, key check, RSS ≤ 10 GB, a hidden-run time estimate;
  - publish the rows and the bank with their scripts (reproduce.md §8).
- Step 5: one LB pair, pube2 vs pube1. Optionally add COCONUT-not-train masked probes of both, if slots allow.
- Later upgrades, each with its own offline gate first:
  - PF-x as the FP channel (D3 GO: +0.042 pf_bern RR over pf_tpu_f);
  - a documentation-aware 0.12 partner if step 0 reads negative;
  - option 3's FPNet as a second FP block only if the CV gaps point to the FP channel.

## 6. Caveats
- **§3.2 compares across harnesses.** Query construction and class splits differ. That is why step 2 measures Engine 2's
  channels on our exact queries before anything is built on them.
- **The 12-molecule list statistics** come from library-hit molecules. They show list composition, not accuracy.
- **No public notebook has measured Engine 2 alone at pubm's settings** (ChEBI / LMSD, PubChemLite, depth 6,
  N_ANALOG 200, 0.88). The 0.333 / 0.337 readings are the closest configuration, and megayak's own seed spread was
  0.017.
- **Eligibility of option R** also depends on the open items of docs/fork_eligibility.md (F5 glue, F6 host ruling) and
  on G19 (our licences). Option R adds no new third-party weights or data.
