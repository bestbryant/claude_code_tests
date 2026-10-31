# Eligibility audit of casmi26-pubm (public 0.479 lineage fork) and the replacement plan

Status 2026-10-31. Read-only audit: nothing was executed, pushed or submitted.

**Subject.** `kernels/casmi26-pubm` is our fork of kunaldesale2408/enveda-casmi-2026-mid with the midx final layer. Its run config is `VARIANT CUSTOM .25/.75`, `PHYSICAL True`, `PROVENANCE False`, `FRAG_ON_V2 True`, `OURS False`, `PF_MID_W 0`, `ISO_ALPHA 0`.

**Sources.**
- The notebook (NB) and its decoded frozen 0.465 pipeline (FZ, sha256 d1e757c2...).
- The run logs `/home/user/work/kout/casmi26-pubm/`, and the 593 mounted-file hashes in `v43_mounted_input_hashes.json`.
- Upstream dataset cards and READMEs, saved under `scratchpad/elig/inv/` and `scratchpad/elig/a1_*`.

**Rules applied** (plan.md, "Rules that shape the solution", hosts' threads 748882 / 748065):
- no NIST data and no NIST-trained models;
- no data that is non-commercial-only, except NC terms that come from `train.parquet` alone, which the host tolerates;
- every pretrained weight must be reproducible from public training code and permitted data;
- all code must be open source under an OSI-approved licence (rule 2.c);
- winners must license the solution under MIT.

**Explicitly allowed by the host:** train.parquet, MassSpecGym-trained ICEBERG and GLACIER, DreaMS, FRIGID, CFM-ID 4, PubChem (and its CID-SID / PMID / patent counts), COCONUT, LOTUS, NPAtlas, ChEBI / LIPID MAPS, GNPS / MassBank / MoNA releases, and Enveda-180 from Zenodo.

The template is our own audit: docs/ATTRIBUTION.md and docs/reproduce.md (gaps G1-G20).

**Reconciled conflicts between the auditor rows:**
- **saner train.py.** The input flag said train.py was unpublished. That is wrong: train.py, prepare.py and the histories are in the dataset, with sha256 values matching its sha256.json.
- **NPAtlas.** It is on the host's allowed list. The NPAtlas part of the popularity prior (`pc_np` bit 4, `NP_BONUS`, `NP_W`) is therefore not a violation. The "ineligible" reading of the ADDITIONAL popularity-prior row is withdrawn. What remains open there is the NIST depositor flag.
- **sergiomalv ChEBI/LMSD.** I rate it ineligible, following our own G1 standard rather than the weights-only reading ("unclear").
- **Knowledge-table external spectra.** I rate them unclear: the records are permitted, but they reach us through an NC-labelled compilation with no build script.
- **ICEBERG and GLACIER.** Both are rated the same way: the weights are eligible, and the runner glue is unclear because it has no OSI licence.

## 1. All assets with verdicts

Verdicts: **E** = eligible, **U** = unclear (needs a fix or a host ruling), **I** = ineligible.

**Column key:**
- **NIST:** NIST data or NIST-trained.
- **NC:** non-commercial terms beyond those inherited from train.parquet.
- **repro:** reproducible from public code and permitted data.

| # | asset | role in pubm | licence | NIST | NC | repro | verdict |
|---|---|---|---|---|---|---|---|
| A1 | Competition train/test/sample_submission | library (2,539,608 spectra), queries, pool rows, in-kernel caches | CC BY-NC 4.0, tolerated | no | no | n/a | **E** |
| A2 | Kaggle image (sha256 37c64f7d..., py3.12, torch 2.10) | runtime | OSS (BSD/MIT/Apache) | no | no | pinned | **E** |
| A3 | RDKit 2026.03.3 wheel (metric; v4g copy picked first) | metric canonicalisation | BSD-3 (Kaggle "unknown") | no | no | n/a | **E** (mount metric only) |
| A4 | Simulator wheels in casmi26-iceberg (rdkit 2025.3.6, lightning, torchmetrics, einops) | ICE / GL environment | BSD-3 / Apache-2.0 / MIT | no | no | n/a | **E** |
| A5 | dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets (container) | carrier for engines, pools, models, wheels | **CC BY-NC-SA 4.0** | no | **yes** | copy aggregate | **I** (as a mount) |
| A6 | v4b code: casmi package, fe_v4 families, v1engine, hooks | Engine 1 | MIT (Ahmed Berat Özer; DreaMS port MIT) | no | no | code yes; bundled tables see A10 | **U** (only via A10) |
| A7 | fpnet_full1.pt | Engine 1 retrieval FP bank | weights "other"; code MIT | no | no | partial: reconstructed command, seed and `--holdout` unrecorded | **E** (G4-class caveat) |
| A8 | FPNet A / B (fe_A = fpnet_0, fe_B = fpnet_1) | V1FE model_views; PubChem ModelBank | weights "other"; code MIT | no | no | A: seed 7, logged; B: seed unrecorded | **E** (caveat on B) |
| A9 | fe_D.pt (DreaMS fine-tune) | V1FE model_views | MIT / CC BY 4.0 (DreaMS-derived) | no | no | partial: reconstructed command; interactive EMA re-wrap | **E** (caveat) |
| A10 | fe_v4 tables (fragnet stageA/B, prior_tables, analog_struct, frag2) | ranker_v4b features | MIT / "other" | no | no | **doubtful**: scripts edited after the runs, fragnet `sets` / `fitA` steps have no logs | **U** |
| A11 | ranker_v4b.pkl (4 x 500-tree LightGBM, 160 features) | Engine 1 ranking | "other"; scripts MIT | no | no | partial: "Step 4 (reconstructed)", no training log, not bit-exact | **U** |
| A12 | v3-models: casmi code + fpnet_0/1 (ranker_0.pkl unread) | PubChem channel | code MIT | no | no | as A8 | **E** (mount upstream) |
| A13 | v2-pool (710,701 rows, fp_bits 10,226) | Engine 1 pool; PubChem filters | "other" (train + COCONUT) | no | no | yes: scripts 01/02/05, byte-identical | **E** (mount upstream) |
| A14 | pool_lsid / pool_lpmid | Engine 1 pool popularity re-rank (mu 0.15) | NCBI policy | no | no | yes: published scripts | **E** |
| A15 | PubChem tier pc_mass / off / smiles (105.9M rows) | PubChem channel search space | "other"; PubChem data | no | no | data only; recipe stated, no script | **E** (minor data-build gap) |
| A16 | Popularity-prior channel arrays: pc_lsid / lpmid, pc_src_mask, pc_np, lotus_refs, lsynonyms, delta / growth | PubChem channel prior | "other" (NCBI; LOTUS CC BY; NPAtlas CC BY-NC but host-allowed) | **flag**: SRC_BITS = 1 includes the "NIST Mass Spectrometry Data Center" depositor | no (NPAtlas allowed) | yes: scripts published | **U** (NIST depositor bit) |
| A17 | pool_popularity.csv | final layer POP_W / SPEC_W / NP_W | same dataset | **flag**: `src_spectral_library` is the same NIST-including group | no | yes | **U** (SPEC_W term only) |
| A18 | fold-safe fpnet_fold0.pt | PubChem ModelBank | checkpoint under competition terms; scripts Apache-2.0 | no | no | yes: train_fpnet.py, train log, split published | **E** |
| A19 | Knowledge-table external_reference_spectra (751 spectra / 469 structures) | appended to the Engine 1 library | records CC BY / CC0; compilation labelled NC | screened by text mentions only | carrier yes, records no | no harmonisation script | **U** |
| A20 | Knowledge-table natural_products.parquet; panel benchmark | EDA display only (FZ c16, c18) | "other" (NPAtlas columns) | no | display only | n/a | **E** (detach) |
| A21 | ICEBERG gen.pt / inten.pt + ice_runner.py, fuse.py, shim/ | ICE scores; E006 window-40 fallback; promotion gate | weights MIT (ms-pred, MassSpecGym); **glue: no licence ("other")** | no (msg_all) | no | allowed weights | weights **E**, glue **U** |
| A22 | GLACIER glacier.pt + gl_runner.py, gl_fuse.py, stand-ins | GLACIER H1; fusion re-rank; window-40 overlay | weights MIT (same best.ckpt sha 5a47cecc... as ours); **glue: no licence** | no | no | allowed weights | weights **E**, glue **U** |
| A23 | N1 proxy bench glb_core.py / glb_driver.py | GLACIER proxies for molecules without [M+H]+ | Apache-2.0 (notebook code) | no | no | n/a | **E** (runs on A21/A22 glue) |
| A24 | frag_rescore.py (MetFrag-lite) | FZ c51 gap fill; fragment check | Apache-2.0 (Aksoy) | no | no | n/a | **E** |
| A25 | Frozen 0.465 notebook (FROZEN_B64) | stage 1 of the fork | code Apache-2.0 | via A16 | via A5 / A29-A34 | code pinned | code **E**; **as executed I** |
| A26 | PubChem channel code (CORE, RUNNER, V17_PATCH) | PubChem channel | Apache-2.0 notebook code + MIT casmi | via config (A16) | no | n/a | **E** (set SRC_BITS = 0) |
| A27 | Other frozen overlay code (FZ c43-c54) | fusion, metric audit, window-40, rank-2 overlay | Apache-2.0 notebook code; metric code | no | no | n/a | **E** (outputs inherit Engine 2) |
| A28 | Engine 2 code: pv.py, pv_fp.py, casmi_engine.py, eng_runner.py | Engine 2 subprocess | notebook code (Apache default); pv_fp.py licence on cubres github unverified | no | hard-wires A29-A34 | code yes | **U** (moot if Engine 2 is replaced) |
| A29 | prvsiyan/casmi26-fp-models-v2 (fp_single_s2, fp_merged_m1) | Engine 2 FP logits | **CC BY-NC-SA 4.0** | no | **yes** | **no** ("byte-exact ... not established", seed applied after construction) | **I** |
| A30 | prvsiyan/casmi26-ranker-features rank_train.npz | rows for the prvsiyan ranker | **CC BY-NC-SA 4.0** | n/a | **yes** | **no** (A/B logit-cache lineage unresolved) | **I** |
| A31 | In-kernel "prvsiyan ranker" (8 HistGB) | Engine 2 s_pv (blend 0.88) | inherits A30 / A29 | no | yes | no | **I** |
| A32 | megayak sim_rank_rows_nofp.npz | rows for "our ranker" | CC0 label; built on the NC-SA bank | no | lineage | **no** (simulation driver unpublished) | **I** |
| A33 | In-kernel "our ranker" (4 HistGB) | Engine 2 s_ours (blend 0.12) | inherits A32 | no | lineage | no | **I** |
| A34 | prvsiyan/coconut-casmi26-candidates (436,389 structures + 6,930-bit fp_bits) | Engine 2 pool and bit order | **CC BY-NC-SA 4.0** | no | **yes** | partial | **I** |
| A35 | sergiomalv/casmi-chebi-lmsd-structures | Engine 2 pool (+98,433 structures) | CC BY 4.0; key filter from A34 | no | filter | **no** build code (our G1) | **I** (G1 standard) |
| A36 | franciscoangulo PubChemLite fingerprints (+356,820 structures) | Engine 2 pool | CC0 label is wrong (upstream PubChemLite is CC BY 4.0); pins A34 bits and keys | no | via A34 | partial | **U** |
| A37 | sanpier/casmi-spectrum-model-v1 final_model.pt (+ model.py, features.py, config.json) | neural scorer: consensus, fragment check, final layer | Apache-2.0 | no | no | **yes, partial**: train.py / prepare.py / histories published; prepare.py misses the `normalized_smiles` alias; AMP, not bit-exact | **E** (caveat) |
| A38 | sanpier/catalogue-experiment-saner | attached, not read | "unknown"; ships a stray test.parquet | n/a | unknown | n/a | **E** for this run; detach |
| A39 | Saner's six helper modules | runner, export, neural scoring, RRF | Apache-2.0 (notebook) | no | no | n/a | **E** |
| A40 | Roman Rozen consensus.py / outsiders.py | leader-safe RRF; outside candidates | Apache-2.0 (notebook) | no | no | n/a | **E** |
| A41 | Spectral Parallax physical / provenance | fragment check (provenance off) | Apache-2.0 (notebook) | no | no | n/a | **E** (support votes use Engine 2) |
| A42 | midx final layer (NB c16) + busyaprime / kunaldesale notebook body | final ranks 2-25 | Apache-2.0 (notebook) | via A17 SPEC_W | no | hand constants | **U** (SPEC_W) |
| A43 | bryantyu/casmi26-code, casmi26-models (PF-syn) | not read in pubm; used by the PF and ISO variants | ours; Kaggle shows "unknown" (G19) | no | no | reproduce.md chain; G5 (PF-b init) open | **E** (fix G19) |
| A44 | Embedded tables and pins (FIELD, MODULE_SHA256, MODEL_PINS, N1 pins) | display and integrity | ours | no | no | n/a | **E** |
| A45 | Attached but unread v4g folders (spec2mol, prvsiyan chebi-lipidmaps, hengck23 DreaMS, megayak ours_fpnet / fp16k, v3 ranker_0, top-level pubchem_tier copies, rdkit-wheels) | none | mixed; several CC BY-NC-SA | n/a | some | n/a | **E** only because unread; detach |

**Totals:**
- 9 ineligible: A5, A25 as executed, A29-A35.
- 12 unclear: A6, A10, A11, A16, A17, A19, the A21 and A22 glue, A28, A36, A42 (A6 only through A10).
- The rest are eligible, some with a documented reproducibility caveat of the same class as our G4/G5.
- No asset uses NIST spectra or NIST-trained weights. The only NIST link is a PubChem depositor-group flag (A16, A17).

## 2. Ineligible and unclear components: replacement paths

LB evidence comes from the lineage history:
- **Fork steps** (NB c6, and plan.md Oct 31): base 0.465; saner 0.466 (+0.001); V2 0.468 (+0.002); 0.75 order 0.472 (+0.004); fragment-check leader moves taken from the V2 order 0.477 (+0.005); midx final layer 0.479 (+0.002, claimed; our pubm read lands Nov 1-2).
- **Inside the frozen 0.465 notebook** (FZ c58 version table):
  - 0.450 at its base;
  - spectral-library bonus 0.451 = 0.451 without it;
  - NP bonus +0.000;
  - PubChemLite in Engine 2: −0.003;
  - Engine 2 tree depth 6: +0.007;
  - GLACIER proxies: +0.001;
  - synonyms, supplements and external spectra: +0.000;
  - proxy GLACIER kept where ICEBERG scores exist, plus joint ICEBERG + GLACIER re-ordering: 0.456 to 0.464 (+0.008), from docs/research_oct25.md (the FZ c58 table lists v16 without a score);
  - candidate-specific PubChem promotion: 0.465.
- **Older E006 lineage** (docs/research_oct25.md):
  - pool popularity re-rank: +0.010;
  - Engine-2 rank-2 insert: +0.002 on two bases;
  - Engine 2's pick at rank 1: 0.403;
  - ChEBI/LIPID MAPS: +0.005;
  - ICEBERG fallback: +0.005;
  - GLACIER: +0.011 over two versions;
  - GLACIER restricted to [M+H]+: +0.006;
  - N1 proxies: +0.008.
- **Noise:** one public rank-1 hit is worth ~0.008, and identical reruns differed by up to 0.006.

The fixes are grouped below. F1 cannot ship alone: dropping the container removes Engine 2's inputs, so F1 always goes with F2.

### F1. Mount the original sources instead of the CC BY-NC-SA container (A5; detaches A20, A38, A45)

**Path.**
- New `dataset_sources`:
  - ahmedberatozer: casmi26-v4b-models, casmi26-v3-models, casmi26-fpnet-full1, casmi26-v2-pool, casmi26-iceberg, casmi26-glacier;
  - dmitriigluzdov: casmi26-pubchem-popularity-prior, casmi26-fold-safe-fpnet. The popularity prior already holds the tier copy that `find('pc_smiles.npy')` resolves to.
  - sanpier/casmi-spectrum-model-v1;
  - metric/rdkit-2026-3-3-wheel;
  - the competition data.
  - bryantyu/casmi26-code and casmi26-models only for the PF and ISO variants.
- Drop the v4g container, the knowledge table, the panel benchmark and catalogue-experiment-saner.
- The frozen finders are path globs (`casmi26-v4b-models/**/...`), so they keep working. Patch:
  - FZ c5: V4G, PANEL and TABLE may be None.
  - The display cells FZ c9, c12, c16 and c18: skip them.
  - Re-embed FROZEN_B64 and update the FROZEN_SHA256 pin in NB c8.
- Check that every mounted model or data file hashes to its row in `v43_mounted_input_hashes.json`. The v4b MANIFEST check in FZ c26 already does this for v4b. Pin upstream dataset versions where an author has re-versioned.
- Check `find()` collisions under the shortest-path rule once our datasets are attached.

| | |
|---|---|
| Work | 4-6 h, plus one commit run (~27 min on a T4) |
| Runtime | 0 |
| LB cost | 0. The inputs are byte-identical, and the build is verified by hash. |

### F2. Engine 2 (A28-A36; the inherited parts of A25, A27, A40 and A41)

All seven inputs of Engine 2 fail on licence or reproducibility. The frozen code also asserts that Engine 2 covers every molecule with a non-empty list (FZ c48 L11, c50 L23-27). So Engine 2 must be either substituted or disabled with a patch.

**F2a. Null engine.** Write `eng_lists.json` as Engine 1's own top 40 (schema `{mid: {smiles, keys}}`), after Engine 1 and before FZ c45.
- The asserts hold, RRF of a list with itself keeps Engine 1's order, and the rank-2 overlay can only re-insert Engine 1's pre-fusion leader.

| | |
|---|---|
| Work | 3-4 h |
| Runtime | about −12 min. Engine 2 took 715 s in the smoke run, ~700 s of it in-kernel GBM fits (630 s and 57 s). |
| LB cost | No direct ablation exists. Expected −0.005 to −0.020, central −0.010. |

The evidence for that range:
- Engine 2 entered in the bundled E006 jump (0.384 to 0.440).
- Its rank-2 insert is worth +0.002 on two bases.
- Its ChEBI/LMSD candidates are worth +0.005.
- Its tree depth alone moved the LB by 0.007-0.008.
- Alone it is weak: its pick at rank 1 scored 0.403, and on disagreements the main list was right on ~13 molecules against ~3 for Engine 2.
- It also feeds the ICE union, the fragment check's support votes, the outsiders and the rank-2 overlay (1 insertion in the smoke run).

**F2b. Our stage 1 as Engine 2** (recommended).
- **Path.** Run our retrieval stage 1 in its own subprocess on the test spectra and write `eng_lists.json` (top 40, same schema). The retrieval is the s2v18c / s2v18cue stage 1:
  - pool v18c (`bryantyu/casmi26-pool-v18c`): our COCONUT 10-2026 build (CC0, `scripts/build_coconut.py`), our ChEBI r255 + LMSD build (`scripts/build_bio_structures.py`, G1 closed), PubChemLite v3.3.0 (CC BY, `scripts/fetch_pubchemlite.py`) and PubChem counts (`key_pop.parquet`);
  - FP `fp_model_v1r.pt` (G4 closed);
  - PeakFormer `pf_model_b.pt` or `pf_tpu_f.pt`;
  - deterministic LightGBM `ret_ranker_v18c` / `v18cu`.
- **Replaces:**
  - prvsiyan FPNets → our FP and PF-syn;
  - prvsiyan COCONUT bank and 6,930-bit list → our COCONUT pool and our bit set;
  - sergiomalv and franciscoangulo → our G1 bio set and PubChemLite;
  - rank_train and megayak rows → our rankers, built by `scripts/build_v18c.sh` from permitted data.
- **Eligibility:** every part is eligible per docs/reproduce.md §6. G5 is still open for the PF-b init, and G19 for licences.

| | |
|---|---|
| Work | 10-16 h: new cell, memory check (1.16M-row pool), key and schema check, review |
| Runtime | +22-27 min CPU − 12 min for Engine 2 = **+10 to +15 min**. Basis: s2v18c's 1,249 s for 400 molecules on 4 Kaggle CPUs is the all-in time (its `stage 1 done` clock starts before the 260 s library build), plus ~1 min of score_key re-keying (~16 ms per key, 15.5k keys, in the 4 workers), with the stage-1 part scaled by up to 1.24 for the hidden test's ~1,500 spectra (1,213 visible). The 45-min cap of elig1 leaves a 1.7-2.0x margin. |
| LB cost | Unknown, est. −0.005 to +0.005 against pubm |

On the LB estimate:
- Our lists are reasonably strong on their own: the probes show COCONUT 0.221 and training 0.157, against the fork's 0.252 and 0.155.
- Their evidence (documentation features, PF-syn) differs from Engine 1's, which is what a hedge engine needs.
- Our earlier "second-ranker rank-2 insertion" was refuted offline only for correlated engines inside our own pipeline. It does not predict this case.

**F2c. Clean rebuild of Engine 2 itself.** This needs our own FPNet in the pv_fp architecture, our own bit selection and our own simulated ranker rows. It is 40-60 h plus GPU time. Not recommended before Dec 7.

### F3. NIST depositor flag (A16, A17; the SPEC_W part of A42)

**Path.**
- Set `SRC_BITS = 0` in FZ c26 (CASMI_SRC_BITS) and `SPEC_W = 0` in NB c1.
- To keep the signal instead, rebuild bit 1 as {MassBank Europe, MoNA} only. Use PubChem SID-Map via our `scripts/pc_sid_sources.py`, or the dataset's own `make_source_groups.py` without the NIST row. Then re-align the result to the tier and pool rows and publish it as our dataset.
- Keep `NP_BONUS` and `NP_W`: NPAtlas is host-allowed.
- In parallel, ask the host whether a PubChem depositor-metadata flag that names the NIST Mass Spectrometry Data Center counts as "using NIST". It is not NIST spectra.

| | |
|---|---|
| Work | 0.5 h for the switch-off; 6-10 h for the rebuild (Kaggle internet kernel; local disk ~1 GB free) |
| Runtime | 0 |
| LB cost | SRC bonus ~0: Gluzdov v3 with the bonus scored 0.451, as did main v52 without it. SPEC_W 0 to −0.002: it is one of five parts of the midx step 0.477 → 0.479. |

If the host rules NPAtlas out after all: compute `pc_np` without bit 4 (COCONUT + LOTUS only). The NP bonus was worth ~0 on the LB (v5 0.451 → 0.451).

### F4. External reference spectra (A19)

**Path.** Drop them: gate off `_build_external_reference_cache` in FZ c36. Rebuilding them from the MassBank 2026.03, MoNA and GNPS releases with a CC0 / CC BY filter would take 8-12 h and more than 1 GB of downloads. Not worth it.

| | |
|---|---|
| Work | 0.5 h |
| Runtime | slightly less |
| LB cost | ~0 (at most 0.002). v15 added them together with the supplements at 0.456 → 0.456, and they changed 1 of 12 visible lists. |

### F5. Forward-model runner glue (the A21 and A22 glue; A23 depends on it)

**F5a.** Ask ahmedberatozer to put an OSI licence (MIT, as on his v3 and v4b code) on the casmi26-iceberg and casmi26-glacier glue. The glue is ice_runner.py, fuse.py, gl_runner.py, gl_fuse.py and shim/, about 2,000 lines in our older copy.

| | |
|---|---|
| Work | 0.5 h |
| LB cost | 0 |
| Risk | The outcome depends on his answer. |

**F5b. Our own glue** (if there is no grant by about Nov 7).
- **Input builder:** spectra dicts, including the N1 [M+NH4]+ and [M−H2O+H]+ proxies.
- **GLACIER:** `casmi/fwdrun.py --device cuda` with `fwdshim/` on `bryantyu/casmi26-glacier`: ms-pred @708148c2a8eb (MIT) with the same MassSpecGym best.ckpt; Ahmed's glacier.pt re-saves the same tensors.
- **Fusion:** a within-formula z-fusion re-implemented to the documented gl_fuse semantics.
- **ICEBERG:** write a new runner on ms-pred's ICEBERG inference (the weights are already in our `casmi26-fwd`), or drop ICEBERG.

| | |
|---|---|
| Work | 16-24 h, plus 6-10 h for an ICEBERG runner |
| Runtime | ~0 on the GPU. On CPU our runner took 2,351 s for 22,196 predictions. |
| LB cost | Unknown. ±0.005 if the semantics are reproduced. −0.004 to −0.008 if ICEBERG is dropped. |

Why the cost is unknown:
- Our scores are not identical to the public runner's: isomer top-1 is 0.838 with ours and 0.743 with theirs in our harness. Downstream gates tuned on theirs therefore see different numbers.
- These stages carried real LB gains in the lineage:
  - GLACIER: +0.011;
  - [M+H]+ restriction: +0.006;
  - N1 proxies: +0.008;
  - ICEBERG fallback: +0.005;
  - joint ICEBERG + GLACIER: +0.008.

### F6. Engine 1's learned tables and ranker (A10, A11; A6 through them; the caveats on A7-A9)

There is no eligible replacement short of our own pipeline. That would cost about 0.12 on the LB (0.348 against ~0.465).

**Path.**
- Keep the weights.
- In the fork's reproduce file, document Ahmed's MIT training chain from ahmedberatozer/casmi26-training-code, with his own gap list and the sha256 values.
- Ask the host (thread 748882) whether reconstructed launch commands, unrecorded seeds and post-run script edits meet the reproducibility rule. Our G4/G5 are the same class of gap, so the answer matters for our own line too.
- **Contingency** if the host says no: regenerate the artifacts with his public scripts and deploy them:
  - `res_fragnet.py` on 32,000 spectra;
  - `priors_build.py` on 30,000 COCONUT parents;
  - `simulate_v1ctx.py` → `build_feats.py` → `train_v4_ranker.py` (15,701 sim1x queries; needs FPNet A/B and fe_D features).

| | |
|---|---|
| Work | 1-2 h for the documentation and the question. The contingency is 30-50 h plus days of Kaggle compute, not feasible before Dec 7 on top of F1-F5. It would rely on the "remediation within one week" clause. |
| Runtime | 0 |
| LB cost | 0 if kept. Unknown for regenerated artifacts (not bit-exact). |

### F7. saner neural scorer (A37)

**Path.**
- Make the reproduction demonstrable: run the published train.py, model.py and features.py with the one-line `normalized_smiles` alias and a two-line launcher. Settings: config seed 20261023, 16 x 500 steps, plus the refit, on a Kaggle T4 (~20 min).
- Record the validation metrics against training_result.json.
- Keep the published `final_model.pt`, which is LB-validated and sha-pinned, unless the host asks for our retrain.

| | |
|---|---|
| Work | 3-5 h |
| Runtime | 0 |
| LB cost | 0 if kept. A swap is not free: the lineage lost 0.019 when the neural model changed (e1 0.472 → e3 0.453). |

### F8. Licence and attribution package (all eligible code rows; G19)

**Path.**
- **Code licences:**
  - Add a LICENSE (MIT) for our code. This is the owner's decision.
  - Add a NOTICE file listing the Apache-2.0 sources, which are all notebook code: Gluzdov's frozen notebook, the saner modules, Rozen, Parallax, the midx / busyaprime / kunaldesale cells, frag_rescore, N1, the overlay cells and the PubChem channel.
  - List the MIT sources in the same NOTICE: Ahmed's casmi package, ms-pred and the DreaMS port.
- **Fork documentation:** add `ATTRIBUTION` and `reproduce` files for the fork, built like docs/ATTRIBUTION.md and docs/reproduce.md, with an F-gap list.
- **Dataset licences:** set the Kaggle licences of our datasets.
- **Apache and MIT:** Apache-2.0 §4 allows a derivative work as a whole to carry different terms (here MIT) as long as the notices are kept. If we win, confirm this with the host.

| | |
|---|---|
| Work | 6-8 h |
| Runtime | 0 |
| LB cost | 0 |

### Summary of fixes

| fix | assets | work (h) | runtime | expected LB vs pubm |
|---|---|---:|---|---|
| F1 upstream mounts | A5, A20, A38, A45 | 4-6 | 0 | 0 (hash-verified) |
| F2a null Engine 2 | A28-A36 | 3-4 | −12 min | −0.005 to −0.020 |
| F2b our stage 1 as Engine 2 | A28-A36 | 10-16 | +10 to +15 min net | −0.005 to +0.005 |
| F3 SRC_BITS 0 and SPEC_W 0 | A16, A17, A42 | 0.5 (rebuild 6-10) | 0 | 0 to −0.002 |
| F4 drop external spectra | A19 | 0.5 | ~0 | ~0 |
| F5a licence grant / F5b own glue | A21, A22 glue | 0.5 / 16-34 | ~0 on GPU | 0 / ±0.005 (−0.004 to −0.008 without ICEBERG) |
| F6 keep Ahmed's chain, ask host | A6, A10, A11 (A7-A9) | 1-2 (contingency 30-50) | 0 | 0 |
| F7 saner retrain demonstration | A37 | 3-5 | 0 | 0 (weights kept) |
| F8 licences and attribution | all code | 6-8 | 0 | 0 |

## 3. What an eligible fork should score

- **elig0** = F1 + F2a + F3 + F4: about **0.457-0.476**, central ~0.468.
- **elig1** = elig0 with F2b instead of F2a: about **0.465-0.480**.
- **F5b:** with our own glue, add ±0.005; without ICEBERG, −0.004 to −0.008.
- **On top:** the PF-syn variants (PF_MID_W, ISO) carry over unchanged. They are our own reproducible chain, with the G5 caveat on the PF-b init, so whatever the Nov 1-2 readings show they add also applies to the eligible fork.
- **Compared with our best eligible score** (s2v18cte, 0.348): the eligible fork should land 0.11-0.13 higher.
- **Caveat:** all of this rests on the claimed 0.479 of pubm, which our own LB read only confirms on Nov 1-2.

## 4. Recommendation

**An eligible version of the fork is realistic before Dec 7, with one open risk.**
- The fix needs no new science: config switches (F3, F4), re-mounting (F1), one engine swap (F2b), and possibly a runner rewrite (F5b).
- That is about 35 h of work without F5b and 55-65 h with it, plus 6-8 LB slots, over 2.5 weeks.
- **The open risk:** Engine 1's learned parts (ranker_v4b, the fe_v4 tables) and the partially reproducible FPNets and saner. Their training code is public, but the commands are reconstructed and the fits are not bit-exact.
  - Only the host can rule on that.
  - It is the same class of gap as our own G4/G5.
  - Replacing Engine 1 would give back most of the gain.
- **The selection protocol** sets the real deadline:
  - both picks must have been scored at least 72 h before the Nov 23 selection;
  - no pick may contain an ineligible component.
  - So the eligible fork must be final and scored by **Nov 19-20**, not Dec 7.

**Order of work:**
1. **Nov 1, 1 h, no compute.**
   - Ask the host (748882) about two things: (a) reconstructed or not-bit-exact retrains (Ahmed's chain, saner, our G4/G5); (b) the PubChem depositor flag that names the NIST MS Data Center.
   - Ask ahmedberatozer for an MIT licence on the casmi26-iceberg / casmi26-glacier glue (F5a).
2. **Nov 1-3: build `casmi26-pubm-elig0`** = F1 + F3 + F4 + F2a, as a new `scripts/make_pub_variant.py` mode.
   - Verify it by commit run and hash check.
   - One LB slot: elig0 against the pubm control. This measures the whole eligibility cost directly, Engine 2 included.
3. **Nov 3-7: build `casmi26-pubm-elig1`** = elig0 with F2b (our stage 1 as Engine 2).
   - LB read paired with elig0.
   - Keep the better of the two as the eligible base.
   - Add the PF-syn PF_MID / ISO settings the Nov 1-2 readings support.
4. **Nov 7-12: F5b**, only if F5a has no answer by Nov 7.
   - GLACIER through `casmi/fwdrun.py` first; check score equivalence on the 12 visible molecules and on our validation set; then an LB read.
   - ICEBERG runner, or drop ICEBERG, by measured cost.
5. **Nov 8-14: F7 and F8.**
   - F7: saner retrain demonstration.
   - F8: fork ATTRIBUTION / reproduce files with the F-gap list, Ahmed's chain documented; LICENSE and NOTICE (owner); Kaggle licences (G19).
   - Close the PF-syn chain in reproduce.md §7.
6. **By Nov 19-20:** final eligible-fork version scored.
7. **Nov 23 selection:**
   - Pick #1: the eligible fork.
   - Pick #2: our own v18c-lineage build, as insurance if the host rejects Engine 1's reconstructed chain (F6).
   - Never select pubm or any variant that still mounts the v4g container or runs Engine 2 as published.
8. **Only if the host rules against Ahmed's chain:** start the F6 contingency (regenerate with casmi26-training-code) under the one-week remediation window. Until then, our own line stays the eligible pick.
