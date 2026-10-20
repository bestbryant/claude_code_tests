# Decision tree after the public-notebook source probes (written Oct 20)

Paths are relative to /home/user/claude_code_tests unless they are absolute.

**Read-out (0.5 h, as soon as probe5-* are scored).**
- For each source s in {train, coco, other}: Δs = fprobe-s − probe5-s. Both use the same srckeys and both restrict the final top-25 lists (run_own cuts at 25; fprobe cell 128 filters submission.csv), so they are like-for-like.
- Composition: ρ = full / (train + coco + other), computed for each pipeline (fork 0.451, ours a0.5 0.323).
- A Δ is real only if it is ≥ 0.04. Each restricted probe is about ±0.02.
- Write the predictions into plan.md before reading the scores. The fork's own PubChem-only list scored 0.037, so expect Δother ≤ 0.02 and Δcoco to be the largest. Branch A or D is the likeliest.

## A. Δcoco ≥ 0.05: the fork keeps or ranks COCONUT answers better (likeliest)
1. **Demotion A/B (2 h of work + 1 h kernel run):**
   - Change: in `casmi/run_own.stage2`, subtract d × (per-molecule SD of the stage-2 score) from PubChemLite-only candidates (in_pcl and not in_train/in_coco/in_bio; all these flags are already in the frame). Env `SRC_DEMOTE`. Kernel `casmi26-own-s2src` with d = 1.0.
   - Decide on the LB only. CV cannot judge it, because about 98% of reachable simulated class-2 truths are in COCONUT.
   - Expected 0 to +0.04. This is the only cheap action that could be ≥ 0.03.
   - It fixes only the part of Δcoco where our top 25 crowds the answer out, not the order inside COCONUT.
2. **Bound the fingerprint lever before building anything (1.5 h; `scripts/cv_stage1.py`, `train_stage2.py cv`):**
   - Replace fp_cos with a synthetic score whose top-1 within formula groups on rare natural products is 0.45 (now 0.34).
   - Go to step 3 only if the unweighted class-2 stage-2 CV gains ≥ 0.02.
3. **FP v5, our own peak-set transformer:**
   - Features: exact-m/z and neutral-loss sinusoidal embeddings. Targets: Morgan r2/r3 + RDKit path + MACCS, about 6k bits.
   - Our own code only, no prvsiyan code (CC-BY-NC-SA).
   - GPU after the Oct 24 quota refresh. 20–30 h including add_fp_feats, OOF, incremental GLACIER and stage 2.
   - Expected +0.01 to +0.04, and probably the low end: swapping the fork's own fingerprint bank moved its LB only 0.380 → 0.378.
   - Do not resume fp_v1: fp_v2 already did that and plateaued at 0.548.

## B. Δother ≥ 0.04: the fork's PubChem channel
1. **Split 'other' (1 h of work + 2 fork GPU runs, 2 slots):**
   - Add src='pool4' to `/home/user/work/srckeys_ds/casmi_probe_srckeys.parquet`: the 433,920 keys in `/home/user/work/pool4/pool.parquet` with in_train = 0 and in_coco = 0. Pool keys are standard InChIKey14 (all 300 sampled matched).
   - Build kernels fprobe-inpool4 and fprobe-outpool4 by copying fprobe-other cell 128.
   - Also submit the existing `casmi26-probe-pubchem` (our documented-PubChem tier; 0 h, 1 slot).
   - Gain 0; this decides reach versus ranking.
2. **Then one of two branches:**
   - **2a. out-pool4 ≥ 0.03 or probe-pubchem ≥ 0.02 (reach):** cheap channel, 10–12 h, no new store, no retraining.
     - Candidates: the pool-v3 rows already on disk (`/home/user/work/pool3`), scored by our fp_cos as z(fp) + 0.25·pop.
     - Merge: fork-style gate. Always fill tail slots 21–25. Promote to rank 1 only if lib_max < 0.9, S ≥ 6 and pop ≥ 5.
     - Expected +0.01 to +0.03.
   - **2b. Mostly in-pool4 (ranking):** apply the same promotion gate to our existing PubChemLite/ChEBI candidates, post hoc. 3 h, 0 to +0.03.
3. **Full-PubChem store on Kaggle CPU (about 70 h).** Only if 2a moves the LB by ≥ 0.02 or out-pool4 is ≥ 0.04.
   - Weight the PubChem-only simulated class toward non-Enveda (natural-product) truths.
   - Inject a simulated truth only if it survives the shortlist; count misses as unreachable.
   - Expected +0.015 to +0.04. Downside −0.02 (what pool v3 cost).

## C. Δtrain ≥ 0.04: class 1
1. **Library-cap fix (2.5 h; ship it in every branch, class 2 changes by −0.002):**
   - The library-match channel uses all spectra instead of lib10 (`LIB_CAP` in `casmi/run_own.py`).
   - Keep lib_n on the old scale: Σ min(n, 10).
   - Re-check stage 2 on the GLACIER rows (20 min). Kernel `casmi26-own-s2v9lib`.
   - Expected +0.005 to +0.02, more if the class-1 share is ≥ 28%.
2. **Stage-2 class-1 diagnostic (1.5 h):**
   - Run GLACIER on the stage-1 top 30 of the test-like validation set (about 15k predictions).
   - If stage 2 demotes library hits: keep the stage-1 order when lib ≥ 0.9 and it leads by ≥ 0.1.
   - 0 to +0.01.
3. **If Δtrain is still ≥ 0.05 after s2v9lib:**
   - LB A/B of the fork's hard rule (lib ≥ 0.9 → rank 1): 1 h, 1 slot. Simulation says −0.02 on class 2; it pays +0.01 to +0.03 only at a class-1 share of about 28%.
   - Then select spectra after hiding in `scripts/dump_ret_rows.py` and retrain with test-like rows on 5-fold CV: about 1 day, +0.003 to +0.01.

## D. All Δ < 0.04 but ρ_fork − ρ_ours ≥ 0.10: we bury answers under other-source isomers
1. **Demotion A/B at two doses, d = 0.5 and 1.0** (2 h, 2 slots). 0 to +0.04.
2. **If a dose wins: the task #13 gate** (2.5 h, 1 slot).
   - PubChemLite-only candidates stay out of ranks 1–3 unless their stage-2 score beats the best train/COCONUT/ChEBI candidate by m.
   - Otherwise they are interleaved at ranks 4, 8, 12…
   - About +0.01 on top of the winning dose.
3. **Our lists cut to the top 1** (1 h, 1 slot). About 0.30 means the rest of the gap is reach (go to B1). 0.26 or lower means the answers sit at ranks 2–25 (go to A2).

## Mixed or other outcomes
- **Several Δ ≥ 0.04:** do C1 first (cheap, ships anyway), then A1/D1, then B1, then A2–A3.
- **Δother ≤ −0.03** (our PubChemLite/ChEBI candidates hold answers the fork misses): use only d = 0.5 and never drop PubChemLite.
- **No Δ ≥ 0.04 and ρ similar:** the gap is in the fork's fusion arithmetic. Stop chasing the LB, ship C1, run the A2 bound, and decide FP v5 on CV.

## Independent of the probes
- Stop deciding on "cls2 weighted"; the no-prior anchor contradicts it. No more slots for alpha or DreaMS (s2v8e) variants: their effects are ≤ 0.01 and invisible on the LB.
- **Side check (3 h, no slot):** download only `enveda-180-filtered.parquet.gz` from Zenodo 21346580 (366 MB; 4.7 GB of disk is free). Check whether its 816 structures that are not in train are in COCONUT or PubChem. If they are the removed COCONUT overlap, add their spectra to the library and use them as a real validation set.

## Slots
- **Oct 21 (5):**
  1–3. probe5-train, probe5-coco, probe5-other. Required by every branch.
  4. s2v9lib (C1).
  5. fprobe-outpool4 if fprobe-other alone is ≥ 0.06 (well above the fork author's 0.037). Otherwise s2src at d = 1.0 (A1/D1).
- **Oct 22 (up to 5), by branch:**
  - A: s2src d = 0.5 (or d = 1.0 if not yet in), plus our top-1 cut.
  - B: fprobe-inpool4 plus probe-pubchem.
  - C: the library-gate A/B.
  - D: the second dose plus the #13 gate.
- Leave the remaining slots unused. Never spend one on a variant with expected |Δ| < 0.02.

_Scores of the four proposal angles (judge, 0-10 value per hour): pool 3.5, np_isomers 3, class1 6, validation 4._
