# Plan to close the COCONUT ranking gap (+0.061) and the merge loss (+0.03) in our own code

All code is our own: the public sources are read for ideas only. One public LB submission covers only 124 molecules, about ±0.03 noise, so it can only show effects of about 0.05 or more. Each step is therefore decided offline, and the LB is used for one combined check plus source probes.

## 0. Validation harness (1 h, Oct 21, before building anything)
- **Primary metric:** stage-2 5-fold CV (`scripts/train_stage2.py cv`), reporting cls2w and cls2 deltas over at least 2 fold seeds with intervals. Guard: cls1 must not drop by more than 0.003.
- **Component metrics:** on class-2 COCONUT-not-train truths, within-formula MRR and pairwise wins against train and COCONUT decoys.
  - Baseline: fp_cos .408 / .947 / .893; frag top-30 MRR .359.
  - Our formula groups average about 80 candidates, so gates compare against our own baseline, not the public numbers.
- **Offline merge loss:** for class-2 COCONUT truths, MRR of the COCONUT-only list minus MRR of the full list, using the same scores. Also record which source (train, PubChemLite, ChEBI) holds each decoy that outranks the truth. Record now and recompute after every step.
- **LB read-out:** the probe5-{train,coco,other} results (Oct 21) give our like-for-like source split. Write predictions into `docs/plan.md` before reading the scores.

## Priority 2, done first because it is cheap and CPU-only: fragment evidence v2 (5 h, Oct 21)
- **New `casmi/fragscore.py`**, computed per query spectrum on the stage-1 top 30 and used as stage-2 features:
  - depth-mixed explained share (1 cut = 1, 2-cut only = 0.6, 0.005 Da, H shifts -2..+2);
  - intensity-weighted 1-cut cleavage coverage;
  - rank and gap within the query and within the formula group.
  - Base it on the head-to-head measurements in `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/fragh2h.py`, not on the public `frag2.py`. Cost is about 0.15 s per query on 4 processes.
- **Leave out:**
  - the adduct-carrier and per-spectrum binary variants (measured -0.001 to -0.003);
  - excl_gain/excl_loss and the `*_vs_lead` columns;
  - formula-relative columns (noise-level in both pipelines).
- **Gate:** cls2w at least +0.004 over 2 seeds, with cls1 not worse than -0.003 (depth + coverage measured cls2w +0.006 [-0.001, +0.012] and cls1 -0.003, so watch cls1).
- **Expected:** +0.002 to +0.006 LB. It gets no LB slot of its own; it ships with Priority 1.

## Priority 1, the main lever: FP v5, a peak-token transformer scored by f.z (28 h; CPU prep Oct 21-23, GPU from Oct 24)
- **a. Data, 6 h CPU.** New `scripts/prep_fpnet_data.py`:
  - 7 RDKit blocks (ECFP4/6, FCFP4, path6, AtomPair, Torsion, MACCS) on pool-v4 SMILES;
  - keep bits whose training-structure frequency is between 0.5% and 99.5%;
  - 63 negatives from the ±10 ppm window per structure (same formula first, random fill when the window has fewer than 8);
  - an index of merge peers per (structure, polarity);
  - rank_keys and val_keys never trained on, so stage-1 FP features stay out-of-sample.
- **b. Model and trainer, 8 h.**
  - `casmi/fpnet_own.py`:
    - inputs: exact-m/z and precursor-loss sinusoidal embeddings (shortest wavelength 0.01 Da), linear intensity, and a global token for precursor, adduct, CE, n_merged and polarity;
    - pooling: CLS plus mean;
    - outputs: a bit head and an element-count head.
  - `scripts/train_fpnet.py`:
    - loss: softmax CE over f.z of the truth plus 63 negatives, plus BCE, plus element-count MSE;
    - augmentation: merge 2-4 peer spectra with p = 0.4, peak dropout, ppm and intensity jitter, CE masking;
    - sampling: half the batch sampled per structure.
  - Smoke-test on a 5% CPU subset before the GPU quota refreshes.
- **c. GPU, kernels/casmi26-fp-train, about 2 runs of 5 h.**
  - Run seed 1 with the full config, then check the component gate.
  - Gate: within-formula MRR on class-2 COCONUT truths at least .46 (baseline .408) and wins against COCONUT decoys at least .93 (baseline .893).
  - Kill: below .43 / .91. Then spend one ablation (current 2215-bit target) to tell whether the architecture or the target is at fault, and stop.
  - Train seed 2 only if the gate passes.
- **d. Features and retrain, 8 h.**
  - `casmi/fpscore.py`: per-spectrum mean logits and per-polarity merged logits.
  - `casmi/own.fp_feats`: fz, fz_rank, fz_gap, fz_z, fz_softmax, group entropy, top-2 gap, fz_single/fz_merged, fp_ll, per-block f.z rank, and rank spread (between the 2 seeds, or between the binned model and the transformer).
  - Pipeline: `scripts/add_fp_feats.py` on ret10 rows, then stage-1 CV, out-of-fold scores, GLACIER only on the top-30 entries that changed, then stage 2.
- **e. Inference.** Compute candidate blocks with RDKit for window candidates only, or ship a packed bit dataset. Check the 9 h notebook budget.
- **Expected:**
  - Oracle stage-1 CV with a +0.03 truth shift gave +0.025 all, +0.03 cls2 and +0.10 cls2w.
  - A realistic build should give +0.01 to +0.03 LB.
  - This is also the only verified mechanism that acts on the merge loss: f.z separates truths from train decoys better (public .964 vs our .947), and those decoys are cross-source isomers.

## Priority 3, optional (8 h, after Priority 1): learned cut-class LLR
- Build it only if, after Priorities 1 and 2, the class-2 COCONUT errors are still mostly same-formula isomers.
- Count tables keyed (polarity, cut class) -> match class, with Dirichlet backoff, trained on split-0 spectra against same-formula pool decoys, val keys held out.
- Expected +0.002 to +0.004. Gate on stage-2 cls2w only.

## LB use (5 slots per day)
- **Slot A, about Oct 27-28:** Priorities 1 + 2 against the best compliant entry (s2v4 0.330 / v10). Expect +0.015 to +0.035; only +0.05 or more is conclusive.
- **Slots B and C, same day:** a COCONUT-restricted probe and the full list, using the probe5 key sets.
  - Targets: restricted COCONUT MRR from 0.191 to at least 0.22, and merge loss at most 0.03.
  - Reuse the train and other probes unless those lists changed.
- **If merge loss is still 0.03 or more after Priority 1:** no verified mechanism covers the rest, since the list-composition ideas were rejected as causes. Break the offline merge loss down by decoy source before building anything new.

## Order and total
Oct 21: harness + fragment v2 (6 h) -> Oct 21-23: FP v5 steps a-b (14 h) -> Oct 24-26: steps c-e (14 h) -> Oct 27-28: LB -> optional Priority 3 (8 h). Total is about 34 h without Priority 3 and 42 h with it.

## Verdicts per mechanism (workflow wf_2a12e406-61f)

- **Spectrum-to-fingerprint transformer scored as f.z, trained with a pool-window contrastive loss over a multi-block fingerprint, with merged and per-spectrum views and a model ensemble** (ranker): +0.02 to +0.04 public (the largest single lever for the COCONUT restricted gap of 0.252 vs 0.191). Note: the earlier bound in plan.md (+0.07 SD on fp_cos gives about +0.01) tested a much smaller improvement than this measured gap, inside a ranker that leans on priors.
- **Learned fragmentation scorer (count-based cleavage-class likelihood ratio plus a small LambdaRank) and fragment contrast against the leading candidate** (ranker): +0.005 to +0.015 public. Part of this overlaps with GLACIER on [M+H]+ molecules; the gain should be larger on negative-mode molecules and other adducts.
- **Fragment evidence: per-spectrum, adduct-aware, depth-weighted and learned (bond-class LLR) explanation instead of one merged binary explained-intensity score** (spectral): +0.005 to +0.015 (public measured +0.016 on NP queries over a stronger baseline; partly overlaps with our GLACIER stage 2, which covers about 91% of molecules)

Rejected:
- **Separate candidate sets per source, with outside sources limited to fixed slots (main list from training structures plus COCONUT only)**: The public code does work this way, and we only partly have it. But on our current pool v4 the mechanism accounts for almost none of the remaining COCONUT gap. The crowding it fixes was mostly a pool-v3 problem, and moving to pool v4 already removed most of it.

1) What the public code does. Verified, with corrections.
- The main engine searches only the 710,701 train+COCONUT structures (cell 70; pool_meta src 2/1/3 = 436,515 / 248,440 / 25,746).
- The window is 10 ppm, falling back to 30 ppm (pool.py:38-42; engine.py:32-33 and 210-216).
- The main list is cut to 60 distinct keys (cell106:26-3
- **Deeper shortlist before the forward-model stage (public carries 60 + 40 identities; we carry 30)**: The public pipeline does not send 60 main-engine identities into the forward-model stage, so a deeper shortlist cannot explain why it beats us on COCONUT. Our recall numbers are correct, but the depth itself would add very little.

1) What the public code actually does with the 60:
- TOPN=60 is real (cell93_main.py:13, cell106_main.py:36). At depth 60 the only rerank is the popularity one, z(score)+0.15*pop (cell106_main.py:41-52). That rerank favours documented compounds, which works against undocumented COCONUT truths.
- In cell108, `BASE_ICE = False` (line 9). So line 65 yields an empty can
- **Key the pool and training labels by metric identity (tautomer-canonical InChIKey14), not raw InChIKey14**: The code checks hold up, but the claimed effect on COCONUT does not. A direct test shows no gain.

PUBLIC CODE (confirmed): chem.py:203-214 score_key does TautomerEnumerator.Canonicalize and then InChIKey14. structs.py stores key=score_key alongside rawkey. traindata.py:22-25 maps sid to pid through that key. /home/user/pub/pool/pool_meta.parquet has 710,701 rows with unique score keys. Its train rows (src 1+3) number 274,186 against 275,810 raw train inchikey14, so about 1,600 train tautomer twins were merged. cell106_main.py:33-35 deduplicates by C.key after scoring, which is the same post-r
- **Provenance-free evidence ranker with documentation added afterwards as z(score) + 0.15*pop**: I checked the public code, our code and the logs, and ran one offline CV (read-only, results in scratchpad). Bottom line: the public lineage does work roughly as claimed, and we don't run that exact combination. But it is unlikely to explain the COCONUT gap. The cited evidence doesn't support it and my CV contradicts it.

1) PUBLIC CODE: mostly confirmed, with caveats.
- engine.py:66-79 has exactly 94 base features: spectral, structural and query-level, with no documentation counts. The MANIFEST says 94 base + 66 family features = 160.
- fe_v4/ctxlib.py:9-12 states the rule 'no provenance: nev
- **Source-tiered list composition: PubChem kept out of the ranked pool, added only through a library gate, capped slots and a margin-only promotion**: The public code does this. We partly lack it. But the claim that it explains part of the COCONUT gap fails a check on our own data, and a re-implementation is net negative in simulation.

1) PUBLIC CODE: confirmed, with corrections.
- cell93_main.py:12-14 sets LIB_TAU=0.9, REL_TH=600, SLOTS_AGG [2,4,6,8,10] and SLOTS_GENTLE [4,8,12,16,20].
- In cell110_main.py, merge() at 2-13 interleaves PubChem items at those slots. promote() at 28-34 puts PubChem #1 at rank 1. The gate at 54-68 skips the merge when lib_max>=0.9.
- The PubChem score is zf+0.25*pop at cell101_V17_PATCH.py:68-69, not 42-69. S 
- **Reciprocal-rank fusion with an independent, documentation-blind, class-2-weighted second engine (one vote per metric identity)**: VERDICT: the code facts are mostly right and we really do lack this. But the mechanism offered for why it helps on COCONUT does not survive checking, and our own CV shows fusion does not move COCONUT-restricted ranking.

WHAT CHECKS OUT
- RRF fusion: s = 1/(3+r_main) + 0.6/(3+r_eng2), top 40, in cell112_main.py:1,15-25.
- One vote per metric identity: metric_fusion in cell116_main.py:4-20, called at cell119_main.py:55-62. The final call (bottom of cell119) uses window=40 and ice_fallback=True.
- Engine 2's pool is COCONUT + ChEBI/LMSD + training structures, never PubChem (casmi_engine.py:145-1
- **Documentation as a fixed, core-scoped prior on top of doc-blind rankers (z(score)+0.15*pop on train+COCONUT only), not as learned features**: The cited public lines are accurate, but the mechanism is described wrongly, and it cannot explain the gap it targets.

1) What the public code does. cell106_main.py:6-7 and 39-51 z-score the top-60 main-ranker scores and add 0.15*(log1p SID + log1p PMID) to pool candidates only; generated candidates keep their slots. cell105 credits this with +0.010 on the LB. The main ranker's FEATURES (engine.py:66-79) and the 66 family features in MANIFEST.json have no PMID/SID columns, and neither does the second engine (cell99_ENG_FILES_pv.py rank_features). So the public rankers are doc-blind: TRUE.

Th
- **Analog search construction: merged per-polarity query, entropy-hybrid similarity on uncapped peaks, per-library representatives with NP-library and zero-shift propagation features, 80-100 analogs**: PUBLIC CODE: the citations are accurate. engine.py:26 NP_LIBS (riken, gnps, massbank, mona, spectraverse, msdial, masaryk; enveda-np-examples and pluskal are left out). :36-37 n_analog=100, analog_query='merged'. cell104 runs EngineCfg(generate=True) with these defaults. :114-124 keeps one representative per (sid, pol, lib) with an is_np flag. :130-136 builds one merged query per polarity. :161-169 runs search_shift with a 4*n_analog shortlist and keeps the best per sid, so is_np comes from whichever representative scored highest. :316-326 computes sim^4 ap, mean_tan, ap_sum, n_hi, tan0 and th
- **Tautomer/representation-style invariance of structure comparisons (analog Tanimoto propagation, and the FP-model candidate side)**: The asymmetry is real in our code, but the public lineage has the same one in its fingerprint features, so it cannot explain the composition gap (0.013 vs >= 0.042). The public code does not do the proposed fix, and its own style-invariant features measured near zero on the COCONUT side. My ranking simulation (scratchpad chk3.py) did not finish, so the verdict rests on the code reading, chk1 and chk2.

1. Cited lines and what the public code does. analog_struct.py:5-9 and 53-57 and fragnet.py:5-15 say what is quoted. The public lineage does not tautomer-canonicalise any fingerprint, though:
- 
