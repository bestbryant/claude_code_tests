# CASMI 2026 (Enveda) — working plan and log

Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
Task: up to 25 ranked SMILES per test molecule; MRR@25 on tautomer-canonical InChIKey14 (RDKit 2026.03.3).
Hidden test ~400 molecules (~1,500 timsTOF spectra), public LB = 31% (~124 molecules), private 69%.
Code competition: notebook ≤ 9 h, no internet. Deadline Dec 14 2026 (merge/entry Dec 7).

## Working rules (session)
- Never end a turn with work remaining without arming the next self check-in (send_later): the owner is away
  and should not have to say "keep going". Submissions open 00:00 UTC (5/day).

## Rules that shape the solution
- Prize-eligible solutions may not use NIST or models trained on NIST; no non-commercial-only data.
- Every pretrained weight must be reproducible from public training code + permitted data
  (host, thread 748882/748065). Hosts "reserve the right to check all submissions".
- Allowed explicitly: train.parquet, MassSpecGym-trained ICEBERG/GLACIER, DreaMS, FRIGID, CFM-ID 4, PubChem
  (+ CID-SID/PMID/patent counts), COCONUT, LOTUS, NPAtlas, ChEBI/LIPID MAPS, GNPS/MassBank/MoNA releases, Enveda-180 Zenodo.
- Hidden test positive-mode m/z carry an electron-mass correction (train enveda-180 does not: subtract 0.000548 Da).

## Classes (host definitions) and what we know
1. structure has public spectra (library search works, MRR ~0.93)
2. structure in PubChem or COCONUT, no public spectra (database retrieval + ranking)
3. structure NOT in PubChem (must be generated)
Community LB-probe estimates of the mix: 16/45/39% or 28/20/52%. Public pipelines (≈0.45) barely score class 3.

## Key findings (this repo)
- 16,917 training structures (6%) are absent from PubChem (GNPS 21%, spectraverse 19%, enveda-180 3%).
  They are the closest available proxy for class 3: real measured molecules with no PubChem entry.
- 45% of them hydrolyse (one amide/ester bond) into two PubChem-known pieces: N-acyl amino acids/amines,
  bile-acid amidates/esters, FAHFAs — "reverse metabolomics" style hypothesised natural products, exactly the
  host's description of the test chemistry. 88% of those have both pieces among other training structures.
- Transform-table derivation from the top-20 spectral analogs (modified cosine) reaches 24% of absent structures
  (29% of COCONUT natural products). Most misses have a close analog (median Tanimoto 0.66) and differ by CH2
  homologation or isomerisation -> transform table extended (chain_ext/short, C-methyl, double-bond shift, ...).
- Modification-site localisation (parent peaks shifted vs unshifted in the query, explained by in-silico fragments
  of the parent) raises derivative MRR on the absent set from 0.061 to 0.107 (n=300).
- NovelEngine (derivatives + conjugates, PubChem/train-known candidates removed): MRR 0.102 on absent proxy,
  90/297 truths in a top-25 list.

## Submissions
| date | kernel | what | public |
|---|---|---|---|
| 10-17 | casmi26-s2s-fork v1 | unchanged fork of dmitriigluzdov s2s (E006 lineage, v51 = 0.450), image pinned | 0.451 |
| 10-17 | casmi26-known-probe | engine candidates that ARE known, site+analog ranking | 0.085 |
| 10-17 | casmi26-s2s-np | fork + q_npeaks on cleaned peaks | 0.451 (no change) |
| 10-18 | casmi26-own v2 | OWN compliant pipeline v1 (pool train+COCONUT+PCL, lib/analog/frag/priors/dual encoder, LambdaRank) | 0.295 |
| 10-18 | casmi26-own v5 | OWN v3: pool v3 (+2.12M PubMed/patent-documented PubChem), ranker v2 | 0.295 (-0.020: more decoys) |
| 10-18 | casmi26-own v4 | OWN v2: + PubChem doc-count priors (own FTP build), ranker v2 (class-1 x2) | 0.315 |
| 10-18 | casmi26-own v3 | same, but the kernel mounted a stale pool version (all rows failed) | 0.000 |
| 10-18 | casmi26-own-noprior v2 | same without documentation priors | 0.245 |
| 10-17 | casmi26-s2s-np-eu | fork + q_npeaks + electron undo on rerun | 0.430 (-0.021: do NOT undo the correction) |
| 10-17 | casmi26-novel-probe v2 | ONLY novel candidates (class-3 reach probe) | 0.010 |
| 10-19 | casmi26-own-v2c v1 | OWN stage 1 only: ranker v2c (artifact-free), pool v2 | 0.307 |
| 10-19 | casmi26-own-s2 v2 | OWN v4: pool v2, stage 1 v2c, stage 2 with own GLACIER runner | 0.317 (+0.010 GLACIER) |
| 10-19 | casmi26-own-s2v4 v1 | pool v4 (+52k ChEBI/LIPID MAPS, in_bio flag), stage 1 v4c, stage 2 GLACIER | **0.330** (best compliant) |
| 10-19 | casmi26-probe-train v1 | pool-v3/ranker-v2 lists restricted to training-structure candidates | 0.146 |
| 10-19 | casmi26-probe-coco v1 | same, restricted to COCONUT candidates | 0.191 |
| 10-20 | casmi26-own-s2v5w-a05 v1 | pool v4, doc-shift weighted class-2 simulation alpha 0.5, stage 2 GLACIER | 0.323 |
| 10-20 | casmi26-own-s2v5w v1 | same, alpha 1 | 0.332 |
| 10-20 | casmi26-fprobe-train v3 | public 0.451 lineage restricted to training structures | 0.155 |
| 10-20 | casmi26-fprobe-coco v3 | public lineage restricted to COCONUT-not-train | 0.252 |
| 10-20 | casmi26-fprobe-other v3 | public lineage restricted to other candidates | 0.057 |
| 10-21 | casmi26-pc-only v1 | PROBE: PubChem channel only (documented tier outside the pool, PeakFormer local epoch 2) | 0.006 |
| 10-21 | casmi26-probe5-other v1 | PROBE: own alpha-0.5 lists restricted to other candidates (pool v4) | 0.007 |
| 10-21 | casmi26-own-s2v10 v1 | v10: full-library matching + alpha 0.5, stage 2 GLACIER | 0.314 |
| 10-21 | casmi26-own-s2v11l v1 | v11l: v10 + PeakFormer features (local), alpha 1 | 0.320 |
| 10-21 | casmi26-own-s2v11lpc-ad v1 | v11l + PubChem channel, adaptive slots (list confidence > 0.7: 1,3,5,7,9 else 4,8,..,20) | 0.307 |
| 10-22 | casmi26-enam-only v1 | PROBE: Enamine tier only (4.56M Enamine-deposited compounds outside train + COCONUT), PeakFormer order | **0.037** |
| 10-22 | casmi26-pcfull-only2 v1 | PROBE: full PubChem tier only (110M, two-pass, train + COCONUT excluded) | 0.003 |
| 10-22 | casmi26-own-s2v12 v1 | v12: v11l + depositor features + vendor-target class-2 weights | 0.312 |
| 10-22 | casmi26-own-s2v12purch v1 | v12 + purchasable-tier channel (slots 4..20, 2..10 if rel > 15, #1 promoted if rel > 60) | 0.294 |
| 10-22 | casmi26-own-s2v12purchrr v1 | same + channel re-ranker | 0.296 |
| 10-23 | casmi26-purch-only-rr v1 | PROBE: purchasable tier only, channel re-ranker order | 0.034 |
| 10-23 | casmi26-own-s2v11lprop95 v1 | v11l + promotion only (re-ranked #1 to rank 1 if rr_top > 3.79) | 0.310 |
| 10-23 | casmi26-own-s2v11lprop98 v1 | v11l + promotion only (rr_top > 4.6) | 0.316 |
| 10-23 | casmi26-probe11-coco v1 | PROBE: v11l lists restricted to COCONUT-not-train | 0.190 |
| 10-23 | casmi26-probe12-coco v1 | PROBE: v12 lists restricted to COCONUT-not-train | 0.187 |

## Enamine tier (Oct 21): Enveda's library is Enamine's catalogue
The test is 100% timsTOF = Enveda's instrument (training: enveda-180 1.15M spectra / 183k structures, enveda-np-examples).
PubChem depositors (scripts/pc_sid_sources.py streams SID-Map: 335M deposits with a CID, 893 sources), share of keys
with a deposit from the source:
| population | Enamine | AKos | ZINC | Jena (COCONUT) | ChEBI |
|---|---|---|---|---|---|
| enveda-180 training structures (178k) | **0.948** | 0.706 | 0.573 | 0.000 | 0.000 |
| Zenodo extras absent from train (3.6k; Enveda, enamine_catalog_id) | **0.943** | 0.642 | 0.552 | 0.000 | 0.000 |
| COCONUT-not-train (428k) | 0.027 | 0.276 | 0.320 | 0.792 | 0.133 |
| ChEBI/LMSD-not-train (64k) | 0.071 | 0.229 | 0.260 | 0.010 | 0.743 |
| random PubChem (300k) | 0.040 | 0.191 | 0.110 | 0.007 | 0.002 |
-> an Enveda-measured molecule outside COCONUT is almost surely an Enamine compound (likelihood ratio ~24 vs PubChem
at large). Enamine-deposited PubChem keys: 4.88M (339k in pool v4, 189k training structures). 484 of 955 class-2
validation queries have an Enamine truth, 236 of them unreachable through COCONUT / PubChemLite / ChEBI-LMSD.
Enamine tier (SEL_CIDS in scripts/build_pubchem_tier.py, exclusion = train + COCONUT keys): 4,556,691 compounds,
dataset bryantyu/casmi26-pc-enam (tier_enam.npz). Offline (scripts/eval_pcchannel.py PC_TRUTH_KEYS=enamine keys
PC_NOREACH=1, 236 unreachable Enamine truths placed into their window, local PeakFormer): median window 866,
| lam (pop) | 0 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|
| MRR | **0.666** | 0.653 | 0.624 | 0.550 |
| top-1 | 0.525 | 0.513 | 0.483 | 0.394 |
(the documented full tier's median window is 12.8k at test masses: same truths, ~15x more decoys). Probe kernel
casmi26-enam-only (lists = the Enamine tier's top 25, train / COCONUT keys excluded, lam 0) for Oct 22; policy
simulation scripts/sim_enam_policy.py (present vs other populations, rel / lib gate) for the combined kernel.

## COCONUT gap study (workflow wf_4186feca-c4e, Oct 23)
Public lineage, read for ideas only: engine 1 = train + COCONUT pool, 160-feature LightGBM with NO documentation /
source features and FPNets WITHOUT an instrument input (4 models incl. a DreaMS fine-tune), learned fragment LLR,
analog-structure families; then z(score) + 0.15 pop re-rank of the top 60 (+0.010 LB per their notes); engine 2 =
COCONUT (prvsiyan) + ChEBI/LMSD + train, doc-blind HistGB rankers on different evidence, fused by one-vote rank fusion
(+0.015 / +0.018 LB in their history); GLACIER slot-keeping re-order within formula groups; PubChem channel does not
touch COCONUT ordering.
Our losses (620 class-2 COCONUT validation queries, kernel token, stage 1 / stage 2, MRR): documentation kept 0.919 /
0.917; random-donor documentation 0.551 / 0.591; purchasable-donor 0.652 / 0.685. On the probe's own restricted metric
(stage 2): 0.934 / 0.650 / 0.741 vs LB-implied 0.655 (random-donor matches; the public lineage ~0.87). Loss buckets:
same-formula COCONUT isomers 58-68% of the lost MRR (not near-duplicates: median Tanimoto 0.34; they win on PeakFormer
and analog evidence), training-structure isomers 21-31% (their own library spectra match, lib ~0.8 -> largely
irreducible), PubChemLite / other < 10%. Truth in the full top 25 for 96.6% -> reach is not the problem.
Within-formula PeakFormer MRR 0.38 (token 1) / 0.49 (token 3) vs the public f.z 0.639; win rate vs COCONUT decoys 0.84 /
0.91 vs public 0.95. Parity emulation: raising the win rate to 0.965-0.977 lifts restricted MRR to 0.83-0.87.
Documentation handling (learned vs public-style additive pop, transplant-trained, reweighting) moves it by <= +-0.03; post-hoc
counterfactuals (doc weight, training-isomer demotion, formula normalisation) <= +0.003 HL-LB (COCONUT gains paid ~1:1
by class 1). GLACIER: COCONUT +0.033 / +0.040 (+0.048 on covered queries), class 1 -0.014, HL-LB +0.009 (= the LB's
+0.010). Fusing two of OUR rankers adds nothing (same evidence).
-> Levers in order: (1) the token-3 override (Oct 24 LB); (2) within-formula spectral discrimination: an instrument-free
or token-dropout PeakFormer (GPU Oct 24, PF_INST_DROP=0.5), more / better FP models, ensembles judged on within-formula
COCONUT MRR; (3) GLACIER coverage; (4) a second engine on genuinely different evidence (candidate set, FP model,
analog construction), fused by rank.

## Population prior in our spectral models (Oct 23, COCONUT-study synthesis)
Within the truth's formula group at matched Tanimoto, a simulated class-2 COCONUT truth beats COCONUT-not-train decoys
far more often than training-structure decoys: excess win rate pf_bern +0.137 (token 3 +0.141), fp_bern +0.107, an_best
+0.070 (partly legitimate), frag +0.013, GLACIER +0.019. Not memorisation (PeakFormer-unseen training structures show
+0.105). Likely cause: the PeakFormer / FP window negatives come from pool v1, whose COCONUT-not-train structures (30% of
negative entries) are never positives -> the models learn "COCONUT-not-train is never the answer", exactly the hidden
COCONUT population. Shifting the truths' pf / fp scores to remove the prior reproduces the LB-implied COCONUT MRR (0.660
vs 0.655), reconciling "hidden answers are well documented" (no-prior gap) with the 0.190 probe; the Zenodo proxy
(PeakFormer-only 0.275 on true non-training structures) agrees. Fix in training: scripts/pf_negtrain.py flags training-
structure negative rows (43% of rows, 70% of entries; 35 negatives per structure remain), PF_NEG_TRAIN in
scripts/train_pf.py keeps only those -> GPU variant b (kernels/casmi26-pf-train-b, Oct 24): instrument-free
(PF_INST_DROP=1.0), PF_TIMS_W=1.0, PF_NEG_TRAIN; variant a keeps the old negatives with token dropout 0.5. Judge them
on a membership-neutral metric: the truth's matched win rate vs same-formula TRAINING decoys (pf_bern now 0.663 kernel /
0.731 token 3) and the CO-TR excess (should shrink), not on raw within-formula MRR (which rewards the prior).
Ranker-side alternative (EVID_SHIFT: train stage 1 with class-2 truths' pf / fp shifted down) kept in reserve.
FP model variant b (kernels/casmi26-fp-train-b, GPU Oct 24): FP_NEG_TRAIN + FP_INST_DROP=1.0 + FP_TIMS_W=1.0 in
scripts/train_fp.py (baseline fp_bern excess +0.109, fp_cos +0.143 under token 1). Paired CPU pre-test
(scripts/fpb_local.sh, Oct 23): base / b / negtrain-only on an identical short schedule (1 x 2000 steps), fp features
recomputed on the class-2 COCONUT same-formula subset of ret14 val rows (620 queries), scored by
scripts/eval_pf_neutral.py under token 1 and 0. Note the training-script val MRR is not comparable across variants
(variant b validates against its own restricted negatives).
Result (620 queries; win rate vs training decoys / vs COCONUT decoys / excess; near-analog excess):
| model | token | fp_cos | fp_bern |
|---|---|---|---|
| deployed (30k steps) | 1 | .644 / .786 / +.143 (+.183) | .599 / .709 / +.109 (+.144) |
| deployed | 0 | .692 / .842 / +.150 (+.237) | .655 / .752 / +.097 (+.158) |
| base (2k steps) | 1 | .676 / .789 / +.112 (+.157) | .566 / .673 / +.107 (+.122) |
| base | 0 | .716 / .834 / +.118 (+.208) | .602 / .705 / +.103 (+.146) |
| **b** (2k steps) | 1 | **.743** / .782 / **+.039** (+.100) | .616 / .712 / +.096 (+.137) |
| **b** | 0 | **.743** / .784 / **+.041** (+.106) | .611 / .711 / +.100 (+.140) |
-> training-structure negatives remove most of the contrastive embedding's prior (excess -0.073..-0.077, paired
bootstrap CI excludes 0 at both tokens). Token 1: win vs training decoys +0.066 [+.038, +.096], vs COCONUT decoys
-0.007 (n.s.); token 0: +0.027 [-.002, +.056] and -0.050 [-.075, -.024] -> at token 0 most of the excess reduction comes
from losing vs COCONUT decoys (the prior removed), the discrimination gain is not significant. The token no longer
matters for b. The Bernoulli head (BCE on positives only,
never sees negatives) keeps its prior: it favours bits typical of training structures. (negtrain-only run stopped: the
token is irrelevant once dropped, and the GPU decision is clear.)
PMI fix for the Bernoulli score (scripts/eval_bern_pmi.py, casmi/fpscore.py FP_PMI = scripts/fp_bitprior.py output):
bern - lam * log p(bits | spectrum-weighted training-structure marginal). Deployed model, token 1: lam 0.5 .634 / .732
/ +.098, **lam 1 .690 / .761 / +.071**; token 0: lam 1 **.717 / .773 / +.055** (raw .655 / .752) -> both win rates
rise, an inference-time change. Same on the 2k-step models (b lam 1: .706 / .742 / +.035). NOT for the PeakFormer
(scripts/eval_pf_pmi.py): its f.z is trained contrastively against window negatives and PMI destroys discrimination
(lam 1: .599 / .555; lam 0.5 .633 / .644 vs raw .667 / .804) -> PeakFormer prior only via retraining (GPU variant b).
Ranker test: ret15 rows = ret14 with fp_* from the PMI scorer (/home/user/work/v15_rows.sh), rankers v14a1 / v15a1
(v13a1 recipe), HL-val + stage-1 CV a1+pf (/home/user/work/v15_eval.sh): NULL. Val cls2 .920 -> .927, cls1 .812 ->
.813; HL-LB .3374 -> .3372; CV all .8819 -> .8828, cls2w .6714 -> .6718 (se .018). But neither CV nor HL-val can credit
a debiasing: their class-2 truths are held-out training structures that do not suffer the prior. -> scripts/hlval.py
HL_PRIOR=1 (Oct 23): per condition and spectral column, penalty = mean score of held-out training decoys (never
positives) minus COCONUT-not-train (resp. other non-train) decoys in matched (query, Tanimoto bin) cells; class-2
truths shifted down by it inside the draws, derived rank / gap / within-formula features recomputed.
Review wf_362c305b-918 (3 lenses + skeptics, Oct 23) before the GPU runs; fixed:
- FP training was CPU-bound on GPU (per-step numpy unpacking of ~32k fingerprints, ~0.7-1.4 s/step -> ~8 h of GPU
  idling): molecule bits now a device CSR gathered with torch (identical ids / offsets, verified), FP_HOURS guard.
- negtrain flagged ALL in_train rows, incl. held-out validation (split 1) and ranker (split 2) structures -> they stayed
  negative-only in b (the eval truths drawn ~3x more often as negatives than in base): negtrain v2 keeps only split-0
  positives (258,932 rows; dataset v2). Held-out truths are now unseen, like hidden answers.
- short restricted lists made sampling with replacement the norm (b: 96% of structures < 63 negatives, +1.1 nats
  logit offset from repeats): FP_NEG_NOREP / PF_NEG_NOREP draw without replacement and mask the rest.
- INST_DROP=1.0 left tokens 1-5 at random init while validation / inference feed real tokens: token columns tied to
  token 0 before validation and save (any token = token 0).
- PeakFormer validation scored a phantom empty candidate for structures with no negatives (93 / 1500 in b): skipped.
Not changed: filtering after the 64-entry cap (sk2: refiltering the full window barely changes empty / short lists).
FP_PMI must reach a kernel explicitly (casmi26-code fpscore.py + fp_bitprior.npy + FP_PMI env), not yet deployed.
PeakFormer variant a (token dropout, old negatives) dropped from the Oct 24 GPU plan (shared quota): b ~8.5 h + FP b
~1.5-2 h.

## Hidden-like validation (scripts/hlval.py, Oct 23)
Stage-1 MRR on validation strata made hidden-like -- features under the token the kernel sees (kernel = token 1 for every
query, ret14 rows; tok3 = the override, ret13 rows; true = own tokens), class-2 truths given the documentation of a random
non-training candidate of the same source (10 draws) -- combined with the LB-implied mix (class 1 0.18, COCONUT 0.29,
other reachable 0.03; Enamine / unreachable score 0 in the main lists). Closure against the LB (stage 2):
| ranker | HL-LB kernel | HL-LB tok3 | LB |
|---|---|---|---|
| v11l | 0.3386 | 0.3546 | 0.320 |
| v10 | 0.3346 | 0.3388 | 0.314 |
| v12 | 0.3296 | 0.3453 | 0.312 |
Same order as the LB with a near-constant +0.018 offset (the plain simulation ranked v12 above v11l). Predicted token-3
gain for v11l: +0.016 (COCONUT stratum 0.549 -> 0.594); the ranker retrained on token-3 rows (v13a1) adds nothing
(0.3552 vs 0.3546) -> the override alone suffices.
Documentation-transplant TRAINING (DOC_TRANSPLANT in train_ret_ranker.py) scores HL-LB 0.411 (COCONUT stratum 0.829) but
is partly an artifact: scripts/hl_transplant_artifact.py shows the model ranks a transplanted DECOY first in 9.2% of
queries vs 1.9% untransplanted (v13a1: 0.3% / 0.3%), i.e. it detects documentation inconsistent with the structure. With
every non-training candidate transplanted (documentation uninformative) it still beats v13a1 (0.661 vs 0.608): it leans
more on spectral evidence. But the LB says documentation matters on hidden answers (no-prior ranker 0.245 vs 0.315,
Oct 18), the opposite of the full-transplant assumption -> not shipped; HL-val is used for token / feature decisions
(documentation treatment held fixed), not for documentation-weighting decisions until a partial-transplant version
reproduces the 0.07 no-prior LB gap.
Calibration (scripts/hl_calib_doc.sh; stage-1 rankers on kernel-token rows, alpha 0, with / without priors): HL gap
prior - no-prior = +0.028 / +0.022 / +0.015 / +0.010 / +0.002 at transplant share p = 0 / 0.25 / 0.5 / 0.75 / 1, against
+0.070 on the LB (Oct 18). Even without any transplant the simulation under-states how much documentation helps on the
hidden test -> hidden answers are at least as documented as simulated truths (consistent with purchased, known
compounds), the opposite of the documentation-shift premise. Caveat: the LB gap was measured with the pool-v2 rankers,
whose spectral evidence was weaker. Documentation treatment stays as in v11l (the alpha A/Bs were LB noise); HL-val
decides token / feature questions only.

## Lessons from the Oct 23 submissions
- The channel re-ranker does not transfer: re-ranked purchasable-only list 0.034 vs PeakFormer-order Enamine-only 0.037
  (offline +0.18 MRR on simulated / Zenodo-proxy truths). Even confident promotions cost: v11l 0.320 -> 0.310 (p95) /
  0.316 (p98). -> no channel insertions in the main lists; the channel stays a probe.
- The vendor prior does nothing on COCONUT answers either: COCONUT-restricted v12 0.187 vs v11l 0.190 -> retired.
- Our COCONUT component is 0.190 (public lineage 0.252): the target of the instrument-token fix. Oct 24: v11l tok0 /
  tok3 full lists plus paired restricted probes (probe11tok3-coco vs probe11-coco 0.190; probe11-train vs
  probe11tok3-train for the class-1 cost of the override).

## Instrument token = chemistry prior (Oct 23; design-panel strategist "sim2real", reproduced)
PeakFormer and the binned FP model take an instrument one-hot; timsTOF training spectra are 99.9% enveda-180 (Enamine), so
the token learned "timsTOF => Enamine chemistry". Every hidden spectrum is timsTOF. scripts/eval_inst_token.py
(PeakFormer f.z MRR of the truth in its pool window):
| validation queries | true token | token 1 (timsTOF) | token 3 (TOF) | token 0 (unknown) |
|---|---|---|---|---|
| npx NP (246; 206 timsTOF) | 0.326 | 0.314 | **0.482** | 0.480 |
| enveda-180 Enamine (120) | 0.828 | 0.826 | 0.635 | 0.688 |
Strategist's stage-1 check (npx class 2 with hidden-like documentation transplanted from purchasable COCONUT-not-train
decoys, 10 paired draws): v11l 0.622 -> 0.713 with token 3 (+0.091 +- 0.010), class 1 neutral. It also explains the
channel insertion cost: rel > 15 fires on 29% of gated timsTOF NP present queries vs 8% non-timsTOF (present cost
-0.036 to -0.041 per timsTOF molecule = the LB-implied -0.039). Estimated hidden mix: train ~0.18, COCONUT ~0.29,
Enamine ~0.056, other ~0.03, ~0.44 unreachable. Actions: OWN_PF_INST override in casmi/own.query_from_rows (main-pool
FP / PeakFormer scoring only; the channel keeps the true token); kernels casmi26-own-s2v11ltok0 / -tok3 (v11l +
override) for Oct 24, paired against v11l 0.320; GPU PeakFormer (Oct 24) trains with PF_INST_DROP=0.5 token dropout.

## Lessons from the Oct 22 submissions
- The Enamine restriction is right: the Enamine-only list scores 0.037 (= the public lineage's PubChem-only list) where
  the full 110M tier scores 0.003 and the documented tier 0.006. With an offline PeakFormer-only channel MRR of ~0.6 on
  Enamine truths, roughly 5-7% of the hidden molecules are Enamine compounds outside train + COCONUT.
- Inserting the proposals COSTS: v12 0.312 -> v12purch 0.294 / v12purchrr 0.296 (paired: the lists differ only by the
  inserted proposals, so -0.018 is far less noisy than an unpaired A/B). The policy simulation predicted +0.014 at
  pi = 0.05. Its error: the simulated main lists hold their answers at rank 1 (MRR 0.78), the hidden lists (MRR ~0.31)
  hold them at ranks 2-20, exactly where slots 2,4,6,.. / 4,8,.. push them down. Weaker lists make displacement MORE
  costly, not less (my earlier assumption was backwards). The re-ranker adds +0.002 (noise-level).
- v12 (vendor features + vendor-target weights) 0.312 vs v11l 0.320: no gain from the purchasable prior on the full LB
  (inside noise); the COCONUT-restricted probes (probe11-coco vs probe12-coco) isolate it next.
- Next channel policy: precision over recall -- at most one or two proposals, only on high re-ranker confidence, never
  in the 2..10 band by default.
- Calibrating the policy simulation to these two LB points fails (scripts/calib_policy.py): with pi = 0.037 / 0.666 =
  0.056 the observed net implies a present-list cost of -0.039; reweighting the simulated lists towards answers below
  rank 1 reaches only -0.023, and the model predicts +0.003 for v12purchrr where the LB gave -0.016. The simulation
  misses something about the hidden lists beyond the rank of the answer (more gated molecules, higher rel /
  re-ranker scores on the hidden spectra), so policy choices go to the LB directly: promotion-only mode (PC_SLOTS=''
  -> only the channel's re-ranked #1 moves to rank 1, and only when the re-ranker confidence rr_top exceeds a
  threshold; no other insertions), kernels casmi26-own-s2v11lprop95 (rr_top > 3.79) and -prop98 (> 4.6) on the v11l
  base, paired against v11l 0.320 on Oct 23 (trigger trig_01QErcbkQuHu8w8aASQef1hB), with the re-ranked channel probe
  and the two COCONUT-restricted probes.

## Kernel status (Oct 21, 09:50 UTC)
- casmi26-enam-only v1 complete (400 / 400 lists, channel on every molecule, 91 min) -> Oct 22 probe.
- Complete with clean logs (400 / 400 lists): casmi26-own-s2v11lenam (60 min), casmi26-own-s2v12 (66 min; lists differ
  from v11l for 399 / 400, top-1 for 12), casmi26-own-s2v12purch (62 min), casmi26-own-s2v12purchrr (51 min),
  casmi26-own-s2v12purchrr2 (64 min), casmi26-purch-only-rr (89 min, re-ranker on all 400 molecules, top-1 differs from the PeakFormer-only Enamine probe for
  244 / 400), casmi26-probe12-coco pushed. Oct 22 slots: enam-only, v12, v12purchrr, v12purch, pcfull-only2.
- casmi26-pc-calibrun (full-tier calibration) died after ~5 h with an empty log (probably memory: per-worker
  fingerprint caches over 400 windows of up to 150k compounds) -> not re-run; the Enamine tier replaced the full tier
  as the main channel and its policy is calibrated locally (scripts/sim_enam_policy.py).

## Channel re-ranker (Oct 21): analog evidence lifts the purchasable channel to top-1 75%
casmi/pcchannel.rerank_feats on the channel's best 100 proposals: PeakFormer z / ranks / gaps, spectral analogs (top-20
library spectra by analog search, truth key hidden -> Tanimoto of their structures to each proposal, an_* as in the pool),
fragment explanation, RDKit ppm, heavy atoms (no documentation column: a simulated truth's counts are per key, tier rows'
per CID). Rows: scripts/channel_rows.py on the purchasable tier, simulated class-2 queries with Enamine truths (train
2,277 / val 484; workers recycled every 40 queries after OOM kills). LambdaRank (scripts/train_channel_rr.py):
| validation | PeakFormer only MRR / top-1 | re-ranked MRR / top-1 |
|---|---|---|
| all Enamine truths (484) | 0.604 / 0.475 | 0.787 / 0.705 |
| unreachable (236) | 0.666 / 0.525 | **0.844 / 0.754** |
Leak found and fixed: the simulation centred the query mass on the truth's exact mass (truth ppm = 0); with the
precursor-derived mass (as at test time, query_from_rows) the first model's 0.885 dropped to 0.844 (without ppm 0.839).
Exact analog duplicates of the truth (another InChIKey-14 of the same structure in the library) are rare (0.4-1.4%); the
truth's best-analog Tanimoto has median 0.58 vs 0.22 for decoys (Enamine series). Gain led by an_r (50%) and fz_gap
(28%). Policy re-check with the re-ranked channel ranks (sim_enam_rr.parquet): agg 15 / promo 60 -> +0.014 / +0.036 /
+0.082 / +0.127 at pi = 0.05 / 0.1 / 0.2 / 0.3 (PeakFormer only: +0.009 / +0.026 / +0.061 / +0.097). Final model
pc_rerank_v1 (train + val, 2,701 queries); kernels casmi26-own-s2v12purchrr (v12 + purchasable channel + re-ranker) and
casmi26-purch-only-rr (probe: re-ranked channel only).
Leakage audit (workflow wf_b7f8c93a-7df, 3 auditors x 2 skeptics, 47 agents): 22 findings, none a material inflation.
No feature marks the appended truth (fake-truth test: an unrelated Enamine training structure re-ranks at chance, MRR
0.013); splits clean (cr_train truths split 2, cr_val split 1, PeakFormer split 0); misses counted in both orders;
hide={truth} removes every same-structure analog for the unreachable subset. Small effects: window still centred on the
truth mass (+0.003), seed choice (+0.003; 5 seeds 0.832-0.844), query make-up vs test (gain -0.006 to -0.016), pool vs
PubChem SMILES of the truth (-0.004, offline pessimistic), PC_GATE ignored (-0.005). 5-fold CV on 1,489 train
unreachable queries: +0.214. Decisive proxy under kernel conditions: 222 Zenodo Enveda/Enamine compounds absent from
train (own spectra, PubChem SMILES, hide=()): analog evidence transfers (P(an_r = 1) 0.640 vs 0.655 in the simulation)
and the re-ranker lifts PF-only 0.275 -> 0.668 (the PeakFormer alone suffers there from the Q-TOF vs timsTOF shift).
Confidence trigger (scripts/eval_rr_policy.py, train-only re-ranker pc_rerank_v1t, answer-present windows from
scripts/channel_rows.py present mode with class-1 truths' library spectra kept): rr_top = best re-ranker score of the
window; present p90 / p95 = 3.14 / 3.79. B slots if rel > 15 or rr_top > 3.14, rank-1 promotion if rel > 60 or
rr_top > 3.79: d_present -0.020, gains +0.011 / +0.043 / +0.106 / +0.170 at pi = 0.05 / 0.1 / 0.2 / 0.3 (rel only:
+0.014 / +0.036 / +0.082 / +0.127; the simulated main lists are stronger than the test's, so the displacement cost is
overstated). Kernel casmi26-own-s2v12purchrr2 (PC_RERANK=pc_rerank_v1t PC_RR_AGG=3.14 PC_RR_PROMO=3.79) for Oct 23;
choose between it and s2v12purchrr by the enam-only probe (pi).

## PeakFormer SBAL runs e/f/g (Kaggle CPU, 11 h, Oct 21)
Internal validation f.z MRR / sum MRR at the last epoch: e 0.639 / 0.649, f 0.631 / 0.627, **g 0.654 / 0.658** (larger
model, 45 MB) vs local d 0.631 / 0.630 and a 0.651 / 0.650. Next: pf features from g alone on the v12 rows (ret12g,
/home/user/work/pf_g_feats.sh) -> stage-1 CV vf+pf+vend vs ret12 (.9024 / .8882 / .9095 / vf-w .9587); channel check
with g on the unreachable Enamine truths (d: MRR 0.666).
Result (ret12g rows): COCONUT non-train within-formula pf_bern MRR 0.475 (d 0.463), all-candidates 0.447 (0.431), wins
vs COCONUT decoys 0.885 (0.878); stage-1 CV vf+pf+vend all .9035 / cls1 .8902 / cls2 .9102 / vf-w .9599 vs d .9024 /
.8882 / .9095 / .9587 -> +0.001-0.002, noise-level; switching would also need new re-ranker rows -> d stays; next
PeakFormer step = the GPU model (Oct 24).

## Pre-submission review (workflow wf_7a997223-4ef, 4 reviewers, Oct 21)
No blockers. Fixed: (1) molecules without a stage-1 list now also get channel proposals (gate no longer requires a
non-empty list); (2) full-tier rel taken from the proposals' own f.z (the rescoring under -1-cid ids could hit the
per-worker fingerprint cache keyed by -1-row ids); (3) the push retry loop treated any non-slot error (a 500 at 09:34)
as success -> scripts/push_retry.sh exits only on "successfully pushed"; (4) v_nsrc counted spectral-library depositors
(MoNA 35% of simulated class-2 truths vs 1.6% of decoys = "has public spectra" leak; measured effect on v12 <= 0.001
class-2 MRR) -> excluded in scripts/build_vendor_feats.py for the next rebuild (v12 itself unchanged). Verified: ret12
rows = ret10pl + 22 vendor columns, vendor values match by pid, Kaggle pool4 / vendor files have the local md5s, no
VENDORS source is a spectral library. (1) and (2) are in the code of casmi26-own-s2v12purch, not in the running
s2v11lenam / s2v12 (rare molecules only).

## Purchasable-compound prior (Oct 21): the test molecules are physical samples Enveda bought
Same depositor analysis for Enveda's natural-product examples (enveda-np-examples, 250 structures, 99.6% COCONUT):
MedChemExpress 96%, TargetMol 94%, BioCrick 79%, ChemFaces 62%, Selleck 60%, AbMole 58%, ApexBio 55% -- against
2.3% / 3.8% / 1.0% / 1.0% / 0.1% / 0.2% / 0.1% of COCONUT-not-train (likelihood ratios 24-280). Union "purchasable"
flag (Enamine or one of 10 bioactive / NP vendors; pool4/vendor.parquet, scripts/build_vendor_feats.py): npx 98.8%,
enveda-180 92.3%, other training libraries 36.7%, reachable non-train 19.9% (COCONUT 6.1% = 27k of 450k; PubChemLite
35.6%; ChEBI/LMSD 13.2%). Bioactive-vendor structures: 110k keys, 55.7k outside the pool.
Stage-1 CV with the depositor columns (v_* , v_bioact, v_nsrc; alpha-1 doc weights): all .8939 -> .8965, cls1 .8836
-> .8831, cls2 .8991 -> .9033, cls2w .7103 -> .7075 (noise). By truth: COCONUT class-2 truths WITH a bioactive vendor
.9306 -> .9439 (n 1021) / .9652 -> .9735 (n 849, also Enamine), WITHOUT .8615 -> .8589 (n 1755): the columns act as
a prior learned at the simulation's 54% purchasable base rate. Conflict found: the alpha-1 documentation weights give
purchasable class-2 truths only 13% of the class-2 weight (they are documented), while Enveda's real samples are ~99%
purchasable. -> weight tables with a vendor target: vf (w = 0.97 / 0.54 for purchasable truths, 0.03 / 0.46 otherwise;
ESS ~2.2k), vfd05 / vfd1 (vf x doc weight^0.5 / ^1); CV evaluated under the vf target (CV_EVAL_W).
Stage-1 5-fold CV (ret10pl rows; cls2w here = vf-target weighted, last column = the old doc-shift target):
| config | all | cls1 | cls2 | cls2w (vf) | cls2w (doc) |
|---|---|---|---|---|---|
| a1+pf (v11l recipe) | .8939 | .8836 | .8991 | .9314 | .7103 |
| a1+pf+vend | .8965 | .8831 | .9033 | .9415 | .7075 |
| vf+pf | .9005 | .8877 | .9070 | .9484 | .6546 |
| **vf+pf+vend** | **.9024** | **.8882** | .9095 | .9587 | .6746 |
| vfd05+pf+vend | .9007 | .8818 | .9103 | .9606 | .6778 |
The two hypotheses about the hidden class-2 population disagree (vf vs doc target); the physical-sample evidence
favours the vendor target, and vf+pf+vend is also best on the unweighted columns -> pipeline v12 (T=v12 ROWS=ret12
ALPHA=1 DOC_W=pool4/vf_w.parquet scripts/pipeline_v8d.sh; kernels need VEND=1 = dataset bryantyu/casmi26-vendor,
VENDOR_PATH) for an LB A/B against v11l (0.320).

## Second-ranker rank-2 insertion (other-gap item 4, Oct 21) -- refuted offline
Source-blind stage-1 ranker (scripts/train_ret_ranker.py --source-blind: no in_coco / in_bio / PubChemLite / PubChem /
LOTUS / priors; alpha 1, ret10pl rows) and the same without in_train / lib_n (--no-membership); its top-1 moved to rank 2
of the v11l stage-1 list (scripts/eval_r2.py, 979 validation queries; mix = 0.45 cls1 + 0.55 weighted cls2):
| second ranker | alone cls2w | insert at 2: d cls1 | d cls2w | d mix |
|---|---|---|---|---|
| v11sb (source-blind) | .7411 (v11l .7333) | -0.0004 | -0.0036 | -0.0022 |
| v11sbnm (+ no membership) | -- | +0.0029 | -0.0015 | +0.0005 |
Gating on the second ranker's top-1 margin and / or a low main margin never beats +0.0007. The second ranker sees the
same candidates and evidence, so its top-1 is rarely right where v11l's top 2 is wrong -> not shipped (knob OWN_R2 /
OWN_R2_SLOT in casmi/run_own.py kept, off by default).

## Lessons from the Oct 21 submissions
- "Other" answers (not train, not COCONUT) are where the public lineage leads most clearly: its lists restricted to
  them score 0.057; ours 0.007 (pool v4: PubChemLite + ChEBI/LMSD) + 0.006 (documented PubChem tier, sid >= 5 or
  pmid >= 1). The public PubChem tier is (near-)complete PubChem (6.7 GB; 3.5% of a random sample has a PubMed
  article) and its channel ranks the WHOLE 10 ppm window by spectral f.z first (ECFP4-block first pass, top 5000,
  then full fingerprint + 0.25 pop). Our documentation cut removed exactly the long tail where these answers are
  (drug-like vendor compounds - Enveda-180 is an Enamine library). -> full PubChem tier built on Kaggle (internet
  kernel, local disk too small), two-pass channel; Zenodo-extra probe (3,717 Enveda-180 structures absent from
  train, mostly without PubChem CID) on Oct 22.
- Full PubChem tier (kernels/casmi26-pc-full-build, internet kernel, 17 min): 109.8M single-component net-neutral
  compounds of 80-1500 Da, 5.9 GB SMILES; channel casmi/pcchannel.propose_full (pass 1: f.z over the Morgan r2 block
  for the whole 10 ppm window, top 2000; pass 2: full PeakFormer); probe casmi26-pcfull-only, combined kernels
  casmi26-own-s2v11lpcf-ad / -fx.
- Ready for Oct 22 (trigger 00:04): zen-probe, pcfull-only, probe11-coco (v11l restricted to COCONUT-not-train),
  v11l + full-PubChem channel adaptive / fixed. Prepared if the Zenodo probe is positive: casmi/zen.py (direct
  spectral match to the 3,717 Zenodo-extra structures, inserted at rank 1 / 3 above similarity 0.7 / 0.5; ZEN=1 in
  scripts/make_own_kernel.py). PeakFormer CPU runs e/f/g (structure-balanced) finish ~11:40 UTC Oct 21; GPU run
  armed for Oct 24 00:20 UTC.
- Workflow wf_a82a089e-6e1 (docs/other_gap_plan.md): the public "other" 0.057 = PubChem channel ~0.03-0.045
  (standalone list 0.037; near-complete 105.9M tier, median window 12.8k at test masses, 58% vendor-deposited; two-pass
  ECFP4-block f.z with N1 5000 + 200 best-documented rows, score z(f.z) + 0.25 pop; lib < 0.9 gate; slots 2,4,..,10 if
  rel = best PC f.z - best pool f.z > 600 else 4,8,..,20; PC #1 promoted to rank 1 if rel > 150 or S > 6 and pop >= 5
  (promotion credited >= +0.024 LB)) + engine-2 ChEBI/LMSD ~0.005-0.01 (rank-2 insert of engine 2's top-1) + generated
  derivatives ~0.003-0.01. Refuted: DreaMS tail fill (dead code), engine-2 COCONUT copy, source-blind ranker as the
  lever. Our channel fixed accordingly (uniform window thinning, metric-identity dedup, exclusion sets, aligned pop
  prior dataset casmi26-pc-full-pop, pop union, PC_PROMO / PC_REL_AGG / PC_S_PROMO knobs); probes pcfull-only2
  (spectral, train + COCONUT excluded) and pcfull-only3 (+ pop); thresholds from a simulation kernel
  (casmi26-pc-calibrun, 400 class-2 validation queries against the full tier).
- v11l 0.320 vs v10 0.314 (PeakFormer + alpha 1: +0.006, inside LB noise); inserting the documented-tier proposals
  costs 0.013 (0.307): no proposals from that tier until the tier is right.

## Lessons from the Oct 19 submissions
- GLACIER stage 2 adds +0.010 over its stage 1 (0.307 -> 0.317); ChEBI/LMSD membership (pool v4, in_bio) adds
  another +0.013 (0.330). Both are inside the ~+-0.03 public-LB noise (124 molecules), but agree with local CV.
- Source probes (pool v3 lists, full list 0.295): train-only 0.146 + COCONUT-only 0.191 = 0.337 > 0.295 even before
  the PubChem part. Restricting a list to one source promotes its truths, so the sum is not additive, but the
  COCONUT-only list alone scores 65% of the full list: natural products are a large share of the answers and the
  full list buries them under synthetic / PubChem isomers. Levers: source-aware ordering (bio / NP prior in the
  ranker, already partly via in_bio) and better isomer ranking inside COCONUT formula groups.

## Documentation shift of the class-2 simulation (Oct 19)
- A simulated class-2 truth is a training structure, i.e. it HAS public spectra. Such structures are far better
  documented than reachable structures without spectra (COCONUT: PMID share 67% vs 22%; ChEBI/LMSD membership: 22k
  of 98k ChEBI/LMSD structures are training structures, only 1.6k are COCONUT-not-train). A LightGBM classifier on
  documentation / membership columns separates the two populations with AUC 0.93. Hidden class-2 answers have no
  public spectra by definition, so the ranker over-trusts documentation (in_bio, pc_*, pcl_*).
- casmi/docw.py: importance weights p(no spectra | doc) / p(spectra | doc) for class-2 queries (DOC_W, DOC_ALPHA in
  train_ret_ranker.py / oof_stage1.py / train_stage2.py). Stage 1 on pool v4, validation (979 queries):
  | alpha | all | cls1 | cls2 | cls2 doc-shift weighted |
  |---|---|---|---|---|
  | 0 (v4c recipe) | 0.891 | 0.827 | 0.926 | 0.668 |
  | 0.5 | 0.890 | 0.830 | 0.921 | 0.685 |
  | 1 | 0.880 | 0.840 | 0.902 | 0.709 |
  Weighting stage 2 alone does nothing (it inherits the prior through s1_z / s1_r). Weighted class-2 recall@30 of
  stage 1 is 0.976 (top-1 0.54): the loss on undocumented truths is ordering, not reach.
- Full pipeline v5w (scripts/pipeline_v5w.sh: weighted stage 1, weighted OOF, incremental GLACIER, weighted stage 2),
  validation (979 queries; weighted = doc-shift weights of pool v4 / pool v5 tables):
  | model | all | cls1 | cls2 | cls2 weighted |
  |---|---|---|---|---|
  | stage2_v4 (LB 0.330) | 0.893 | 0.832 | 0.926 | 0.677 / 0.675 |
  | stage2_v5w | 0.893 | 0.852 | 0.914 | 0.769 / 0.768 |
  | stage2_v5w_a0.5 | 0.902 | 0.840 | 0.935 | 0.758 / 0.765 |
  Kernels casmi26-own-s2v5w (alpha 1) and casmi26-own-s2v5w-a05 (alpha 0.5; best on every validation column but
  the weighted one, where it is within noise of alpha 1) -> both submitted Oct 20 against s2v4 (0.330).
- 5-fold CV over all 5,856 simulated queries (train + val, stage 1 only; weighted cls2 ESS 442, se ~0.018):
  | config | all | cls1 | cls2 | cls2 weighted |
  |---|---|---|---|---|
  | v4c (alpha 0) | 0.888 | 0.862 | 0.901 | 0.648 |
  | alpha 0.5 | 0.887 | 0.861 | 0.900 | 0.673 |
  | v5w (alpha 1) | 0.867 | 0.863 | 0.869 | 0.678 |
  | v4c + formula-rel | 0.888 | 0.862 | 0.901 | 0.656 |
  | v7w (alpha 1 + formula-rel) | 0.872 | 0.863 | 0.876 | 0.680 |
  | alpha 0.5 + formula-rel | 0.884 | 0.862 | 0.895 | 0.660 |
  alpha 0.5 keeps most of the weighted gain at no unweighted cost -> second LB candidate (pipeline_v5w.sh 0.5,
  kernel casmi26-own-s2v5w_a05). Formula-relative features are noise-level (sign flips with alpha) -> not used.
  The single validation split (979 queries, weighted ESS ~80) is too noisy for these comparisons (se ~0.05).
  Ablations at alpha 0.5 (same CV): + LOTUS columns 0.886 / cls2w 0.675 (no effect); - in_bio 0.823 / cls2 0.813 /
  cls2w 0.667 (ChEBI/LMSD membership is a large shortcut in simulation, 61% of simulated class-2 truths vs ~0-3% of
  structures without spectra, but neutral under the test-like weights); - PubChemLite annotations 0.885 / 0.680.
  timsTOF-query upweighting (x3): slightly worse. Learning curve: 25% of the queries as good as 100%.
  LightGBM (alpha 0.5): num_leaves 63 / min_data_in_leaf 50 0.889 / cls2w 0.678 (base 31 / 100: 0.887 / 0.673);
  lr 0.03 x 900 rounds 0.888 / 0.675; num_leaves 15 / 200 0.879 / 0.674 -> 63 / 50 from the next stage-1 version.
- LOTUS occurrences (pool v5, scripts/add_lotus.py: land-plant / fungal / bacterial organisms, families, genera; CC BY
  4.0) do not help in simulation (stage 1 cls2 weighted 0.665 without / 0.652 with doc weights vs 0.677 / 0.711 without
  LOTUS): simulated truths are not plant-biased, so the ranker can only learn them as more documentation. Parked;
  needs a plant-targeted query weighting or an LB test.
- Counter-evidence to keep in mind: removing the priors entirely cost 0.07 on the LB (0.315 -> 0.245), so the test's
  answers are documented relative to their isomers; full correction may overshoot -> LB A/B (casmi26-own-s2v5w).
- Forum (Aryan Awasthi, Oct 14): popularity puts the answer first for 67% of enveda-np-examples structures (all have
  public spectra elsewhere) but ~0% of enveda-180 structures without public spectra; fragment explained-intensity gap
  was their top ranker feature. DancingLumberjack: PubChem as an extra candidate source moved their LB +0.002.
- enveda-np-examples (250 structures, timsTOF): common plant metabolites (alkaloids, phenylpropanoids, terpenoids;
  Arabidopsis, Ginkgo, tea, Capsicum ...), in_bio 96%, PMID 99% -> class-1-like, solved by priors (val cls2 0.98).

## DreaMS channel (Oct 19, in progress)
- DreaMS (Bushuiev et al., Nat. Biotechnol. 2025; MIT weights on Hugging Face; explicitly allowed by the hosts) as a
  second spectral representation. TorchScript embedder (casmi/dreams_emb.py, preprocessing as the reference
  SpectrumPreprocessor) runs at ~8 spectra/s on 4 CPU threads -> library = one representative per (structure,
  polarity) of the analog library (360k spectra: 251k positive, 109k negative), embedded on 5 Kaggle CPU shards
  (kernels/casmi26-dreams-emb-*); simulated query spectra (23.7k) embedded locally (scripts/embed_query_rows.py).
- Uses: (1) DreaMS-neighbour analog channel (casmi/dreams_feats.py: dr_best / dr_tan_best / dr_sum / dr_top1_tan /
  dr_r / dr_gap, plus dr_self = direct DreaMS library match, instrument-robust); hiding as in the simulation
  (class 2: truth structure, class 1: query spectra). (2) Frozen-embedding dual encoder / fingerprint head trained on
  split-0 structures only (scripts/train_dreams_fp.py), compared with the binned model's fp_cos.
- Inference wiring done (casmi/run_own.py, DREAMS_DIR; active only for rankers whose features include dr_*).
- Results (library 360,144 embeddings, 5 Kaggle shards at 6.3 spectra/s): alone the DreaMS features are weaker than
  the modified-cosine / library ones (validation truth MRR by single feature: dr_best cls2 0.366 vs an_best 0.464;
  dr_self cls1 0.526 vs lib 0.834; corr(an_best, dr_best) 0.83). DreaMS fingerprint head (frozen embeddings, 347k
  split-0 spectra): FP-only MRR 0.443 / top-1 0.296 vs binned fp_cos 0.495 / 0.355 -> not used.
  Stage-1 5-fold CV (alpha 0.5): base 0.887 / cls1 0.861 / cls2 0.900 / cls2w 0.673; + dr_* 0.892 / 0.867 / 0.904 /
  0.693 (gain on every column); dr_* replacing an_* 0.878 / cls2w 0.638 (complementary, not a substitute).
  -> pipeline_v8d.sh (dr_* + alpha 0.5 + 63-leaf trees), kernel casmi26-own-s2v8d.
- Review workflow (4 lenses x 2 adversarial verifiers) found a simulation leak: the library keeps ONE representative
  per (structure, polarity), and class-1 hiding of the query spectra removed it for ~1/3 of class-1 truths, so
  "lib_n > 0 and dr_self == 0" marked the answer (36% precision vs 1.4% base rate) -- impossible at test time.
  Fixed with substitute representatives (scripts/add_dreams_feats.py); simulated enveda-180 positive queries are now
  embedded electron-mass corrected like the hidden test. Corrected CV: + dr_* 0.8868 / cls1 0.8590 / cls2 0.9008 /
  cls2w 0.6845 vs base 0.8870 / 0.8605 / 0.9003 / 0.6734: the class-1 "gain" was the leak; what remains is +0.011
  on weighted class 2.
- Full pipelines (single validation split, stage 2): a0.5 0.902 / cls1 0.840 / cls2 0.935 / weighted 0.758 / 0.765;
  v8d (dr_* + 63 leaves) 0.902 / 0.851 / 0.930 / 0.721 / 0.682 (stage 2 adds nothing over its stage 1, 0.903);
  v8e (dr_* + 31 leaves) 0.902 / 0.852 / 0.929 / 0.729 / 0.696. Net: DreaMS is about neutral after the leak fix
  (class 1 up, class 2 slightly down, weighted within noise). Main entry stays alpha 0.5; v8e (kernel
  casmi26-own-s2v8e) is an optional LB test, not a default.
- Stage-2 5-fold CV on the 4,877 training queries (out-of-fold stage-1 scores; train_stage2.py cv now also prints the
  weighted class-2 MRR, se ~0.02): alpha 0.5 stage 1 0.8807 / w 0.660 -> stage 2 0.8861 / w 0.677;
  v8e stage 1 0.8818 / w 0.669 -> stage 2 0.8868 / w 0.682. DreaMS = +0.001 / +0.005, i.e. noise-level positive.
- casmi26-own-s2v8e v1 ran on Kaggle in 56 min (DreaMS library 360,144 loaded, 1,213 test spectra embedded in ~4 min,
  no per-molecule failures, 400/400 lists, top-1 equal to s2v5w-a05 on 98.2% of the placeholder test) -> ready for an
  LB slot on Oct 21.

## Oct 20 LB: documentation-shift weighting is invisible on the LB
alpha 0.5 0.323, alpha 1 0.332 vs s2v4 0.330 (same pool, same stage-2 design). Non-monotone in alpha and well inside
the +-0.03 noise of 124 public molecules. Arithmetic: a +0.09 weighted class-2 gain on ~20% of the molecules is
~+0.02 at most -- below what one public LB submission can resolve. Consequences: (1) offline-validated changes of
this size cannot be LB-confirmed; decide them on CV and keep the one with the better weighted CV (alpha 1 / a0.5 are
equivalent on the LB -> keep alpha 0.5, the CV-dominant one, as the default); (2) the LB is only informative for
effects >= ~0.05, i.e. the 0.12 gap to the public lineage, which the source probes target.

## Public-lineage source decomposition (Oct 20)
| source restriction | public lineage | ours (pool-v3 / ranker-v2 lists) | gap |
|---|---|---|---|
| training structures | 0.155 | 0.146 | +0.009 (noise) |
| COCONUT-not-train | 0.252 | 0.191 | +0.061 |
| other | 0.057 | (pubchem probe not run) | ? |
| sum of restricted | 0.464 | >= 0.337 | |
| full list | 0.451 | 0.295 | |
| loss from merging sources | 0.013 | >= 0.042 | >= +0.03 |
The 0.12-0.16 gap is (1) ranking inside COCONUT (+0.06 restricted) and (2) list composition (we lose >= 0.04 when
merging sources, the public lineage 0.013). Class 1 is on par. Our current pipeline's own decomposition comes from the
probe5 kernels (Oct 21). LB experiments queued: SRC_DEMOTE (s2src, Oct 21), SRC_GATE (implemented), POP re-rank (the
public lineage adds 0.15 * pop to its ranker z).

## COCONUT-gap workflow and PeakFormer (Oct 20, afternoon)
Workflow wf_2a12e406-61f compared the public lineage's COCONUT ranking with ours mechanism by mechanism, each claim
verified offline (full plan and verdicts: docs/coconut_gap_plan.md). Verdicts:
- Fingerprint model quality: HOLDS, and the bound below was too small. Matching the public model's pairwise win rate
  against COCONUT decoys (.857 -> ~.94) needs a +0.03 shift of the truth's fp_cos, not +0.009: stage-1 CV all .892 ->
  .917, cls2 .901 -> .930, weighted cls2 .677 -> .775 (+0.10). Our within-formula MRR of fp_cos on class-2 COCONUT
  truths is .408 (theirs .639 for f.z). -> Priority 1.
- Fragmentation scorer (depth-mixed explained share + cleavage coverage): holds, small (+0.002..0.007 LB; GLACIER
  already reorders isomers for ~93% of molecules). -> Priority 2, CPU.
- Rejected (offline checks): documentation-blind ranker + z+0.15 pop (-0.027 weighted cls2), source-tiered list
  composition (+0.000..0.005 on pool v4: pool v4 already removed most PubChem crowding), deeper shortlist before GLACIER
  (~+0.001), metric-identity keys (CV -0.0004 +- 0.0016), RRF with a second engine, core-only popularity prior,
  analog-search details, tautomer-invariant Tanimoto.
- Also tested here: slot-keeping within-formula GLACIER swaps (the public fusion) on fwd_v5w_a0.5 OOF: 0.71 vs our
  stage 2 0.89 (weighted cls2 0.59 vs 0.68) -> no.

PeakFormer (casmi/peakformer.py, scripts/train_pf.py), own implementation of the idea:
- peak tokens = exact-m/z and neutral-loss sinusoids (0.01-1000 Da) + intensity; global token = precursor + metadata;
  transformer encoder;
- two heads: bit logits z over a 7-block RDKit fingerprint (casmi/richfp.py: Morgan r2/r3 4096, feature-Morgan,
  paths <= 6, atom pairs, torsions, MACCS; 15,440 informative bits, scripts/prep_pf_data.py) scored as f.z, and a
  cosine embedding;
- trained against in-batch and 10 ppm window negatives plus BCE, with merge augmentation (1-3 same-structure,
  same-adduct spectra), peak dropout, ppm/intensity jitter;
- inference views = each spectrum + merged spectra per adduct.
Kaggle GPU quota is exhausted until Oct 24, so three CPU variants train for 11 h (kernels casmi26-pf-cpu-a/b/c:
d=192 x 4 layers seeds 0/1, d=128 x 3 layers; ~2.3 s/step at batch 256, ~2 epochs). Data: bryantyu/casmi26-pf-data.
Next: pf_* features on ret10 rows -> stage-1 CV gate -> v11 pipeline; a larger model on GPU after Oct 24.

PeakFormer early check (Oct 20, local run d=192 x 4, structure-balanced, checkpoint after 3,000 steps = 0.33
epoch): internal validation f.z MRR 0.584 / top-1 0.439 (FP v3: 0.549 / 0.392 after 4 epochs). Class-2 COCONUT
non-train validation truths (scripts/eval_pf.py, within-formula): pf_bern MRR 0.419 vs fp_cos 0.430, wins vs COCONUT
decoys 0.849 vs 0.829, vs train decoys 0.832 vs 0.794. Stage-1 CV (ret10 rows + pf): all 0.8951 -> 0.8991, cls1
+0.003, cls2 +0.0046, weighted cls2 -0.007 +- 0.010 (paired); replacing fp by pf: weighted -0.016 +- 0.011 -> the
early model adds on top of fp but does not replace it; final models (2 epochs, ensemble) tonight.

PeakFormer final local model (9 h CPU, ~2.6M samples; internal val f.z MRR 0.631): class-2 COCONUT non-train
truths within-formula MRR 0.463 (fp_cos 0.430), all-candidates MRR 0.431 (0.382), wins vs train decoys 0.859 (0.794),
vs COCONUT decoys 0.878 (0.829). Stage-1 CV (ret10pl rows):
| config | all | cls1 | cls2 | cls2w |
| a0.5 | .8951 | .8804 | .9025 | .6931 |
| a0.5+pf | .8989 | .8817 | .9075 | .6894 |
| a0.5+pf-fp | .8996 | .8804 | .9093 | .6850 |
| a1 | .8820 | .8820 | .8820 | .6872 |
| a1+pf | .8939 | .8836 | .8991 | .7103 |
Paired: a1+pf vs a0.5 weighted cls2 +0.017 +- 0.011 (cls1 +0.003, cls2 -0.003); a1+pf vs a1 +0.023 +- 0.012. By
truth source (a0.5 -> a0.5+pf, weighted): COCONUT -0.008, PubChemLite-only +0.080 (n 112), bio-only +0.006. The
PeakFormer pays off when the ranker is pushed off documentation (alpha 1) -> v11 = alpha 1 + pf (pipeline T=v11l
started with the local model; ensemble with the Kaggle CPU models next).

v11l pipeline (alpha 1 + pf from the local PeakFormer; T=v11l ROWS=ret10pl ALPHA=1 scripts/pipeline_v8d.sh): ranker
gain led by pf_bern_gap (6x the next feature). Validation (held-out val rows): stage 1 0.9028 (cls1 .8466, cls2
.9329) vs v10 0.8912 (.8297 / .9241); stage 2 0.9011 (.8374 / .9352) vs v10 0.8938 (.8228 / .9317). Stage-2 5-fold
CV on train queries: all .8971, cls1 .8818, cls2 .9047, weighted .6993 vs v10 .8942 / .8744 / .9040 / .6838. List
confidence retrained on v11l scores (pother_v11l, AUC 0.891). Kernels casmi26-own-s2v11l and -s2v11lpc-ad.

Kaggle CPU PeakFormers (11 h): a d=192x4 seed 0 val f.z 0.651 / sum 0.650; b (merge 0.5) 0.634 / 0.635; c d=128x3
0.633 / 0.638; local d (structure-balanced half batches) 0.631 / 0.630. Ensemble a+b+c+d (cos mean, z-scored f.z
mean): COCONUT non-train within-formula MRR 0.474 (sum) / 0.469 (f.z), wins vs COCONUT decoys 0.888; stage-1 CV
a1+pf .8918 / .8839 / .8959 / w .7056 (local-only .8939 / .8836 / .8991 / .7103); v11 pipeline val stage 2 .8948
(v11l .9011) -> the z-scored ensemble does not beat the single local model; v11a (model a alone) running.

PubChem proposal channel (Oct 20): the public lineage's "other" source (answers outside train + COCONUT) is worth
0.057 on the LB and comes from a separate PubChem channel (10 ppm window, top by documentation, f.z + 0.25 pop, slot
insertion when the library match is weak). Ours: tier = 15.8M documented PubChem compounds outside pool v4 (sid >= 5
or pmid >= 1; scripts/build_pubchem_tier.py, dataset bryantyu/casmi26-pc-tier), casmi/pcchannel.py, run_own PC_TIER.
Offline (scripts/eval_pcchannel.py, 100 class-2 validation truths placed into their tier window, median window 1,394,
PeakFormer after only 1,500 steps): n_pop 2000, lam 0 -> MRR 0.194 (doc-weighted 0.153); lam 0.25 -> 0.395 (0.125);
lam 0.5 -> 0.472 (0.095); n_pop 500 loses the weighted truths (0.06). -> use n_pop >= 2000 and a small lam; LB probe
after the trained PeakFormer exists.
Same check with the epoch-1 checkpoint (250 truths, median window 1,313): n_pop 5000, lam 0 -> MRR 0.380 (weighted
0.463), lam 0.05 -> 0.429 (0.458), lam 0.25 -> 0.543 (0.403); n_pop 2000 lam 0 -> 0.333 (0.378). Optimistic (the
truth is a training-like structure among PubChem decoys), but the channel is strong -> LB probe casmi26-pc-only
(every list = the channel's top 25; epoch-2 checkpoint, n_pop 5000, lam 0.05) replaces the s2src slot on Oct 21; it
is directly comparable to the public lineage's "other" 0.057.
Final local PeakFormer (same check, 250 truths): lam 0 -> MRR 0.485 (weighted 0.581), lam 0.05 -> 0.540 (0.547),
lam 0.1 -> 0.574 (0.540), lam 0.2 -> 0.602 (0.505); n_pop 5000 = 20000 (windows rarely exceed 5000).
Placement: list-confidence model (casmi/pother.py, scripts/train_pother.py): p(answer outside the pool) from the
stage-1 top-30 list (top score, margins in score SDs, leading candidates' evidence and sources), trained on simulated
lists with and without the truth; CV AUC 0.882 (doc-weighted class 2 0.737). Policy simulation (sim lists, assumed
channel quality pessimistic top-1 .12 / MRR .17 or optimistic .30 / .40, share of "other" molecules 0.2 / 0.35):
"slots 1,3,5,7,9 if p > 0.7 else 4,8,12,16,20" is positive in all four cases (+0.001 / +0.015 / +0.021 / +0.050),
fixed 4,8,12,16,20 likewise (+0.006 / +0.012 / +0.016 / +0.031); proposals at the top for every list (1,3,5,...)
cost -0.13 to -0.28. The simulated main lists are far better than the test's (sim MRR 0.77 vs LB ~0.33), so
displacement costs are overstated here. Kernel knobs PC_POTHER=pother_v1.txt PC_TH PC_SLOTS_HI.
Kernels casmi26-own-s2v10pc-ad (adaptive) and -fx (fixed 4,8,12,16,20), local epoch-2 PeakFormer: on the visible
test.parquet both equal v10, because that file is a placeholder made of training spectra (every precursor occurs
exactly in train.parquet) -> library similarity ~1 -> the lib >= 0.9 gate skips 398 of 400 molecules. The hidden
rerun is where the channel acts; visible outputs cannot validate it (runtime with proposals for every molecule:
pc-only kernel 54 min).

Fragment evidence v2 (casmi/fragscore.py, Priority 2): depth-mixed explained share + 1-cut coverage on the stage-1
top 30 (fwd_v10 rows), stage-2 5-fold CV, two fold seeds: weighted cls2 +0.0047 / -0.0019, cls1 +0.0015 / -0.0011,
all +0.0013 / +0.0005 -> noise-level, gate (+0.004 on both seeds) failed; not shipped (S2_EXTRA=frag2 kept).

## Fingerprint-model bound (Oct 20, design-panel A2) -- superseded, see above
Within-formula top-1 of fp_cos on validation is 0.351 (1,321 queries with >= 2 isomers). Adding 0.07 SD to the truth's
fp_cos (top-1 -> ~0.45, i.e. a much better fingerprint model) and re-running stage-1 CV on the v10 rows: all 0.895 ->
0.903, cls2 0.903 -> 0.912, weighted cls2 0.693 -> 0.718. Even that is ~+0.01 on the LB -> a stronger fingerprint
model (FP v5 / GPU) is not the lever for the 0.12 gap; not pursued now.

## Full-library matching (Oct 20, design-panel C1)
The direct library channel used lib10 (<= 10 spectra per structure and polarity, prioritised [M+H]+/[M-H]-, 15-65 eV,
TOF). scripts/add_libfull_feats.py recomputes it against every training spectrum with a test adduct (lib_all.npz,
2,057,184 spectra; same hiding as the dump). Validation, library similarity alone, class 1: 0.834 -> 0.861; with the
truth's same-instrument spectra hidden (XINST=1, the realistic cross-instrument case): 0.781 -> 0.860.
Stage-1 5-fold CV with lib / lib_n / lib_r / lib_gap replaced (ret10 rows): alpha 0.5 0.887 -> 0.895, cls1 0.861 ->
0.880, cls2 0.900 -> 0.903, weighted cls2 0.673 -> 0.693; alpha 0 0.888 -> 0.896. -> pipeline v10 (T=v10
ROWS=ret10), kernel with LIB_CAP=100000.

## Enveda-180 Zenodo release (Oct 20)
Zenodo 21346580 (CC BY 4.0, explicitly allowed): enveda-180-filtered.parquet has 1,175,202 spectra of 184,088
structures; 3,717 of them are not in train.parquet (14,341 spectra), but all carry Enamine catalogue ids (synthetic
screening compounds, e.g. thiazole carboxamides) and only 6 are in our pool (PubChemLite), none in COCONUT -> not the
natural-product test chemistry; not added. The full (unfiltered) release is 2.6 GB and was not checked.
Design-panel decision tree for the probe outcomes: docs/next_steps_oct20.md.

## Post-hoc popularity knob (Oct 20)
The public lineage re-ranks by z(ranker) + 0.15 * pop. On top of stage2_v5w_a0.5 (single validation split, pop =
log PubChem SIDs + log PMIDs): lambda +0.10 all 0.902 -> 0.905, cls1 0.840 -> 0.858, but weighted cls2 0.728 -> 0.698;
lambda -0.10 the reverse (weighted cls2 0.747, cls1 0.799). Popularity only on in-train candidates promotes training
decoys in class-2 queries (cls2 0.935 -> 0.917 at +0.1). No free lunch: a trade-off the LB alpha A/B already probes.
Not used. Probe kernels casmi26-probe5-{train,coco,other} (alpha-0.5 lists restricted per source, same key sets as
the public-lineage probes) are running for a like-for-like comparison.

## Source-decomposition probes of the public lineage (Oct 19)
kernels/casmi26-fprobe-{train,coco,other}: the forked 0.451 notebook (CPU variant) with its final lists restricted
to training structures / COCONUT-not-train / everything else (keys from bryantyu/casmi26-srckeys). Compared with our
probes (pool v3 lists: train 0.146, coco 0.191, pubchem pending; full 0.295) they locate the 0.12 gap by source.

## Lesson from the class-3 probe (Oct 17)
The novel-only list scored 0.010 public, against ~0.045 expected from the PubChem-absent proxy (MRR 0.10 x ~45% class 3).
The proxy overstates reach ~4-5x: absent training molecules (reverse-metabolomics conjugates, synthetic analogs) are
much easier than Enveda's novel test compounds. Class 3 via analog derivation/conjugation is worth at most ~0.005-0.01
as an overlay. Pivot: use the same machinery (analog derivation + site localisation) for class-2 isomer ranking
(known-only probe pending).

## Forward models (Oct 18)
- Isomer discrimination inside the truth's formula group (enveda-180 validation queries, stage-1 top 20, n=77-148):
  GLACIER top-1 0.743, ICEBERG 0.714, own FP encoder 0.568, analog channel 0.74; FP+GLACIER 0.804,
  FP+analog+GLACIER+ICEBERG 0.870. enveda-180 is not in MassSpecGym, so no forward-model leakage.
- GLACIER is ~6x faster than ICEBERG on CPU (8.6 vs 1.35 predictions/s/thread) -> usable as a ranker FEATURE:
  two-stage ranker (stage 1 = current ranker, top 30 -> GLACIER -> stage 2 with GLACIER features).
- Compliance: ms-pred code (MIT) + MassSpecGym weights (allowed by host). LinSATNet stand-in replaced by the original
  MIT LinSATNet 0.1.3 (max diff 4e-4); DGL / torch_scatter stand-ins replaced by our own fwdshim/ (scores identical,
  max diff 0.0). Still third-party: the runner glue (ice_runner.py / gl_runner.py / fuse.py, ahmedberatozer, licence
  "Other") -> own runner on ms-pred's official MIT predict_smis_joint.py (TreeProcessor + IntenDataset.collate_fn),
  then recompute the GLACIER training features with it.
- Stage-2 CV on validation (1107 queries, 76% with a covered positive spectrum): stage 1 0.823, stage 2 without
  forward features 0.834, with GLACIER 0.844 (covered queries 0.840 -> 0.865).
- Own-runner stage 2 (pool v2, stage 1 = ranker v2 with out-of-fold scores, 4/6 training shards, full validation
  of 1093 queries, 76% with a covered spectrum): stage 1 0.825 -> stage 2 without GLACIER 0.827 -> with GLACIER
  0.840 (cls1 +0.005, cls2 +0.021); covered queries 0.842 -> 0.861.
- Stage 2 on v2c (shipped as stage2_v2): validation 0.842 -> 0.850 (covered 0.850 -> 0.861); boosting from the
  stage-1 margin instead: 0.851 (noise level, and it reintroduces the score-scale mismatch) -> not used.
- GLACIER for novel candidates (class 3): on 178 reachable enveda-only absent-from-PubChem proxy queries (not in
  MassSpecGym) novel-LTR MRR 0.624 -> LTR + 2 x GLACIER (z-scores) 0.725. Reach (15.8% of those queries) is the
  bottleneck, so the LB effect would be small (novel-only probe 0.010 today).
- Pool v4 = pool v2 + 52k ChEBI / LIPID MAPS structures (CC BY 4.0), membership flag in_bio (also a reachable
  source in simulation): stage-1 v4c validation cls1 0.827 / cls2 0.926 (v2c 0.806 / 0.862), stage 2 0.893
  (stage 1 0.891). in_bio is the 3rd most important feature -- likely inflated (training structures have public
  spectra, hence are well known and over-represented in ChEBI); only the LB can tell (kernel casmi26-own-s2v4).
- Analog library with 10 spectra per structure instead of 3: best analog Tanimoto to the truth 0.585 -> 0.599
  (>= 0.7: 37.7% -> 40.3%, 300 validation queries) for 3x search cost -> not adopted for now.
- Ranker v2c = v2 recipe without the artifact class-1 queries: validation 0.815 -> 0.842, test-like 0.851 -> 0.860.
- The repository has no LICENSE file yet: the winning entry must be open source -> the owner should add one (MIT
  fits the components used) before the final submission.

## Validation sets (Oct 18)
- Visible test.parquet is a placeholder copied from enveda-180 training rows (every molecule has a spectrum identical
  to a training one); the hidden test replaces it at scoring time. It says nothing about the hidden molecules.
- Test-like validation (tl:, 263 timsTOF val structures): ranker v2 0.860 (class 1 0.739 < pure library similarity
  0.812 -> the ranker under-trusts cross-instrument library matches; class 2 0.938, PMID count alone 0.708).
- Rare natural products (class 2, COCONUT, <= 5 PMIDs, 267 queries): ranker v2 0.702 vs no-prior 0.560 -> priors
  still help on poorly documented truths.

- Stage-1 v3 (FP v3 features, + 2483 test-like training queries): standard validation cls1 0.819 / cls2 0.840
  (v2 0.820 / 0.829); test-like validation cls1 0.706 / cls2 0.934 (v2 0.743 / 0.944) -> not adopted. Class-1
  failures: a COCONUT-only candidate without library spectra outranks the training structure whose cross-instrument
  library similarity is moderate (median 0.59); the learned class prior (most answers are not training structures)
  drives it. Whether that prior is right for the hidden test is what the train-source LB probe measures.
- Original simulation artifact: 27% of class-1 validation queries leave the truth without any library spectrum
  (all its library spectra were the hidden query spectra) -> effectively class 2 with in_train = 1.

## Simulation leak (Oct 18)
Class-2 queries whose truth is NOT reachable without train.parquet still score MRR 0.87 on validation: a hidden
training structure is the only candidate with no source flag. Training already filters these (--reach-pmid), but
validation numbers must use the same filter. Class 1 was too easy (same-instrument spectra of the structure stay in
the library). Test-like mode (tl:) fixes both: timsTOF queries only, class 1 keeps only other-instrument spectra.

## Compliance notes (public pipeline we forked)
The fork mounts dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets, byte copies of:
| component | author | licence | reproducible? |
|---|---|---|---|
| casmi26-v4b/v3 models, fpnet-full1, v2-pool (engine 1 code + FPNet/ranker weights) | ahmedberatozer | "other": non-commercial w/ attribution | code shipped; training not verified |
| casmi26-iceberg / casmi26-glacier | ahmedberatozer (ms-pred MIT, MassSpecGym weights) | MIT weights | allowed by host |
| casmi26-fp-models-v2 (FPNet) | prvsiyan | CC-BY-NC-SA-4.0 | training script published in a kernel |
| casmi26-ranker-features (rank_train.npz) | prvsiyan | CC-BY-NC-SA-4.0 | construction unpublished -> NOT reproducible |
| coconut-casmi26-candidates, chebi-lipidmaps | prvsiyan | CC-BY-NC-SA-4.0 | derived from COCONUT/ChEBI |
| casmi26-simulated-ranker-rows | megayak | CC0 | ? |
| hengck23-dreams-enveda-casmi26 | hengck23 | MIT | ? |
| casmi-chebi-lmsd-structures | sergiomalv | CC-BY-4.0 | yes |
Host rules: winning solution must be licensable to the sponsor (open source, commercial use allowed) and every
weight reproducible from public code + permitted data. NC licences inherited only from train.parquet are tolerated.
=> The forked 0.45 pipeline is at best uncertain for a prize. A prize-eligible final entry needs our own code and
weights for every learned component (allowed third-party: MassSpecGym ICEBERG/GLACIER, DreaMS, FRIGID, PubChem counts).

## Next (updated Oct 18, midday)
Own runner for GLACIER (casmi/fwdrun.py: official ms-pred source + original MassSpecGym checkpoint + fwdshim):
isomer top-1 inside the truth's formula group 0.838 (public runner 0.743, ICEBERG 0.714, own FP encoder 0.568);
ICEBERG adds nothing on top -> dropped. Pipeline under construction:
1. Stage 1 = ranker v2 on pool v2 (LB 0.315). Stage-2 training rows = pool-v2 training queries with OUT-OF-FOLD
   stage-1 scores (scripts/oof_stage1.py) -> top 30 -> GLACIER on Kaggle CPU workers (casmi26-glw-*) ->
   scripts/train_stage2.py -> kernel casmi26-own-s2 (datasets casmi26-pool2, casmi26-glacier).
2. Test-like stage 1: FP v3 (0.1 Da bins, test-like keys held out) -> test-like training dump (tl_keys.txt, pool v2)
   -> stage-1 ranker trained on ret2 + test-like rows (fixes class-1 under-trust of cross-instrument library hits).
3. Oct 19 submissions (all kernels already run and checked):
   scripts/safe_submit.sh casmi26-own-s2 2 "OWN v4: pool v2, stage 1 v2c, stage 2 with own GLACIER runner"
   scripts/safe_submit.sh casmi26-own-v2c 1 "OWN v4 stage 1 only: ranker v2c (artifact-free), pool v2"
   kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k bryantyu/casmi26-srcprobe -v 1 \
     -f submission_train.csv -m "PROBE: lists restricted to training structures"   (same for _coco, _pubchem)
   Pool-v4 alternative: scripts/safe_submit.sh casmi26-own-s2v4 1 "OWN v4 + ChEBI/LIPID MAPS pool (in_bio)"
   (5 slots for 6 candidates: s2, v2c, s2v4, probe_train, probe_coco now; probe_pubchem the day after)
   Earlier plan for these slots: own-ice (pool v3 + ICEBERG fusion vs 0.295), stage 2, source probes
   (submission_train / _coco / _pubchem from casmi26-srcprobe v1: which candidate sources hold the answers).
4. Novel-candidate overlay (expected <= +0.01) and a stronger FP model on GPU (quota refresh Oct 24) after that.
