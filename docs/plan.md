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
| 10-24 | casmi26-own-s2v11ltok3 v1 | v11l + instrument token 3 (TOF) for main-pool PeakFormer / FP scoring | 0.315 |
| 10-24 | casmi26-own-s2v11ltok0 v1 | v11l + instrument token 0 (unknown) | 0.316 |
| 10-24 | casmi26-probe11tok3-coco v1 | PROBE: v11l tok3 lists restricted to COCONUT-not-train | 0.200 |
| 10-24 | casmi26-probe11-train v1 | PROBE: v11l lists restricted to training structures | 0.149 |
| 10-24 | casmi26-own-s2v13 v2 | stage 1 / stage 2 retrained on token-3 features + token-3 override | 0.313 |
| 10-25 | casmi26-probe5w-coco v2 | PROBE: v5w lists restricted to COCONUT-not-train | 0.208 |
| 10-25 | casmi26-probe5w-train v2 | PROBE: v5w lists restricted to training structures | 0.156 |
| 10-25 | casmi26-own-s2v5w-olast v2 | v5w lists, source tiering (train / COCONUT first, other after, nothing dropped) | 0.324 |
| 10-25 | casmi26-own-s2v17 v3 | v5w recipe (31 leaves, alpha 1, lib10) + PeakFormer d features | 0.334 |
| 10-25 | casmi26-own-s2v17b v1 | v17 with the debiased PeakFormer b (training-structure negatives, instrument-free, GPU 5 h) replacing d | **0.339** (best eligible) |
| 10-26 | casmi26-own-s2v17bb v1 | v17b + FP model b (GPU, training-structure negatives, instrument-free) replacing the FP model | 0.322 (FP-b hurts) |
| 10-26 | casmi26-probe17b-coco v1 | v17b lists restricted to COCONUT-not-train | 0.211 |
| 10-26 | casmi26-probe17b-train v1 | v17b lists restricted to training structures | 0.150 |
| 10-26 | casmi26-specpfd-coco v1 | PeakFormer d score alone, COCONUT-not-train candidates | 0.129 |
| 10-26 | casmi26-specpfb-coco v1 | PeakFormer b score alone, COCONUT-not-train candidates | 0.160 |
| 10-27 | casmi26-own-s2v18c v1 | clean-room rebuild: own ChEBI/LMSD set (in_bio leak removed), FP v1 retrain, one fresh GLACIER pass | **0.339** (= v17b; eligible base) |
| 10-27 | casmi26-probe18c-coco v1 | v18c lists restricted to COCONUT-not-train | 0.215 |
| 10-27 | casmi26-probe18c-train v1 | v18c lists restricted to training structures | 0.157 |
| 10-27 | casmi26-zen-probe v1 | direct spectral match to the 3,717 Zenodo-only Enveda-180 structures | 0.000 |
| 10-27 | casmi26-own-s2v17bpop v1 | v17b lists + popularity re-rank z(score) + 0.15 (ln SID + ln PMID) | 0.323 (rejected) |
| 10-28 | casmi26-specpfsyn-coco v1 | PF-syn matched ensemble (s31+s32) score alone, COCONUT-not-train | **0.173** |
| 10-28 | casmi26-specpfctl-coco v1 | PF-b fine-tune control ensemble (s31+s32) score alone, COCONUT-not-train | 0.149 |
| 10-28 | casmi26-own-s2v18cw v1 | v18c retrained without class-1 query duplication (C1_DUP=1) | 0.317 (class-1 weight matters) |
| 10-28 | casmi26-probe18cw-coco v1 | v18cw lists restricted to COCONUT-not-train | 0.208 |
| 10-28 | casmi26-probe18cw-train v1 | v18cw lists restricted to training structures | 0.148 |

## Oct 28 slots (pre-registered Oct 27 01:40 UTC, before the PF-syn matched re-run read and before v18cw is built)
| slot | kernel | pair | prediction | rule |
|---|---|---|---|---|
| 1 | casmi26-specpfsyn-coco (pf_bern alone, COCONUT-not-train, specpfb-coco setup; OWN_PF = matched syn ensemble s31 + s32) | slot 2 | 0.49 x (0.22 a_A + 0.78 a_B) +- 0.012, a = offline coco delta of the ENSEMBLE rows (filled in at the 02:40 read; diagnostic prior +0.014) | delta >= +0.010 transfer confirmed -> start the full-pool simulation (casmi26-simw2-*); (-0.010, +0.010) uninformative; <= -0.010 STOP PF-syn |
| 2 | casmi26-specpfctl-coco (same, OWN_PF = ctl ensemble s31 + s32) | specpfb-coco 0.160 | 0.160 +- 0.012 | outside [0.140, 0.180]: setup drift; F = slot2 - 0.160 >= +0.012: fine-tuning alone pays |
| 3 | casmi26-own-s2v18cw (v18c retrained with C1_DUP=1: class-1 queries not duplicated) | v18c 0.339 | 0.341 +- 0.012 | retrain rule: adopt >= 0.347 with slot 4 >= 0.220 and slot 5 >= 0.151; <= 0.333 class-1 weight matters the other way |
| 4 | casmi26-probe18cw-coco | probe18c-coco 0.215 | 0.222 +- 0.010 | mechanism: COCONUT answers ranked up |
| 5 | casmi26-probe18cw-train | probe18c-train 0.157 | 0.152 +- 0.006 | class-1 cost cap -0.006 |
Rationale (slot 3): the source-restricted probes of v18c sum to ~0.385 (train 0.157 + COCONUT 0.215 + other ~0.013) while
the full lists score 0.339 -> ~0.046 is lost where training and COCONUT candidates compete in one list. Stage 1 / stage 2
duplicate class-1 queries (effective class-1 share 46% of the training weight; comment: "test mix closer to 45:55",
never LB-tested alone), while the training probe implies ~17-27% class 1 among the hidden molecules. C1_DUP=1 trains at
the simulated 30:70 mix. Any kernel not COMPLETE and verified by 00:05 is replaced by nothing (no unverified submits).
PF-syn matched read (Oct 27 03:40; syn arms re-run with PF_STOP_AT 6500 of 7500 = the guard-cut ctl arms; both logs
"PF_STOP_AT reached at step 6500 of 7500", no guard cut -> VALID): half A pf_cos delta win vs training decoys s31 +0.031
[+0.007, +0.055], s32 +0.023 [-0.002, +0.048], seed mean +0.027 [+0.007, +0.048]; vs COCONUT +0.021 / +0.024 / +0.022;
excess syn vs ctl +0.015 vs +0.025 / +0.023 vs +0.022 / +0.019 vs +0.024; pf_cos_gl +0.009 / -0.001 / +0.004 (files
/home/user/work/pfsyn_eval/gate_s31.txt, gate_s32.txt, gate_seedmean.txt). No seed passes (+0.06 size; s32 also fails
CI, excess and gl) -> BRANCH FAIL-near (seed mean >= +0.02 with CI > 0, both seeds > 0, guards hold on the mean): no
simulation; the Oct 28 COCONUT probe pair decides. Offline coco probe on the 2-checkpoint ENSEMBLE rows (the LB kernels'
OWN_PF): syn vs ctl +0.0176 [+0.004, +0.031], A +0.0082, B +0.0270 -> slot-1 prediction 0.49 x (0.22 x 0.0082 + 0.78 x
0.0270) = +0.011 +- 0.012; ctl ensemble vs PF-b +0.0056 -> slot-2 prediction 0.163 +- 0.012. Exposure gap ~0.
Checkpoints in casmi26-models (md5): pf_ctl_s31 00edd01c, pf_ctl_s32 a0521dd0, pf_synm_s31 61d1a1a5, pf_synm_s32 cd6ed368.
Kernels casmi26-specpfsyn-coco / -specpfctl-coco generated (code identical to casmi26-specpfb-coco apart from OWN_PF);
pushed when the v18cw GLACIER workers free their sessions.
v18cw stage D (01:25; C1_DUP=1 BASE=v18c scripts/build_v18c.sh ... v18cw, FP rows hard-linked from ret18c): stage 1 val
cls1 .815 / cls2 .8227 (v18c .8268 / .8159), OOF .8229. Stage E waits for free sessions (<= 4 of ours).
v18cw DONE 05:12 (4 CPU GLACIER workers casmi26-glwv18cw-0..3, coverage .783 / .754): stage 2 val .8341 (cls1 .8237 /
cls2 .8396) vs v18c .8314 (.8301 / .8321). Models ret_ranker_v18cw (md5 1d126e99), stage2_v18cw (09384a68) in
casmi26-models; kernel casmi26-own-s2v18cw pushed 05:20 (code identical to s2v18c apart from the ranker names), probes
casmi26-probe18cw-coco / -train generated (pushed when sessions allow); spectral probes casmi26-specpfsyn-coco /
-specpfctl-coco pushed 05:11.
Verified 05:35: casmi26-specpfsyn-coco and -specpfctl-coco COMPLETE, 394 / 400 rows, PeakFormer members 2 (Oct 28 slots 1-2
ready). casmi26-probe18cw-coco / -train pushed 05:36.
Verified 07:05: casmi26-own-s2v18cw v1 (400 / 400 rows with candidates, GLACIER 371 / 400, no Traceback),
casmi26-probe18cw-coco v1 (393 / 400 COCONUT rows), casmi26-probe18cw-train v1 (400 / 400) -> all five Oct 28 kernels ready.
Oct 29 masked probes casmi26-mask18c-train / -coco pushed 07:05.
Masked probes COMPLETE (logs 09:52): train kept 3,045 / masked 6,798 slots, coco kept 4,331 / masked 5,512 (both 9,843 =
the same lists), 400 rows, GLACIER 371 / 400, no Traceback. List identity (output endpoint 429 all morning) checked by
source composition instead: the local v18c v1 lists hold 3,043 train / 4,331 COCONUT / 2,469 other slots (9,843) vs
the masked kernels' kept 3,045 train / 4,331 COCONUT -> NEAR-identical (2 of 9,843 slots differ: the known CPU
nondeterminism of GLACIER-dependent reruns; each LB reading comes from its own hidden rerun anyway). Interpretation of
the VOID rule recorded here: valid unless the reading fails M_T + M_C in [F - 0.016, F + 0.001].

## Oct 29 slots (pre-registered Oct 27 07:00, before the Oct 28 readings)
Masked split probes E1 on the base (v18c unless v18cw is adopted on Oct 28; kernels casmi26-mask18c-train / -coco,
scripts/make_probe_kernel.py ... mask): M_s = full-list RR carried by source-s answers, L_s = P_s - M_s.
| slot | kernel | prediction (80%) | derived |
|---|---|---|---|
| 1 | casmi26-mask18c-train | 0.139 [0.128, 0.150] | L_T = 0.157 - M_T = 0.018 [0.007, 0.029] |
| 2 | casmi26-mask18c-coco | 0.194 [0.184, 0.204] | L_C = 0.215 - M_C = 0.022 [0.012, 0.031]; M_O = 0.339 - M_T - M_C = 0.007 [0.002, 0.013] |
VOID (fix + rerun) if the base lists differ from the v18c v1 output, or M_T + M_C > F + 0.001, or < F - 0.016. Rules:
COCONUT-heavy (L_T <= 0.010 and L_C >= 0.028) -> Oct 30 library-gated COCONUT prior SRC_COCO_LIBGATE=0.3,0.8 (fixed now;
adopt >= +0.005); train-heavy (L_T >= 0.025) -> offline class-1 library-trust work only; balanced -> close the
interleaving line for this window. If valid and M_T <= 0.150, mask-train replaces probe-train as the class-1 cost cap
(>= parent - 0.006).
Slots 3-5 (pre-registered Oct 28 01:30, after the v18cw reading, before v18c3 is built):
| slot | kernel | pair | prediction | rule |
|---|---|---|---|---|
| 3 | casmi26-own-s2v18c3 (v18c with C1_DUP=3: class-1 queries tripled) | v18c 0.339 | 0.345 +- 0.012 (monotone in the class-1 weight, diminishing: the 1 -> 2 step was +0.022) | retrain rule: adopt >= 0.347 with slot 4 >= 0.220 and slot 5 >= 0.151; <= 0.333: the class-1 weight optimum is 2 |
| 4 | casmi26-probe18c3-coco | probe18c-coco 0.215 | 0.218 +- 0.010 | |
| 5 | casmi26-probe18c3-train | probe18c-train 0.157 | 0.160 +- 0.006 | |
v18c3 stage D (01:24): stage 1 val cls1 .8316 / cls2 .8171 (v18c .8268 / .8159); stage E 3 CPU GLACIER workers
casmi26-glwv18c3-0..2 pushed 01:32 (GLW_N=3). Full-pool simulation: casmi26-simw2-0 pushed 01:20 (1 of 24).
v18c3 DONE 04:24: stage 2 val .8351 (cls1 .837 / cls2 .834) vs v18c .8314 (.8301 / .8321), v18cw .8341; models
ret_ranker_v18c3 (md5 69622e2c), stage2_v18c3 (ba659dd5) in casmi26-models; kernels casmi26-own-s2v18c3 and
casmi26-probe18c3-coco pushed 04:32 (code identical to s2v18c apart from the ranker names); probe18c3-train waits for a
session (owner's ntl-dv-full running -> cap 3).
Verified 06:05: casmi26-own-s2v18c3 v1 (400 / 400 rows, GLACIER 371 / 400, no Traceback), casmi26-probe18c3-coco v1
(394 / 400 COCONUT rows); casmi26-probe18c3-train pushed 06:05. Simulation: casmi26-simw2-0 COMPLETE (sim2.jobs_0
69,507 / 69,507 and sim2.jobs_24 69,903 / 69,903 predictions, ~2.0 h per file -> ~4 h per worker); simw2-1 and -2 pushed
06:05; simw2-3 refused: "Maximum batch CPU session count of 5 reached" = the account-wide Kaggle limit (5 CPU sessions,
shared with the owner) -> keep <= 3 of ours.
Verified 07:35: casmi26-probe18c3-train v1 (400 / 400 training rows, GLACIER 371 / 400) -> all five Oct 29 kernels
verified. Simulation: simw2-3 pushed 07:35 (running: 1, 2, 3; done: 0).
Rollout automation: recurring trigger trig_01AiNse4as9CLD231rSo5Vuv (every 4 h at :34) pushes the next workers within the
cap, verifies completed ones, pushes casmi26-syn-prep2 when all 24 are done and then deletes itself.
07:56: rollout check made HOURLY (cron 34 * * * *; at 4 h a finished worker waited up to ~2.5 h). A 4th worker
(simw2-4) was refused at 07:55 although the owner's kernels were all COMPLETE ("Maximum batch CPU session count of 5
reached": interactive notebook sessions count too) -> stays at 3 of ours.
TPU line (08:00): the owner's last TPU kernel (ntl-tpu-probe) finished Oct 26; this week's 20 h TPU quota (resets
Sat Oct 31 00:00) is otherwise unused. Our XLA path (scripts/train_pf_xla.py) has never run on a TPU (only PF_DEV=cpu).
kernels/casmi26-pf-tpu-smoke (DEV=tpu scripts/make_pf_cpu_pilot.py casmi26-pf-tpu-smoke ctl,syn 300 0.4 200 25 31: the
TPU pilot at 300 steps per arm, pilot syn set) pushed 07:59 (QUEUED). Pass = both arms reach "PF_STOP"/step 300 without
a Traceback or watchdog exit, a stable s/step after compilation, val numbers comparable to the CPU arms. If it passes:
(1) synP-partial on the TPU = PF-b fine-tuned with PF_SYN = syn-prep of the simulation files finished so far (workers 0-3
= sim2.jobs_0-3 + 24-27, ~1/6 of the pool, a deterministic subset -> eligible), PF_SYN_P 0.25, step count from the smoke's
s/step (<= 7.5 h); syn-prep on a CPU kernel (scripts/make_syn_prep.py, prep needs RDKit); read on Oct 30 (spectral COCONUT
probe vs specpfb-coco 0.160 / specpfsyn-coco 0.173, plus a v18c PeakFormer swap if built in time);
(2) synP-full on the TPU once all 24 workers are done (before the quota reset if hours remain, else the next week) --
replaces the two-seed CPU synP fine-tune.
GLACIER cache for LB-read builds (08:10): scripts/build_v18c.sh GL_HAVE=/home/user/work/fwd_v18c* passes
build_fwd_jobs.py --have (pairs already scored by earlier passes on the same rows are skipped; stage F reads those files
too); GLW_N=1 runs the rest on one worker. Dry run (v18c3 val vs the v18c cache): 26,261 top-30 pairs -> 2,323 GLACIER
predictions left (572 items) -> a PeakFormer swap needs ~15k predictions (~30 min on one CPU worker) instead of 3 workers
x 3 h. Not a fresh pass (G2): a final pick gets a GL_HAVE-free rebuild.
Simulation 10:41: simw2-1 COMPLETE (sim2.jobs_1 69,471 / 69,471 md5 4fe7a334, jobs_25 69,993 / 69,993 caa15adb),
simw2-2 COMPLETE (jobs_2 69,981 / 69,981 1567bd7a, jobs_26 69,651 / 69,651 89ce385d); simw2-4 and -5 pushed 10:41.
Running 3, 4, 5; done 0-2 (6 of 48 files); TPU smoke still QUEUED (2 h 40 min).
12:36: simw2-3 COMPLETE (jobs_3 69,750 / 69,750 md5 d9ffc7d0, jobs_27 69,774 / 69,774 ea34c3e8); simw2-6 pushed.
Running 4, 5, 6; done 0-3 (8 of 48 files); TPU smoke QUEUED 4 h 40 min.
15:36: simw2-4 COMPLETE (jobs_4 69,558 / 69,558 md5 cb4a651c; jobs_28 69,879 / 69,882 00dc6c96 -- 3 predictions = one
structure's 3 CEs failed in fwdrun, not a deadline cut (16,567 s): no top-up), simw2-5 COMPLETE (jobs_5 70,119 / 70,119
271c1cda, jobs_29 69,681 / 69,681 c1a5c7e5); simw2-7 and -8 pushed. Running 6, 7, 8; done 0-5 (12 of 48 files); TPU smoke
QUEUED 7 h 40 min.
17:34: simw2-6 COMPLETE (jobs_6 69,888 / 69,888 md5 8d67ccc1, jobs_30 70,023 / 70,023 dff7dff5); simw2-9 pushed. Running
7, 8, 9; done 0-6 (14 of 48 files); TPU smoke QUEUED 9 h 35 min.

## Interleaving loss (Oct 27, workflow wf_006468f5-611; full write-up docs/interleaving_oct27.md)
- Loss = P_train + P_coco - F + P_other = 0.157 + 0.215 - 0.339 + ~0.007 = ~0.040 (not 0.046; the old 0.013 'other'
  included the pc-only channel v18c does not run); almost all train <-> COCONUT displacement, per-query confusion (a
  training decoy above a class-2 truth leads by a median 0.59 SD), not a source-wide bias (best global shift recovers
  14-30% offline). The public chain merges ~3x better (0.013) but its recipe (provenance-free rankers, popularity after,
  RRF) is negative for us in every eligible copy (offline -0.001 to -0.010; LB v17bpop -0.016).
- Gap to the public 0.451 (0.112): 'other' 0.049, COCONUT within-source ranking 0.037, interleaving 0.027, train -0.002.
- Hidden class mix (public 124): train 0.19 [0.165, 0.225], COCONUT-not-train 0.32 [0.26, 0.40], other reachable 0.08
  [0.06, 0.12] (Enamine part ~0.056), unreachable 0.42 [0.30, 0.50]; 28/20/52 ruled out; class-1 share among
  train + COCONUT answers w1 = 0.37 [0.31, 0.42] (v18c trains at 46%, v18cw at 30%).
- v18cw predicted 0.339 [-0.003, +0.004] (moves the loss between classes). Next: masked split probes E1 (off-source
  candidates replaced in place by n-alkane placeholders: exact full-list RR per source), conditional library-gated
  COCONUT prior (only if COCONUT-heavy); STOP list: unconditional source shifts, re-picked deltas, provenance-free /
  factorised rankers, class-mix dose retrains (v18cq / v18c3), tiering, POP_BOOST, RRF / rank-2 insertion.

## GLACIER is in-sample on public-library truths (Oct 28 08:00; scratchpad ana/glpub.py, glpub2.py) -- no fix found
GLACIER (MassSpecGym weights) was trained on GNPS / MassBank / MoNA / MSnLib spectra. Proxy for membership: the truth
has a public-library [M+H]+ spectrum in train (gnps, massbank, mona, pluskal_ms2) = pub. Within-formula truth RR among
GLACIER-covered top-30 candidates (fwd_v18c train, groups with >= 3 same-formula candidates):
| class / pub | n | GLACIER | PeakFormer cos | stage 1 |
|---|---|---|---|---|
| 1 / 0 | 422 | 0.785 | 0.814 | 0.965 |
| 1 / 1 | 629 | 0.706 | 0.589 | 0.909 |
| 2 / 0 | 260 | 0.397 | 0.445 | 0.715 |
| 2 / 1 | 1914 | 0.634 | 0.521 | 0.833 |
(val: 2/1 0.626 vs 0.460, 2/0 0.345 vs 0.376.) GLACIER beats the PeakFormer only where it has seen the truth's spectra; on
Enveda-only truths (not in MassSpecGym; the hidden COCONUT-not-train answers are all of this kind) it is slightly worse.
74-88% of the simulated class-2 truths are pub, so stage 2 learns GLACIER on mostly in-sample truths. Stage 2 vs stage 1
by group, val: 2/1 covered 0.844 -> 0.876, 2/0 covered 0.701 -> 0.680, 2/0 uncovered 0.724 -> 0.713; 5-fold CV on the
4,877 training queries: 2/0 0.742 -> 0.735 (s1 -> base stage 2).
Fixes tried (CV train / val, MRR of the test-like group 2/0, overall): down-weighting pub class-2 queries x0.25 .732 /
.728 (overall .821 / .826 vs base .824 / .831), x0.1 .733 / .723, x0 .715 / .696, plus pub class-1 x0.5 .730 / .710;
stage 2 without GLACIER .729 / .731; minimal stage 2 (s1_z, s1_r + GLACIER columns) .708 / .711, with PeakFormer
within-formula columns .710 / .713. Nothing beats stage 1 alone on 2/0 (.742 CV) and every variant costs overall ->
no build, no slot. The LB credited GLACIER stage 2 with +0.010 (Oct 19), so its gain must come from other answers
(class 1 with public spectra) or the simulated 2/0 group is not representative; recorded as a known bias. Relevance
for PF-syn: GLACIER-simulated spectra are near-real for pub structures and out-of-sample for COCONUT ones.

## Lessons from the Oct 28 submissions
| kernel | prediction | LB | pair | rule outcome |
|---|---|---|---|---|
| specpfsyn-coco | slot 2 + 0.011 +- 0.012 | 0.173 | specpfctl-coco 0.149: +0.024 | >= +0.010: PF-syn TRANSFER CONFIRMED -> start the full-pool simulation |
| specpfctl-coco | 0.163 +- 0.012 | 0.149 | specpfb-coco 0.160: -0.011 | in [0.140, 0.180] (no drift); F = -0.011: fine-tuning PF-b further on real data alone HURTS COCONUT ranking |
| v18cw | 0.341 +- 0.012 (analysis 0.339) | 0.317 | v18c 0.339: -0.022 | <= 0.333: class-1 weight matters the OTHER way (not adopted) |
| probe18cw-coco | 0.222 +- 0.010 | 0.208 | probe18c-coco 0.215: -0.007 | COCONUT answers ranked DOWN, not up |
| probe18cw-train | 0.152 +- 0.006 | 0.148 | probe18c-train 0.157: -0.009 | beyond the -0.006 cap |
- PF-syn: the synthetic arm beats its matched control by +0.024 and PF-b by +0.013 on the spectral-only COCONUT probe
  (offline analog predicted +0.011: the LB / offline ratio of this pair is ~1.1, vs 0.49 for PF-d -> PF-b). The
  control alone loses -0.011 vs PF-b: more fine-tuning on real spectra deepens the population prior; synthetic
  positives of pool structures counteract it and add. -> full-pool simulation (casmi26-simw2-*) started; production
  fine-tune (synP) on the full-pool syn2 set next; pilot checkpoints stay probe-only (their syn set used the
  neutral-set windows from sub.parquet, G15).
- Class-1 weight: halving it (C1_DUP 2 -> 1) cost -0.022 on the LB with BOTH source probes down (train -0.009,
  COCONUT -0.007; G_cw = 0.148 + 0.208 - 0.317 = 0.039 vs 0.033): the duplicated class-1 queries are not just a mix
  prior, they carry general training signal (clean library-supported labels). The interleaving analysis' prediction
  (0.339 [-0.003, +0.004]) and its "no mirror dose" advice are refuted by the LB -> the opposite dose C1_DUP=3 (v18c3)
  is the natural next test.

## Lessons from the Oct 27 submissions
Predictions (pre-registered Oct 25 16:40, "Oct 27 slots") vs LB, and the rule outcomes:
| kernel | prediction | LB | pair | rule outcome |
|---|---|---|---|---|
| v18c | 0.345 +- 0.012 | 0.339 | v17b 0.339: +0.000 | 0.331-0.346: v18c = BASE (eligible by construction, no cost) |
| probe18c-coco | 0.226 +- 0.012 | 0.215 | probe17b-coco 0.211: +0.004 | between 0.211 and 0.221: mechanism NOT confirmed (the leak cost the hidden COCONUT answers little) |
| probe18c-train | 0.145 +- 0.006 | 0.157 | probe17b-train 0.150: +0.007 | >= 0.144: no class-1 cost (better, within noise) |
| zen-probe | 0.005-0.02 | 0.000 | -- | < 0.008: no hidden answer among the Zenodo-only Enveda-180 structures -> ZEN line CLOSED |
| v17bpop | 0.342 +- 0.008 | 0.323 | v17b 0.339: -0.016 | <= 0: popularity re-rank REJECTED (no v18cpop) |
- The in_bio leak removal is LB-neutral: the simulated cls-2 drop (.92 -> .82) did not cost the hidden test anything,
  and the expected COCONUT gain (+0.015) did not appear (+0.004). The shortcut fired on simulated class-2 truths only;
  hidden COCONUT answers never carried in_bio (0.36%), so the ranker's use of it did not move them either way.
- v18c (0.339) replaces v17b as pick #1: same score, prize-eligible chain (docs/reproduce.md section 6).
- Documentation popularity (POP_BOOST) hurts our lists (-0.016) although it helped the public chain (+0.010 there):
  our rankers already use pc_sid / pc_pmid; adding them again over-weights documented (training-like) candidates.
- The Zenodo-only Enveda-180 structures are not hidden answers (0.000): the 'other' gap is not Enveda's own published
  library; the remaining 'other' levers are the PubChem tiers (closed earlier) -> 'other' stays conceded.

## Lessons from the Oct 26 submissions
Predictions (pre-registered Oct 25 01:30 in the "Oct 26 slots" table) vs LB, and the rule outcomes:
| kernel | prediction | LB | pair | rule outcome |
|---|---|---|---|---|
| v17bb | 0.340 +- 0.010 | 0.322 | v17b 0.339: -0.017 | <= 0.333: FP-b HURTS -> not adopted; v18cb (FP-b on the clean base, val .838 vs .8314) NOT pushed |
| probe17b-coco | 0.216 +- 0.010 | 0.211 | probe5w-coco 0.208: +0.003 | 0.208-0.213: ambiguous (v17b's +0.007 over v5w is not located in COCONUT) |
| probe17b-train | 0.153 +- 0.006 | 0.150 | probe5w-train 0.156: -0.006 | at the cap (>= 0.150) |
| specpfd-coco | 0.14 (0.10-0.18) | 0.129 | -- | reference |
| specpfb-coco | 0.16 (0.12-0.20) | 0.160 | specpfd-coco: +0.031 | b - d >= +0.015: spectral gains TRANSFER to the hidden COCONUT answers -> spectral scaling (PF-x, PF-syn) is the main lever, PF-syn GO; donor rule (>= probe17b-coco + 0.010 = 0.221) not met |
- The debiased-negatives recipe helps the PeakFormer (v17 -> v17b +0.005; spectral-only COCONUT +0.031) but hurts the
  FP model (-0.017). The FP model and the PeakFormer share most of their signal; FP-b's instrument-free / training-only
  negatives may have shifted its scores toward the PeakFormer's (less complementary) -- the stage-1 ranker learned the
  combination on simulated rows where both looked better (v18cb val +0.0066), which the LB does not credit. Simulated
  validation again failed to rank an FP / PF change; LB pairs stay the only judge.
- v17b remains pick #1 (0.339, best eligible) though its mechanism reading is ambiguous; v18c (clean, FP v1r) is its
  eligible replacement, read Oct 27.
- Oct 27 slots unchanged from the pre-registered table. The conditional probe17-coco (matched control for the ambiguous
  COCONUT reading) is NOT added: specpfb - specpfd (+0.031) already answers whether PF-b's spectral gain reaches the
  COCONUT answers, and slot 5 (v17bpop) can add score; recorded deviation from the conditional rule.

## Compute after the Oct 26 reading (PF-syn GO)
- TPU: casmi26-pf-train-x5 (queued since Oct 25 04:39, never started) DELETED Oct 26 01:24: the owner's own
  ntl-tpu-probe (TpuV5E8) has been queued since Oct 25 16:43 and Kaggle allows one TPU session per account, queued
  ones included -> owner first. No new TPU push while the owner uses the TPU.
- GPU: the shared week runs to Oct 31 (12.1 h left before casmi26-simg's 1.5 h; the owner ran three T4 kernels since,
  ntl-p641-s3a / s3ab / s3b, the last at 00:11 today) -> no 3.5 h pilot now.
- Local CPU pilot started 01:25 (scripts/run_pf_pilot_local.sh 10000 /home/user/work/pfsyn_local): the pre-reviewed
  pilot (kernels/casmi26-pf-pilot settings, PF_SYN_P 0.25, NEUTRAL_W 0.18, half B dropped) at batch 256, 10,000
  steps per arm, ctl then syn (~2.3 s/step -> ~6.5 h per arm). 2.56M samples per arm = ~6% of the TPU design (41M);
  a PASS of the pre-registered gate is meaningful, a FAIL is weak evidence. Its role: decide whether the Oct 31 GPU
  week goes to PF-syn (pilot at T4 scale + full-pool simulation on free CPU workers + production run).
- The local run died with the container (reclaimed while the session was idle; uptime 0 at the 04:28 check-in; no
  step logged): long local jobs only survive inside active turns. Moved to Kaggle CPU 04:30: kernels
  casmi26-pf-cpu-ctl / casmi26-pf-cpu-syn (scripts/make_pf_cpu_pilot.py <slug> <arm> 7500 11.3 800 250), the two arms
  in parallel, step-matched at 7,500 steps of batch 256 (~4.96 s/step -> ~10.3 h; 1.92M samples per arm, ~4.7% of the
  TPU design). Evaluation as pre-registered: scripts/eval_pf_pair.py on the 620-query neutral set (halves A / B).
  Seed replicate (04:40): casmi26-pf-cpu-ctl2 / -syn2, the same pair at PF_SEED 32 (the paired bootstrap covers query
  sampling only, not training-seed noise). Reading: the gate must hold on BOTH seeds (and the mean) to count as a pass.
  Evaluation dry run (04:50) on the 150-step calibration checkpoints (casmi26-pf-cpu-cal; GLACIER column
  /home/user/work/pfsyn_eval/sub_gl.parquet: 620 queries, 418 with [M+H]+, 27,155 / 42,421 rows): pf_cos delta win vs
  training decoys all +0.022 [+0.005, +0.038], half A +0.018 [-0.009, +0.045], half B +0.026 [+0.004, +0.048]; vs
  COCONUT +0.003; excess +0.025 -> +0.006; pf_cos_gl (covered) +0.004 [-0.010, +0.018]. Direction right after 150
  steps; the gate (+0.06 on half A) is read on the 7,500-step arms.

## PF-syn decision tree and LB rules (pre-registered Oct 26 07:00, before the pilot read; design workflow wf_adf4f08f-b33)
Pilot = kernels casmi26-pf-cpu-ctl / -syn (seed 31) and -ctl2 / -syn2 (seed 32), 7,500 steps each.
Validity: a seed is VOID if either arm's log contains "PF_HOURS reached" or the two arms' last "step N/7500" differ
(re-run it; a VOID seed is never a gate fail). Gate per seed (half A, scripts/eval_pf_pair.py): pf_cos delta win vs
training decoys >= +0.06 with 95% CI > 0; delta vs COCONUT >= -0.01; excess_syn <= excess_ctl and >= -0.02; pf_cos_gl
delta > 0. Seed-mean reading: comma-separated row files.
- PASS-2 (both seeds pass; or one passes and the other misses only on size, delta in [+0.03, +0.06) with CI > 0 and
  guards ok, seed mean >= +0.06): push 3 simulation workers casmi26-simw2-* at once (full pool v18c, 24 workers x
  ~6 h, rolling, <= 3 long sessions of ours, <= 2 while the owner has a CPU kernel running/queued); Oct 28: pilot
  spectral probes (below); then the CPU production fine-tune pfs1 on the full-pool syn2 set; GPU only from Oct 31 per
  the LB-2 rule.
- PASS-1 (one seed passes, the other fails on sign / CI / a guard or is VOID): seed-33 tie-breaker pair (2 sessions)
  + 1 simulation worker; read ~12 h later: 2 of 3 seeds pass AND the 3-seed mean >= +0.04 -> PASS-2, else FAIL-near.
- FAIL-near (no seed passes, seed-mean half-A delta >= +0.02 with CI > 0, both seeds > 0, guards on the mean): no
  simulation; Oct 28 coco probe pair only; delta-coco >= +0.010 -> start the simulation then (chain ~1.5 days later).
- FAIL-hard (anything else): Oct 28 coco pair only as a gate calibration; PF-syn closed unless delta-coco >= +0.015.
LB-1 (Oct 28; pool4 / ret_ranker_v17 / no stage 2 / LIB_CAP 10, ordered by pf_bern = the casmi26-specpfb-coco setup;
2-seed ensembles OWN_PF=pf_syn_s31.pt,pf_syn_s32.pt vs pf_ctl_s31.pt,pf_ctl_s32.pt):
| slot | kernel | pair | prediction |
|---|---|---|---|
| a | specpfsyn-coco | specpfctl-coco | delta = 0.49 x (0.22 a_A + 0.78 a_B) +- 0.012 (a = offline coco RR delta, halves A/B; 0.49 = LB / offline ratio of PF-d -> PF-b) |
| b | specpfctl-coco | specpfb-coco 0.160 | 0.160 +- 0.012 (anchor: outside [0.140, 0.180] -> setup drift, read paired deltas only) |
| c | specpfctl-cotr | -- | anchor (training + COCONUT-not-train candidates, OWN_SPEC_SRC=cotr) |
| d | specpfsyn-cotr | specpfctl-cotr | diagnostic only (first slot to give way) |
Rules: delta-coco >= +0.010 transfer confirmed; (-0.010, +0.010) uninformative (CPU path continues, LB-2 decides the
GPU); <= -0.010 STOP (no pfs1, no GPU). F = specpfctl-coco - 0.160 >= +0.012 -> fine-tuning alone pays (ctl becomes a
deploy candidate). LB-2 (pfs1 production probes, same setup, paired with the Oct 28 controls if the inference-module
sha256 in casmi26-code MANIFEST is unchanged): GPU go needs pfs1 CPU gate passed AND delta-p-cotr >= -0.005 AND (delta-p-coco
>= +0.015 or mean(delta-coco, delta-p-coco) >= +0.007 or delta-p-cotr >= +0.015). Adoption of any PF-syn pipeline
(v18c rebuild with the new PeakFormer): house retrain rule (>= v18c + 0.008, COCONUT probe >= +0.005, training probe
>= -0.006), +0.002 on the bar per extra candidate, at most 3 candidates by Nov 10. Honest prior: P(adopt) ~20-25%.
Owner note: simulating the candidate pool with GLACIER uses pool structures only (no labels), like inference-time
GLACIER; recorded for the owner, not a compute blocker.

## PF-syn CPU pilot result (Oct 26 16:20)
Validity: BOTH seeds VOID by the pre-registered rule -- both ctl arms were cut by the 11.3 h guard at step 6500 of 7500
(ctl ~6.3 s/step vs syn ~5.4 on Kaggle CPU: the real-row host augmentation is the slower path; the 150-step calibration
did not show it), both syn arms completed 7500 (syn2 logged "PF_HOURS reached at step 7500 of 7500" = complete).
The missing ctl tail carries < 0.5% of the integrated LR (cosine at 87%: lr 4% of peak), so the readings below are
near-matched DIAGNOSTICS, not gate readings. Rule applied: VOID -> re-run; the syn arms are re-run with PF_STOP_AT=6500
(LR schedule still 7500 = exactly the cut ctl arms): kernels casmi26-pf-cpu-synm (s31) / -syn2m (s32), pushed 16:25,
~9.8 h -> read ~02:30 Oct 27. Real-val of the arms (764 split-1 structures, cos MRR): ctl .801 / .802, syn .796 / .812.
Diagnostic gate (scripts/eval_pf_pair.py, files in /home/user/work/pfsyn_eval/gate_*_void.txt), pf_cos delta win vs
training decoys, half A: s31 +0.032 [+0.009, +0.056], s32 +0.022 [-0.003, +0.046]; seed mean +0.027 [+0.006, +0.048];
vs COCONUT s31 +0.019, s32 +0.026, mean +0.022; excess (syn vs ctl) s31 +0.012 vs +0.025, s32 +0.026 vs +0.022, mean
+0.019 vs +0.024; pf_cos_gl (covered A) s31 +0.010, s32 -0.000, mean +0.005. Exposure gap (pf_bern, COCONUT decoys
simulated vs not) -0.0002 / -0.0022: no membership bias. Offline COCONUT probe (scripts/eval_syn_probe.py, pf_bern):
s31 +0.029 [+0.015, +0.042] (A +0.024, B +0.034), s32 +0.019 [+0.005, +0.033] (A +0.008, B +0.030); cotr s31 +0.023.
Reading: synthetic rows give a GENERAL discrimination gain (~+0.02 against training AND COCONUT decoys alike), not the
targeted removal of the membership prior the design aimed at (excess moves -0.005 on the mean). Provisional branch
(if the matched re-runs read like this): FAIL-near (no seed passes the +0.06 size; seed mean >= +0.02 with CI > 0,
both seeds > 0, guards hold on the mean) -> no simulation yet; Oct 28: the COCONUT probe pair only (specpfsyn-coco =
ensemble of the two matched syn checkpoints vs specpfctl-coco = the two ctl checkpoints, specpfb-coco setup);
delta-coco >= +0.010 -> start the full-pool simulation (casmi26-simw2-*). Diagnostic LB-1 prediction:
0.49 x (0.22 x 0.016 + 0.78 x 0.032) = +0.014 +- 0.012 (recomputed on the matched checkpoints before pushing).

## Lessons from the Oct 24 submissions
- The token override does not pay on the LB: v11l 0.320 -> tok3 0.315 / tok0 0.316 / v13 (retrained on token-3 rows)
  0.313. Within COCONUT it helps (tok3 COCONUT-restricted probe 0.200 vs 0.190), so the rest of the list loses ~0.015:
  most likely class-1 / Enamine-like answers, for which the true timsTOF token is genuinely informative (the hidden
  class-1 molecules are Enveda timsTOF measurements of training-like chemistry). All deltas are ~1 paired se (~0.01),
  and the three kernels share one perturbation -> effectively one observation: "token override ~ -0.005".
- The pre-registered HL-val predictions (HL_PRIOR, p=1 lam=1) were +0.018 / +0.020 / +0.021 -> they FAILED by ~0.02.
  HL-val does not model the hidden class-1 / timsTOF behaviour (its T stratum mixes instruments and applies the token
  to every spectrum) and its prior emulation is not identified (review wf_b897a7fd-fd5). HL-val is NOT a decision tool
  for token / spectral-model changes; only LB A/Bs with restricted probes are.
- Our class-1 component is near parity: probe11-train 0.149 vs the public lineage 0.155. The gaps are COCONUT (0.190 -
  0.200 vs 0.252) and "other" (~0.01 vs 0.057).
- Reminder from the table: the best eligible LB is still s2v5w (alpha 1, Oct 20) at 0.332; v10 (full-library
  matching, alpha 0.5) 0.314, v11l (+PeakFormer, alpha 1) 0.320, v12 0.312 -- the Oct 21+ feature additions did not
  beat it (alpha 1 > 0.5 in both pairs: +0.009 / +0.006; full-library matching v10 vs v5w-a05: -0.009).
- Consequence for variant b (instrument-free PeakFormer / FP): it removes the token like tok0 -> the class-1 cost may
  repeat. Deploy debiased models only as ADDITIONAL features next to the token-1 ones, or for COCONUT candidates only,
  and judge by the COCONUT-restricted probe plus the full list.

## Lessons from the Oct 25 submissions
Predictions (pre-registered in the Oct 25 trigger) vs LB, and the rule outcomes:
| kernel | prediction | LB | pair | rule outcome |
|---|---|---|---|---|
| probe5w-coco | 0.195 (0.185-0.210) | 0.208 | probe11-coco 0.190: +0.018 | in range (high end) |
| probe5w-train | 0.148 (0.140-0.156) | 0.156 | probe11-train 0.149: +0.007 | in range (top) |
| s2v5w-olast | 0.334; adopt >= 0.337, reject <= 0.332 | 0.324 | v5w 0.332: -0.008 | REJECT (source tiering) |
| s2v17 | 0.328 +- 0.009; >= 0.326 neutral | 0.334 | v5w 0.332: +0.002 | PeakFormer d neutral at lib10 -> v17 = the PF-b control |
| s2v17b | 0.330 +- 0.010 | **0.339** | v17 0.334: +0.005; v5w: +0.007 | >= 0.336 -> candidate pick #1; retrain rule (>= +0.008 vs the matched parent) not met by the LB alone -> second consistent point = the Oct 26 probes |
- PeakFormer b (debiased + ~11x the training of d) is the first spectral change that moved the LB up since Oct 20: v17b 0.339
  is the best eligible score (v5w 0.332, s2v4 0.330). v17 (PF-d at lib10) was neutral, so the +0.005 belongs to PF-b's
  replacement as a whole (debiasing and the longer GPU training are confounded by design; v17b vs v17 measures both).
- The v5w (lib10, 31 leaves) family is better than v11l on both restricted probes: COCONUT +0.018, train +0.007 ->
  the v10-family loss (-0.012 overall) is spread over both sources, mostly COCONUT. Confirms "never the v10 family".
- Source tiering FAILS (-0.008; the prediction +0.002 missed by 0.010): v5w's "other" candidates that sit high in the
  list are right often enough (probe5-other 0.007) that pushing them below train / COCONUT costs about all of it. The
  ranker's source mixing is already calibrated; no hard source rules (STOP list: source tiering / promotions).
- Both v5w probes came in at the top of their ranges -> the v5w lists are a bit better than the v11l-based predictions
  assumed; the public-lineage gaps shrink to COCONUT 0.252 - 0.208 = 0.044 and other ~0.05 (train: parity, 0.155 vs 0.156).
Adoption rule for v17b (second consistent point, Oct 26): probe17b-coco >= probe5w-coco + 0.005 (>= 0.213; the
targeted probe of a COCONUT-debiasing change) AND probe17b-train >= probe5w-train - 0.006 (>= 0.150; class-1 cost cap).
probe5w is the proxy parent (no probe17 exists; v17 - v5w = +0.002 on the LB). Both hold -> v17b ADOPTED as the base
(pick #1); COCONUT probe < 0.208 -> v17b stays pick #1 on the LB but the mechanism is unconfirmed (push probe17-coco
as the matched control before building on it); train probe < 0.150 -> the class-1 cost of instrument-free scoring
repeats (cf. the token override) -> test PF-b for COCONUT candidates only (slot-keeping donor).

## Oct 26 slots (pre-registered Oct 25 01:30 UTC; all five kernels COMPLETE and verified clean on Oct 24)
| slot | kernel | pair | prediction | rule |
|---|---|---|---|---|
| 1 | casmi26-own-s2v17bb v1 (v17b + FP model b) | v17b 0.339 | 0.340 +- 0.010 | >= 0.347: adopt as base; 0.340-0.346: needs a second point (probe17bb-coco); <= 0.339: FP-b not adopted; <= 0.333: FP-b hurts |
| 2 | casmi26-probe17b-coco v1 | probe5w-coco 0.208 | 0.216 +- 0.010 | >= 0.213 (with slot 3 ok): v17b ADOPTED; < 0.208: mechanism unconfirmed |
| 3 | casmi26-probe17b-train v1 | probe5w-train 0.156 | 0.153 +- 0.006 | >= 0.150: class-1 cost within the cap |
| 4 | casmi26-specpfd-coco v1 (PeakFormer d alone, COCONUT-not-train) | -- | 0.14 (0.10-0.18) | reference for slot 5 |
| 5 | casmi26-specpfb-coco v1 (PeakFormer b alone, COCONUT-not-train) | specpfd-coco | 0.16 (0.12-0.20) | b - d >= +0.015: spectral gains transfer to the hidden COCONUT answers -> GPU spectral scaling is the main lever; <= 0: they do not -> re-plan |
Donor rule (panel, conditional): specpfb-coco >= probe17b-coco + 0.010 -> build a slot-keeping kernel (v17b lists, the
COCONUT-not-train slots re-ordered by pf_bern, every other slot unchanged); post-processing rule applies (adopt >= +0.005).

## Oct 27 slots (pre-registered Oct 25 16:40 UTC, before any Oct 26 reading; kernels pushed Oct 25 15:54-16:40)
| slot | kernel | pair | prediction | rule |
|---|---|---|---|---|
| 1 | casmi26-own-s2v18c v1 (clean-room rebuild: no in_bio leak, FP v1r, fresh GLACIER pass) | v17b 0.339 | 0.345 +- 0.012 | >= 0.347: leak removal pays, v18c = base; 0.331-0.346: v18c = base (eligible, cost within noise); <= 0.330: leak removal costs -> read slots 2-3 before anything else |
| 2 | casmi26-probe18c-coco v1 | probe17b-coco (Oct 26) | probe17b-coco + 0.015 +- 0.012 | >= +0.010: mechanism confirmed (COCONUT answers freed); <= 0: not confirmed |
| 3 | casmi26-probe18c-train v1 | probe17b-train (Oct 26) | probe17b-train - 0.005 +- 0.006 | >= -0.006: class-1 cost within the cap; <= -0.012: in_bio carried real class-1 evidence -> consider a leak-free class-1 flag |
| 4 | casmi26-zen-probe | -- | 0.005-0.02 | >= 0.008: integrate the Zenodo-only structures as a library channel |
| 5 | casmi26-own-s2v17bpop | v17b 0.339 | 0.342 +- 0.008 | post-processing rule: adopt >= +0.005 (then as v18cpop on the new base), reject <= 0 |
Verified 19:45 (execution logs): casmi26-own-s2v18c v1 400 / 400 rows with candidates, GLACIER 371 / 400, no
Traceback; casmi26-probe18c-coco v1 393 / 400 COCONUT rows; casmi26-probe18c-train v1 400 / 400 training rows.
v18c vs v17b on the VISIBLE test (placeholder of training spectra, so not informative about the hidden mix): top-1
identical 97.5%, top-5 overlap .74, top-1 a training structure 399 / 400 in both (by construction of the placeholder).
If the Oct 26 trigger fills slots with conditional follow-ups (probe17-coco, v17bb second point), they take priority
over slot 5; slots 1-3 are kept together (one mechanism reading).

## in_bio was a training-membership leak (found Oct 25 by the v18c clean rebuild)
pool4's in_bio came from the third-party ChEBI/LMSD set, which removed the keys of prvsiyan/coconut-casmi26-candidates,
a COCONUT-NOT-TRAIN candidate list. Result: among COCONUT structures in_bio = 1 for 60.75% of TRAINING structures
(25,969) but 0.36% of COCONUT-not-train (450,388) -> in_bio is a hidden in_train flag. The class-2 simulation sets the
truth's in_train to 0 to mimic hidden COCONUT answers, but in_bio still gives the truth away (label leak), and docw's
documentation model (which uses in_bio) learned the same shortcut for the doc-shift weights. Since pool v4 (Oct 19)
every stage-1 ranker learned "COCONUT + in_bio = truth" from simulation; on the hidden test, class-2 COCONUT answers
never carry it (0.36%), while COCONUT-and-train decoys do (61%) -> a population prior against exactly the answers of
our COCONUT gap (probe 0.208 vs public 0.252).
Evidence: v18c rows (clean rebuild; own bio set = ChEBI/LMSD minus OUR COCONUT, so in_bio = 0 on all COCONUT) give
stage-1 val cls1 .8268 / cls2 .8150 (OOF .8232); the SAME rows with pool4's in_bio and doc weights give .8432 / .9168
= v17b's .8426 / .9145. So the clean pipeline reproduces v17b exactly and the whole simulated drop is the leak.
2x2 (stage-1 val cls1 / cls2 on v18c rows): old in_bio feature + new doc weights .8468 / .9278; new in_bio + old doc
weights .8315 / .8165 -> the leak acts through the in_bio FEATURE (a binary flag: low split gain, decisive on the
simulated truths), not through the doc-shift weights.
Leak audit (all 58 stage-1 features, AUC for training membership among 79,475 COCONUT decoys of the v18c val rows):
library match lib_n / lib .95 / .94 (legitimate: training structures have library spectra at test time too),
documentation pc_sid .85, pc_pmid .79, pc_patent .78, n_syn .77, PubChemLite / COCONUT annotations .62-.73 (the known
documentation shift the doc-shift weights address), spectral ranks .62-.68; old in_bio .80 -- but as a construction
artifact (a list built relative to the competition train set: 61% vs 0.36% = 170x rate ratio) rather than an organic
documentation signal. No other construction leak found.
Simulated validation cannot judge this change (its truths carry the leak's target); only the LB can: v18c vs v17b is
a pre-registered A/B (difference = the leak removal + FP-v1 retrain + 15 pool rows). Prediction: v18c >= v17b - 0.008
(class 2 freed from the shortcut; the pool-v4 +0.013 was mostly reach of the 52k new ChEBI/LMSD structures), with the
COCONUT-restricted probe of v18c the mechanism check. v18c is the eligible base either way.

## Reproducibility manifest (Oct 25, workflow wf_029f119e-b7e; [reproduce.md](reproduce.md), [ATTRIBUTION.md](ATTRIBUTION.md))
44 agents traced every artifact of the v17b submission and verified each gap against the repo (several rebuilds run:
pool4 from pool2 + build_pool_bio + add_pubchem_pop reproduces pool.parquet exactly; build_pool.py is deterministic).
20 confirmed gaps, two of them blocking prize eligibility as the submission stands:
- G1: in_bio and 52,371 pool4 rows come from sergiomalv bio_structures.parquet (no build code; its key set is ChEBI
  r255 + LMSD minus prvsiyan/coconut-casmi26-candidates keys, a CC-BY-NC-SA list). Fix: our own
  scripts/build_bio_structures.py from raw ChEBI r255 + LMSD, then rebuild pool, doc_w, rows, rankers.
- G2: stage2_v17b was fit on GLACIER scores merged from 17 historical fwd dirs (fwd2/3/4 jobs built with third-party
  fuse.build_ice_input on deleted inputs; batch-dependent scores). Fix: clean rescoring of every top-30 pair with one
  scripted fwdrun pass (GPU now: casmi/fwdrun --device cuda), retrain stage 2.
Others: LightGBM determinism and version pinning (G3); fp_model.pt / pf_model_b.pt training commands and step counts
(G4/G5: FP_STOP_EPOCH, PF_STEPS); missing scripts for split keys, pool v1, COCONUT tables, train_meta, PubChemLite,
pool2->pool4 chain, ret4 dump, ret17 build (G6-G14); gate inputs (G15); code dataset packaging and version pinning
(G16); fpx / richfp logs (G17/G18); LICENSE (owner) and dataset licences (G19); stale docs (G20).
G1 done (Oct 25): scripts/build_bio_structures.py on ChEBI rel255 chebi.sdf.gz (md5 4adcf594; 2/3-star) + LMSD.sdf.zip
(downloaded 2026-10-25, md5 505ec732): 250,207 records -> 235,655 standardised -> 162,195 unique keys (third-party set:
162,180; same drop counts) -> 80,232 outside our COCONUT 2026-10. Versus the third-party set: 80,197 shared, its
18,236 extra keys are ALL in our COCONUT (already pool rows), 35 new -> pool5 = pool4 with in_bio = 0 for 18k COCONUT
rows (+35 rows): the eligibility fix barely changes the candidate set. G3 done: deterministic row-wise LightGBM
everywhere, requirements-pipeline.txt. scripts/build_v18c.sh (+ make_gl_gpu_kernel.py) scripts the whole chain;
stages A-B (pool5, row dumps) running.
GLACIER determinism (G2 detail, Oct 25): predictions depend on batch composition and device -- CPU batch 1 vs 32
mean predicted-spectrum entropy similarity 0.995 (min 0.889, identical 33%), CPU (batch 32) vs GPU (batch 128)
0.988 (min 0.885); batch 1 costs 2.6x on CPU (too slow for the 4 h inference budget). So exact equality between the
GPU training pass and CPU inference is not attainable; the v18c pass is one scripted GPU run with a fixed batch policy
(bs 128, jobs sorted by SMILES length, 8 + 2 files), reproducible run to run, documented device / batch noise ~0.99.
v18c stage A (08:24): pool_v18c = pool2 + own bio set: 1,160,133 rows (pool4 + 15), in_bio 71,589 (pool4 89,054;
17,520 rows changed, 17,400 of them COCONUT) -- a stale $W/pool5 from Oct 19 was first picked up by mistake (now: tag-named
directory + completion marker). Stage B (row dumps, seeds 12 / 11) running; stage C waits for the Oct 26 v17bb reading
(FP model: FP-b from its GPU kernel if v17bb >= v17b, else a recorded retrain of the v1 FP model, G4).
v18c stage B (09:42): ret18c dumps 7,788 queries / 980,913 rows and 1,463 / 182,179 (ret4: 980,910 / 182,177; the
15 new bio structures add 3 + 2 rows). Stage C1 (10:01, 18 min without the richfp cache): PeakFormer-b features equal
ret17b's on the 182,177 matched validation rows (pf_bern identical, pf_cos <= 1.2e-7). C2-F wait for the FP choice
(Oct 26 v17bb reading, or the recorded FP-v1 retrain kernels/casmi26-fp-train-v1 running on Kaggle CPU) so that the
GPU GLACIER pass runs once.
Scripted fixes G6-G20 (workflow wf_fd86ad46-d15, 7 implementers + 7 reviewers, all reviews ok; details in
docs/reproduce.md section 5): make_split_keys.py (byte-identical key lists), build_pool_v1.py (byte-identical pool v1) +
prep_fp_data.py arguments (all 16 fp-data arrays equal) + check_fp_data.py, build_coconut.py / build_train_meta.py /
fetch_pubchemlite.py (md5-checked), streamed add_pubchem_pop.py + PubChem download manifest + build_pool4.sh,
build_v17b.sh / build_pfswap.sh on ret4 rows with GATE=0 and a pool-checked richfp cache, push_code.sh (clean tree,
MANIFEST with sha256), atomic fpx write, docs/ATTRIBUTION.md and corrected docs. Open: LICENSE (owner); key lists not
committed (competition-derived); pool2 half of build_pool4.sh needs ~12 GB RAM to verify end to end. Six stray empty
error logs /ret17b_{train,val}_fp.parquet.part{0,1,2}.log in the filesystem root (left by an agent's shell heredoc)
could not be removed by the session (root-path safety check) -- harmless.
G4 resolved (13:13): the recorded FP-v1 retrain (kernels/casmi26-fp-train-v1, Kaggle CPU, FP_STOP_EPOCH=3, same
command as the deployed fp_model.pt) gives val MRR .5109 / .5395 / .5569 at epochs 0 / 1 / 2 vs the original
.5118 / .5346 / .5461 -> equivalent (not bit-identical: CPU nondeterminism). Its checkpoint (own_models/fp_model_v1r.pt,
md5 a9748a4a) is the FP model of v18c, which is therefore a clean replica of v17b (FP-b variant after the Oct 26 reading).
v18c stages C2-E launched 13:20 (E = 4 GLACIER CPU worker kernels).
v18c DONE (15:48): stage 1 val .8197 (cls1 .8268 / cls2 .8159), stage 2 .8314 (.8301 / .8321); fresh GLACIER pass
coverage train .783 / val .754 (= v17b's merged history). v17b: .8895 (.8426 / .9146) -> .894 (.8432 / .9212); the
cls-2 gap is the in_bio leak (see above), the cls-1 gap (.8301 vs .8432) is mostly the same flag on class-1 COCONUT
truths. Models ret_ranker_v18c (md5 3f12b795), stage2_v18c (6af95879) + fp_model_v1r.pt in casmi26-models; pool as
dataset casmi26-pool-v18c; kernel casmi26-own-s2v18c (make_own_kernel.py POOL_DS=casmi26-pool-v18c) = the Oct 27 LB
candidate for the pre-registered A/B above.
v18cb (FP-b swap on the same pool and B / C1 rows: BASE=v18c STOP_AFTER=D scripts/build_v18c.sh ... fp_model_b.pt
v18cb, 16:22): stage 1 val cls1 .8249 / cls2 .8212 (v18c .8268 / .8159), OOF .8204 (v18c .8232). Stage E (4 CPU GLACIER
workers) waits until the v18c kernels free their CPU sessions; built only as far as the Oct 26 v17bb reading needs.
v18cb DONE 22:41 (4 CPU GLACIER workers casmi26-glwv18cb-0..3, coverage train .783 / val .754): stage 1 .8231
(.8249 / .8221), stage 2 .838 (cls1 .8389 / cls2 .8375) vs v18c .8314 (.8301 / .8321): +0.0066 on simulated val.
Models ret_ranker_v18cb (md5 2cdb147d), stage2_v18cb (1dfd20fd), local only; upload + kernel casmi26-own-s2v18cb only if
the Oct 26 v17bb reading is >= v17b (otherwise the val gain alone does not justify a slot).
Plan: a clean-room rebuild track ("v18c") that fixes G1-G5 structurally and is the base for every later model, so the
final picks are eligible by construction; the scripted fixes (G6-G18) land as they are touched. The LB-validated v17b
stays the reference until v18c is built and checked against it on the LB.

## Roadmap after the lever research (Oct 25, workflow wf_2081278c-964; full write-up docs/research_oct25.md)
Critic's order (realistic combined gain +0.015 to +0.04 over 0.339; 0.45-0.50 is not reachable eligibly in 7 weeks):
0. Compliance package (LICENSE = owner's decision; public training code for every weight; one end-to-end rebuild
   from public inputs before the freeze) -- weeks 1-2, not week 7.
1. Read the Oct 26 probes; PF-x (TPU, bigger model) + a 2-3 model PeakFormer ensemble as the new spectral base;
   slot-keeping COCONUT donor only if specpfb-coco >= probe17b-coco + 0.010.
2. PF-syn (GLACIER-simulated [M+H]+ spectra of pool structures as extra positives: COCONUT-not-train finally appears
   as positive) -- conditional on specpfb-coco - specpfd-coco >= 0; pilot + matched control on the TPU, gated on the
   neutral metric, then the full-pool simulation. Population-balance rules: synthetic rows draw negatives inside the
   simulated set, real rows keep training-positive negatives, in-batch negatives never cross looks, no source token.
3. GLACIER evidence (reduced): one global CE offset and per-adduct stage-2 features, [M+NH4]+ / [M-H2O+H]+ proxies
   only for molecules without [M+H]+ (public chain: +0.008 per coverage step; ours covers 371/400 already).
4. ZEN probe (kernel casmi26-zen-probe, ready since Oct 21, never submitted): direct match to the 3,717
   Zenodo-only Enveda structures; integrate as a library channel only if >= 0.008.
5. Enamine-tier promotion: offline first (PF-b misfire on the 206 npx natural-product present queries must fall
   from 29% to <= 8%), one paired A/B, then close the "other" line either way.
6. Test-test networking (host-allowed 2026-10-24, thread 752297): half-day density check first (>= 15% networked).
7. Class 3: CPU go/no-go only (PeakFormer morgan2 Tanimoto bins); otherwise parked for good.
Dropped: contrastive GLACIER fine-tuning on pool isomers (would teach GLACIER the population prior), late-interaction
MaxSim head (most expensive, rises or falls with PF-syn). Electron-mass correction of the training data: fold into
the next data build (no dedicated run). Also from the research: the public chain's post-hoc popularity re-rank
z(score) + 0.15 (ln SID + ln PMID) was worth +0.010 there; ours exists (POP_BOOST, never LB-tested; offline a
cls1-up / cls2-down trade-off that HL-val cannot judge) -> kernel casmi26-own-s2v17bpop (v17b lists + POP_BOOST
0.15, post-processing on identical lists: adopt >= +0.005, reject <= 0).
Test-test networking density check (item 6, Oct 25) on enveda-np-examples (250 timsTOF natural products, "same
pipeline as the test", 1,184 spectra): structures with a Tanimoto >= 0.7 neighbour in the set 19% (>= 0.5: 44%);
with a spectral neighbour (direct entropy similarity, same polarity) >= 0.6: 36% (>= 0.7: 20%); with a neighbour that
is both spectrally (>= 0.6) and structurally (>= 0.5) close 21%; with a transform-delta neighbour (CH2, O, H2,
hexose, deoxyhexose, pentose, glucuronide, CH2O, acetyl, C2H4, H2O, malonyl, sulfate; 3 mDa) AND Tanimoto >= 0.5:
28% (any transform delta: 62%, mostly coincidental); same-mass isomer in the set 7%. -> above the 15% build
threshold, but the set is a curated example list (the hidden test may be denser: "natural molecules and their
analogs" from one isolation platform). Build after items 1-4; offline evaluation needs a pseudo-test of held-out
timsTOF families.
Networking offline test (item 6, Oct 25 16:30; scripts/net_prep.py + net_eval.py): pseudo-test = the 979 v18c val
queries as one test set (stage-2 top 30, query spectra rebuilt from q_rows, modified cosine over every same-polarity
spectrum pair). Spectral neighbours are weak structural evidence: modcos >= 0.7 with >= 6 matched peaks -> truths
Tanimoto >= 0.5 in 26% of pairs; 10-17% of queries have a structurally close high-modcos neighbour (enveda-np-examples:
19-28%, comparable). NAP-style consensus boost (neighbour-weighted softmax of stage-2 x Tanimoto): -0.004 to -0.16
overall at every setting (48 configs). Confident-neighbour variant (max Tanimoto to the top-1 of neighbours with
stage-2 gap >= 1-3, optional transform-delta filter, applied only to unconfident queries): at most +0.0013 overall
(one or two of 979 queries), larger only on post-hoc subsets of 8-60 queries. Below the research gate (+0.05 on
networked queries without a post-hoc pick) and far below LB resolution -> item 6 CLOSED; not built into run_own.
Verified Oct 25: casmi26-own-s2v17bpop v1 (400/400 lists, GLACIER 371/400, 400/400 rows with candidates, no
Traceback; from the execution log while the output endpoint returned 429 for ~4 h, later also from the output) and
casmi26-probe17-coco v1 (393/400 rows, GLACIER 371; placeholder top-1 agreement with probe17b-coco only 0.42 -> PF-d
and PF-b order the COCONUT candidates very differently). scripts/safe_submit.sh now falls back to the execution log
('rows with candidates N of M' >= 90%, no Traceback) when the output download fails.
Oct 27 candidates: casmi26-own-s2v17bpop, casmi26-zen-probe, casmi26-probe17-coco (matched control, only if the
Oct 26 probe17b-coco reading is ambiguous), plus whatever the Oct 26 rules trigger.
PF-syn pilot (started 03:20): scripts/build_sim_jobs.py -> 238,055 structures (38,055 neutral-set window members
incl. truths, 100k train, 100k COCONUT-not-train) x CE 20/40/60 = 714k GLACIER predictions on 4 Kaggle CPU workers
(kernels/casmi26-simw-0..3, scripts/make_sim_workers.py) -> kernels/casmi26-syn-prep (scripts/prep_syn_data.py) ->
TPU pilot kernels/casmi26-pf-train-syn vs matched control casmi26-pf-train-ctl (both 4 h fine-tunes of PF-b);
design under adversarial review (workflow wf_4f2803ec-ac0) before any TPU hour is spent.
PF-syn review (wf_4f2803ec-ac0, 3 lenses, 17 findings -> 4 confirmed, all fixed): (1) the 620 evaluation windows were
16% of synthetic draws with complete windows (~3% in a full-pool run) -> pilot gain inflated: PF_SYN_NEUTRAL_W 0.18
(full-pool density) and half B of the windows held out (scripts/syn_ab_split.py: 19,104 B-only structures dropped
via PF_SYN_DROP; half A = decision half, half B = generalisation check); (2) the gate was the pre-critic +0.03 ->
now pre-registered in the kernel docstrings: half A pf_cos delta win vs training decoys >= +0.06 with the 95%
paired-bootstrap interval > 0, delta vs COCONUT decoys >= -0.01, excess_syn <= excess_ctl and >= -0.02;
(3) stage 2 already has GLACIER, so PF-syn must add beyond it: pf_cos_gl = z(pf_cos) + z(GLACIER) delta > 0 on
covered A queries (scripts/add_gl_feats.py computes the GLACIER column from the simulations); (4) arms step-matched
(PF_STEPS; set PILOT_STEPS from the PF-x v5e step time before pushing). scripts/eval_pf_pair.py reproduces the known
PF-d -> PF-b numbers exactly (pf_bern +0.124 [+0.095, +0.154], = +0.005 LB).
GLACIER simulation speed (Oct 25): ~0.95 predictions/s on 4 CPU threads (597 in 625 s locally; the network forward
pass is ~95% of the time, fragment-tree featurisation ~2%) -> the 4 CPU workers (per-file budget 2.75 h) cover only
~21% of the 714k pilot predictions, shortest SMILES first. casmi/fwdrun.py --device cuda added; GPU simulation
kernel casmi26-simg (T4 x 2, one fwdrun per GPU, hard deadline 1.5 h = the GPU quota this costs; 12.1 h were left,
shared with the owner) replaces them for the pilot; the CPU outputs serve as a CPU-vs-GPU equivalence check.
GPU simulation done (08:15): 704,636 of 714,165 predictions in 5,379 s on T4 x 2 (~65 / s per GPU, ~70x a 4-core
CPU kernel); only the longest-SMILES tails of 2 of 16 files were cut by the deadline -> kernels/casmi26-syn-prep
pushed. Kaggle allows ONE TPU session at a time (queued ones included): the two pilot arms are merged into one kernel
kernels/casmi26-pf-pilot (ctl then syn, PF_STEPS 20,000 each, 3.5 h guard each), pushed when PF-x (x5) has finished.
syn-prep done (09:00): syn.npz 704,636 synthetic spectra of 234,880 structures (all matched to pool ids, fingerprints
computed in the kernel, ~1,037 on-bits), within-set window negatives per structure mean 31.9 (neutral 58.2, train 32.9,
COCONUT 20.8; 9,810 without any). The pilot kernel waits for the single TPU slot (PF-x x5 queued since 04:39).
Kaggle CPU kernels simulate GLACIER at ~6.5 predictions/s each (the 4 workers finished all 714k pilot predictions in
~7.5 h; ~7x this container's CPU): CPU workers are a free alternative to the GPU for simulation and for v18c's
GLACIER pass (stage E now defaults to 4 CPU worker kernels, GL_DEVICE=gpu optional; same device type as inference).
TPU: Tpu1VmV38 queued 2 h without starting; current Kaggle docs list TpuV5E8 -> all TPU kernels moved to v5e-8, PF-x
re-pushed as casmi26-pf-train-x5 (04:39).

## TPU training of the PeakFormer (Oct 25)
Why: PeakFormer b is the only spectral change that moved the LB since Oct 20 (v17b +0.005 over v17), its validation
plateaued after ~7 of 20 epochs (f.z MRR .80-.81, cosine still creeping to .807) -> capacity-bound; the GPU quota is
shared with the owner (17.9 h used, 12.1 h left until Oct 31) while the 20 h / week TPU quota is untouched.
scripts/train_pf_xla.py: same model, losses, sampling and augmentation as train_pf.py with static shapes for XLA:
batches built on the host (numpy, prefetch thread; the float64 sinusoid phase reduction is done there), molecule bits
as one packed bit matrix on the device (898,753 rows x 1,930 bytes = 1.73 GB; ~1,000 on-bits per molecule) unpacked
into a dense multi-hot matrix (EmbeddingBag sum -> multi-hot @ bits.weight, f.z -> batched matmul), data parallel over
the 8 cores, step-based cosine schedule with the step budget calibrated on core 0. CPU checks: embeddings equal to the
original path to 1e-7, f.z to 3e-5; smoke run trains, validates and saves a checkpoint PFScorer loads.
Review workflow wf_e7bb7b73-ae2 (3 lenses + adversarial verification, 12 findings -> 3 confirmed): BLOCKER -- on TPU
v2/v3 PJRT runs two cores as threads of one process sharing the CPU RNG, so the 8 replicas started from different
weights and, with only gradients all-reduced, would have stayed different for the whole run (reproduced on CPU: 43 of
71 tensors differ; fixed with an init lock + broadcast of core 0's parameters); Adam's bias correction reaching 1.0
would have recompiled mid-run (XLA_NO_SPECIAL_SCALARS=1); plus per-core dropout RNG, exit on a failing core and a
30-min no-progress watchdog (a hung collective would hold the TPU until the session limit).
Kernels: casmi26-tpu-probe (framework / speed probe, queued since 01:24 -- Kaggle TPU queue) and casmi26-pf-train-x
(PF-x: variant-b data, d=384 x 6 layers, 8 heads, d_mol 512, dropout 0.1, batch 256 per core x 8, 63 negatives, LR
5e-4, 8 h budget). When PF-x exists: neutral gate + v17x via scripts/build_pfswap.sh <pf_model.pt> v17x (generic
build_v17b.sh), compared with v17b 0.339 on the LB (retrain rule: adopt at >= +0.008).
CPU fallback measured (Oct 25 16:40, kernels/casmi26-pf-cpu-cal, scripts/make_pf_cpu_pilot.py): a Kaggle CPU kernel runs
the pilot model at ~5.0 s/step (batch 256, both arms; this container 2.3 s/step) -> ~8k steps in 11.5 h = 2M samples,
~0.7 exposures per synthetic spectrum (TPU pilot: 41M samples, ~14 exposures) -> too small for the pre-registered gate;
CPU pilot dropped. (150-step smoke: syn arm loss 2.93 -> 2.42, real-only val cos .870 -> .845 on 94 structures, ctl
unchanged; noise-level.) If the Oct 26 specpfb - specpfd reading keeps PF-syn alive and the TPU is still queued: T4
pilot (2 arms x ~3 h, batch 256, ~35k steps each, ~3 exposures per synthetic spectrum), only when the owner's
kernels are not on the GPU.

## v5w drift check (Oct 24): current code reproduces the Oct 20 lists only approximately
The Oct 24 v5w re-runs (probe5w-coco / -train / olast, current casmi26-code, same models and pool) keep the Oct 20 top-1
on 399 / 400 molecules, but only 122 / 400 lists keep their full order (16% of candidate scores move > 0.01; top-5
differs on 46 molecules). NOT the Kaggle image: the default image did change between Oct 22 and 23 (s2v4 / v10 / v11l /
v12 / probe11-coco ran on sha256:81b1e7c4b4f0..., tok3 / v13 / probe11-train on sha256:17b2fc7f27c2...), but re-runs pinned
to the Oct 20 image are bit-identical to the unpinned ones. Models (ret_ranker_v5w, stage2_v5w, fp_model.pt), pool4 and
the competition data are unchanged and every new run_own / own / retrieval / fpscore knob defaults to off; the cause
is an unidentified code-path change since the Oct 19 code the v5w kernel ran on. With the top-1 identical on 99.75% of
molecules, the effect on MRR is expected to be well below LB noise (|d| <~ 0.003) -> new kernels are compared with v5w
0.332 directly, and the Oct 24 token readings are NOT confounded by the image. Kernels stay pinned to the Oct 20 image
(scripts/make_own_kernel.py) so that future default-image changes cannot move results.

## Plan from design panel wf_f100d853-7f9 (Oct 24, after the token results)
Facts verified by the judge: the lib10 family (direct library matching capped at 10 spectra per structure: s2v4 0.330,
v5w-a05 0.323, v5w 0.332; mean 0.328) beats every full-library kernel (v10 0.314, v11l 0.320, v12 0.312, tok3 0.315, tok0
0.316, v13 0.313; mean 0.315; ordering p = 1/84); v10 also switched stage 1 to 63 leaves / min_data 50 -> the family gap
= library mode + trees (+ code date), dropped together by returning to the v5w recipe. Bugs: pipeline_v8d.sh stage 2
loaded only fwd dirs up to fwd_v10 while the job builders skip pairs scored in ANY fwd dir -> v12 / v13 / v13p05 / v16r
stage 2 saw NaN GLACIER on already-scored decoys (fixed Oct 24: every fwd dir loaded); make_own_kernel.py writes LIB_CAP
100000 -> pass LIB_CAP=10 for lib10 kernels.
Builds (Oct 24): thin kernels casmi26-probe5w-coco / -train (v5w lists restricted by source) and casmi26-own-s2v5w-olast
(v5w lists, train / COCONUT candidates first, others after, nothing dropped); v17 = ret4 rows + PeakFormer d columns
(ret17), v5w recipe (31 leaves, alpha 1), kernel with LIB_CAP=10; v17b = v17 with PeakFormer b replacing d after a
neutral gate (win vs same-formula training decoys >= 0.70 and excess <= +0.087 on the 620-query subset).
Adoption rules (pre-registered): post-processing on identical base lists (se ~0.005): adopt at >= +0.005, reject <= 0,
replicate in between; retrain / feature changes vs the matched parent (se ~0.008-0.010): adopt at >= +0.008 with the
targeted restricted probe moving >= +0.005 the same way, else a second consistent point; a new pick #1 needs >= 0.336 or
a paired post-processing gain >= +0.005 on v5w's lists; an adopted change may not cost > 0.006-0.008 on its train probe.
Final selection: pick #1 = v5w + adopted layers unless a retrain base cleared its rule; pick #2 = best adopted config
with different spectral evidence within 0.008 of #1; never a v10-family config. Do not select by the max public score
(0.332 is the max of ~20 related kernels; winner's curse ~0.003-0.005).
Stop: v16r / v13p05 submissions; v10-family builds; token micro-variants (incl. per-candidate mixed token); debiased
models as EXTRA columns (replace, or use as a COCONUT order donor via slot-keeping); channel insertions / promotions;
fusion of rankers with shared evidence; alpha / vendor / PMI / seed / DreaMS / LOTUS / formula-relative variants; CV /
HL-val / HL_PRIOR as decision tools for sim2real-sensitive changes; reading single LB deltas < 0.008 (retrain) or
< 0.005 (post-processing) as signal; class-3 generation until the main score is > ~0.40.
Later (conditional): slot-keeping with a COCONUT order donor (when the donor's COCONUT probe beats the base's by >=
0.010); RRF fusion only of engines with different spectral evidence (v5w + v17b, k = 3) as the pick-#2 hedge;
spectral-only COCONUT probes (PeakFormer d vs b vs the public f.z) to locate the COCONUT gap; spectral scaling
(FP-b, larger instrument-free PeakFormer, DreaMS fine-tune) when GPU windows open (the owner's jobs hold both slots).
GPU Oct 24 03:11: quota 5.34 h used / 24.66 h left; FP-b pushed (one slot freed); PF-b refused again (the owner holds the
other slot) -> PF-b relies on the local CPU run; GPU retry after FP-b.
CONTAINER RESTARTS (Oct 23-24): the session container is reclaimed when idle / restarted every few hours (uptime reset
at ~09:07 Oct 23, ~01:29, ~02:31 and ~05:30 Oct 24); detached local jobs die with it (the local PF-b CPU run stopped at
step 1600, ~02:31, after one epoch). Rule: long training only on Kaggle (GPU / CPU kernels); locally only jobs < ~1-2 h
under a harness-tracked background waiter; every chain must be restartable (skip finished parts).
PF-b GPU (model-d config, PF_HOURS 5) pushed 05:31 (quota 7.5 h used) -> v17b now uses it (scripts/build_v17b.sh
default /home/user/work/kout/pfb_gpu/pf_model.pt), ready ~10:40.
PF-b GPU done 10:35 (57k steps x batch 512 = ~29M samples, ~11x model d; val f.z MRR .805 vs its own restricted
negatives). NEUTRAL GATE PASSED (620 queries, token tied): pf_bern .791 vs training decoys / .835 vs COCONUT / excess
+.044 (d: .667 / .804 / +.137 at token 1, .744 / .876 / +.132 at token 0); pf_cos .810 / .831 / +.022 -> v17b build
(scripts/build_v17b.sh) running 10:47 under a harness-tracked waiter, then v17bb. Part of the gain is the 11x training,
not only the debiasing; v17b vs v17 measures the replacement as a whole.
v17b / v17bb built 11:47 (scripts/build_v17b.sh, build_v17bb.sh): val (unshifted sim, cannot see debiasing) v17 stage 1
.8864 / stage 2 .8949 (.838 / .925); v17b .8895 / .8940 (.843 / .921); v17bb (+ FP-b) .8869 / .8967 (.846 / .924); v17b
stage-1 gain now led by pf_cos_gap (PeakFormer b's cosine head). Kernels casmi26-own-s2v17b (Oct 25 slot 5) and
casmi26-own-s2v17bb (Oct 26) pushed, pinned to the Oct 20 image. v17b COMPLETE 12:08 (400 / 400 lists, GLACIER 371 / 400,
no fallback rows; placeholder top-1 agreement with v17 .953). Probe kernels casmi26-probe17b-coco / -train (pinned)
pushed 11:55 for Oct 26. Verified 13:20: v17bb (400 / 400, GLACIER 371), probe17b-coco (394 / 400 rows), probe17b-train (400 / 400) all clean.
Spectral-only COCONUT probes (panel action 3d; casmi/own.py OWN_SPEC_ONLY=<score> orders the whole window by one
spectral score, no ranker; OWN_SPEC_SRC=coco keeps COCONUT-not-train candidates): casmi26-specpfd-coco (PeakFormer d,
token 1 = deployed) and casmi26-specpfb-coco (PeakFormer b) pushed 13:35 for Oct 26 -> measure the spectral models
directly on the hidden COCONUT answers (compare: v11l COCONUT probe 0.190, public lineage 0.252). specpfb COMPLETE 14:27
(394 / 400 rows, mean list length 16.7; sampled candidates 2,997 / 2,997 COCONUT-not-train by the probe key sets);
specpfd COMPLETE 16:43 (queued long, ran 20 min; 394 / 400 rows, 1,997 / 1,997 sampled COCONUT-not-train). Top-1
agreement PeakFormer d vs b on these lists: 0.62 -> the two models order COCONUT candidates quite differently; the
Oct 26 pair (specpfd vs specpfb) measures which is right on the hidden COCONUT answers.
FP-b GPU (Oct 24 03:11-04:25, 30k steps, 0.135 s/step after the CSR fix; val MRR .756 vs its own restricted
negatives) -> own_models/fp_model_b.pt. Neutral metric (620 queries, token 0 = tied): fp_cos .749 / .822 / +.073 (deployed
token 1 .644 / .786 / +.143, token 0 .692 / .842 / +.150); fp_bern .674 / .767 / +.093, with PMI .745 / .800 / +.054.
-> clearly better prior-free discrimination; one change at a time: v17bb (= v17b + FP-b replacing the FP model, OWN_FP)
after v17b, for Oct 26.
v17 built Oct 24 02:25 (pipeline_v8d.sh T=v17 ROWS=ret17 ALPHA=1 RANKER_PARAMS='{"num_threads": 1}' GL_THREADS=1, i.e. the
v5w recipe; features = v5w's 45 + 12 PeakFormer d columns): OOF stage 1 .8816; val stage 1 .8864 (cls1 .829 / cls2
.917), stage 2 .8949 (.838 / .925) vs v5w stage 2 .893 (.852 / .914); GLACIER: 62 of 101,211 train rows in covered
queries without a score (0.06%, no truth; v16r had 5.1%) -> GL_GLOB fix works. Kernel casmi26-own-s2v17 (LIB_CAP 10,
OWN_PF pf_local_final.pt) pushed 02:35.
Compliance: casmi_probe_srckeys.parquet is reproducible with scripts/build_srckeys.py (Oct 24: identical to the deployed
file, 726,198 keys); an open-source LICENSE file is still missing (owner's decision, needed for prize eligibility).

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
| deployed (9k of 24k steps, reproduce.md G4) | 1 | .644 / .786 / +.143 (+.183) | .599 / .709 / +.109 (+.144) |
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
HL_PRIOR=1 results (rankers trained on their own rows; penalty C: pf_bern 15.8 tok1 / 14.0 tok3, fp_bern 8.46 raw /
3.44 PMI, fp_cos .022, pf_cos .018 / .012):
| ranker / rows | HL-LB | T | C | O |
|---|---|---|---|---|
| v14a1 / ret14 (token 1 = kernel) | .3255 | .874 | .528 | .506 |
| v15a1 / ret15 (token 1 + PMI) | .3250 | .875 | .531 | .452 |
| v13a1 / ret13 (token 3) | **.3385** | .907 | .547 | .556 |
(cross: v15a1 on tok3 rows .3294, v13a1 on kernel rows .3226.) -> PMI null under the prior emulation too (fp_bern carries
little ranker weight; v14a1 vs v15a1 share the recipe) -> not deployed.
CONFOUND: v13a1 = v13 used the pipeline recipe (63 leaves, min_data 50, doc weights alpha 1) while v14a1 / v15a1 / v16a1
were trained with the script defaults (31 leaves, no doc weights) -> the "+.013 for token 3" mixes recipe and token
(the recipe alone moves T by ~.03). Fixed-ranker comparison (v11l weights, rows under each token policy) and v13:
| ranker | token 0 rows | token 3 rows |
|---|---|---|
| v11l (the Oct 24 override kernels) | .3379 | .3361 |
| v13 (pipeline recipe on ret13 rows; kernel casmi26-own-s2v13) | .3393 | .3385 |
| v16a1 (31 leaves, ret16 = token-0 rows) | .3296 | .3274 |
-> retraining on token-3 rows adds ~+.002 over the override; token 0 vs 3 within noise (sd over draws ~.002).
Pipeline T=v13 (ret13 rows) built and kernel pushed Oct 23 11:40 (Oct 24 slot 5).
Fixed-ranker closure: v11l on token-1 rows (= the LB 0.320 kernel) HL-LB **.3179** with HL_PRIOR (plain HL overshot by
+.018) -> pre-registered Oct 24 predictions: tok3 +.018, tok0 +.020, v13 +.021 over 0.320.
EVID_SHIFT (scripts/train_ret_ranker.py PRIOR_SHIFT=lam, casmi/hlprior.py; penalty estimated on the reachable training
queries: C pf_bern 13.2 pf_cos .010 fp_bern 8.3 fp_cos .020, O 9.6 / .008 / 2.4 / .010; v13 recipe on ret13 rows):
| ranker | HL-LB with prior emulation | without |
|---|---|---|
| v13 | .3385 (C .547) | .3555 (C .598) |
| v13p1 (lam 1) | .3453 (C .573) | .3328 |
| v13p05 (lam 0.5) | **.3473** (C .577) | .3451 |
-> lam 0.5 = hedge (+.009 if the prior is real, -.010 if not; the closure says it is). Penalty robustness: C stable on
COCONUT-truth windows (train 13.2 / val 13.9 pf_bern, fp_bern 8.1 both); the O penalty is small on O-truth windows (1.3
/ 0.6, 118 / 18 queries) but ~10-14 in COCONUT windows; unfiltered rows (incl. unreachable synthetic queries) give C 9.7
/ O -0.7 -> the penalty depends on the window chemistry; estimate it on the matching query population.
Pipeline T=v13p05 (ret13p05 rows = ret13 with class-2 truths shifted by 0.5 x the reachable-train penalty): val (shifted)
stage 1 .8948 / stage 2 .8975; kernel casmi26-own-s2v13p05 pushed Oct 23 15:35 (candidate for Oct 25; review
wf_b897a7fd-fd5 running).
Review wf_b897a7fd-fd5 (3 lenses + skeptics, Oct 23): implementation correct (lam 0 reproduces every stored feature;
the v13p05 pipeline's OOF / stage 2 did see shifted rows) BUT EVID_SHIFT NOT ADOPTED:
- artifact (confirmed): PRIOR_SHIFT rankers learned the shifted-truth pattern, mainly through fp_bern -> shifting a
  same-formula COCONUT decoy DOWN raises its ranker score (lam 1: v13 -0.324, v13p05 -0.054, v13p1 +0.196; rank-1 share
  v13p1 .0125 -> .0202) -- the DOC_TRANSPLANT failure again (scripts/check_decoy_shift.py);
- circular evidence: HL_PRIOR scores the transform the rankers were trained on; at other (transplant p, lam_eval)
  settings v13p05 - v13 is -0.009..+0.008 (p=0: -.006 / +.001 / +.003 at lam 0 / .5 / 1);
- the v11l closure (.3179 vs .320) fixes the transplant p and lam only jointly (p=1, lam 0: .3380; p=0: ~.44); the
  analog features (an_*) show a held-minus-COCONUT gap as large as the spectral one (in within-query sd) that HL_PRIOR does
  not emulate; vs PURCHASABLE COCONUT decoys the penalty is 35-75% smaller;
- LB power: paired se of v13p05 vs v13 ~.007-.010 on the public 124 -> adopt only if >= +.010 over v13; slot only if spare.
-> the prior is removed at its source instead (PF-b / FP-b GPU retrain, Oct 24); HL_PRIOR stays a diagnostic with
known identification limits. Pre-registered Oct 24 token predictions remain (they test p=1, lam=1).
Adoption chain scripts/adopt_spectral_models.sh <pf> <fp> <tag> [token] (neutral eval -> rows -> v13-recipe ranker ->
HL-val with / without prior vs v13 -> full pipeline) validated on the current models at token 0 (tag v16r): PeakFormer d
under token 0 pf_bern .744 / .876 / +.132 (token 1 .667 / .804 / +.137, token 3 .734 / .871 / +.137); v16r HL-LB .3378
(prior) / .3559 (no prior) vs v13 .3385 / .3555 -> token 0 = token 3 at matched recipe; val stage 1 .9008 / stage 2 .8964.
Kernel casmi26-own-s2v16r pushed Oct 23 16:45 (insurance if token 0 wins on Oct 24); v13 v2 and v13p05 COMPLETE.
Pre-submission check (Oct 23 21:30, output endpoint recovered from 429): v13 v2 / v13p05 / v16r all clean (own lists 400
of 400, no Traceback, 0 fallback rows, GLACIER 371 / 400, 50-61 min); placeholder top-1 agreement v13 vs v16r .96,
v13 vs v13p05 .975.
GPU Oct 24: PF-b (d=384 x 6, batch 512) died at step 1 with CUDA OOM on the T4 (11.8 GB in use + 3.1 GB requested;
the config had never run on GPU). The owner's own GPU jobs (ntl-c-lora-aug6a / 6b, from 00:20) hold both batch GPU
slots ("Maximum batch GPU session count of 2 reached" for FP-b and the PF-b retry). Quota at 01:10: 1.63 h used of 30.
Re-plan (shared quota, owner first): FP-b first (~1.5-2 h); PF-b resized to the deployed model-d config (d=192 x 4,
6 heads, d_mol 256, no dropout, batch 512) so that b vs d isolates the training-data change, PF_HOURS 5; pushed only
when a slot frees and >= 8 h quota remain. Fallback started 01:10 on local CPU (no GPU quota): PF-b in the exact model-d recipe (PF_D=192 PF_LAYERS=4
PF_HEADS=6 PF_DMOL=256 PF_DROP=0 PF_SBAL=0.5 PF_SEED=3, 40 x 1500 x batch 256 x 23 negatives, 3 threads, PF_HOURS=9)
+ PF_INST_DROP=1 PF_TIMS_W=1 PF_NEG_TRAIN=negtrain v2 PF_NEG_NOREP=1 -> /home/user/work/pf_local_b/pf_model_b.pt
(~10:10 UTC); paired with d for the neutral metric and scripts/adopt_spectral_models.sh.
Seed bagging (v13 recipe, ret13 rows, HL_PRIOR): seed default .3385, seed 1 .3353, seed 2 .3363, 3-seed mean-score bag
.3379 -> seed sd ~.002, bagging +.001 over the average single model -> not worth 3x OOF / kernel cost.
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
