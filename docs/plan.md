# CASMI 2026 (Enveda) — working plan and log

Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
Task: up to 25 ranked SMILES per test molecule; MRR@25 on tautomer-canonical InChIKey14 (RDKit 2026.03.3).
Hidden test ~400 molecules (~1,500 timsTOF spectra), public LB = 31% (~124 molecules), private 69%.
Code competition: notebook ≤ 9 h, no internet. Deadline Dec 14 2026 (merge/entry Dec 7).

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
  Kernel casmi26-own-s2v5w (models in bryantyu/casmi26-models).
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
