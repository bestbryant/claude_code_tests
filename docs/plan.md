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
| 10-20 | casmi26-fprobe-train v3 | public 0.451 lineage restricted to training structures | pending |
| 10-20 | casmi26-fprobe-coco v3 | public lineage restricted to COCONUT-not-train | pending |
| 10-20 | casmi26-fprobe-other v3 | public lineage restricted to other candidates | pending |

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
