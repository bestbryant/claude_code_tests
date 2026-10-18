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
| 10-18 | casmi26-own-noprior v2 | same without documentation priors | 0.245 |
| 10-17 | casmi26-s2s-np-eu | fork + q_npeaks + electron undo on rerun | 0.430 (-0.021: do NOT undo the correction) |
| 10-17 | casmi26-novel-probe v2 | ONLY novel candidates (class-3 reach probe) | 0.010 |

## Lesson from the class-3 probe (Oct 17)
The novel-only list scored 0.010 public, against ~0.045 expected from the PubChem-absent proxy (MRR 0.10 x ~45% class 3).
The proxy overstates reach ~4-5x: absent training molecules (reverse-metabolomics conjugates, synthetic analogs) are
much easier than Enveda's novel test compounds. Class 3 via analog derivation/conjugation is worth at most ~0.005-0.01
as an overlay. Pivot: use the same machinery (analog derivation + site localisation) for class-2 isomer ranking
(known-only probe pending).

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

## Next
1. LTR (LightGBM lambdarank) over novel candidates (rows dump running) -> better class-3 top-1.
2. Overlay into the 0.45 pipeline, gated by library similarity; choose slots from probe/LB evidence.
3. mmpdb (chunked) transform mining from COCONUT for wider derivation recall; COCONUT building blocks.
4. Compliance rebuild.
