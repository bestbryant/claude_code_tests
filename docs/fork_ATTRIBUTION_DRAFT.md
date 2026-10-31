# DRAFT: Attribution for the eligible forks casmi26-pube0 and casmi26-pube1

> **DRAFT, not in force.** This file waits for the owner's LICENSE decision (docs/reproduce.md G19,
> docs/fork_eligibility.md F8). It grants no licence. No LICENSE file is added to the repository (owner's decision).
> Until the owner decides, our own code in these notebooks has no licence statement beyond what the notebooks and
> casmi26-elig-assets say. Companion draft: [`NOTICE_DRAFT.md`](NOTICE_DRAFT.md) (Apache-2.0 and MIT code).

Status 2026-10-31. Scope: everything the two eligible fork kernels mount (`kernels/casmi26-pube0/kernel-metadata.json`,
`kernels/casmi26-pube1/kernel-metadata.json`) or embed (notebook cells 0-23, including the pinned sources of cell 8 and
the frozen 0.465 notebook in `FROZEN_B64`).
- Each entry gives the licence, the source URL and the notice that licence requires.
- Asset ids `A#` and verdicts (E eligible, U unclear, I ineligible) are those of
  [`fork_eligibility.md`](fork_eligibility.md), section 1.
- Sources: that audit, the two kernels' metadata, cell 23 ("Credits") of each notebook, section 6 and the credits cell of
  the embedded 0.465 notebook, and the dataset cards and LICENSE-NOTICE files saved during the audit. These were not
  re-checked on 2026-10-31.
- Our own data sources and their citations are in [`ATTRIBUTION.md`](ATTRIBUTION.md).

The two kernels share everything except Engine 2:
- **pube0 (elig0)** mounts 10 upstream datasets and runs the null Engine 2 (fix F2a).
- **pube1 (elig1)** adds our bryantyu/casmi26-elig-assets and bryantyu/casmi26-pool-v18c and runs our stage 1 as Engine
  2 (fix F2b).

In both, `OURS = False` and `ISO_ALPHA = 0`, so our PF-syn cell (11) and our ISO cell (17) are embedded but do not
run. Image: `gcr.io/kaggle-private-byod/python@sha256:37c64f7d...` (A2), GPU T4, internet off.

## 1. Competition data and runtime

| asset | used for | licence | required notice | source |
|---|---|---|---|---|
| Competition data: train.parquet, test.parquet, sample_submission.csv (A1) | library, queries, pool rows, in-kernel caches; training data of every model below | CC BY-NC 4.0 (rules page, as quoted in plan.md roadmap section 0). The host tolerates non-commercial terms inherited only from train.parquet (threads 748882 / 748065; Ahmed Berat Özer's cards cite discussion 745072). | Attribution to the competition and Enveda; non-commercial use. Do not redistribute the raw files (rule 2.4.b, as cited in `.gitignore`). The enveda-180 rows also cite Enveda-180 (Zenodo 10.5281/zenodo.21346580, CC BY 4.0; ATTRIBUTION.md 1.2). | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra |
| Kaggle image python@sha256:37c64f7d... (A2): Python 3.12, torch 2.10, numpy, pandas, LightGBM | runtime | open-source components (BSD / MIT / Apache-2.0) | none: not redistributed by us | Kaggle |

## 2. Mounted datasets

### 2.1 Mounted by both kernels

**ahmedberatozer/casmi26-v4b-models** (A6, A8-A11; A6 U through A10, A8 E, A9 E with caveat, A10 U, A11 U)
- URL: https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models
- Used for: Engine 1, that is the `casmi` package, the fe_v4 families and tables, FPNet A / B, fe_D and ranker_v4b.
- Licence: Kaggle "other". The card (licence section updated 2026-10-13) puts all of Ahmed Berat Özer's code under MIT,
  "Copyright (c) 2026 Ahmed Berat Özer".
  - Weights and tables are built only from train.parquet and COCONUT.
  - fe_D.pt is derived from DreaMS (Bushuiev et al., Nat. Biotechnol. 2025; weights MIT / CC BY 4.0).
  - `code/casmi/dreams_model.py` re-implements DreaMS: MIT, "Copyright (c) 2024 Roman Bushuiev" (LICENSE-NOTICE.txt).
- Required notice:
  - MIT copyright and permission notice wherever his code is copied. The fork mounts it and does not copy it; the NOTICE
    draft carries the notice anyway, see A26 there.
  - Keep the COCONUT and DreaMS attributions, as the card asks.
  - The weights inherit the competition terms (non-commercial, attribution).

**ahmedberatozer/casmi26-v3-models** (A12, E)
- URL: https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v3-models
- Used for: the PubChem channel, that is the v3 `casmi` code and FPNet A / B. ranker_0.pkl is not read.
- Licence: Kaggle "other". Code MIT ("Copyright (c) 2026 Ahmed Berat Özer"); weights from train.parquet and COCONUT.
- Required notice: as v4b (MIT notice; COCONUT attribution; competition terms for the weights).

**ahmedberatozer/casmi26-fpnet-full1** (A7, E with a G4-class caveat)
- URL: https://www.kaggle.com/datasets/ahmedberatozer/casmi26-fpnet-full1
- Used for: Engine 1's retrieval fingerprint bank (fpnet_full1.pt).
- Licence: Kaggle "other". Weights only, trained on train.parquet; the model code is the MIT `casmi` package of v4b.
- Required notice: competition terms for the weights; credit Ahmed Berat Özer.

**ahmedberatozer/casmi26-v2-pool** (A13, A14, E)
- URL: https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool
- Used for: Engine 1's candidate pool (710,701 structures: training structures and COCONUT), fingerprints, fragment
  masses, and pool popularity (pool_lsid / pool_lpmid through the popularity prior).
- Licence: Kaggle "other". No code. Built from train.parquet and COCONUT.
- Required notice: keep the COCONUT attribution (Chandrasekhar et al. 2025, doi:10.1093/nar/gkae1063). The card calls
  COCONUT CC BY 4.0; COCONUT's own page says CC0 1.0 (ATTRIBUTION.md 1.3), and either way the attribution is kept.
  Competition terms apply to the training-structure rows.

**ahmedberatozer/casmi26-iceberg** (A21: weights E, glue U; A4 E)
- URL: https://www.kaggle.com/datasets/ahmedberatozer/casmi26-iceberg
- Used for: ICEBERG scores, the E006 window-40 fallback and the promotion gate. It also supplies the simulator wheels
  that GLACIER uses.
- Licence: Kaggle "other". Its LICENSE-NOTICE.txt (text as in the copy inside the v4g container) lists:
  - `src/ms_pred/`: a subset of ms-pred @708148c2a8eb, MIT, "Copyright (c) 2023 Samuel Goldman" (full text in
    `src/ms_pred/LICENSE`);
  - `ckpt/gen.pt`, `ckpt/inten.pt`: ICEBERG msg_all weights, MassSpecGym-trained, published with ms-pred (MIT);
  - `shim/`: Ahmed Berat Özer's own code, with no licence stated;
  - `wheels/*.whl.ice`, unmodified PyPI wheels: rdkit 2025.3.6 (BSD-3-Clause), pytorch_lightning 2.6.6, torchmetrics
    1.9.0 and lightning_utilities 0.15.3 (Apache-2.0), einops 0.8.2 (MIT).
  - The runner glue `ice_runner.py` and `fuse.py` carries **no OSI licence** (fix F5a open).
- Required notice:
  - The ms-pred MIT notice.
  - The licence notices of the wheels: they are redistributed by the dataset, not by us.
  - Cite ICEBERG: Goldman, Li, Coley, Anal. Chem. 96:3419-3428 (2024), doi:10.1021/acs.analchem.3c04654.
  - Cite MassSpecGym: Bushuiev et al., NeurIPS 2024, arXiv:2410.23326.
  - The glue cannot be relicensed by us; it needs the author's grant (F5a) or our own runner (F5b).

**ahmedberatozer/casmi26-glacier** (A22: weights E, glue U)
- URL: https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier
- Used for: GLACIER H1 scores, the fusion re-rank and the window-40 overlay. It also runs the N1 proxies (A23).
- Licence: Kaggle "other". Its LICENSE-NOTICE.txt lists:
  - the ms-pred subset (MIT, as above);
  - `ckpt/glacier.pt`: the GLACIER MassSpecGym checkpoint of the ms-pred authors (best.ckpt sha256 5a47cecca707d3ab...,
    the same tensors as our casmi26-glacier), used under the ms-pred MIT licence;
  - `shim/`: own code, including an independent re-implementation of `LinSATNet.linsat_layer` (LinSATNet: Wang et al.,
    ICML 2023, MIT).
  - It reuses the iceberg wheels.
  - The glue `gl_runner.py` and `gl_fuse.py` carries **no OSI licence** (F5a).
- Required notice:
  - The ms-pred MIT notice.
  - Cite GLACIER: Wang, Wang, Coley, arXiv:2606.29161 (2026).
  - Cite MassSpecGym.
  - Glue: as for iceberg.

**dmitriigluzdov/casmi26-pubchem-popularity-prior** (A15 E, A16 U, A17 U)
- URL: https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior
- Used for:
  - the PubChem tier (105,875,489 rows; folder `casmi26-pubchem-tier/`, a byte copy of ahmedberatozer/casmi26-pubchem-tier);
  - the popularity-prior arrays of the PubChem channel;
  - pool_popularity.csv (POP_W and NP_W of the final layer).
  - In these builds the depositor-source bonus is off (SRC_BITS 0) and SPEC_W is 0 (fix F3), so the
    `src_spectral_library` flag column is not loaded.
- Licence: Kaggle "other". The card's "Licence and credits" section says:
  - PubChem is "Courtesy of the U.S. National Library of Medicine", reused under the NCBI data-use policy, the NLM FTP
    terms and PubChem's source guidance; NLM does not endorse the dataset.
  - LOTUS 2026-04-13 (Zenodo 10.5281/zenodo.19360665) is CC BY 4.0. The frozen notebook's own section 6 still says
    "LOTUS membership CC0"; the card corrected this on 2026-10-20.
  - NPAtlas is CC BY-NC 4.0. NPAtlas is on the host's allowed list.
  - The tier folder is Ahmed Berat Özer's. Its card asks for use under the source licences, non-commercial and with
    attribution.
- Required notice:
  - The NLM courtesy line and its no-endorsement sentence.
  - Cite Kim et al., NAR 53:D1516 (2025), doi:10.1093/nar/gkae1059.
  - LOTUS: credit Rutz et al., eLife 2022, doi:10.7554/eLife.70780, with the CC BY 4.0 link.
  - NPAtlas: non-commercial use, attribution to van Santen et al., NAR 2022, doi:10.1093/nar/gkab941.
  - Credit Ahmed Berat Özer for the tier and wangpenghua for the popularity formulation, as the card does.

**dmitriigluzdov/casmi26-fold-safe-fpnet** (A18, E)
- URL: https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet
- Used for: the fold-safe FPNet (fpnet_fold0.pt) in the PubChem channel's model bank.
- Licence: Kaggle "other".
  - The checkpoints and competition-derived files follow the competition data terms: non-commercial use with attribution.
  - The training and evaluation scripts are Apache-2.0.
  - The architecture and the fingerprint bit selection are Ahmed Berat Özer's: "Copied code and bit indices retain their
    original notices."
- Required notice: competition terms (NC, attribution) for the checkpoint; the Apache-2.0 notice if the scripts are
  copied (the fork does not copy them); credit Dmitrii Gluzdov and Ahmed Berat Özer.

**sanpier/casmi-spectrum-model-v1** (A37, E with caveat; F7 open)
- URL: https://www.kaggle.com/datasets/sanpier/casmi-spectrum-model-v1
- Used for: the neural spectrum-candidate scorer, read as model.py, features.py, config.json and final_model.pt.
  `MODEL_PINS` in cell 8 holds their sha256: a97ad9b3..., 25474188..., 563fb2a2... and 44436b2d....
- Licence: Kaggle "apache-2.0". The card: "The original PyTorch checkpoints, author-created training scripts,
  configurations, and supporting artifacts distributed in this release are licensed under the Apache License, Version
  2.0"; third-party dependencies and datasets keep their licences. The model is trained on the competition data.
- Required notice:
  - Apache-2.0 (a copy of the licence; keep the attribution).
  - "Developed and published by Saner Turhaner (kaggle.com/sanpier)".
  - The competition terms for the training data.

**metric/rdkit-2026-3-3-wheel** (A3, E)
- URL: https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel
- Used for: RDKit 2026.03.3, the metric's canonicalisation and the pipeline's chemistry. The v4g copy is no longer
  mounted.
- Licence: Kaggle "unknown"; RDKit itself is BSD-3-Clause.
- Required notice: BSD-3 notices travel with the wheel, which Kaggle redistributes, not us. Cite "RDKit: Open-source
  cheminformatics", doi:10.5281/zenodo.591637.

### 2.2 Mounted by pube1 only (ours)

**bryantyu/casmi26-elig-assets** (ours; prepared by `scripts/pub_variant/make_elig_assets.py`, **not created on Kaggle yet**)
- Used for: Engine 2 of pube1 (fix F2b). It holds:
  - 16 `casmi` modules (the import closure of `casmi.run_eng2`);
  - fp_model_v1r.pt (a9748a4a) and pf_tpu_f.pt (3f594bf0);
  - ret_ranker_v18cu.txt (1bfde65f) and ret_ranker_v18cur1.txt (d3b10b6b) with their feature lists;
  - README.md and MANIFEST.sha256.
- Licence as prepared: Kaggle "other". The `*.py` files are MIT (LICENSE.txt, "Copyright (c) 2026 bryantyu (Kaggle user
  bryantyu, CASMI 2026 entry)"). The six model files are CC BY-NC 4.0 (MODELS_LICENSE.txt), because they are trained on
  the competition data.
- Third-party data behind the models:
  - train.parquet;
  - COCONUT 10-2026 (CC0);
  - ChEBI r255 (CC BY 4.0);
  - LIPID MAPS LMSD (CC BY 4.0);
  - PubChemLite for Exposomics 3.3.0 (CC BY 4.0);
  - PubChem counts (NCBI policy);
  - GLACIER predictions for PF-syn (ms-pred MIT, the MassSpecGym checkpoint of its authors).
  - Citations: ATTRIBUTION.md section 1. Provenance: reproduce.md sections 6 and 7.
- Required notice: the MIT notice for the code; for the models, CC BY-NC 4.0 attribution plus the source attributions in
  MODELS_LICENSE.txt and README.md. The MIT part depends on the owner's LICENSE decision.

**bryantyu/casmi26-pool-v18c** (ours)
- Used for: Engine 2's candidate pool in pube1: 1,160,133 structures with fingerprints and COCONUT priors.
  - Files: pool.parquet 8afda40a, pool_fp.npy cef50fc5, coconut_prior.parquet f8b83ce5.
  - Built by `scripts/build_v18c.sh` stage A (reproduce.md section 6).
- Licence: Kaggle "CC-BY-4.0". However, its 275,810 `in_train` rows are training structures under the competition terms
  (CC BY-NC 4.0), so the CC BY label conflicts with them. The dataset page has no attribution text yet (G19, open).
- Sources:
  - COCONUT 10-2026: CC0;
  - ChEBI r255: CC BY 4.0;
  - LIPID MAPS LMSD: CC BY 4.0;
  - PubChemLite for Exposomics 3.3.0: CC BY 4.0;
  - PubChem documentation counts: NCBI policy.
- Required notice:
  - CC BY 4.0 attribution for ChEBI, LMSD and PubChemLite, with the licence link and the note that the data were
    standardised and filtered (`build_bio_structures.py`, `fetch_pubchemlite.py`).
  - The NCBI/NLM courtesy line.
  - The competition terms for the in_train rows.
  - Fix: ship ATTRIBUTION.md in the dataset and choose a licence label that keeps the in_train rows under the
    competition terms (roadmap section 7: CC BY-NC 4.0 for competition-derived content).

## 3. Code embedded in the notebooks

All third-party code below is notebook code published on Kaggle. The audit rates it as Apache-2.0, Kaggle's licence for
public notebooks; that rating was not re-checked against the Kaggle terms page. The embedded 0.465 notebook states this
itself: "The public notebook declares Apache-2.0; upstream datasets/components retain their own licences."
- **Modified by us** means patched by `scripts/make_pub_variant.py` (ELIG=0 / ELIG=1).
- **Runs** means it executes in the pube0 / pube1 run.

| part | where | author and source | licence | modified by us | runs |
|---|---|---|---|---|---|
| Notebook body (configuration, session and stage machinery, run, fragment-check and merge cells, sections 1-7 text) | cells 0-3, 7, 9, 12, 13, 15, 18-22 | Roman Tamrazov, [Spectral Parallax](https://www.kaggle.com/code/romantamrazov/casmi-2026-spectral-parallax) v22, the notebook this one runs. It was extended to "Variant M" (0.477; merge cell, run table, docs) by busyaprime (kernel 33-casmi26-px-m1; slug as named in section 5 of the notebook, not opened) and published as kunaldesale2408/enveda-casmi-2026-mid (https://www.kaggle.com/code/kunaldesale2408/enveda-casmi-2026-mid, URL built from the slug in plan.md). Lineage credits (Kozykorpesh, Roman Rozen, alex chilton, Saner Turhaner, Dmitrii Gluzdov) are in cell 23. The authorship of individual cells is only as these credits state. | Apache-2.0 (A42) | yes: cell 0 (eligibility notes), cell 1 (ELIG, OURS, PF, ISO and final-layer knobs, SPEC_W 0), cells 20, 21 and 23 (text) | yes |
| Figure helpers and the public-score field record (FIELD, taken 2026-10-28) | cells 4-6 | busyaprime (`engine/33_style.py` at their repo commit 4ebb73e9) | Apache-2.0 | no | display only |
| Six helper modules: runner.py, export_evidence.py, neural_scores.py, submission.py, ranking.py, bundle.py | cell 8 `MODULES` (sha256 pinned in `MODULE_SHA256`) | Saner Turhaner, [PubChem + Neural025](https://www.kaggle.com/code/sanpier/pubchem-neural025-lb-0-466) | Apache-2.0 (A39) | no (byte-identical, hash-checked) | yes |
| consensus.py (leader-safe RRF), outsiders.py | cell 8 `MODULES` | Roman Rozen, [From Peaks to Structures](https://www.kaggle.com/code/romanrozen/0-468-casmi-from-peaks-to-structures) | Apache-2.0 (A40) | no | yes |
| The frozen 0.465 pipeline notebook (`FROZEN_B64`; original sha256 d1e757c2b71b46e6...) | cell 8, run by runner.py | Dmitrii Gluzdov, [The Data Behind Spectra to Structures](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures). It embeds code by the authors in its own section 6 table and credits cell, kept unchanged (list in NOTICE_DRAFT.md part 1) | Apache-2.0 as declared by that notebook (A25, A26, A27) | yes. pube0: cells 5, 9, 12, 14, 16, 18, 26, 32, 36, 39, 45, 54 (sha256 47db9220...). pube1: the same plus cell 31 (sha256 67764f55...). Each patched cell starts with an "elig0 / elig1 (ours, fix F#)" comment. | yes; display cells 9, 12, 14, 16 and 18 skipped |
| frag_rescore.py (untrained MetFrag-lite fragment scorer) | cell 10 (`_physical`, sha256-checked) and FZ cell 51 | Huseyin Emre Aksoy ([profile](https://www.kaggle.com/huseyinemreaksoy)), reached through Ahmed Berat Özer's [v7c notebook](https://www.kaggle.com/code/ahmedberatozer/casmi26-v7c-e006-fraggap) | Apache-2.0 (A24) | no | yes |
| parallax_physical.py (fragment check) | cell 10 (`_post`) | Roman Tamrazov, Spectral Parallax v22 | Apache-2.0 (A41) | no | yes (PHYSICAL True) |
| Provenance-aware reranker (parallax_public_provenance.py) | cells 14-15 | Roman Tamrazov, Spectral Parallax v22; idea credited there to heliosli | Apache-2.0 (A41) | no | no (PROVENANCE False) |
| midx final layer | cell 16 | baidalinadilzhan, casmi26-midx-exact-dedup-fill, its cell 15 (https://www.kaggle.com/code/baidalinadilzhan/casmi26-midx-exact-dedup-fill, URL built from the slug in cell 1) | Apache-2.0 (A42) | yes: our `_mid_neural_score` hook (PF_MID_W; 0 here = midx exactly); configuration lines moved to cell 1; SPEC_W 0 (F3) | yes |
| PF-syn cell | cell 11 | ours | pending owner (MIT planned) | n/a | no (OURS False) |
| ISO isomer reranker ("heliosli-style" idea; our code) | cell 17 | ours | pending owner | n/a | no (ISO_ALPHA 0) |
| Eligibility code: mount and hash check, `ELIG_*` tables, `V43_HASHES_JSON` (hashes of casmi26-pubm's mounted inputs), null_engine.py; pube1 also eng2_merge.py | cell 8 | ours (`scripts/pub_variant/elig.py`, `elig_cell.py`, `null_engine.py`, `eng2_merge.py`; module headers say "Ours, MIT") | pending owner (MIT in the module headers) | n/a | yes |

Ideas credited without code: alex chilton (rank 2-25 weight 0.75, [notebook](https://www.kaggle.com/code/alexchilton/casmi26-lb-0-472-leader-safe-ranks-2-25-at-0-75)),
Kozykorpesh (V2 consensus pipeline, [notebook](https://www.kaggle.com/code/kozykappa/lb-0-468-balanced-acquisition-evidence)) and heliosli
(training-structure protection, [notebook](https://www.kaggle.com/code/heliosli/happy-to-make-the-dataset-public-if-it-helps)).

## 4. Credited in the notebooks, not mounted and not read (lineage only)

The embedded notebook's section 6 and cell 23 still name these. They keep their credit; neither kernel mounts or reads
them (fixes F1 and F2), so no notice obligation arises from them:
- dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets (CC BY-NC-SA 4.0, A5);
- casmi26-natural-product-knowledge-table and casmi26-natural-product-panel-benchmark (A19, A20);
- sanpier/catalogue-experiment-saner (A38);
- the Engine 2 inputs of the original: prvsiyan casmi26-fp-models-v2, casmi26-ranker-features,
  coconut-casmi26-candidates and chebi-lipidmaps-casmi26; megayak casmi26-simulated-ranker-rows;
  sergiomalv/casmi-chebi-lmsd-structures; franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints;
  cosmicbe/casmi26-pubchemlite (A29-A36);
- hengck23's DreaMS resource (A45);
- ahmedberatozer/casmi26-spectrum-model-long (runs e3-e5 of the lineage only);
- the MassBank / MoNA / GNPS external reference spectra (F4 off).

## 5. What each licence requires of the fork (summary)

- **Apache-2.0 (section 3, and A37 / A18 scripts):** §4 of the licence.
  - (a) Ship a copy of the licence with the notebook.
  - (b) State our changes in every modified file: the patched frozen notebook cells, cell 16, and the body cells patched
    by make_pub_variant.py. The patched cells already carry "ours, fix F#" comments; NOTICE_DRAFT.md lists them.
  - (c) Keep the attribution notices: the frozen notebook's section 6 and credits cell are kept unchanged.
  - (d) Pass on any NOTICE file. None of the notebooks ships one.
  - The fork as a whole may carry other terms (MIT for our parts) as long as these conditions are met (F8). If we win,
    confirm this with the host.
- **MIT (Ahmed Berat Özer's `casmi`, DreaMS port, ms-pred, einops, LinSATNet):** keep the copyright line and the
  permission notice in every copy or substantial portion. The fork runs this code from mounted datasets and copies none
  of it, apart from what A26 rates as MIT-derived PubChem-channel code inside the frozen notebook. NOTICE_DRAFT.md carries
  the notices.
- **CC BY 4.0 (LOTUS; ChEBI, LMSD and PubChemLite through pool-v18c and our models; Enveda-180 inside train.parquet):**
  - credit the creators, link the licence, and state changes (standardisation, filtering);
  - no added restrictions on the CC BY material itself.
- **CC BY-NC 4.0 (competition data, NPAtlas, every model trained on train.parquet):** attribution and non-commercial use
  only. This covers saner's model, Ahmed Berat Özer's weights, the fold-safe FPNet and our models.
  - The host tolerates NC inherited from train.parquet.
  - The winner licence ("Open Source - MIT", roadmap section 0) applies to code. How it interacts with NC model weights
    is a host question; it is already listed under F6 / F8.
- **CC0 (COCONUT):** no requirement; citation kept.
- **NCBI / NLM (PubChem):** the courtesy line and the no-endorsement sentence; cite PubChem 2025.
- **BSD-3 (RDKit wheels):** notices travel with the wheels. They are redistributed by Kaggle and Ahmed Berat Özer, not
  by us.
- **No licence (ICEBERG / GLACIER runner glue, A21 / A22):** it cannot be redistributed or relicensed by us. Fix F5a
  (MIT grant from its author) or F5b (our own runner).

## 6. Open points before this draft can be shipped (owner)

1. LICENSE for our code (MIT planned; reproduce.md G19). This governs:
   - cells 11 and 17 and the eligibility code of cell 8;
   - the casmi26-elig-assets `*.py` files;
   - the repository.
2. Set casmi26-pool-v18c's licence label and attribution text: in_train rows under the competition terms (2.2).
3. Create casmi26-elig-assets after reviewing LICENSE.txt and MODELS_LICENSE.txt.
4. F5a: ask ahmedberatozer for an OSI licence on ice_runner.py, fuse.py, gl_runner.py, gl_fuse.py and shim/. Until he
   answers, the fork is not eligible (fork notebooks, cell 0).
5. F8: ship NOTICE (from NOTICE_DRAFT.md) and the Apache-2.0 text with the published fork notebook. State in the
   notebook text which cells are ours (MIT) and which are third-party (Apache-2.0 / MIT).
6. Re-check every licence statement above on the live pages before publishing; this draft relies on copies saved
   during the Oct 31 audit.
