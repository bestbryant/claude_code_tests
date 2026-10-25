# CASMI 2026 (Enveda) — molecule identification from MS/MS

Code for the Kaggle competition [enveda-CASMI26-molecule-id-mass-spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra):
rank up to 25 candidate structures per test molecule from its timsTOF MS/MS spectra (metric MRR@25 on the
tautomer-canonical InChIKey first block). Working notes, results and the submission log are in
[`docs/plan.md`](docs/plan.md); a short illustrated overview is in [`docs/strategy_report.pdf`](docs/strategy_report.pdf).

Every learned component is trained here from `train.parquet` and permitted public data, or is a weight the
competition explicitly allows (MassSpecGym-trained GLACIER from ms-pred). Competition data are never committed.
Not every intermediate can be rebuilt yet: the rebuild recipes, md5s and open gaps of the submitted entry
(`kernels/casmi26-own-s2v17b`, LB 0.339) are in [`docs/reproduce.md`](docs/reproduce.md); third-party data and models
with licences and citations in [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md).

## Pipeline (kernels `kernels/casmi26-own-s2v*`; submitted entry `casmi26-own-s2v17b`)

1. **Candidate pool** (pool v4, `scripts/build_pool4.sh`): `scripts/build_pool.py` merges training structures,
   COCONUT and PubChemLite (with the PubChemLite annotation counts `pcl_*`). The other documentation priors come from
   separate steps: the COCONUT organism / collection / DOI counts are `coconut_prior.parquet`, built from the full
   COCONUT CSV by `scripts/build_coconut.py`; the PubChem SID / PMID / patent counts (`pc_*`, tables built from the
   PubChem FTP by `scripts/build_pubchem_tables.py`) are joined by `scripts/add_pubchem_pop.py`; ChEBI / LIPID MAPS
   rows (`in_bio`) are added by `scripts/build_pool_bio.py`. Candidates are taken in a 10 ppm window of the neutral
   mass.
2. **Stage-1 features** (`casmi/retrieval.py`, `casmi/own.py`): weighted spectral-entropy library similarity,
   modified-cosine spectral analogues propagated by Tanimoto similarity, explained intensity of 1–2-bond fragments,
   a spectrum↔fingerprint dual encoder (`casmi/fpmodel.py`, trained by `scripts/train_fp.py`), PeakFormer scores (below)
   and the priors. The dual encoder and PeakFormer are trained against negatives from pool v1
   (`scripts/build_pool_v1.py`: training structures + COCONUT only, no PubChemLite or ChEBI / LIPID MAPS), not the
   candidate pool above.
3. **Stage-1 ranker**: LightGBM LambdaRank trained on simulated queries (`scripts/dump_ret_rows.py`): held-out
   training structures, class 1 = only the query spectra hidden, class 2 = all spectra hidden and the truth reachable
   without `train.parquet` (`scripts/train_ret_ranker.py --drop-artifact`).
4. **GLACIER re-scoring** (`casmi/fwdrun.py`): the stage-1 top 30 are scored by the weighted entropy similarity
   between each candidate's GLACIER-predicted spectrum and the query (positive [M+H]+ / [M+Na]+ spectra). Runs the
   official ms-pred source with the original MassSpecGym checkpoint; `fwdshim/` is our pure-PyTorch stand-in for
   the DGL / torch_scatter calls (reproduces the original scores exactly).
5. **Stage-2 ranker** (`scripts/train_stage2.py`, `casmi/fwdfeat.py`): LambdaRank on stage-1 features + GLACIER
   features (within-formula z-score / rank / gap), trained on out-of-fold stage-1 scores (`scripts/oof_stage1.py`)
   with GLACIER computed on Kaggle CPU workers (`scripts/build_fwd_jobs.py`, `scripts/make_gl_workers.py`).

**Documentation-shift weighting** (`casmi/docw.py`): a simulated class-2 truth is a training structure, i.e. it has
public spectra and is far better documented than the reachable structures without spectra that hidden class-2
answers come from. Class-2 training queries are importance-weighted towards that population (`DOC_W`, `DOC_ALPHA`
in the training scripts; `scripts/pipeline_v5w.sh`), so the rankers rely less on documentation priors.

**DreaMS channel** (in progress; `casmi/dreams_emb.py`, `casmi/dreams_feats.py`): DreaMS embeddings (MIT weights)
of one library spectrum per structure and polarity, computed on Kaggle CPU shards (`kernels/casmi26-dreams-emb-*`);
embedding neighbours give a second analog channel and a frozen-embedding fingerprint head
(`scripts/train_dreams_fp.py`).

**PeakFormer** (`casmi/peakformer.py`, `scripts/train_pf.py`): a peak-token transformer that reads
each peak's exact m/z and neutral loss as multi-scale sinusoids (0.01–1000 Da) and predicts logits over a 7-block
RDKit structure fingerprint (`casmi/richfp.py`); a candidate scores f·z (its fingerprint's Bernoulli log-likelihood)
plus a cosine embedding score. Trained against same-mass negatives from pool v1 (training structures + COCONUT, no
PubChemLite; `scripts/prep_fp_data.py`, `scripts/prep_pf_data.py`) and in-batch negatives with merged-spectrum
augmentation (`kernels/casmi26-pf-cpu-*`, `kernels/casmi26-pf-train`, `kernels/casmi26-pf-train-b`); its scores are
stage-1 features (`pf_*`, `scripts/pipeline_v11.sh`; `pf_model_b.pt` in the submitted entry).

**PubChem proposal channel** (`casmi/pcchannel.py`, `scripts/build_pubchem_tier.py`; in progress): 15.8M documented
PubChem compounds outside the pool, mass-sorted; for a test molecule the 10 ppm window is scored by the PeakFormer
(+ a small documentation term) and the best proposals are inserted at fixed list positions when the library match
is weak (`PC_TIER`, `PC_SLOTS`, `PC_GATE` in `casmi/run_own.py`).

Also here: the novel-structure engine for molecules absent from PubChem (`casmi/novel.py`: analogue derivation with
modification-site localisation, conjugate assembly from hydrolysis building blocks) and LB probe kernels.

## Layout

- `casmi/` — library code (shipped to Kaggle as the `casmi26-code` dataset)
- `fwdshim/` — DGL / torch_scatter stand-ins for ms-pred inference
- `scripts/` — data building, simulation, training, evaluation, Kaggle helpers (`safe_submit.sh` refuses to submit
  a kernel whose output looks broken)
- `kernels/` — Kaggle notebooks / scripts with their metadata
- `docs/` — plan and results log, strategy report, reproduction manifest (`reproduce.md`), attribution
  (`ATTRIBUTION.md`)

## Third-party components

ms-pred (MIT), GLACIER MassSpecGym weights (ms-pred authors), DreaMS (MIT), RDKit (BSD-3), LightGBM (MIT), PyTorch (BSD),
LinSATNet (MIT), PubChem / COCONUT / PubChemLite / ChEBI / LIPID MAPS / LOTUS data under their terms. Licences and
citations of every third-party data source and model behind the pool and model datasets are in
[`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md); compliance notes are in `docs/plan.md`. The repository has no LICENSE file
yet (owner's decision, `docs/reproduce.md` G19).
