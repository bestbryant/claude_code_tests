# CASMI 2026 (Enveda) — molecule identification from MS/MS

Code for the Kaggle competition [enveda-CASMI26-molecule-id-mass-spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra):
rank up to 25 candidate structures per test molecule from its timsTOF MS/MS spectra (metric MRR@25 on the
tautomer-canonical InChIKey first block). Working notes, results and the submission log are in
[`docs/plan.md`](docs/plan.md); a short illustrated overview is in [`docs/strategy_report.pdf`](docs/strategy_report.pdf).

Every learned component is trained here from `train.parquet` and permitted public data, or is a weight the
competition explicitly allows (MassSpecGym-trained GLACIER from ms-pred). Competition data are never committed.

## Pipeline (kernels `kernels/casmi26-own-s2v4`, `-s2v5w`, `-s2v5w-a05`)

1. **Candidate pool** (`scripts/build_pool.py`): training structures + COCONUT + PubChemLite, with documentation
   priors (COCONUT organism/DOI counts, PubChemLite annotations, PubChem SID / PMID / patent counts built from the
   PubChem FTP by `scripts/build_pubchem_tables.py`). Candidates are taken in a 10 ppm window of the neutral mass.
2. **Stage-1 features** (`casmi/retrieval.py`, `casmi/own.py`): weighted spectral-entropy library similarity,
   modified-cosine spectral analogues propagated by Tanimoto similarity, explained intensity of 1–2-bond fragments,
   a spectrum↔fingerprint dual encoder (`casmi/fpmodel.py`, trained by `scripts/train_fp.py`), and the priors.
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

Also here: the novel-structure engine for molecules absent from PubChem (`casmi/novel.py`: analogue derivation with
modification-site localisation, conjugate assembly from hydrolysis building blocks) and LB probe kernels.

## Layout

- `casmi/` — library code (shipped to Kaggle as the `casmi26-code` dataset)
- `fwdshim/` — DGL / torch_scatter stand-ins for ms-pred inference
- `scripts/` — data building, simulation, training, evaluation, Kaggle helpers (`safe_submit.sh` refuses to submit
  a kernel whose output looks broken)
- `kernels/` — Kaggle notebooks / scripts with their metadata
- `docs/` — plan and results log, strategy report

## Third-party components

ms-pred (MIT), GLACIER MassSpecGym weights (ms-pred authors), DreaMS (MIT), RDKit (BSD-3), LightGBM (MIT), PyTorch (BSD),
LinSATNet (MIT), PubChem / COCONUT / PubChemLite / ChEBI / LIPID MAPS / LOTUS data under their terms. See `docs/plan.md` for the compliance notes.
