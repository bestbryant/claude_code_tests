# Third-party data and models: licences and attribution

Status 2026-10-25. This file lists every third-party data source and pretrained model that the Kaggle datasets of this
entry (candidate pools, training arrays, model weights, GLACIER bundle) are built from or contain, with licence and
citation. Licence statements were re-checked on 2026-10-25 on the pages linked below, except the competition terms
(kaggle.com was not reachable from the build environment). Rebuild recipes and md5s: [`reproduce.md`](reproduce.md).

This file grants no licence. Our datasets do not change the licences of the sources; keep this attribution with any
copy. To ship it with a dataset, copy it into the dataset directory (as `ATTRIBUTION.md`) before
`kaggle datasets version`. The repository's own code (`casmi/`, `scripts/`, `fwdshim/`, `kernels/`) has no LICENSE
file yet. Choosing one is the owner's decision (reproduce.md G19).

## 1. Sources

### 1.1 Competition data (Enveda CASMI 2026)
- Files: `train.parquet` (2,539,608 spectra of 275,810 structures), `test.parquet`, `sample_submission.csv` from
  https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra. Spectra per `ingest_lib`: enveda-180
  1,153,785; pluskal_ms2 527,581; riken 347,171; gnps 220,849; massbank 101,727; mona 92,416; spectraverse 50,933;
  msdial 40,765; drug_plus 2,545; enveda-np-examples 1,184; masaryk 652.
- Used for: library and analog channels (built in the kernel from the mounted competition data), the training-structure
  rows of every pool (`in_train`, SMILES and keys), all training queries and spectra of the FP, PeakFormer and LightGBM
  models, and the derived arrays in casmi26-fp-data / casmi26-pf-data (top-64 peaks, fingerprints).
- Terms: the competition rules and data description on Kaggle. Competition data must not be redistributed (rule
  2.4.b, as cited in `.gitignore`). The spectra come from the libraries listed above, each with its own terms, some of
  them non-commercial. The host tolerates non-commercial terms that are inherited only from `train.parquet` (plan.md,
  "Compliance notes"). We never commit or redistribute the raw competition files.
- Cite: the competition page. For the enveda-180 part, also cite Enveda-180 (1.2).

### 1.2 Enveda-180
- Krettler C.A., Kind T., DeBloois E., Killian M., Prince J., Taylor J., Gadiya Y., Domingo-Fernández D., Allen A.,
  Colluru V., Healey D. "Enveda-180: generation of a large open multimodal MS/MS and ion mobility spectral library for
  drug-like small molecules." Zenodo, 2026-07-13. doi:10.5281/zenodo.21346580
- Licence: CC BY 4.0 (Zenodo record).
- Used for: the enveda-180 rows of `train.parquet` (182,941 structures), which reach every use in 1.1. Of the Zenodo
  release itself, we use only the 3,717 structures that are absent from `train.parquet`, and only in
  casmi26-zen-extra. The s2v17b submission does not use casmi26-zen-extra.

### 1.3 COCONUT (Collection of Open Natural Products) 2.0, release 10-2026
- Lite and full CSV from https://coconut.s3.uni-jena.de/prod/downloads/2026-10/. URLs and sha256 are in
  `scripts/build_coconut.py`.
- Licence: CC0 1.0. "Complete curated dataset is available for download under CC0 License"
  (https://coconut.naturalproducts.net/about). The COCONUT infrastructure code is MIT.
- Used for: pool rows and `in_coco` (476,357 rows of pool4); `coconut_prior.parquet` (479,690 keys: organism,
  collection, DOI and synonym counts, CAS / name flags, annotation level, NP-likeness, pathway); pool v1, which
  supplies the FP / PeakFormer training negatives; doc_w.
- Cite: Chandrasekhar V. et al. "COCONUT 2.0: a comprehensive overhaul and curation of the collection of open natural
  products database." Nucleic Acids Res. 53(D1):D634-D643 (2025). doi:10.1093/nar/gkae1063
  -- Sorokina M., Merseburger P., Rajan K., Yirik M.A., Steinbeck C. "COCONUT online: Collection of Open Natural
  Products database." J. Cheminform. 13:2 (2021). doi:10.1186/s13321-020-00478-9

### 1.4 PubChemLite for Exposomics 3.3.0
- Schymanski E., Elapavalore A., Kondic T., Thiessen P., Zhang J., Bolton E. "PubChemLite for Exposomics", version
  3.3.0 (2026-09-25). Zenodo. doi:10.5281/zenodo.22953038. File `PubChemLite_exposomics_20260925.csv`,
  200,142,750 B, md5 3023e89b (`scripts/fetch_pubchemlite.py`).
- Licence: CC BY 4.0 (Zenodo record). It is derived from PubChem (1.5).
- Used for: pool rows and `in_pcl` (467,093 rows of pool2 / pool4; 381,549 of them come from PubChemLite alone), and the
  `pcl_*` annotation counts (PubMed, patent, annotation types, BioPathway, FoodRelated). Pool v1 does not contain
  PubChemLite, so neither do the FP / PeakFormer training data.
- Cite: the dataset above, and Schymanski E.L., Kondić T., Neumann S., Thiessen P.A., Zhang J., Bolton E.E.
  "Empowering large chemical knowledge bases for exposomics: PubChemLite meets MetFrag." J. Cheminform. 13:19 (2021).
  doi:10.1186/s13321-021-00489-0

### 1.5 PubChem
- FTP `Compound/Extras` CID-InChI-Key, CID-SID, CID-PMID and CID-Patent (https://ftp.ncbi.nlm.nih.gov/pubchem/),
  snapshot of 2026-10-18. `scripts/build_pubchem_tables.py` builds `key_pop.parquet` (md5 ce9c3671) from them.
- Licence: NCBI data policy (https://www.ncbi.nlm.nih.gov/home/about/policies/): "NCBI itself places no restrictions
  on the use or distribution of the data contained therein. However, some submitters of the original data (or the
  country of origin of such data) may claim patent, copyright, or other intellectual property rights in all or a
  portion of the data".
- Used for: `pc_sid`, `pc_pmid`, `pc_patent` and `pc_ncid` of pool2 / pool4. These are aggregate counts per InChIKey
  first block; pool4 contains no structures taken directly from PubChem. PubChem-derived datasets that s2v17b does not
  use: casmi26-pubchem-keys, casmi26-pc-tier, casmi26-pc-full-pop, casmi26-pc-enam, casmi26-pc-purch and
  casmi26-vendor.
- Cite: Kim S. et al. "PubChem 2025 update." Nucleic Acids Res. 53(D1):D1516-D1525 (2025). doi:10.1093/nar/gkae1059

### 1.6 ChEBI, release 255
- `chebi.sdf.gz` (release of 2026-09-09; 2- and 3-star entries) from https://ftp.ebi.ac.uk/pub/databases/chebi/.
- Licence: CC BY 4.0. "The data on this website is available under the Creative Commons License (CC BY 4.0)"
  (https://www.ebi.ac.uk/chebi/aboutChebiForward.do).
- Used for: `in_bio` (89,054 rows of pool4) and the ChEBI share of the 52,371 rows that only ChEBI / LMSD contribute.
  Pool4 gets these through the third-party set in 1.8. The G1 replacement (`scripts/build_bio_structures.py`, pool
  v18c) reads the raw release instead (md5 4adcf594).
- Cite: Hastings J. et al. "ChEBI in 2016: Improved services and an expanding collection of metabolites." Nucleic
  Acids Res. 44(D1):D1214-D1219 (2016). doi:10.1093/nar/gkv1031

### 1.7 LIPID MAPS Structure Database (LMSD)
- `LMSD.sdf.zip` from https://www.lipidmaps.org/databases/lmsd/download. Upstream does not version it: the 1.8 set
  used the download of 2026-10-03, and our G1 rebuild uses the download of 2026-10-25 (md5 505ec732).
- Licence: CC BY 4.0. "All LIPID MAPS® databases are licensed under a Creative Commons Attribution 4.0
  International License" (https://www.lipidmaps.org/terms-of-use).
- Used for: the same purposes as ChEBI (1.6).
- Cite: Sud M. et al. "LMSD: LIPID MAPS structure database." Nucleic Acids Res. 35:D527-D532 (2007).
  doi:10.1093/nar/gkl838 -- Conroy M.J. et al. "LIPID MAPS: update to databases and tools for the lipidomics
  community." Nucleic Acids Res. 52(D1):D1677-D1682 (2024). doi:10.1093/nar/gkad896

### 1.8 sergiomalv/casmi-chebi-lmsd-structures (pool4 only)
- `bio_structures.parquet` (98,433 rows, md5 ba20c5ab), from
  https://www.kaggle.com/datasets/sergiomalv/casmi-chebi-lmsd-structures. It holds ChEBI r255 and LMSD structures,
  standardised with RDKit 2026.03.3. Keys present in prvsiyan/coconut-casmi26-candidates were removed.
- Licence: CC BY 4.0 (Kaggle card), but two problems remain (G1). The removed-key list comes from
  prvsiyan/coconut-casmi26-candidates, whose Kaggle licence is CC-BY-NC-SA-4.0; the set's own ATTRIBUTION.md calls it
  CC BY 4.0. The set also publishes no build code. Pool v18c drops this set and uses `scripts/build_bio_structures.py`
  on the raw releases (1.6, 1.7).
- Cite: the Kaggle dataset, together with ChEBI and LMSD.

### 1.9 GLACIER checkpoint, ms-pred and MassSpecGym
- ms-pred at commit 708148c2a8eb (https://github.com/coleygroup/ms-pred), unmodified. Licence: MIT, "Copyright (c) 2023
  Samuel Goldman", shipped as `mspred/LICENSE-ms-pred` in casmi26-glacier.
- GLACIER checkpoint trained on MassSpecGym, published by the ms-pred authors (README link "GLACIER"). We use it
  unmodified as `ckpt/best.ckpt` (md5 d7d3e3ec, sha256 5a47cecca707d3ab...). It comes with no separate licence file:
  the ms-pred README offers it as open-source MassSpecGym-trained weights. The hosts explicitly allow
  MassSpecGym-trained GLACIER (plan.md, "Rules").
- MassSpecGym is the checkpoint's training data; we do not use it directly. Licence: MIT (Hugging Face card
  `roman-bushuiev/MassSpecGym` and the paper's data statement).
- Used for: the `gl_*` stage-2 features, both at training time (fwd_* GLACIER scores behind stage2_v17b) and in the
  kernel.
- Cite: Wang R.-X., Wang R., Coley C.W. "GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection Problem."
  arXiv:2606.29161 (2026) -- Bushuiev R. et al. "MassSpecGym: A benchmark for the discovery and identification of
  molecules." NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2410.23326 -- Goldman S., Li J., Coley C.W.
  "Generating molecular fragmentation graphs with autoregressive neural networks." Anal. Chem. 96:3419-3428 (2024).
  The ms-pred README asks users to cite the works behind the parts they use.

### 1.10 Software shipped in or used to build the datasets
- RDKit 2026.03.3: the main pipeline, pool SMILES / keys / fingerprints (the metric/rdkit-2026-3-3-wheel dataset;
  Kaggle shows its licence as "unknown"). Licence: BSD-3-Clause. Cite: "RDKit: Open-source cheminformatics",
  https://www.rdkit.org, doi:10.5281/zenodo.591637
- Wheels in casmi26-glacier, unmodified from PyPI: rdkit 2025.3.6 (BSD-3-Clause); pytorch_lightning 2.6.6,
  torchmetrics 1.9.0 and lightning_utilities 0.15.3 (Apache-2.0); einops 0.8.2 and LinSATNet 0.1.3 (MIT).
- Not redistributed (Kaggle image / local environment): LightGBM 4.7.0 (MIT), PyTorch 2.14.1 (BSD-3-Clause), numpy and
  pandas (BSD-3-Clause), pyarrow (Apache-2.0), numba (BSD-2-Clause).

## 2. Contents of each dataset

| Kaggle dataset | files | built by | third-party content | Kaggle licence now |
|---|---|---|---|---|
| bryantyu/casmi26-pool4 | pool.parquet (1,160,118 rows), pool_fp.npy, coconut_prior.parquet | `scripts/build_pool4.sh` | competition structures (275,810 `in_train`), COCONUT (476,357 `in_coco` + prior), PubChemLite (467,093 `in_pcl` + `pcl_*`), PubChem counts (`pc_*`), ChEBI / LMSD through 1.8 (89,054 `in_bio`, 52,371 rows from ChEBI / LMSD alone) | other |
| bryantyu/casmi26-pool2 | the same files, 1,107,747 rows | `scripts/build_pool.py`, then the prior and `add_pubchem_pop.py` | as pool4, without ChEBI / LMSD | other |
| bryantyu/casmi26-fp-data | spectra.npz, structs.npz, negs.npy, negfp.npy | `scripts/prep_fp_data.py` | competition spectra (top 64 peaks) and structures; negatives from pool v1 (competition structures + COCONUT; no PubChemLite, ChEBI or LMSD) | CC-BY-4.0 |
| bryantyu/casmi26-pf-data | fpx.npz | `scripts/prep_pf_data.py` | fingerprints of the fp-data structures and the pool v1 negatives | CC-BY-4.0 |
| bryantyu/casmi26-pf-negtrain | negtrain.npy | `scripts/pf_negtrain.py` | flags over the pool v1 negatives | CC0-1.0 |
| bryantyu/casmi26-models | fp_model.pt, pf_model_b.pt, ret_ranker_v17b, stage2_v17b (+ `_feats.txt`), other variants | `train_fp.py`, `train_pf.py`, `train_ret_ranker.py`, `train_stage2.py` | weights we trained on the data above: FP / PeakFormer on fp-data and pf-data; rankers on pool4 features and GLACIER scores. No third-party weights. | Apache 2.0 |
| bryantyu/casmi26-glacier | mspred/, ckpt/best.ckpt, wheels/, fwdshim/, NOTICE.txt | copied unmodified (fwdshim/ is ours) | ms-pred, GLACIER checkpoint, wheels (1.9, 1.10) | other |
| bryantyu/casmi26-code | casmi/*.py, train_fp_script.py, train_pf_script.py | hand copy (`scripts/push_code.sh`, G16) | none | Apache 2.0 in the local metadata, "unknown" on Kaggle |

Other datasets published from this repository and not used by s2v17b: casmi26-zen-extra (Enveda-180 Zenodo spectra,
1.2); casmi26-dreams (DreaMS embeddings; DreaMS weights are MIT); the PubChem-derived sets listed in 1.5;
casmi26-pc-calib (competition spectra, 1.1).

## 3. Licence settings still open (owner, G19)
- casmi26-pool4 / casmi26-pool2: proposed change from "other" to CC BY 4.0, with this file added to the dataset. The
  sources allow it: COCONUT is CC0, the PubChem counts are under the NCBI policy, and PubChemLite / ChEBI / LMSD are
  CC BY 4.0 with the attribution in section 1. Two caveats remain. The competition terms still apply to the
  `in_train` structure rows. Pool4's ChEBI / LMSD rows carry the G1 problem (1.8), which pool v18c removes.
- casmi26-code: the Kaggle licence should match the repository LICENSE once the owner has chosen one.
- casmi26-models (Apache 2.0), casmi26-fp-data / casmi26-pf-data (CC BY 4.0): these are our weights and arrays, but
  the competition terms (1.1) apply to everything derived from `train.parquet`. That covers the spectra arrays in
  fp-data (top-64 peaks of the training spectra) and every model trained on them. Check rule 2.4.b before making
  these datasets public.
- casmi26-glacier: keep "other". The content is mixed (MIT, Apache-2.0, BSD-3-Clause, and the ms-pred authors'
  checkpoint), and NOTICE.txt lists each part.
