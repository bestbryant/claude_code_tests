# Reproducing the eligible submission (casmi26-own-s2v17b, LB 0.339)

Status as of 2026-10-25. Host rule: every learned weight must be reproducible from public training code + permitted
data, including the code that generates the intermediates. Gap ids `G#` refer to section 3.

Paths: `R=/home/user/claude_code_tests` (repo), `W=/home/user/work`, `EXT=/home/user/ext`, `DS=/home/user/ds`
(local mirrors of the Kaggle datasets). md5 values are given as 8-character prefixes.

Reference environment (local): Python 3.13.16, numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0, RDKit 2026.03.3,
LightGBM 4.7.0, torch 2.14.1+cpu, x86-64 CPU with AVX-512 (pool row order depends on it, G12).

## 1. Rebuild, in order

### 1.0 Raw permitted inputs

```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p /home/user/data   # unzip: train.parquet, test.parquet, sample_submission.csv
# COCONUT 10-2026 (lite: row order of coconut_2026_10.parquet; full: coconut_prior.parquet). Full CSV sha256 ad4b1744...b46a
curl -O https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv_lite-10-2026.zip
curl -O https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv-10-2026.zip
# PubChemLite for Exposomics v3.3.0 (DOI 10.5281/zenodo.22953038), md5 3023e89b
curl -o $EXT/pcl/pcl.csv https://zenodo.org/api/records/22953038/files/PubChemLite_exposomics_20260925.csv/content
# PubChem FTP Compound/Extras CID-InChI-Key, CID-SID, CID-PMID, CID-Patent (live snapshot; the Oct 18 state is gone, G11)
python3 scripts/build_pubchem_tables.py $EXT/pubchem keys,sid,pmid,patent,agg       # -> key_pop.parquet (md5 ce9c3671), cid_ik.npy
# ChEBI rel 255 + LMSD, third-party standardised set (G1)
kaggle datasets download sergiomalv/casmi-chebi-lmsd-structures -p $EXT/chebi_lmsd --unzip   # bio_structures.parquet (md5 ba20c5ab)
# GLACIER: bryantyu/casmi26-glacier (ckpt/best.ckpt md5 d7d3e3ec, ms-pred @708148c2a8eb, wheels, fwdshim).
# Local copies used for training: $EXT/glacier_orig/best.ckpt, $EXT/mspred_src/ms-pred/src, $EXT/fwd_site.
```

### 1.1 Preprocessed inputs (no scripts exist; recipes verified against the local files)

```bash
# NO SCRIPT (G9): train_meta.parquet = train.parquet columns inchikey14, normalized_smiles, molecular_formula, ingest_lib,
#   adduct, instrument_type, precursor_mz, collision_energy_ev, num_peaks, ionization_mode; no filter/reorder (md5 82d7df60)
# NO SCRIPT (G8): $EXT/coconut/coconut_2026_10.parquet = LITE CSV in file order, columns identifier, canonical_smiles,
#   ik14 = standard_inchi_key[:14], exact_molecular_weight, np_likeness (738,794 rows; byte-identical rebuild, md5 8e2fcf7e)
# NO SCRIPT (G8): $EXT/coconut/coconut_prior.parquet from the FULL CSV, grouped by ik14: n_org/n_coll/n_doi/n_syn = max of
#   '|'-separated counts (0 if empty), has_cas/has_name = max(notna), ann = max(annotation_level), npl = max(np_likeness),
#   pathway = first(np_classifier_pathway), n_stereo = size; df.to_parquet (479,690 rows; byte-identical, md5 f8b83ce5)
# NO SCRIPT (G10): $EXT/pcl/pcl.parquet = read_csv(pcl.csv, usecols=[Identifier, FirstBlock, PubMed_Count, Patent_Count,
#   SMILES, MonoisotopicMass, AnnoTypeCount, BioPathway, FoodRelated]) (+ unused in_pool column; 469,602 rows)
```

### 1.2 Spectral-model training data

```bash
python3 scripts/build_spec_store.py /home/user/data/train.parquet $W/train_store 64
git show 26b2f28:scripts/build_pool.py > /tmp/build_pool_v1.py                 # pool v1 builder exists only in history (G7)
python3 /tmp/build_pool_v1.py $W/pool 4                                        # 726,198 rows; pool.parquet 707055a9, pool_fp.npy 6df2c180
# NO SCRIPT (G6): $W/val_keys.txt (seed 42: 250 enveda-np-examples + 750 COCONUT&train + 500 train-only) and
#   $W/rank_keys.txt (seed 7: 5,000 COCONUT&train + 3,000 train-only, val excluded); test-adduct keys, 120 < mass < 1300.
#   md5 e2c96647 / 0cf23d75. Output identical for pool v1, pool2, pool4.
python3 scripts/prep_fp_data.py $DS/casmi26-fp-data          # hard-codes $W/train_store, $W/pool, $W/{val,rank}_keys.txt
PROCS=4 python3 scripts/prep_pf_data.py $DS/casmi26-fp-data $W/pool $W/pf_data           # fpx.npz; needs ~3.6 GB free (G17)
python3 scripts/pf_negtrain.py $DS/casmi26-fp-data $W/pool $W/negtrain2/negtrain.npy     # md5 6d4923ae (byte-identical rerun)
# publish: bryantyu/casmi26-fp-data, casmi26-pf-data (fpx.npz), casmi26-pf-negtrain (v2)
```

### 1.3 Spectral models

```bash
# FP dual encoder (fp_model.pt, md5 ad9b7de6): local CPU run, killed after the "EPOCH 2" save = step 9,000 of 24,000 (G4)
python3 scripts/train_fp.py $DS/casmi26-fp-data $W/fp_v1.pt 8 3000 512 15 3   # expect val MRR .5118 / .5346 / .5461 at epochs 0/1/2
cp $W/fp_v1.pt $DS/casmi26-models/fp_model.pt
# casmi26-code, assembled by hand (G16): casmi/*.py (v38 omits hlprior.py) + scripts/train_fp.py -> casmi/train_fp_script.py,
#   scripts/train_pf.py -> casmi/train_pf_script.py; kaggle datasets version -p $DS/casmi26-code
# PeakFormer b (pf_model_b.pt, md5 560e194f): Kaggle T4, PF_HOURS=5 wall-clock stop at ~57.3k of 180k steps (G5)
kaggle kernels push -p kernels/casmi26-pf-train-b    # train_pf_script.py <fp-data> fpx.npz pf_model.pt 60 3000 512 63; PF_SEED=11
kaggle kernels output bryantyu/casmi26-pf-train-b -p $W/kout/pfb_gpu
```

### 1.4 Candidate pool v4 (bryantyu/casmi26-pool4); chain not scripted (G12)

```bash
python3 scripts/build_pool.py $W/pool2 4               # reads train_meta, coconut_2026_10, pcl (hard-coded); 1,107,747 rows
cp $EXT/coconut/coconut_prior.parquet $W/pool2/        # manual step
python3 scripts/add_pubchem_pop.py $EXT/pubchem/key_pop.parquet $W/pool2      # loads all 105.9M rows; OOM with ~12 GB free (G11)
python3 scripts/build_pool_bio.py $W/pool2 $EXT/chebi_lmsd/bio_structures.parquet $W/pool4 4   # +52,371 rows, in_bio
python3 scripts/add_pubchem_pop.py $EXT/pubchem/key_pop.parquet $W/pool4
# expect pool2: pool.parquet c433147c, pool_fp.npy 5abdd519; pool4: pool.parquet aa45930c, pool_fp.npy a077de63
kaggle datasets version -p $DS/casmi26-pool4 -m pool4  # ships pool.parquet, pool_fp.npy, coconut_prior.parquet only
python3 -m casmi.docw $W/pool4 $W/pool4/doc_w.parquet  # md5 971fd343 (byte-identical rerun); pipeline_v8d.sh does not build it
```

### 1.5 Stage-1 training rows

```bash
python3 scripts/build_analog_lib.py                    # $W/analog_lib.npz (N=3, reads $W/train_store)
python3 -c "from casmi.build import build_analog_lib; build_analog_lib('/home/user/data/train.parquet','/home/user/work/lib10.npz',per_struct=10)"
# ret4 rows; invocation unrecorded, recovered by seed search (G13). Delete stale *.parquet.parts dirs first.
CASMI_POOL=$W/pool4 VENDOR_PATH=/nonexistent python3 scripts/dump_ret_rows.py keys:$W/rank_keys.txt 8000 $W/ret4_train.parquet 12 4
CASMI_POOL=$W/pool4 VENDOR_PATH=/nonexistent python3 scripts/dump_ret_rows.py keys:$W/val_keys.txt 1500 $W/ret4_val.parquet 11 4
# expect 7,788 queries / 980,910 rows and 1,463 / 182,177
# fp_* columns: fp() of scripts/pipeline_v4.sh only (the rest of that script trains v4 models), per split s:
for i in 0 1 2; do python3 scripts/add_fp_feats.py $W/ret4_${s}.parquet $W/fp_v1.pt $W/ret4_${s}_fp.parquet.part$i $W/pool4 $i 3 & done; wait
#   then concat the 3 parts -> $W/ret4_${s}_fp.parquet
# ret17_${s}_fp.parquet has no builder (G14); every column build_v17b.sh keeps equals ret4_${s}_fp.parquet.
# Optional speed cache (identical outputs without it, G18):
PROCS=4 python3 scripts/cache_richfp.py $W/pool4 $W/richfp_ret10.npz $W/ret10_train_fp.parquet $W/ret10_val_fp.parquet
```

### 1.6 v17b rankers, GLACIER, upload, kernel

```bash
scripts/build_v17b.sh $W/kout/pfb_gpu/pf_model.pt
```

What it runs:
1. Gate: `add_fp_feats.py $W/fpb_local/sub.parquet <PF> $W/v17b/sub_pf.parquet $W/pool4` then `eval_pf_neutral.py`
   (pf_bern .791 / excess +.044). Input comes from `scripts/fpb_local.sh` on the deleted `ret14_val_fp.parquet` (G15).
2. ret17b rows: `add_fp_feats.py $W/ret17_${s}_fp.parquet <PF> <out>.part$i $W/pool4 $i 3` (i = 0..2,
   `RICHFP_CACHE=$W/richfp_ret10.npz`), concat, `casmi.own.formula_rel(D, PF_FREL_SRC)`.
3. `T=v17b ROWS=ret17b ALPHA=1 RANKER_PARAMS='{"num_threads": 4}' GL_THREADS=4 scripts/pipeline_v8d.sh`, with
   `DOC_W=$W/pool4/doc_w.parquet DOC_ALPHA=1`:
   - `train_ret_ranker.py $W/ret17b_train_fp.parquet $W/ret17b_val_fp.parquet $W/own_models/ret_ranker_v17b.txt $W/pool4 --drop-artifact`
   - `oof_stage1.py $W/ret17b_train_fp.parquet $W/own_models/ret_ranker_v17b_feats.txt $W/ret_v17b_train_oof.parquet --drop-artifact`
   - inline: `$W/ret17b_val_reach_c.parquet`
   - in `$W`: `build_fwd_jobs.py ret_v17b_train_oof.parquet col:s1_oof pool4 30 fwd_v17b/train 1 "--have=$W/fwd[_234]*/train.jobs_*.gl.json"`
     and `build_fwd_jobs.py ret17b_val_reach_c.parquet own_models/ret_ranker_v17b.txt pool4 30 fwd_v17b/val 1 "--have=..."`
   - `python3 -m casmi.fwdrun $W/fwd_v17b/{train,val}.jobs_0.json <same>.gl.json --src $EXT/mspred_src/ms-pred/src --ckpt $EXT/glacier_orig/best.ckpt --site $EXT/fwd_site --threads 4`
   - `GL_GLOB=<every $W/fwd[_234]*/train.jobs_*.gl.json> GL_GLOB_VAL=<same, val> train_stage2.py fit fwd_v17b/train own_models/ret_ranker_v17b_feats.txt own_models/stage2_v17b.txt fwd_v17b/val` (G2)
4. Copy `pf_model_b.pt`, `ret_ranker_v17b{,_feats}.txt`, `stage2_v17b{,_feats}.txt` to `$DS/casmi26-models`;
   `kaggle datasets version` (models v22).
5. `python3 scripts/make_own_kernel.py casmi26-own-s2v17b ret_ranker_v17b stage2_v17b OWN_PF=pf_model_b.pt LIB_CAP=10`;
   `kaggle kernels push -p kernels/casmi26-own-s2v17b`.

Expected: ret_ranker_v17b.txt md5 3a975604, stage2_v17b.txt md5 c8090321 (neither reproducible byte for byte, G3).

### 1.7 Kernel run (no internet, CPU, image sha256:81b1e7c4...)

`notebook.py` installs the RDKit 2026.3.3 wheel, copies casmi/ to `/kaggle/working/pkg/casmi` and runs
`python -m casmi.run_own {COMP}/test.parquet {COMP}/train.parquet {POOL} {MODELS} /kaggle/working/own_lists.json /kaggle/working/own 4`
with `OWN_RANKER=ret_ranker_v17b OWN_STAGE2=stage2_v17b OWN_PF=pf_model_b.pt LIB_CAP=10 GLACIER_DIR=<casmi26-glacier>
GL_THREADS=4 GL_BUDGET=14400 NUMBA_NUM_THREADS=1` (`OWN_FP` unset, so `fp_model.pt`). It builds `analog_lib.npz`
(per_struct 3) and `lib10.npz` (per_struct 10) from train.parquet and writes submission.csv (25 SMILES per molecule_id,
`C` if empty). Inputs mounted by the LB run (from version history; confirm on the Input tab): casmi26-code v38
(= repo 79a542d casmi/), casmi26-models v22, casmi26-pool4 v1, casmi26-glacier v1, rdkit wheel v1.

## 2. Artifacts

| artifact | used by | built by | inputs | status |
|---|---|---|---|---|
| kernels/casmi26-own-s2v17b (notebook.py, kernel-metadata.json) | submission.csv, LB 0.339 | make_own_kernel.py (build_v17b.sh step 5) | code, pool4, models, rdkit wheel, glacier, competition data; image pinned | complete; dataset versions unpinned (G16) |
| runtime analog_lib.npz, lib10.npz | Retrieval analog (an_*) and library (lib*) channels | casmi.build.build_analog_lib in run_own.main (per_struct 3 / LIB_CAP 10) | train.parquet | complete |
| competition test.parquet, train.parquet, sample_submission.csv | run_own, build_analog_lib, row order | host | – | third-party |
| casmi26-code v38 (casmi/*.py, *_script.py) | every kernel | hand copy of casmi/ + renamed scripts/train_{fp,pf}.py | repo 79a542d | partial (G16) |
| casmi26-models ret_ranker_v17b + feats (58 features; 3a975604) | run_own stage 1; stage-2 feature list | train_ret_ranker.py via pipeline_v8d.sh | ret17b_*_fp, doc_w, pool4 | partial: refit within 4.4e-14, not byte-identical (G3); upstream G4-G6, G13 |
| casmi26-models stage2_v17b + feats (67 features; c8090321) | run_own stage 2, final top 25 | train_stage2.py fit, GL_GLOB over all fwd[_234]* dirs | fwd_v17b top.parquet, GLACIER scores from 17 dirs, doc_w | partial (G2, G3) |
| casmi26-models fp_model.pt (= $W/fp_v1.pt; ad9b7de6) | FPScorer fp_* at inference and in ret4_fp rows | local CPU train_fp.py `8 3000 512 15 3`, stopped after epoch 2 | casmi26-fp-data | partial (G4) |
| casmi26-models pf_model_b.pt (= $W/kout/pfb_gpu/pf_model.pt; 560e194f) | PFEnsemble pf_* at inference and in ret17b rows | Kaggle kernel casmi26-pf-train-b | code, fp-data, pf-data, pf-negtrain v2 | partial (G5) |
| casmi26-pool4 pool.parquet (1,160,118 rows; aa45930c) | candidates, in_*/pcl_*/pc_*/in_bio features, dumps, doc_w | build_pool.py, add_pubchem_pop.py, build_pool_bio.py, add_pubchem_pop.py (manual chain) | pool2, bio_structures, key_pop | partial (G1, G11, G12) |
| casmi26-pool4 pool_fp.npy (a077de63) | analog Tanimoto, FPScorer molecule bits | props() in build_pool.py / build_pool_bio.py | pool SMILES | partial (G12) |
| casmi26-pool4 coconut_prior.parquet (479,690 rows; f8b83ce5) | 8 prior features (stage 1 and 2), docw | NO SCRIPT (inline Oct 17 22:14), manual cp into pool2 | COCONUT full CSV | broken (G8) |
| casmi26-glacier ckpt/best.ckpt (d7d3e3ec) | fwdrun gl_* | ms-pred authors (MassSpecGym GLACIER) | – | third-party |
| casmi26-glacier mspred/ (ms-pred @708148c2a8eb) | fwdrun | unmodified | – | third-party |
| casmi26-glacier wheels/*.whl.ice | fwdrun site dir | unmodified PyPI wheels | – | third-party |
| casmi26-glacier fwdshim/ | fwdrun (DGL / torch_scatter stand-ins) | repo fwdshim/ | – | complete |
| metric/rdkit-2026-3-3-wheel | main-pipeline RDKit | Kaggle | – | third-party |
| Kaggle image sha256:81b1e7c4... | lightgbm, torch, numba, numpy, pandas, pyarrow | Kaggle | – | third-party |
| /home/user/data/train_meta.parquet (82d7df60) | build_pool.py, train_ret_ranker.py, split keys, 6 other scripts | NO SCRIPT (10-column projection) | train.parquet | broken (G9) |
| $EXT/coconut/coconut_2026_10.parquet (738,794; 8e2fcf7e) | build_pool.py (v1, v2) | NO SCRIPT; lite CSV in file order | COCONUT lite CSV | broken (G8) |
| $EXT/pcl/pcl.parquet (469,602) | build_pool.py v2 (in_pcl, pcl_*) | NO SCRIPT (9-column read_csv) | PubChemLite 20260925 CSV | broken (G10) |
| $EXT/pubchem/key_pop.parquet + cid_ik.npy (ce9c3671) | add_pubchem_pop.py (pc_*) | build_pubchem_tables.py | PubChem FTP Extras, Oct 18 snapshot | partial (G11) |
| $EXT/chebi_lmsd/bio_structures.parquet (98,433; ba20c5ab) | build_pool_bio.py: in_bio, 52,371 pool rows | sergiomalv/casmi-chebi-lmsd-structures (no build code) | ChEBI r255, LMSD, prvsiyan filter | third-party; blocker (G1) |
| $W/train_store (top 64 peaks) | prep_fp_data, add_fp_feats, dump_ret_rows, build_analog_lib | build_spec_store.py | train.parquet | complete |
| pool v1 $W/pool (726,198; fp 6df2c180) | prep_fp_data (hard-coded), prep_pf_data, pf_negtrain | `git show 26b2f28:scripts/build_pool.py` | train_meta, coconut_2026_10 | partial (G7) |
| $W/val_keys.txt (1,500; e2c96647), rank_keys.txt (8,000; 0cf23d75) | fp-data split, dump_ret_rows | NO SCRIPT (inline, seeds 42 / 7) | pool key/mass/in_train/in_coco, train_meta | broken (G6) |
| casmi26-fp-data (spectra, structs, negs, negfp) | train_fp, train_pf, prep_pf_data, pf_negtrain | prep_fp_data.py | train_store, pool v1, key files | partial (G6, G7) |
| casmi26-pf-data fpx.npz (bdd0e338) | PF-b training | prep_pf_data.py (528d8ca); final save unlogged | fp-data, pool v1 | partial (G17) |
| casmi26-pf-negtrain v2 negtrain.npy (6d4923ae) | PF-b PF_NEG_TRAIN | pf_negtrain.py | fp-data, pool v1 | complete |
| pool2 $W/pool2 (1,107,747; = casmi26-pool2) | build_pool_bio.py | build_pool.py + cp prior + add_pubchem_pop.py | train_meta, coconut_2026_10, pcl, key_pop | partial (G12) |
| $W/pool4/doc_w.parquet (971fd343) | DOC_W in train_ret_ranker, oof_stage1, train_stage2 | python3 -m casmi.docw | pool4 | complete |
| $W/analog_lib.npz, $W/lib10.npz (training) | dump_ret_rows | build_analog_lib.py (N=3); lib10 command unrecorded | train.parquet / train_store | complete |
| $W/ret4_{train,val}.parquet | ret4_fp | dump_ret_rows.py, seeds 12 / 11 (unrecorded) | key files, pool4, coconut_prior, libs, train_store | partial (G13) |
| $W/ret4_{train,val}_fp.parquet | ret17 / ret17b rows | pipeline_v4.sh fp() | ret4, fp_v1.pt, pool4, train_store | complete step; weights G4 |
| $W/ret17_{train,val}_fp.parquet | build_v17b.sh input only | NO SCRIPT (inline merge with ret10pl pf columns, source deleted) | ret4_fp, PF-d | broken; content not needed (G14) |
| $W/richfp_ret10.npz | add_fp_feats RICHFP_CACHE (speed only) | cache_richfp.py (args unrecorded) | pool4 SMILES | partial (G18) |
| $W/ret17b_{train,val}_fp.parquet | train_ret_ranker, oof_stage1, reach_c | build_v17b.sh step 2 | ret17_fp (ret4_fp columns), pf_model_b, pool4 | partial: deterministic; G5, G14 |
| $W/ret_v17b_train_oof.parquet | build_fwd_jobs train top 30 | oof_stage1.py | ret17b_train_fp, feats, doc_w | partial (G3: deployed fold 0 not regenerable) |
| $W/ret17b_val_reach_c.parquet | build_fwd_jobs val | inline in pipeline_v8d.sh | ret17b_val_fp | complete |
| $W/fwd_v17b/{train,val}.top.parquet + jobs_0.json | fwdrun, train_stage2 | build_fwd_jobs.py with --have over all fwd dirs | OOF / ranker, pool4, train.parquet, earlier gl.json | partial (G2) |
| GLACIER scores fwd_v17b, fwd3 ... fwd_v17 (*.gl.json) | train_stage2 GL_GLOB | casmi.fwdrun, local | job JSONs, ckpt, ms-pred, fwd_site | partial (G2) |
| GLACIER scores fwd2 (76,066 of 101,144 scored train rows) | train_stage2 GL_GLOB | Kaggle casmi26-glw-* + local reruns, merged by hand; jobs from build_fwd_jobs@ea3a22f with third-party fuse.build_ice_input | ret2 / pool2 OOF (deleted) | broken (G2) |
| $W/v17b/gate.txt | go / no-go only | build_v17b.sh step 1 | fpb_local/sub.parquet (from deleted ret14_val_fp) | partial (G15) |
| $W/pool4/vendor.parquet, vend_w, vf_w, vfd*_w | not used by v17b | build_vendor_feats.py, casmi.docw purch | PubChem depositors | n/a; picked up by re-dumps (G13) |

## 3. Open gaps (confirmed) and fixes

| id | gap | fix |
|---|---|---|
| G1 | in_bio and 52,371 pool rows come from sergiomalv bio_structures.parquet: standardisation code not public; key set filtered by prvsiyan/coconut-casmi26-candidates (Kaggle licence CC-BY-NC-SA-4.0; ATTRIBUTION.md wrongly says CC BY 4.0). Not rebuildable from permitted data (18,236 of its keys are in our COCONUT). Blocker. | Add `scripts/build_bio_structures.py` from ChEBI r255 chebi.sdf.gz (2/3-star) + LMSD.sdf (RDKit 2026.03.3: largest organic fragment, uncharge, drop R/dummy, isotope and carbon-free entries, key = InChIKey14). Define in_bio as membership with no prvsiyan filter (or minus our COCONUT keys). Rebuild pool4, doc_w, ret rows; retrain stage 1, OOF, stage 2; resubmit and check LB. |
| G2 | stage2_v17b was fit on GLACIER scores merged from 17 historical fwd dirs (only 3,065 train rows scored in fwd_v17b; fwd2 76,066, fwd4 6,805, fwd3 6,262). fwd2/3/4 jobs were built with third-party fuse.build_ice_input from deleted ret2 inputs, scored on Kaggle and locally, merged by hand. Scores depend on batch composition (up to ~0.01-0.018) and crashes/timeouts give None. GL_GLOB grows with whatever fwd dirs exist. | Add a clean-room mode (`FWD_FRESH=1` in pipeline_v8d.sh or `scripts/rebuild_stage2_clean.sh`): build_fwd_jobs without `--have`, fwdrun on every top-30 pair (~155k pairs) with a fixed batch policy and per-item retry, `GL_GLOB= GL_GLOB_VAL=` for train_stage2. Retrain stage 2 (stage2_v17bc), gate on val MRR (.894; cls1 .8432 / cls2 .9212), push a new kernel and submit it as the eligible entry. Write a gl manifest (files + md5) at fit time. |
| G3 | LightGBM runs without `deterministic` or a forced histogram mode, so the mode is chosen by timing. Deployed OOF fold 0 cannot be regenerated (max delta 3.0; 7,030 of 128,982 top-30 pairs change). stage2_v17b refits are byte-identical only with row-wise mode. LightGBM version not recorded. | Add `deterministic=True`, a forced mode and explicit `num_threads` to train_ret_ranker.py, oof_stage1.py, train_stage2.py and casmi/docw.py (stage 2: `force_row_wise=True` reproduces the deployed model exactly; stage 1: `force_row_wise` changes the model, defaults match within 4.4e-14). Pin lightgbm==4.7.0 in a requirements file. Rebuild the v17b chain with pinned params (with G2) and record md5s. |
| G4 | fp_model.pt is the epoch-2 checkpoint (9,000 of a 24,000-step OneCycle schedule) of a local CPU run that died; no command recorded; train_fp.py has no epoch stop; kernels/casmi26-fp-train looks like the producer but was replaced by a stub. plan.md says "deployed (30k steps)". | Add `FP_STOP_EPOCH` to train_fp.py (break after epoch N, schedule still epochs*steps). Add `scripts/repro_fp_v1.sh`: `FP_STOP_EPOCH=3 python3 scripts/train_fp.py <casmi26-fp-data> fp_model.pt 8 3000 512 15 3`, checking step 200 loss 8.254 and epoch MRR .5118/.5346/.5461. Commit fp_v1.log. Fix plan.md and the casmi26-fp-train docstring. |
| G5 | pf_model_b.pt training stops on a 5 h wall clock and its cosine LR uses max(step fraction, time fraction), so step count (~57.3k, exact value not logged) and schedule depend on hardware; no CUDA determinism; kernel-metadata has no docker_image. | Add `PF_STEPS` to train_pf.py (LR fraction = step/total, stop at total, log final step, save `_step`/`_env` in the checkpoint) and `PF_DETERMINISTIC`. Pin `docker_image` gcr.io/kaggle-private-byod/python@sha256:9a6bcfda... in casmi26-pf-train-b. Rerun with `PF_STEPS=57300`; accept on epoch-19 val MRR (cos .807, f.z .805, sum .820). |
| G6 | val_keys.txt / rank_keys.txt (set the ranker split and the FP/PF split 1/2) have no producer; made by inline snippets on Oct 17. | Add `scripts/make_split_keys.py <pool.parquet> <train_meta.parquet> <out_dir>` with the recipe in 1.2 (seeds 42 and 7, no trailing newline), asserting md5 e2c96647 / 0cf23d75. Commit the two lists as data as well. |
| G7 | Pool v1 builder exists only as `git show 26b2f28:scripts/build_pool.py`; HEAD builds pool v2. prep_fp_data.py hard-codes $W/pool, train_store and the key files. Pool v1 is not published. | Add `scripts/build_pool_v1.py` (verbatim copy; assert pool_fp md5 6df2c180). Make prep_fp_data.py take `<pool_dir> <out_dir>` and the key-file paths as arguments. Publish pool v1 or add a check script for structs.key / negs.npy. |
| G8 | coconut_2026_10.parquet and coconut_prior.parquet have no script. Row order of the former matters (build_pool.py keeps the first row per ik14): the full-CSV order changes 3,050 stereo-stripped SMILES. | Add `scripts/build_coconut.py <lite_csv> <full_csv> <out_dir>` with the recipes in 1.1 (lite CSV for the parquet, full CSV for the prior); assert md5 8e2fcf7e / f8b83ce5. Record URLs and sha256. Let build_pool.py copy the prior into its output dir. |
| G9 | train_meta.parquet has no script. | Add `scripts/build_train_meta.py` (10-column projection, assert 2,539,608 rows). |
| G10 | pcl.parquet has no download or conversion script. | Add `scripts/fetch_pubchemlite.py`: Zenodo record 22953038 URL, size 200,142,750 B, md5 3023e89b, 9-column read_csv; drop the unused in_pool column. |
| G11 | PubChem FTP Extras were regenerated on Oct 24 (3 of 4 files), so a rebuild gives different pc_* counts; key_pop.parquet is unpublished; add_pubchem_pop.py loads all rows (OOM at ~12 GB). | Publish key_pop.parquet (md5 ce9c3671) as a CC0 Kaggle dataset dated 2026-10-18. Record file sizes, Last-Modified and md5 in a manifest in future runs of build_pubchem_tables.py. Stream key_pop in add_pubchem_pop.py. |
| G12 | Pool v2 -> v4 chain (1.4) is not in any script; build_pool.py paths are hard-coded; `sort_values('mass')` is an unstable quicksort whose tie order depends on numpy SIMD dispatch (862,621 of 1,107,747 rows move without AVX-512), and training rows key pool rows by pid. | Add `scripts/build_pool4.sh` running 1.4 with md5 checks; document the environment above. Keep the current sort for pool4; use `sort_values(['mass','key'], kind='stable')` for any new pool. |
| G13 | ret4 dump command (seeds 12 / 11, CASMI_POOL=pool4) unrecorded; pool4/vendor.parquet (Oct 21) is now auto-loaded and adds 22 v_* features; stale .parts dirs are silently reused. | Add `scripts/dump_ret4.sh` with the 1.5 commands, `VENDOR_PATH=/nonexistent` and `rm -rf` of the .parts dirs. |
| G14 | ret17_*_fp has no builder (inline merge; source ret10pl deleted); build_v17b.sh and build_pfswap.sh read it. | Point build_v17b.sh line 24 and build_pfswap.sh line 26 at `ret4_${s}_fp.parquet` (verified identical ret17b values and row order); reorder columns to PF_FEATS + PF_FREL. |
| G15 | build_v17b.sh step 1 needs $W/fpb_local/sub.parquet, built by fpb_local.sh from the deleted ret14_val_fp.parquet; a fresh run stops there. | Add `GATE=0` to skip the gate (it does not feed any weight). |
| G16 | casmi26-code is assembled by hand; kernel-metadata does not pin dataset versions (code has been re-versioned to v44, models to v23 since the run). | Add `scripts/push_code.sh` (clean-tree check, copy + renames, MANIFEST with commit and sha256). Record code v38 / models v22 / pool4 v1 / glacier v1 after confirming the Input tab. Pin versions in make_own_kernel.py output. |
| G17 | fpx.npz was saved by an unlogged rerun after the logged run hit ENOSPC (sampled rows verified equal). | Rerun prep_pf_data.py with >= 4 GB free, compare all arrays to the published file, commit the log; write via temp file + os.replace. |
| G18 | richfp_ret10.npz command unrecorded (pure cache; outputs identical without it); prep_syn_data.py defaults to it; no pool-order check. | Add a guarded cache_richfp.py line to build_v17b.sh (1.5); store pool row count/hash in the npz and check on load; drop the default path in prep_syn_data.py. |
| G19 | Licences: no LICENSE file in the repo; casmi26-code licence shows "unknown" on Kaggle (local metadata says Apache 2.0); casmi26-pool4 and casmi26-pool2 licence "other". | Add LICENSE (owner's choice); set the casmi26-code licence on Kaggle; set pool datasets to CC BY 4.0 with an ATTRIBUTION file (PubChemLite, ChEBI/LMSD, COCONUT, PubChem). |
| G20 | Stale docs: notebook says ret_ranker_v17b uses "full-library matching, alpha 0.5" (actual LIB_CAP 10, alpha 1) and claims full reproducibility; README says build_pool.py makes the COCONUT priors and implies the FP/PF pool includes PubChemLite; plan.md "deployed (30k steps)" (actual 9k). | Correct the three texts; link this file from README and plan.md. |

## 4. Third-party components and licences

| component | used for | licence / status |
|---|---|---|
| Competition data (train.parquet, test.parquet, sample_submission.csv) | library, pool, training rows, queries | competition rules (permitted) |
| COCONUT 10-2026 CSV (lite + full) | pool rows, in_coco, prior features | permitted by host; licence not yet checked on the COCONUT site (CC0 vs CC BY 4.0 claims) |
| PubChemLite for Exposomics v3.3.0 (Zenodo 10.5281/zenodo.22953038) | in_pcl, pcl_* | CC BY 4.0 (permitted) |
| PubChem FTP Compound/Extras (CID-InChI-Key, -SID, -PMID, -Patent) | pc_* counts | NCBI public domain (permitted) |
| ChEBI rel 255, LIPID MAPS LMSD | in_bio, bio pool rows | CC BY 4.0 (permitted) |
| sergiomalv/casmi-chebi-lmsd-structures | standardised ChEBI/LMSD set | CC BY 4.0; no build code |
| prvsiyan/coconut-casmi26-candidates (key filter inside the above) | defines which ChEBI/LMSD keys were removed | CC-BY-NC-SA-4.0 (G1) |
| GLACIER MassSpecGym checkpoint (ms-pred README link) | gl_* features | allowed by host; download URL and full sha256 not recorded |
| ms-pred @708148c2a8eb | GLACIER model code | MIT |
| rdkit 2025.3.6 wheels (casmi26-glacier) | GLACIER featurisation | BSD-3-Clause |
| pytorch_lightning 2.6.6, torchmetrics 1.9.0, lightning_utilities 0.15.3 | GLACIER loading | Apache-2.0 |
| einops 0.8.2, LinSATNet 0.1.3 | GLACIER model | MIT |
| RDKit 2026.3.3 wheel (metric/rdkit-2026-3-3-wheel) | main pipeline | BSD-3-Clause |
| Kaggle image: LightGBM / torch / numba / numpy / pandas / pyarrow | rankers, models, numerics | MIT / BSD-3 / BSD-2 / BSD-3 / BSD-3 / Apache-2.0 |
| Own code (casmi/, scripts/, fwdshim/, kernels/) | everything else | no LICENSE file yet (G19) |
