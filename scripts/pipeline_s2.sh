#!/usr/bin/env bash
# Stage-2 v2 on stage-1 v2c (artifact-free ranker): incremental GLACIER for the changed top-30 lists, then fit.
#   scripts/pipeline_s2.sh   (expects fwd2/*.gl.json complete, ret2_train_oof_c.parquet, ret2_val_reach_c.parquet)
set -euo pipefail
W=/home/user/work; R=/home/user/claude_code_tests
cd $W; mkdir -p fwd3
python3 $R/scripts/build_fwd_jobs.py ret2_train_oof_c.parquet col:s1_oof pool2 30 fwd3/train 1 "--have=$W/fwd2/train.jobs_*.gl.json"
python3 $R/scripts/build_fwd_jobs.py ret2_val_reach_c.parquet own_models/ret_ranker_v2c.txt pool2 30 fwd3/val 1 "--have=$W/fwd2/val.jobs_*.gl.json"
cd $R
for f in train val; do
  python3 -m casmi.fwdrun $W/fwd3/$f.jobs_0.json $W/fwd3/$f.jobs_0.gl.json --src /home/user/ext/mspred_src/ms-pred/src \
    --ckpt /home/user/ext/glacier_orig/best.ckpt --site /home/user/ext/fwd_site --threads 4 > $W/fwd3/$f.fwdrun.log 2>&1
done
cd $W
GL_GLOB="$W/fwd2/train.jobs_*.gl.json" GL_GLOB_VAL="$W/fwd2/val.jobs_*.gl.json" \
  python3 $R/scripts/train_stage2.py fit fwd3/train own_models/ret_ranker_v2c_feats.txt own_models/stage2_v2.txt fwd3/val
echo S2_DONE
