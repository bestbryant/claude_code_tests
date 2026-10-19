#!/usr/bin/env bash
# Pool v4 with documentation-shift weights (casmi/docw.py) on simulated class-2 queries: stage-1 ranker v5w,
# out-of-fold stage-1 scores, incremental GLACIER (earlier fwd2-4 scores reused), stage-2 v5w.
# scripts/pipeline_v5w.sh [alpha]     (expects ret4_train_fp.parquet, ret4_val_fp.parquet from pipeline_v4.sh)
set -euo pipefail
A=${1:-1}; T=v5w$( [ "$A" = 1 ] && echo "" || echo "_a$A" )
W=/home/user/work; R=/home/user/claude_code_tests
export DOC_W=$W/pool4/doc_w.parquet DOC_ALPHA=$A
cd $R
[ -f $DOC_W ] || python3 -m casmi.docw $W/pool4 $DOC_W
python3 scripts/train_ret_ranker.py $W/ret4_train_fp.parquet $W/ret4_val_fp.parquet $W/own_models/ret_ranker_$T.txt $W/pool4 --drop-artifact
python3 scripts/oof_stage1.py $W/ret4_train_fp.parquet $W/own_models/ret_ranker_${T}_feats.txt $W/ret_${T}_train_oof.parquet --drop-artifact
cd $W; mkdir -p fwd_$T
python3 $R/scripts/build_fwd_jobs.py ret_${T}_train_oof.parquet col:s1_oof pool4 30 fwd_$T/train 1 "--have=$W/fwd[234]/train.jobs_*.gl.json"
python3 $R/scripts/build_fwd_jobs.py ret4_val_reach_c.parquet own_models/ret_ranker_$T.txt pool4 30 fwd_$T/val 1 "--have=$W/fwd[234]/val.jobs_*.gl.json"
cd $R
for f in train val; do
  python3 -m casmi.fwdrun $W/fwd_$T/$f.jobs_0.json $W/fwd_$T/$f.jobs_0.gl.json --src /home/user/ext/mspred_src/ms-pred/src \
    --ckpt /home/user/ext/glacier_orig/best.ckpt --site /home/user/ext/fwd_site --threads 4 > $W/fwd_$T/$f.fwdrun.log 2>&1
done
cd $W
GL_GLOB="$W/fwd2/train.jobs_*.gl.json,$W/fwd3/train.jobs_*.gl.json,$W/fwd4/train.jobs_*.gl.json" \
GL_GLOB_VAL="$W/fwd2/val.jobs_*.gl.json,$W/fwd3/val.jobs_*.gl.json,$W/fwd4/val.jobs_*.gl.json" \
  python3 $R/scripts/train_stage2.py fit fwd_$T/train own_models/ret_ranker_${T}_feats.txt own_models/stage2_$T.txt fwd_$T/val
echo ${T}_DONE
