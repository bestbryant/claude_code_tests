#!/usr/bin/env bash
# v8d = DreaMS analog channel (casmi/dreams_feats.py) + documentation-shift weights (alpha 0.5) + 63-leaf stage-1 trees.
# [T=name RANKER_PARAMS='{}'] scripts/pipeline_v8d.sh      (expects ret4_{train,val}_fp.parquet, dreams_qemb.npz and work/dreams_lib/dreams_lib_*.npz)
set -euo pipefail
T=${T:-v8d}; ROWS=${ROWS:-ret8}
W=/home/user/work; R=/home/user/claude_code_tests
export DOC_W=$W/pool4/doc_w.parquet DOC_ALPHA=0.5 RANKER_PARAMS=${RANKER_PARAMS:-'{"num_leaves": 63, "min_data_in_leaf": 50}'}
cd $R
for s in train val; do
  [ -f $W/${ROWS}_${s}_fp.parquet ] || python3 scripts/add_dreams_feats.py $W/ret4_${s}_fp.parquet $W/${ROWS}_${s}_fp.parquet \
      $W/pool4 /home/user/ext/dreams/DreaMS_embedding_model_torchscript.pt $W/dreams_qemb.npz $W/dreams_lib/dreams_lib_*.npz
done
[ -f $W/own_models/ret_ranker_$T.txt ] || python3 scripts/train_ret_ranker.py $W/${ROWS}_train_fp.parquet $W/${ROWS}_val_fp.parquet $W/own_models/ret_ranker_$T.txt $W/pool4 --drop-artifact
[ -f $W/ret_${T}_train_oof.parquet ] || python3 scripts/oof_stage1.py $W/${ROWS}_train_fp.parquet $W/own_models/ret_ranker_${T}_feats.txt $W/ret_${T}_train_oof.parquet --drop-artifact
cd $W; mkdir -p fwd_$T
[ -f ${ROWS}_val_reach_c.parquet ] || python3 - $ROWS <<'PY'
import sys, pandas as pd
r = sys.argv[1]
D = pd.read_parquet(f'/home/user/work/{r}_val_fp.parquet')
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
t = D[D.label == 1].set_index('qid')
reach = (t.in_coco == 1) | (t.in_pcl == 1) | (t.in_bio == 1)
art = t.index[(t.cls == 1) & (t.lib_n == 0)]
D = D[((D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)) & ~D.qid.isin(art)]
D.to_parquet(f'/home/user/work/{r}_val_reach_c.parquet'); print('val queries', D.qid.nunique())
PY
python3 $R/scripts/build_fwd_jobs.py ret_${T}_train_oof.parquet col:s1_oof pool4 30 fwd_$T/train 1 "--have=$W/fwd[_234]*/train.jobs_*.gl.json"
python3 $R/scripts/build_fwd_jobs.py ${ROWS}_val_reach_c.parquet own_models/ret_ranker_$T.txt pool4 30 fwd_$T/val 1 "--have=$W/fwd[_234]*/val.jobs_*.gl.json"
cd $R
for f in train val; do
  python3 -m casmi.fwdrun $W/fwd_$T/$f.jobs_0.json $W/fwd_$T/$f.jobs_0.gl.json --src /home/user/ext/mspred_src/ms-pred/src \
    --ckpt /home/user/ext/glacier_orig/best.ckpt --site /home/user/ext/fwd_site --threads 4 > $W/fwd_$T/$f.fwdrun.log 2>&1
done
cd $W
G=""; GV=""
for d in fwd2 fwd3 fwd4 fwd_v5w fwd_v7w fwd_v5w_a0.5 fwd_v8d fwd_v8e; do G="$G,$W/$d/train.jobs_*.gl.json"; GV="$GV,$W/$d/val.jobs_*.gl.json"; done
GL_GLOB="${G#,}" GL_GLOB_VAL="${GV#,}" \
  python3 $R/scripts/train_stage2.py fit fwd_$T/train own_models/ret_ranker_${T}_feats.txt own_models/stage2_$T.txt fwd_$T/val
echo ${T}_DONE
