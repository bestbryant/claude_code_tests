#!/usr/bin/env bash
# v7w = v5w + within-formula relative features (casmi.own.formula_rel; rows from scripts/add_formula_rel.py).
# Documentation-shift weighted stage 1, out-of-fold scores, incremental GLACIER (fwd2-4 and fwd_v5w scores reused), stage 2.
# scripts/pipeline_v7w.sh [alpha]     (expects ret7_{train,val}_fp.parquet from scripts/add_formula_rel.py)
set -euo pipefail
A=${1:-1}; T=v7w; ROWS=ret7
W=/home/user/work; R=/home/user/claude_code_tests
export DOC_W=$W/pool4/doc_w.parquet DOC_ALPHA=$A
cd $R
[ -f $DOC_W ] || python3 -m casmi.docw $W/pool4 $DOC_W
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
GL_GLOB="$W/fwd2/train.jobs_*.gl.json,$W/fwd3/train.jobs_*.gl.json,$W/fwd4/train.jobs_*.gl.json,$W/fwd_v5w/train.jobs_*.gl.json" \
GL_GLOB_VAL="$W/fwd2/val.jobs_*.gl.json,$W/fwd3/val.jobs_*.gl.json,$W/fwd4/val.jobs_*.gl.json,$W/fwd_v5w/val.jobs_*.gl.json" \
  python3 $R/scripts/train_stage2.py fit fwd_$T/train own_models/ret_ranker_${T}_feats.txt own_models/stage2_$T.txt fwd_$T/val
echo ${T}_DONE
