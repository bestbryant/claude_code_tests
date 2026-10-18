#!/usr/bin/env bash
# Pool v4 (+ ChEBI / LIPID MAPS, in_bio): FP features, artifact-free stage-1 ranker v4c, out-of-fold stage-1 scores,
# incremental GLACIER (query ids and spectra equal those of the pool-v2 dumps, so earlier scores are reused) and
# stage-2 v4.   scripts/pipeline_v4.sh   (expects ret4_train.parquet and ret4_val.parquet)
set -euo pipefail
W=/home/user/work; R=/home/user/claude_code_tests
cd $R
fp() {  # rows out pool shards
  local rows=$1 out=$2 pool=$3 n=$4
  for i in $(seq 0 $((n - 1))); do
    python3 scripts/add_fp_feats.py $rows $W/fp_v1.pt $out.part$i $pool $i $n > $out.part$i.log 2>&1 &
  done
  wait
  python3 - "$out" "$n" <<'PY'
import sys, pandas as pd
out, n = sys.argv[1], int(sys.argv[2])
pd.concat([pd.read_parquet(f'{out}.part{i}') for i in range(n)], ignore_index=True).to_parquet(out)
PY
  rm -f $out.part*
}
fp $W/ret4_train.parquet $W/ret4_train_fp.parquet $W/pool4 3
fp $W/ret4_val.parquet $W/ret4_val_fp.parquet $W/pool4 3
python3 scripts/train_ret_ranker.py $W/ret4_train_fp.parquet $W/ret4_val_fp.parquet $W/own_models/ret_ranker_v4c.txt $W/pool4 --drop-artifact
python3 scripts/oof_stage1.py $W/ret4_train_fp.parquet $W/own_models/ret_ranker_v4c_feats.txt $W/ret4_train_oof.parquet --drop-artifact
python3 - <<'PY'
import pandas as pd
D = pd.read_parquet('/home/user/work/ret4_val_fp.parquet')
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
t = D[D.label == 1].set_index('qid')
reach = (t.in_coco == 1) | (t.in_pcl == 1) | (t.in_bio == 1)
art = t.index[(t.cls == 1) & (t.lib_n == 0)]
D = D[((D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)) & ~D.qid.isin(art)]
D.to_parquet('/home/user/work/ret4_val_reach_c.parquet'); print('val queries', D.qid.nunique())
PY
cd $W; mkdir -p fwd4
python3 $R/scripts/build_fwd_jobs.py ret4_train_oof.parquet col:s1_oof pool4 30 fwd4/train 1 "--have=$W/fwd[23]/train.jobs_*.gl.json"
python3 $R/scripts/build_fwd_jobs.py ret4_val_reach_c.parquet own_models/ret_ranker_v4c.txt pool4 30 fwd4/val 1 "--have=$W/fwd[23]/val.jobs_*.gl.json"
cd $R
for f in train val; do
  python3 -m casmi.fwdrun $W/fwd4/$f.jobs_0.json $W/fwd4/$f.jobs_0.gl.json --src /home/user/ext/mspred_src/ms-pred/src \
    --ckpt /home/user/ext/glacier_orig/best.ckpt --site /home/user/ext/fwd_site --threads 4 > $W/fwd4/$f.fwdrun.log 2>&1
done
cd $W
GL_GLOB="$W/fwd2/train.jobs_*.gl.json,$W/fwd3/train.jobs_*.gl.json" GL_GLOB_VAL="$W/fwd2/val.jobs_*.gl.json,$W/fwd3/val.jobs_*.gl.json" \
  python3 $R/scripts/train_stage2.py fit fwd4/train own_models/ret_ranker_v4c_feats.txt own_models/stage2_v4.txt fwd4/val
echo V4_DONE
