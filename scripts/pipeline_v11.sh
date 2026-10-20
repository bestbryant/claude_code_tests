#!/usr/bin/env bash
# v11 = v10 (full-library matching, doc-shift alpha 0.5) + PeakFormer features (pf_*, casmi/peakformer.py).
# PF=<comma-separated checkpoints> [T=v11] scripts/pipeline_v11.sh
# 1. pf_* columns on the ret10 rows (fingerprints from work/richfp_ret10.npz) -> ret10pf_{train,val}_fp.parquet
# 2. stage-1 5-fold CV gate (a0.5 vs a0.5+pf vs a0.5+pf-fp)
# 3. GATE=1: ranker, out-of-fold scores, incremental GLACIER and stage 2 via pipeline_v8d.sh (ROWS=ret10pf)
set -euo pipefail
T=${T:-v11}; W=/home/user/work; R=/home/user/claude_code_tests
cd $R
export RICHFP_CACHE=$W/richfp_ret10.npz
for s in train val; do
  out=$W/ret10pf_${s}_fp.parquet
  [ -f $out ] && continue
  for i in 0 1 2; do
    python3 scripts/add_fp_feats.py $W/ret10_${s}_fp.parquet "$PF" $out.part$i $W/pool4 $i 3 > $out.part$i.log 2>&1 &
  done
  wait
  python3 -c "
import sys, pandas as pd
pd.concat([pd.read_parquet(f'$out.part{i}') for i in range(3)], ignore_index=True).to_parquet('$out')"
  rm -f $out.part?
done
python3 scripts/eval_pf.py $W/ret10pf_val_fp.parquet
CV_ROWS=ret10pf python3 scripts/cv_stage1.py a0.5 a0.5+pf a0.5+pf-fp
[ "${GATE:-0}" = "1" ] || exit 0
T=$T ROWS=ret10pf bash scripts/pipeline_v8d.sh
