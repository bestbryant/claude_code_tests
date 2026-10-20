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
  N=${NSH:-2}          # each worker holds the rows, the fingerprint cache and every checkpoint (~5 GB)
  for i in $(seq 0 $((N - 1))); do
    python3 scripts/add_fp_feats.py $W/ret10_${s}_fp.parquet "$PF" $out.part$i $W/pool4 $i $N > $out.part$i.log 2>&1 &
  done
  wait
  python3 -c "
import sys, pandas as pd
sys.path.insert(0, '$R')
from casmi.own import formula_rel, PF_FREL_SRC
D = pd.concat([pd.read_parquet(f'$out.part{i}') for i in range(${NSH:-2})], ignore_index=True)
D['formula'] = pd.read_parquet('$W/pool4/pool.parquet', columns=['formula']).formula.to_numpy(dtype=object)[D.pid.values]
formula_rel(D, PF_FREL_SRC)
D.drop(columns='formula').to_parquet('$out')"
  rm -f $out.part*
done
python3 scripts/eval_pf.py $W/ret10pf_val_fp.parquet
CV_ROWS=ret10pf python3 scripts/cv_stage1.py ${CV_CONFIGS:-a0.5 a0.5+pf a0.5+pf+frel a0.5+pf-fp}
[ "${GATE:-0}" = "1" ] || exit 0
T=$T ROWS=ret10pf ALPHA=${ALPHA:-1} bash scripts/pipeline_v8d.sh
