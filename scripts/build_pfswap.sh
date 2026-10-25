#!/bin/bash
# Generic PeakFormer swap on the v17 recipe (v5w: 31 leaves, alpha 1, lib10): scripts/build_v17b.sh with the model
# and tag as arguments. scripts/build_pfswap.sh <pf_model.pt> <tag> [fp_model.pt]
# 1. neutral gate on the 620-query class-2 COCONUT subset (as build_v17b.sh: pf_bern vs training decoys >= 0.70,
#    excess <= +0.087); also printed: PeakFormer b's reference .791 / +.044;
# 2. ret<tag> rows = ret17 with pf_* (+ within-formula pf columns) rescored by the new model (token tied); with a third
#    argument the fp_* columns are rescored too (FP model swap, as build_v17bb.sh);
# 3. pipeline T=<tag>; 4. models upload (pf_model_<tag>.pt) + kernel casmi26-own-s2<tag> (LIB_CAP 10).
set -e
PF=$1; TAG=$2; FP=$3
[ -n "$PF" ] && [ -n "$TAG" ] || { echo "usage: $0 <pf_model.pt> <tag> [fp_model.pt]"; exit 1; }
W=/home/user/work; R=/home/user/claude_code_tests; cd $R
export RICHFP_CACHE=$W/richfp_ret10.npz
O=$W/$TAG; mkdir -p $O
python3 scripts/add_fp_feats.py $W/fpb_local/sub.parquet $PF $O/sub_pf.parquet $W/pool4 > /dev/null
python3 scripts/eval_pf_neutral.py $O/sub_pf.parquet pf_bern,pf_cos | tee $O/gate.txt
python3 - $O/gate.txt <<'EOF'
import re, sys
line = [l for l in open(sys.argv[1]) if l.startswith('pf_bern')][0]
tr, ex = float(re.search(r'training decoys ([0-9.]+)', line).group(1)), float(re.search(r'excess \(prior\) ([-+0-9.]+)', line).group(1))
print('GATE pf_bern: vs training decoys', tr, 'excess', ex, '(PeakFormer b: .791 / +.044)')
sys.exit(0 if tr >= 0.70 and ex <= 0.087 else 'GATE FAILED')
EOF
for s in train val; do
  out=$W/ret${TAG}_${s}_fp.parquet; [ -f $out ] && continue
  for i in 0 1 2; do python3 scripts/add_fp_feats.py $W/ret17_${s}_fp.parquet $PF $out.part$i $W/pool4 $i 3 > $out.part$i.log 2>&1 & done; wait
  python3 -c "
import sys, pandas as pd
sys.path.insert(0, '$R')
from casmi.own import formula_rel, PF_FREL_SRC
D = pd.concat([pd.read_parquet(f'$out.part{i}') for i in range(3)], ignore_index=True)
D['formula'] = pd.read_parquet('$W/pool4/pool.parquet', columns=['formula']).formula.to_numpy(dtype=object)[D.pid.values]
formula_rel(D, PF_FREL_SRC)
D.drop(columns='formula').to_parquet('$out')"
  rm -f $out.part0 $out.part1 $out.part2
  if [ -n "$FP" ]; then
    for i in 0 1 2; do python3 scripts/add_fp_feats.py $out $FP $out.fp$i $W/pool4 $i 3 > $out.fp$i.log 2>&1 & done; wait
    python3 -c "
import pandas as pd
pd.concat([pd.read_parquet(f'$out.fp{i}') for i in range(3)], ignore_index=True).to_parquet('$out')"
    rm -f $out.fp0 $out.fp1 $out.fp2
  fi
done
T=$TAG ROWS=ret$TAG ALPHA=1 RANKER_PARAMS='{"num_threads": 4}' GL_THREADS=4 scripts/pipeline_v8d.sh
M=/home/user/ds/casmi26-models
cp $PF $M/pf_model_$TAG.pt
EXTRA=""
if [ -n "$FP" ]; then cp $FP $M/fp_model_$TAG.pt; EXTRA="OWN_FP=fp_model_$TAG.pt"; fi
cp $W/own_models/ret_ranker_$TAG.txt $W/own_models/ret_ranker_${TAG}_feats.txt $W/own_models/stage2_$TAG.txt $W/own_models/stage2_${TAG}_feats.txt $M/
kaggle datasets version -p $M -m "$TAG: PeakFormer $(basename $PF) + $TAG rankers" -q 2>&1 | grep -v Warning | tail -1
for i in $(seq 1 30); do n=$(kaggle datasets files bryantyu/casmi26-models --page-size 200 2>&1 | grep -c "_$TAG"); [ "$n" -ge 5 ] && break; sleep 30; done
sleep 120          # kernels pushed right after a dataset version have failed at startup (v13 v1, v17 v1)
python3 scripts/make_own_kernel.py casmi26-own-s2$TAG ret_ranker_$TAG stage2_$TAG OWN_PF=pf_model_$TAG.pt $EXTRA LIB_CAP=10 | tail -1
kaggle kernels push -p kernels/casmi26-own-s2$TAG 2>&1 | grep -v Warning | tail -1
echo PFSWAP_DONE $TAG
