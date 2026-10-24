#!/bin/bash
# v17bb = v17b with the debiased FP model b (GPU, training-structure negatives, instrument-free; own_models/fp_model_b.pt)
# replacing the deployed FP model -- the second single change after v17b (design panel wf_f100d853-7f9).
# scripts/build_v17bb.sh   (needs ret17b rows and pf_model_b.pt already in the models dataset from scripts/build_v17b.sh)
set -e
FP=/home/user/work/own_models/fp_model_b.pt
W=/home/user/work; R=/home/user/claude_code_tests; cd $R
for s in train val; do
  out=$W/ret17bb_${s}_fp.parquet; [ -f $out ] && continue
  for i in 0 1 2; do python3 scripts/add_fp_feats.py $W/ret17b_${s}_fp.parquet $FP $out.part$i $W/pool4 $i 3 > $out.part$i.log 2>&1 & done; wait
  python3 -c "
import pandas as pd
pd.concat([pd.read_parquet(f'$out.part{i}') for i in range(3)], ignore_index=True).to_parquet('$out')"
  rm -f $out.part0 $out.part1 $out.part2
done
T=v17bb ROWS=ret17bb ALPHA=1 RANKER_PARAMS='{"num_threads": 4}' GL_THREADS=4 scripts/pipeline_v8d.sh
M=/home/user/ds/casmi26-models
cp $FP $M/fp_model_b.pt
cp $W/own_models/ret_ranker_v17bb.txt $W/own_models/ret_ranker_v17bb_feats.txt $W/own_models/stage2_v17bb.txt $W/own_models/stage2_v17bb_feats.txt $M/
kaggle datasets version -p $M -m "v17bb: FP model b + v17bb rankers" -q 2>&1 | grep -v Warning | tail -1
for i in $(seq 1 30); do n=$(kaggle datasets files bryantyu/casmi26-models --page-size 200 2>&1 | grep -c "_v17bb\|fp_model_b.pt"); [ "$n" -ge 5 ] && break; sleep 30; done
sleep 120          # kernels pushed right after a dataset version have failed at startup (v13 v1, v17 v1)
python3 scripts/make_own_kernel.py casmi26-own-s2v17bb ret_ranker_v17bb stage2_v17bb OWN_PF=pf_model_b.pt OWN_FP=fp_model_b.pt LIB_CAP=10 | tail -1
kaggle kernels push -p kernels/casmi26-own-s2v17bb 2>&1 | grep -v Warning | tail -1
echo V17BB_DONE
