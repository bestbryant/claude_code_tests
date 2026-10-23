#!/bin/bash
# Calibrate HL_TRANSPLANT_P: the LB gap between stage-1 rankers with and without documentation priors was
# 0.245 - 0.315 = -0.070 (Oct 18, alpha 0); find the transplant share at which HL-val reproduces it.
W=/home/user/work; R=/home/user/claude_code_tests; cd $R
export RANKER_PARAMS='{"num_leaves": 63, "min_data_in_leaf": 50, "num_threads": 4}'
[ -f $W/own_models/ret_ranker_hlp.txt ] || python3 scripts/train_ret_ranker.py $W/ret14_train_fp.parquet $W/ret14_val_fp.parquet $W/own_models/ret_ranker_hlp.txt $W/pool4 --drop-artifact > $W/hlp.log 2>&1
[ -f $W/own_models/ret_ranker_hlnp.txt ] || python3 scripts/train_ret_ranker.py $W/ret14_train_fp.parquet $W/ret14_val_fp.parquet $W/own_models/ret_ranker_hlnp.txt $W/pool4 --drop-artifact --no-prior > $W/hlnp.log 2>&1
for p in 0 0.25 0.5 0.75 1; do
  echo "== HL_TRANSPLANT_P=$p"
  HL_TRANSPLANT_P=$p HL_DRAWS=5 HL_ROWS_kernel=$W/ret14_val_fp.parquet python3 scripts/hlval.py $W/own_models/ret_ranker_hlp.txt $W/own_models/ret_ranker_hlnp.txt 2>&1 | grep -v Warning
done
echo CALIB_DONE
