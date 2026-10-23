#!/bin/bash
# Adopt new spectral models (e.g. GPU variant b): neutral eval -> rows -> stage-1 ranker -> HL-val -> full pipeline.
# scripts/adopt_spectral_models.sh <pf_model.pt> <fp_model.pt> <tag e.g. v17> [token, default 0]
# Rows = ret10pl (all other features) with pf_* / fp_* recomputed by the new models under the token policy; ranker and
# pipeline use the v11l / v13 recipe (doc weights alpha 1, 63 leaves). Compare against v13 on ret13 rows.
set -e
PF=$1; FP=$2; T=$3; TOK=${4:-0}
W=/home/user/work; R=/home/user/claude_code_tests; cd $R
export RICHFP_CACHE=$W/richfp_ret10.npz OWN_PF_INST=$TOK
O=$W/adopt_$T; mkdir -p $O
echo "== neutral eval (class-2 COCONUT subset, token $TOK)"
python3 scripts/add_fp_feats.py $W/fpb_local/sub.parquet $PF $O/sub_pf.parquet $W/pool4 > /dev/null
python3 scripts/eval_pf_neutral.py $O/sub_pf.parquet pf_bern,pf_cos | tail -2
python3 scripts/add_fp_feats.py $W/fpb_local/sub.parquet $FP $O/sub_fp.parquet $W/pool4 > /dev/null
python3 scripts/eval_pf_neutral.py $O/sub_fp.parquet fp_bern,fp_cos | tail -2
for s in train val; do
  a=$O/a_${s}.parquet; out=$W/ret_${T}_${s}_fp.parquet
  [ -f $out ] && continue
  for i in 0 1; do python3 scripts/add_fp_feats.py $W/ret10pl_${s}_fp.parquet $PF $a.part$i $W/pool4 $i 2 > $a.part$i.log 2>&1 & done; wait
  python3 -c "
import sys, pandas as pd
sys.path.insert(0, '$R')
from casmi.own import formula_rel, PF_FREL_SRC
D = pd.concat([pd.read_parquet(f'$a.part{i}') for i in range(2)], ignore_index=True)
D['formula'] = pd.read_parquet('$W/pool4/pool.parquet', columns=['formula']).formula.to_numpy(dtype=object)[D.pid.values]
formula_rel(D, PF_FREL_SRC)
D.drop(columns='formula').to_parquet('$a')"
  rm -f $a.part0 $a.part1
  for i in 0 1; do python3 scripts/add_fp_feats.py $a $FP $out.part$i $W/pool4 $i 2 > $out.part$i.log 2>&1 & done; wait
  python3 -c "
import pandas as pd
pd.concat([pd.read_parquet(f'$out.part{i}') for i in range(2)], ignore_index=True).to_parquet('$out')"
  rm -f $out.part0 $out.part1 $a
done
export DOC_W=$W/pool4/doc_w.parquet DOC_ALPHA=1 RANKER_PARAMS='{"num_leaves": 63, "min_data_in_leaf": 50}'
[ -f $W/own_models/ret_ranker_$T.txt ] || python3 scripts/train_ret_ranker.py $W/ret_${T}_train_fp.parquet $W/ret_${T}_val_fp.parquet $W/own_models/ret_ranker_$T.txt $W/pool4 --drop-artifact > $O/ranker.log 2>&1
grep "ALL cls" $O/ranker.log
M=$W/own_models
for pr in 1 0; do
  echo "== HL-val HL_PRIOR=$pr: new ranker on its rows vs v13 on ret13 rows"
  HL_PRIOR=$pr HL_ROWS_new=$W/ret_${T}_val_fp.parquet python3 scripts/hlval.py $M/ret_ranker_$T.txt 2>&1 | grep -v Warning
  HL_PRIOR=$pr HL_ROWS_tok3=$W/ret13_val_fp.parquet python3 scripts/hlval.py $M/ret_ranker_v13.txt 2>&1 | grep -v Warning
done
T=$T ROWS=ret_$T ALPHA=1 scripts/pipeline_v8d.sh
echo ADOPT_${T}_DONE
