#!/usr/bin/env bash
# Stage-1 ranker v3: FP v3 features (test-like keys held out of FP training) on the pool-v2 ranker rows and the
# test-like (timsTOF, instrument hold-out) rows, then one LambdaRank on both, validated on pool-v2 validation rows
# and on the test-like validation rows.
#   scripts/pipeline_s1v3.sh            (expects /home/user/work/fp_v3.pt and tl_train.parquet)
set -euo pipefail
W=/home/user/work; R=/home/user/claude_code_tests
cd $R
fp() {  # rows out pool shards
  local rows=$1 out=$2 pool=$3 n=$4
  for i in $(seq 0 $((n - 1))); do
    python3 scripts/add_fp_feats.py $rows $W/fp_v3.pt $out.part$i $pool $i $n > $out.part$i.log 2>&1 &
  done
  wait
  python3 - "$out" "$n" <<'EOF'
import sys, pandas as pd
out, n = sys.argv[1], int(sys.argv[2])
pd.concat([pd.read_parquet(f'{out}.part{i}') for i in range(n)], ignore_index=True).to_parquet(out)
EOF
  rm -f $out.part*
}
fp $W/ret2_train_fp.parquet $W/ret2_train_fp3.parquet $W/pool2 3
fp $W/tl_train.parquet $W/tl_train_fp3.parquet $W/pool2 3
fp $W/ret2_val_reach.parquet $W/ret2_val_fp3.parquet $W/pool2 3
fp $W/tl_val_fp.parquet $W/tl_val_fp3.parquet $W/pool3 3
python3 - <<'EOF'
import pandas as pd
W = '/home/user/work'
a = pd.read_parquet(f'{W}/ret2_train_fp3.parquet'); b = pd.read_parquet(f'{W}/tl_train_fp3.parquet')
b['qid'] = b.qid + 1_000_000
pd.concat([a, b[a.columns.intersection(b.columns)]], ignore_index=True).to_parquet(f'{W}/s1v3_train.parquet')
print('rows', len(a), '+', len(b))
EOF
python3 scripts/train_ret_ranker.py $W/s1v3_train.parquet $W/ret2_val_fp3.parquet $W/own_models/ret_ranker_v3.txt $W/pool2
echo PIPELINE_DONE
