#!/bin/bash
# v17b = v17 (v5w recipe + PeakFormer features on lib10 rows) with the debiased PeakFormer b replacing PeakFormer d.
# scripts/build_v17b.sh [pf_model.pt]   (default: the GPU run kernels/casmi26-pf-train-b, downloaded to /home/user/work/kout/pfb_gpu)
# 1. neutral gate on the 620-query class-2 COCONUT subset (design panel wf_f100d853-7f9): pf_bern win rate vs same-formula
#    training decoys >= 0.70 and excess (COCONUT - training) <= +0.087 (PeakFormer d: .667 / +.137 at token 1);
# 2. ret17b rows = ret17 with pf_* (+ within-formula pf columns) rescored by PeakFormer b (instrument-free: token tied);
# 3. pipeline (v5w recipe), 4. models upload + kernel casmi26-own-s2v17b (LIB_CAP 10, OWN_PF = the b checkpoint).
set -e
PF=${1:-/home/user/work/kout/pfb_gpu/pf_model.pt}
W=/home/user/work; R=/home/user/claude_code_tests; cd $R
export RICHFP_CACHE=$W/richfp_ret10.npz
O=$W/v17b; mkdir -p $O
python3 scripts/add_fp_feats.py $W/fpb_local/sub.parquet $PF $O/sub_pf.parquet $W/pool4 > /dev/null
python3 scripts/eval_pf_neutral.py $O/sub_pf.parquet pf_bern,pf_cos | tee $O/gate.txt
python3 - $O/gate.txt <<'EOF'
import re, sys
line = [l for l in open(sys.argv[1]) if l.startswith('pf_bern')][0]
tr, ex = float(re.search(r'training decoys ([0-9.]+)', line).group(1)), float(re.search(r'excess \(prior\) ([-+0-9.]+)', line).group(1))
print('GATE pf_bern: vs training decoys', tr, 'excess', ex)
sys.exit(0 if tr >= 0.70 and ex <= 0.087 else 'GATE FAILED: no v17b (PF-b goes to the spectral-only probe only)')
EOF
for s in train val; do
  out=$W/ret17b_${s}_fp.parquet; [ -f $out ] && continue
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
done
T=v17b ROWS=ret17b ALPHA=1 RANKER_PARAMS='{"num_threads": 4}' GL_THREADS=4 scripts/pipeline_v8d.sh
M=/home/user/ds/casmi26-models
cp $PF $M/pf_model_b.pt
cp $W/own_models/ret_ranker_v17b.txt $W/own_models/ret_ranker_v17b_feats.txt $W/own_models/stage2_v17b.txt $W/own_models/stage2_v17b_feats.txt $M/
kaggle datasets version -p $M -m "v17b: PeakFormer b (training-structure negatives, instrument-free) + v17b rankers" -q 2>&1 | grep -v Warning | tail -1
for i in $(seq 1 30); do n=$(kaggle datasets files bryantyu/casmi26-models --page-size 200 2>&1 | grep -c "_v17b\|pf_model_b.pt"); [ "$n" -ge 5 ] && break; sleep 30; done
sleep 120          # kernels pushed right after a dataset version have failed at startup (v13 v1, v17 v1)
python3 scripts/make_own_kernel.py casmi26-own-s2v17b ret_ranker_v17b stage2_v17b OWN_PF=pf_model_b.pt LIB_CAP=10 | tail -1
kaggle kernels push -p kernels/casmi26-own-s2v17b 2>&1 | grep -v Warning | tail -1
echo V17B_DONE
