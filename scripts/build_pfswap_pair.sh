#!/bin/bash
# PeakFormer swap on the v18c base as a seed pair: a single retrain moves the LB by up to +-0.006 (seed replicate v18cr1,
# Oct 29-30), so a swap is read as the 2-seed ensemble <tag>e against v18ce (v18c + v18cr1), not as one model.
#   scripts/build_pfswap_pair.sh <pf_model.pt[,pf2.pt]> <tag>      e.g. ... /home/user/work/pfs/pf_synp.pt v18cs
# 1. <tag>   = BASE=v18c REUSE_C1=0 build_v18c.sh (fresh PeakFormer columns, stage 1 / OOF / stage 2 with seed 0),
# 2. <tag>r1 = the same rows (hard links) with LightGBM seed 1 (RANKER_PARAMS / S2_PARAMS),
# both with GL_HAVE=/home/user/work/fwd_v18c* and GLW_N=1 (only GLACIER pairs no earlier pass scored; LB-read builds,
# not a fresh pass G2). Run it again after each "waiting for the GLACIER kernels" stop; it resumes where it left off.
# Then: models into casmi26-models, kernels by sed from casmi26-own-s2v18cr1 / -s2v18ce with OWN_PF = the new PeakFormer.
set -euo pipefail
PF=$1; T=$2
W=/home/user/work; R=/home/user/claude_code_tests
FP=$W/own_models/fp_model_v1r.pt
cd $R
export BASE=v18c REUSE_C1=0 GL_HAVE="$W/fwd_v18c*" GLW_N=1
bash scripts/build_v18c.sh "$PF" "$FP" $T
[ -f $W/own_models/stage2_$T.txt ] || { echo "$T not finished (GLACIER pending?)"; exit 0; }
RT=ret${T#v}; RT1=ret${T#v}r1
for s in train val; do
  for k in pf fp; do [ -f $W/${RT1}_${s}_$k.parquet ] || ln $W/${RT}_${s}_$k.parquet $W/${RT1}_${s}_$k.parquet; done
done
RANKER_PARAMS='{"num_threads": 4, "seed": 1}' S2_PARAMS='{"seed": 1}' bash scripts/build_v18c.sh "$PF" "$FP" ${T}r1
