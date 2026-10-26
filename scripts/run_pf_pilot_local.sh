#!/bin/bash
# PF-syn pilot on this container's CPU (no TPU / GPU quota): the arms of kernels/casmi26-pf-pilot (same model, data, LR,
# negatives, synthetic-row settings) run one after the other on PF_DEV=cpu with batch 256 and a fixed step count, so
# both arms are step-matched. Outputs $OUT/pf_ctl.pt, $OUT/pf_syn.pt and their logs.
# scripts/run_pf_pilot_local.sh <steps> <out_dir> [arms (default "ctl syn")]
# Inputs: casmi26-fp-data, casmi26-pf-data (fpx.npz), casmi26-pf-negtrain, casmi26-models (pf_model_b.pt),
# casmi26-sim-jobs (syn_drop_B.txt) under /home/user/ds, syn.npz (kernels/casmi26-syn-prep output) under $SYN.
set -euo pipefail
STEPS=$1; OUT=$2; ARMS=${3:-"ctl syn"}
R=$(cd "$(dirname "$0")/.." && pwd); DS=/home/user/ds; SYN=${SYN:-/home/user/work/synprep/syn.npz}
mkdir -p "$OUT"
for arm in $ARMS; do
  [ -f "$OUT/pf_$arm.pt" ] && { echo "$arm: $OUT/pf_$arm.pt exists, skipped"; continue; }
  extra=()
  [ "$arm" = syn ] && extra=(PF_SYN=$SYN PF_SYN_P=0.25 PF_SYN_NEUTRAL_W=0.18 PF_SYN_DROP=$DS/casmi26-sim-jobs/syn_drop_B.txt)
  echo "=== arm $arm $(date -u +%H:%M)"
  env PF_DEV=cpu PF_HOURS=${PF_HOURS:-9} PF_VAL_N=800 PF_D=192 PF_LAYERS=4 PF_HEADS=6 PF_DMOL=256 PF_DROP=0.0 \
      PF_SBAL=0.5 PF_MERGE=0.35 PF_SEED=31 PF_LR=2e-4 PF_LOG=200 PF_INST_DROP=1.0 PF_TIMS_W=1.0 \
      PF_NEG_TRAIN=$DS/casmi26-pf-negtrain/negtrain.npy PF_NEG_NOREP=1 PF_CAL=200 PF_INIT=$DS/casmi26-models/pf_model_b.pt \
      PF_STEPS=$STEPS "${extra[@]}" \
      python3 "$R/scripts/train_pf_xla.py" $DS/casmi26-fp-data $DS/casmi26-pf-data/fpx.npz "$OUT/pf_$arm.pt.tmp" 1 "$STEPS" 256 63 \
      > "$OUT/pf_$arm.log" 2>&1
  for f in "$OUT/pf_$arm.pt.tmp"*; do mv "$f" "${f/.pt.tmp/.pt}"; done
  echo "=== arm $arm done $(date -u +%H:%M)"
done
echo PILOT_DONE
