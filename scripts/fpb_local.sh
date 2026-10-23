#!/bin/bash
# Paired CPU test of the FP-model population-prior fix before the GPU runs: baseline / variant b (instrument-free +
# training-structure negatives) / negtrain only, identical short schedule (1 epoch x 2000 steps, batch 1024, 31 negatives),
# then fp features recomputed on the class-2 COCONUT same-formula subset of ret14 val rows and scored with
# scripts/eval_pf_neutral.py (token 1 = kernel policy, token 0 = token-free).
set -e
cd /home/user/claude_code_tests
NT=/home/user/ds/casmi26-pf-negtrain/negtrain.npy; DATA=/home/user/ds/casmi26-fp-data; O=/home/user/work/fpb_local
mkdir -p $O
[ -f $O/fp_base.pt ] || python scripts/train_fp.py $DATA $O/fp_base.pt 1 2000 1024 31 4 > $O/base.log 2>&1
[ -f $O/fp_b.pt ] || FP_NEG_TRAIN=$NT FP_INST_DROP=1.0 FP_TIMS_W=1.0 python scripts/train_fp.py $DATA $O/fp_b.pt 1 2000 1024 31 4 > $O/b.log 2>&1
[ -f $O/fp_nt.pt ] || FP_NEG_TRAIN=$NT python scripts/train_fp.py $DATA $O/fp_nt.pt 1 2000 1024 31 4 > $O/nt.log 2>&1
python - <<'PY'
import pandas as pd
W = '/home/user/work'
D = pd.read_parquet(f'{W}/ret14_val_fp.parquet')
D['tr0'] = D.in_train; D.loc[(D.cls == 2) & (D.label == 1), 'tr0'] = 0
f = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['formula']).formula.to_numpy(dtype=object)
D['formula'] = f[D.pid.values]
t = D[(D.label == 1) & (D.cls == 2) & (D.in_coco == 1)][['qid', 'formula']].rename(columns={'formula': 'tf'})
M = D.merge(t, on='qid')
M = M[(M.formula == M.tf) & ((M.label == 1) | (M.tr0 == 1) | (M.in_coco == 1))].drop(columns=['tr0', 'formula', 'tf'])
M.to_parquet(f'{W}/fpb_local/sub.parquet'); print('subset', len(M), M.qid.nunique())
PY
for m in own_models/fp_model base b nt; do
  w=$O/fp_$(basename $m).pt; [ $m = own_models/fp_model ] && w=/home/user/work/own_models/fp_model.pt
  for tok in 1 0; do
    OWN_PF_INST=$tok python scripts/add_fp_feats.py $O/sub.parquet $w $O/sub_$(basename $m)_t$tok.parquet /home/user/work/pool4 > /dev/null
    echo "== $(basename $m) token $tok"; python scripts/eval_pf_neutral.py $O/sub_$(basename $m)_t$tok.parquet fp_bern,fp_cos | tail -2
  done
done
echo FPB_DONE
