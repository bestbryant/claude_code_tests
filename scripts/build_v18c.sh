#!/bin/bash
# v18c = clean-room rebuild of the v17b pipeline (docs/reproduce.md): every input built by code in this repository from
# permitted raw data, every stage restartable (skips finished outputs), deterministic LightGBM (G3).
#   G1: own ChEBI rel255 + LMSD structure set (scripts/build_bio_structures.py) -> pool5 (replaces pool4's third-party set)
#   G2: GLACIER scores for every stage-2 training / validation pair computed in ONE fresh pass (no --have, no merged
#       history) on a Kaggle GPU kernel (casmi.fwdrun --device cuda); stage 2 reads only that pass.
#   G13/G14: row dumps with recorded seeds, VENDOR_PATH disabled, stale part dirs removed; features added in one place.
# scripts/build_v18c.sh <pf_model.pt> <fp_model.pt> [tag]      (STOP_AFTER=B: pool5 + row dumps only)
# Stage E stops after pushing the GPU GLACIER kernel; run the script again when kernels/casmi26-gl<tag> is COMPLETE.
set -euo pipefail
PF=$1; FP=$2; T=${3:-v18c}
W=/home/user/work; R=/home/user/claude_code_tests; EXT=/home/user/ext; DS=/home/user/ds
P5=$W/pool5; ROWS=ret${T#v}            # e.g. ret18c
cd $R
log() { echo "[$(date -u +%H:%M)] $*"; }

# A. pool5 = pool2 + our ChEBI / LMSD set; PubChem documentation counts; COCONUT priors; doc-shift weights
BIO=$EXT/bio_raw/bio_structures_own.parquet
[ -f $BIO ] || python3 scripts/build_bio_structures.py $EXT/bio_raw/chebi_rel255.sdf.gz $EXT/bio_raw/LMSD.sdf.zip \
    $EXT/coconut/coconut_2026_10.parquet $BIO 4
if [ ! -f $P5/pool.parquet ]; then
  log "A: pool5"
  python3 scripts/build_pool_bio.py $W/pool2 $BIO $P5 4
  python3 scripts/add_pubchem_pop.py $EXT/pubchem/key_pop.parquet $P5
  cp $EXT/coconut/coconut_prior.parquet $P5/
fi
[ -f $P5/doc_w.parquet ] || python3 -m casmi.docw $P5 $P5/doc_w.parquet
md5sum $P5/pool.parquet $P5/pool_fp.npy $BIO | tee $W/$T.md5

# B. retrieval rows (seeds as ret4: 12 train / 11 val; split keys $W/rank_keys.txt / val_keys.txt, G6)
for s in train val; do
  out=$W/${ROWS}_$s.parquet; [ -f $out ] && continue
  rm -rf $out.parts
  [ $s = train ] && args="keys:$W/rank_keys.txt 8000 $out 12 4" || args="keys:$W/val_keys.txt 1500 $out 11 4"
  log "B: dump $s"
  CASMI_POOL=$P5 VENDOR_PATH=/nonexistent python3 scripts/dump_ret_rows.py $args > $out.log 2>&1
done

[ "${STOP_AFTER:-}" = B ] && { log "STOP_AFTER=B"; exit 0; }

# C. spectral features: FP model (fp_*), PeakFormer (pf_* + within-formula columns); no richfp cache (pool5 ids differ)
for s in train val; do
  out=$W/${ROWS}_${s}_fp.parquet; [ -f $out ] && continue
  log "C: features $s"
  for i in 0 1 2; do python3 scripts/add_fp_feats.py $W/${ROWS}_$s.parquet $FP $out.fp$i $P5 $i 3 > $out.fp$i.log 2>&1 & done; wait
  python3 -c "
import pandas as pd
pd.concat([pd.read_parquet(f'$out.fp{i}') for i in range(3)], ignore_index=True).to_parquet('$out.fp')"
  rm -f $out.fp0 $out.fp1 $out.fp2
  for i in 0 1 2; do RICHFP_CACHE= python3 scripts/add_fp_feats.py $out.fp $PF $out.pf$i $P5 $i 3 > $out.pf$i.log 2>&1 & done; wait
  python3 -c "
import sys, pandas as pd
sys.path.insert(0, '$R')
from casmi.own import formula_rel, PF_FREL_SRC
D = pd.concat([pd.read_parquet(f'$out.pf{i}') for i in range(3)], ignore_index=True)
D['formula'] = pd.read_parquet('$P5/pool.parquet', columns=['formula']).formula.to_numpy(dtype=object)[D.pid.values]
formula_rel(D, PF_FREL_SRC)
D.drop(columns='formula').to_parquet('$out')"
  rm -f $out.fp $out.pf0 $out.pf1 $out.pf2
done

# D. stage 1 (v5w recipe: 31 leaves, alpha 1) + out-of-fold scores
export DOC_W=$P5/doc_w.parquet DOC_ALPHA=1 RANKER_PARAMS='{"num_threads": 4}'
[ -f $W/own_models/ret_ranker_$T.txt ] || python3 scripts/train_ret_ranker.py $W/${ROWS}_train_fp.parquet $W/${ROWS}_val_fp.parquet $W/own_models/ret_ranker_$T.txt $P5 --drop-artifact
[ -f $W/ret_${T}_train_oof.parquet ] || python3 scripts/oof_stage1.py $W/${ROWS}_train_fp.parquet $W/own_models/ret_ranker_${T}_feats.txt $W/ret_${T}_train_oof.parquet --drop-artifact
[ -f $W/${ROWS}_val_reach_c.parquet ] || python3 - $W/${ROWS}_val_fp.parquet $W/${ROWS}_val_reach_c.parquet <<'PY'
import sys, pandas as pd
D = pd.read_parquet(sys.argv[1])
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
t = D[D.label == 1].set_index('qid')
reach = (t.in_coco == 1) | (t.in_pcl == 1) | (t.in_bio == 1)
art = t.index[(t.cls == 1) & (t.lib_n == 0)]
D = D[((D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)) & ~D.qid.isin(art)]
D.to_parquet(sys.argv[2]); print('val queries', D.qid.nunique())
PY

# E. GLACIER, one fresh pass for every top-30 pair (no --have), on a Kaggle GPU kernel
F=$W/fwd_$T; GJ=$DS/casmi26-gl-$T; K=kernels/casmi26-gl$T
mkdir -p $F
[ -f $F/train.jobs_0.json ] || (cd $W && python3 $R/scripts/build_fwd_jobs.py ret_${T}_train_oof.parquet col:s1_oof $P5 30 $F/train 8)
[ -f $F/val.jobs_0.json ] || (cd $W && python3 $R/scripts/build_fwd_jobs.py ${ROWS}_val_reach_c.parquet own_models/ret_ranker_$T.txt $P5 30 $F/val 2)
if ! ls $F/train.jobs_*.gl.json > /dev/null 2>&1; then
  if [ ! -f $K/kernel-metadata.json ]; then
    log "E: upload GLACIER jobs + push GPU kernel"
    mkdir -p $GJ && cp $F/*.jobs_*.json $GJ/
    echo "{\"title\": \"casmi26-gl-$T\", \"id\": \"bryantyu/casmi26-gl-$T\", \"licenses\": [{\"name\": \"Apache 2.0\"}]}" > $GJ/dataset-metadata.json
    kaggle datasets create -p $GJ -q 2>&1 | grep -v Warning | tail -1
    python3 scripts/make_gl_gpu_kernel.py casmi26-gl$T casmi26-gl-$T
    for i in $(seq 1 30); do st=$(kaggle datasets status bryantyu/casmi26-gl-$T 2>&1 | grep -v Warning); [ "$st" = ready ] && break; sleep 20; done
    sleep 120
    kaggle kernels push -p $K 2>&1 | grep -v Warning | tail -1
  fi
  log "E: waiting for kernels/casmi26-gl$T; run this script again when it is COMPLETE"
  st=$(kaggle kernels status bryantyu/casmi26-gl$T 2>&1 | grep -v Warning)
  case "$st" in *COMPLETE*) kaggle kernels output bryantyu/casmi26-gl$T -p $F --file-pattern 'gl\.json$' 2>&1 | grep -v Warning | tail -1;;
                *) echo "$st"; exit 0;; esac
fi

# F. stage 2 on the fresh GLACIER pass only
[ -f $W/own_models/stage2_$T.txt ] || (cd $W && GL_GLOB="$F/train.jobs_*.gl.json" GL_GLOB_VAL="$F/val.jobs_*.gl.json" \
  python3 $R/scripts/train_stage2.py fit fwd_$T/train own_models/ret_ranker_${T}_feats.txt own_models/stage2_$T.txt fwd_$T/val)
md5sum $W/own_models/ret_ranker_$T.txt $W/own_models/stage2_$T.txt | tee -a $W/$T.md5
log "${T}_DONE"
