#!/bin/bash
# v18c = clean-room rebuild of the v17b pipeline (docs/reproduce.md): every input built by code in this repository from
# permitted raw data, every stage restartable (skips finished outputs), deterministic LightGBM (G3).
#   G1: own ChEBI rel255 + LMSD structure set (scripts/build_bio_structures.py) -> pool5 (replaces pool4's third-party set)
#   G2: GLACIER scores for every stage-2 training / validation pair computed in ONE fresh pass (no --have, no merged
#       history) on a Kaggle GPU kernel (casmi.fwdrun --device cuda); stage 2 reads only that pass.
#   G13/G14: row dumps with recorded seeds, VENDOR_PATH disabled, stale part dirs removed; features added in one place.
# scripts/build_v18c.sh <pf_model.pt> <fp_model.pt> [tag]      (STOP_AFTER=B: pool + row dumps; C1: + PeakFormer features; D: + stage 1)
# BASE=<tag> reuses that build's pool and stage B rows; its C1 rows (PeakFormer features) too unless REUSE_C1=0:
#   FP-model swap:  BASE=v18c scripts/build_v18c.sh <pf_model_b.pt> <fp_model_b.pt> v18cb
#   PeakFormer swap: BASE=v18c REUSE_C1=0 scripts/build_v18c.sh <new pf_model.pt> <fp_model_v1r.pt> v18cs
# Stage E stops after pushing the GPU GLACIER kernel; run the script again when kernels/casmi26-gl<tag> is COMPLETE.
# GL_HAVE=<dir glob> (e.g. /home/user/work/fwd_v18c*): LB-read builds only -- GLACIER jobs skip (query, candidate) pairs
#   already scored in <dir>/{train,val}.jobs_*.gl.json of earlier builds on the same rows (build_fwd_jobs.py --have) and
#   stage F reads those files too; GLW_N=1 then runs the missing pairs on one worker. Not a fresh pass (G2): rebuild
#   without GL_HAVE before a final pick (CPU GLACIER differs between passes by a median 0, 99% < 0.003 anyway).
set -euo pipefail
PF=$1; FP=$2; T=${3:-v18c}
W=/home/user/work; R=/home/user/claude_code_tests; EXT=/home/user/ext; DS=/home/user/ds
B0=${BASE:-$T}
P5=$W/pool_$B0; RB=ret${B0#v}; ROWS=ret${T#v}   # e.g. pool_v18c, ret18c (fresh names: an old $W/pool5 from Oct 19 exists)
if [ -n "${BASE:-}" ] && [ "${REUSE_C1:-1}" = 0 ]; then RC=$ROWS; else RC=$RB; fi   # rows carrying the PeakFormer columns
cd $R
log() { echo "[$(date -u +%H:%M)] $*"; }

# A. pool_<tag> = pool2 + our ChEBI / LMSD set; PubChem documentation counts; COCONUT priors; doc-shift weights
BIO=$EXT/bio_raw/bio_structures_own.parquet
[ -f $BIO ] || python3 scripts/build_bio_structures.py $EXT/bio_raw/chebi_rel255.sdf.gz $EXT/bio_raw/LMSD.sdf.zip \
    $EXT/coconut/coconut_2026_10.parquet $BIO 4
[ -z "${BASE:-}" ] || [ -f $P5/.built ] || { echo "BASE=$BASE: $P5 is not a finished build"; exit 1; }
if [ ! -f $P5/.built ]; then        # marker written only by this stage: never reuse a directory of unknown origin
  log "A: $P5"
  rm -rf $P5
  python3 scripts/build_pool_bio.py $W/pool2 $BIO $P5 4
  python3 scripts/add_pubchem_pop.py $EXT/pubchem/key_pop.parquet $P5
  cp $EXT/coconut/coconut_prior.parquet $P5/
  touch $P5/.built
fi
[ -f $P5/doc_w.parquet ] || python3 -m casmi.docw $P5 $P5/doc_w.parquet
md5sum $P5/pool.parquet $P5/pool_fp.npy $BIO | tee $W/$T.md5

# B. retrieval rows (seeds as ret4: 12 train / 11 val; split keys $W/rank_keys.txt / val_keys.txt, G6)
for s in train val; do
  out=$W/${RB}_$s.parquet; [ -f $out ] && continue
  rm -rf $out.parts
  [ $s = train ] && args="keys:$W/rank_keys.txt 8000 $out 12 4" || args="keys:$W/val_keys.txt 1500 $out 11 4"
  log "B: dump $s"
  CASMI_POOL=$P5 VENDOR_PATH=/nonexistent python3 scripts/dump_ret_rows.py $args > $out.log 2>&1
done

[ "${STOP_AFTER:-}" = B ] && { log "STOP_AFTER=B"; exit 0; }

# C. spectral features without a richfp cache (pool ids differ from pool4): C1 PeakFormer (pf_* + within-formula
#    columns) on the raw rows, C2 FP model (fp_*) on top -- C1 does not depend on the FP choice
for s in train val; do
  out=$W/${RC}_${s}_pf.parquet; [ -f $out ] && continue
  log "C1: PeakFormer features $s"
  for i in 0 1 2; do RICHFP_CACHE= python3 scripts/add_fp_feats.py $W/${RB}_$s.parquet $PF $out.p$i $P5 $i 3 > $out.p$i.log 2>&1 & done; wait
  python3 -c "
import sys, pandas as pd
sys.path.insert(0, '$R')
from casmi.own import formula_rel, PF_FREL_SRC
D = pd.concat([pd.read_parquet(f'$out.p{i}') for i in range(3)], ignore_index=True)
D['formula'] = pd.read_parquet('$P5/pool.parquet', columns=['formula']).formula.to_numpy(dtype=object)[D.pid.values]
formula_rel(D, PF_FREL_SRC)
D.drop(columns='formula').to_parquet('$out')"
  rm -f $out.p0 $out.p1 $out.p2
done
[ "${STOP_AFTER:-}" = C1 ] && { log "STOP_AFTER=C1"; exit 0; }
for s in train val; do
  out=$W/${ROWS}_${s}_fp.parquet; [ -f $out ] && continue
  log "C2: FP features $s"
  for i in 0 1 2; do python3 scripts/add_fp_feats.py $W/${RC}_${s}_pf.parquet $FP $out.f$i $P5 $i 3 > $out.f$i.log 2>&1 & done; wait
  python3 -c "
import pandas as pd
pd.concat([pd.read_parquet(f'$out.f{i}') for i in range(3)], ignore_index=True).to_parquet('$out')"
  rm -f $out.f0 $out.f1 $out.f2
done

# D. stage 1 (v5w recipe: 31 leaves, alpha 1) + out-of-fold scores
export DOC_W=$P5/doc_w.parquet DOC_ALPHA=1 RANKER_PARAMS=${RANKER_PARAMS:-'{"num_threads": 4}'}   # e.g. '{"num_threads": 4, "seed": 1}' (seed replicate; S2_PARAMS for stage 2)
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

[ "${STOP_AFTER:-}" = D ] && { log "STOP_AFTER=D"; exit 0; }

# E. GLACIER, one fresh pass for every top-30 pair (no --have), on Kaggle: 4 CPU worker kernels by default (Kaggle CPU
#    kernels run ~6.5 predictions/s each, free, and match the CPU device of the inference kernel), GL_DEVICE=gpu for one
#    T4 x 2 kernel (scripts/make_gl_gpu_kernel.py; ~65 predictions/s per GPU, shared GPU quota)
F=$W/fwd_$T; GJ=$DS/casmi26-gl-$T
mkdir -p $F
[ -f $F/train.jobs_0.json ] || (cd $W && python3 $R/scripts/build_fwd_jobs.py ret_${T}_train_oof.parquet col:s1_oof $P5 30 $F/train 6 ${GL_HAVE:+"--have=$GL_HAVE/train.jobs_*.gl.json"})
[ -f $F/val.jobs_0.json ] || (cd $W && python3 $R/scripts/build_fwd_jobs.py ${ROWS}_val_reach_c.parquet own_models/ret_ranker_$T.txt $P5 30 $F/val 2 ${GL_HAVE:+"--have=$GL_HAVE/val.jobs_*.gl.json"})
GLW_N=${GLW_N:-4}                   # CPU workers: 4, or 3 when fewer Kaggle sessions are free (the owner's come first)
if [ "${GL_DEVICE:-cpu}" = gpu ]; then KS="casmi26-gl$T"; else KS=$(for i in $(seq 0 $((GLW_N - 1))); do echo -n "casmi26-glw$T-$i "; done); fi
if ! ls $F/train.jobs_*.gl.json > /dev/null 2>&1; then
  if [ ! -f kernels/${KS%% *}/kernel-metadata.json ]; then
    log "E: upload GLACIER jobs + push ${GL_DEVICE:-cpu} kernels"
    mkdir -p $GJ && cp $F/*.jobs_*.json $GJ/
    echo "{\"title\": \"casmi26-gl-$T\", \"id\": \"bryantyu/casmi26-gl-$T\", \"licenses\": [{\"name\": \"Apache 2.0\"}]}" > $GJ/dataset-metadata.json
    kaggle datasets create -p $GJ -q 2>&1 | grep -v Warning | tail -1
    if [ "${GL_DEVICE:-cpu}" = gpu ]; then
      python3 scripts/make_gl_gpu_kernel.py casmi26-gl$T casmi26-gl-$T
    else
      if [ "$GLW_N" = 1 ]; then MAP="0:$(cd $F && ls train.jobs_*.json val.jobs_*.json | grep -v gl.json | paste -sd,)"
      elif [ "$GLW_N" = 3 ]; then MAP="0:train.jobs_0.json,train.jobs_3.json,val.jobs_0.json 1:train.jobs_1.json,train.jobs_4.json,val.jobs_1.json 2:train.jobs_2.json,train.jobs_5.json"
      else MAP="0:train.jobs_0.json,train.jobs_4.json 1:train.jobs_1.json,train.jobs_5.json 2:train.jobs_2.json,val.jobs_0.json 3:train.jobs_3.json,val.jobs_1.json"; fi
      GLW_DS=casmi26-gl-$T GLW_PREFIX=casmi26-glw$T python3 scripts/make_gl_workers.py $MAP
    fi
    for i in $(seq 1 30); do st=$(kaggle datasets status bryantyu/casmi26-gl-$T 2>&1 | grep -v Warning); [ "$st" = ready ] && break; sleep 20; done
    sleep 120
    for k in $KS; do kaggle kernels push -p kernels/$k 2>&1 | grep -v Warning | tail -1; done
  fi
  done_all=1
  for k in $KS; do
    st=$(kaggle kernels status bryantyu/$k 2>&1 | grep -v Warning)
    case "$st" in *COMPLETE*) ;; *) echo "$k: $st"; done_all=0;; esac
  done
  [ $done_all = 1 ] || { log "E: waiting for the GLACIER kernels; run this script again when they are COMPLETE"; exit 0; }
  for k in $KS; do kaggle kernels output bryantyu/$k -p $F --file-pattern 'gl\.json$' 2>&1 | grep -v Warning | tail -1; done
fi

# F. stage 2 on the fresh GLACIER pass only
[ -f $W/own_models/stage2_$T.txt ] || (cd $W && GL_GLOB="$F/train.jobs_*.gl.json${GL_HAVE:+,$GL_HAVE/train.jobs_*.gl.json}" GL_GLOB_VAL="$F/val.jobs_*.gl.json${GL_HAVE:+,$GL_HAVE/val.jobs_*.gl.json}" \
  python3 $R/scripts/train_stage2.py fit fwd_$T/train own_models/ret_ranker_${T}_feats.txt own_models/stage2_$T.txt fwd_$T/val)
md5sum $W/own_models/ret_ranker_$T.txt $W/own_models/stage2_$T.txt | tee -a $W/$T.md5
log "${T}_DONE"
