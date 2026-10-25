#!/bin/bash
# Candidate pool v4 (bryantyu/casmi26-pool4): docs/reproduce.md 1.4 chain (G12), md5-checked against the deployed files.
#   pool2 = build_pool.py (train_meta + COCONUT 2026-10 + PubChemLite) + coconut_prior.parquet + PubChem counts
#   pool4 = pool2 + ChEBI / LMSD rows of bio_structures.parquet (in_bio) + PubChem counts; doc_w.parquet (casmi.docw)
# scripts/build_pool4.sh <out_dir> [procs]          -> <out_dir>/pool2, <out_dir>/pool4
#   POOL2=<dir>  start from an existing pool2 (md5-checked, read only; e.g. $W/pool2) instead of building one
#   DRY=1        print the commands and checks without running them
# build_pool.py needs ~12 GB RAM. Its sort_values('mass') is an unstable quicksort whose tie order follows numpy's SIMD
# dispatch, so the md5s hold on x86-64 with AVX-512 under Python 3.13.16, numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0,
# RDKit 2026.03.3 (G12). A pool dir gets .built once its checks pass and is only re-checked on a rerun; a dir this
# script did not start (no .building / .built marker) is never reused or deleted.
set -euo pipefail
OUT=$(realpath -m ${1:?usage: scripts/build_pool4.sh <out_dir> [procs]}); PROCS=${2:-4}
R=/home/user/claude_code_tests; EXT=/home/user/ext; DATA=/home/user/data
KEY_POP=$EXT/pubchem/key_pop.parquet; BIO=$EXT/chebi_lmsd/bio_structures.parquet
P2=$(realpath -m ${POOL2:-$OUT/pool2}); P4=$OUT/pool4
cd $R
log() { echo "[$(date -u +%H:%M)] $*"; }
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }
chk() {          # chk <file> <md5> [<md5> ...]: stop unless the file has one of them
  local f=$1 m w; shift
  if [ -n "${DRY:-}" ]; then echo "+ md5 $f = $*"; return; fi
  m=$(md5sum < $f | cut -d' ' -f1)
  for w in "$@"; do [ $m = $w ] && { echo "ok $m $f"; return; }; done
  echo "md5 mismatch: $f is $m, expected $*" >&2; exit 1
}
rows() {         # rows <parquet> <n>
  if [ -n "${DRY:-}" ]; then echo "+ rows $1 = $2"; return; fi
  local n; n=$(python3 -c "import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)" $1)
  [ $n = $2 ] || { echo "row count: $1 has $n, expected $2" >&2; exit 1; }
}
fresh() {        # fresh <dir>: empty dir marked .building (a partial dir of an earlier run is removed)
  if [ -f $1/.building ]; then run rm -rf $1
  elif [ -e $1 ]; then echo "$1 exists but this script did not make it: remove it or use another out_dir" >&2; exit 1
  fi
  run mkdir -p $1; run touch $1/.building
}

python3 -c "import sys, numpy, pandas, pyarrow, rdkit; print('python', sys.version.split()[0], 'numpy',
numpy.__version__, 'pandas', pandas.__version__, 'pyarrow', pyarrow.__version__, 'rdkit', rdkit.__version__)"
python3 -c "from numpy._core._multiarray_umath import __cpu_features__ as f; assert f['AVX512_SKX']" 2>/dev/null \
  || echo "warning: no AVX-512: pool row order and md5s will differ (G12)" >&2
chk $KEY_POP ce9c3671d1d8f8521c5477b02f417aa7               # PubChem Extras, Oct 18 snapshot (G11)
chk $BIO ba20c5ab5358dc2225a829b493d30b72                   # sergiomalv/casmi-chebi-lmsd-structures (G1)

# pool2
built2=0
if [ -n "${POOL2:-}" ] || [ -f $P2/.built ]; then
  log "pool2: checking $P2"
else
  log "pool2: building $P2"; built2=1
  chk $DATA/train_meta.parquet 82d7df60fee38cecca498c3dff1e5b79
  chk $EXT/coconut/coconut_2026_10.parquet 8e2fcf7e6234aa3b332f7a9bbedeaf22
  chk $EXT/pcl/pcl.parquet d9d51dd1cf71131667324cf1d3bcbc97 ac08d8f5025fda5f45f2e6940064fc97  # +-in_pool (G10)
  fresh $P2
  run python3 scripts/build_pool.py $P2 $PROCS        # reads the three inputs above from hard-coded paths
  rows $P2/pool.parquet 1107747
  run cp $EXT/coconut/coconut_prior.parquet $P2/
  run python3 scripts/add_pubchem_pop.py $KEY_POP $P2
fi
chk $P2/pool.parquet c433147c07bddb1b0c13f1eab2aa33b3
chk $P2/pool_fp.npy 5abdd51951456113f7f12e5a08ab61aa
chk $P2/coconut_prior.parquet f8b83ce5512deabd27431a9ae8b48c87
[ $built2 = 0 ] || run mv $P2/.building $P2/.built

# pool4 (+52,371 ChEBI / LMSD rows; build_pool_bio.py copies coconut_prior.parquet from pool2)
built4=0
if [ -f $P4/.built ]; then
  log "pool4: checking $P4"
else
  log "pool4: building $P4"; built4=1
  fresh $P4
  run python3 scripts/build_pool_bio.py $P2 $BIO $P4 $PROCS
  rows $P4/pool.parquet 1160118
  run python3 scripts/add_pubchem_pop.py $KEY_POP $P4
fi
chk $P4/pool.parquet aa45930cb164bbd684eb4a4b58c1e28b
chk $P4/pool_fp.npy a077de6357ef627dbfd4ad646346b864
chk $P4/coconut_prior.parquet f8b83ce5512deabd27431a9ae8b48c87
[ $built4 = 0 ] || run mv $P4/.building $P4/.built

# documentation-shift weights (DOC_W of the rankers; not shipped in casmi26-pool4)
[ -f $P4/doc_w.parquet ] || { log "doc_w"; run python3 -m casmi.docw $P4 $P4/doc_w.parquet; }
chk $P4/doc_w.parquet 971fd3438326ec8dfdf5479d54c8b78a
log "pool4 done; publish: copy pool.parquet, pool_fp.npy, coconut_prior.parquet to \$DS/casmi26-pool4, then"
echo "  kaggle datasets version -p \$DS/casmi26-pool4 -m pool4"
