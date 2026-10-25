#!/bin/bash
# casmi26-code (Kaggle dataset bryantyu/casmi26-code) from a clean git tree (docs/reproduce.md G16): every tracked
# casmi/*.py at HEAD plus the trainer scripts renamed into the package (scripts/train_fp.py -> casmi/train_fp_script.py,
# train_pf.py, train_pf_xla.py, prep_syn_data.py likewise) and casmi/MANIFEST.txt (commit, sha256 of every shipped file;
# check with: cd casmi && grep -v '^#' MANIFEST.txt | sha256sum -c). Files come from `git show HEAD:<path>`, so the
# dataset is exactly the commit named in the manifest. v38 (the LB run) omitted casmi/hlprior.py; this ships all of casmi/.
#   scripts/push_code.sh                     dry run: assemble into $OUT (default: new mktemp dir), diff against $DS copy
#   scripts/push_code.sh --push [message]    assemble into /home/user/ds/casmi26-code, kaggle datasets version
# Refuses on any change (staged, unstaged, untracked) under casmi/ or scripts/; --push also refuses a commit that is on
# no remote-tracking branch (ALLOW_UNPUSHED=1 overrides).
set -euo pipefail
PUSH=0; [ "${1:-}" = --push ] && { PUSH=1; shift; }
R=$(cd "$(dirname "$0")/.." && pwd); DS=/home/user/ds/casmi26-code
RENAME="train_fp train_pf train_pf_xla prep_syn_data"    # scripts/<n>.py -> casmi/<n>_script.py
OUT=${OUT:+$(realpath -m "$OUT")}
cd "$R"

dirty=$(git status --porcelain --untracked-files=all -- casmi scripts)
[ -z "$dirty" ] || { echo "REFUSING: uncommitted changes under casmi/ or scripts/ (commit or stash them):"; echo "$dirty"; exit 1; }
C=$(git rev-parse HEAD)
remote=$(git branch -r --contains "$C" | awk '!/->/ && !r {r = $1} END {print r}')
if [ -z "$remote" ]; then
  echo "WARNING: commit ${C:0:7} is on no remote-tracking branch (push it so the manifest commit is public)"
  [ $PUSH = 1 ] && [ "${ALLOW_UNPUSHED:-0}" != 1 ] && { echo "REFUSING to push (ALLOW_UNPUSHED=1 overrides)"; exit 1; }
fi

if [ $PUSH = 1 ]; then D=$DS; S=$DS/casmi.new; rm -rf "$S"
else
  D=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/casmi26-code.XXXXXX")}; S=$D/casmi
  [ -z "$(ls -A "$D" 2>/dev/null)" ] || { echo "REFUSING: OUT=$D is not empty"; exit 1; }
  mkdir -p "$D"; cp "$DS/dataset-metadata.json" "$D/"
fi
mkdir -p "$S"

# 1. files at HEAD
py=$(git ls-tree --name-only HEAD casmi/ | grep '\.py$')
for f in $py; do git show "HEAD:$f" > "$S/$(basename "$f")"; done
for n in $RENAME; do
  [ -e "$S/${n}_script.py" ] && { echo "name clash: casmi/${n}_script.py is tracked"; exit 1; }
  git show "HEAD:scripts/$n.py" > "$S/${n}_script.py"
done
n_py=$(echo "$py" | wc -l); n_all=$(ls "$S" | wc -l)
[ "$n_all" -eq $((n_py + $(echo $RENAME | wc -w))) ] || { echo "file count $n_all != $n_py + renamed"; exit 1; }

# 2. manifest (deterministic for a given commit)
files=$(cd "$S" && ls | LC_ALL=C sort)
{
  echo "# bryantyu/casmi26-code assembled by scripts/push_code.sh"
  echo "# commit $C ($(git log -1 --format=%cI "$C")) ${remote:-not-on-remote}"
  echo "# casmi/*.py at that commit ($n_py files) + renamed: $(for n in $RENAME; do echo "${n}_script.py <- scripts/$n.py"; done | paste -sd, | sed 's/,/, /g')"
  echo "# check: cd casmi && grep -v '^#' MANIFEST.txt | sha256sum -c"
  (cd "$S" && sha256sum $files)
} > "$S/MANIFEST.txt"
(cd "$S" && grep -v '^#' MANIFEST.txt | sha256sum -c --quiet)
echo "assembled $n_all files + MANIFEST.txt from ${C:0:7} in $S (manifest sha256 $(sha256sum "$S/MANIFEST.txt" | cut -c1-12))"

# 3. what a push changes relative to the current local copy of the dataset
cur=$DS/casmi
diff -rq "$cur" "$S" | sed -e "s|$S|new|g" -e "s|$cur|ds|g" || true

if [ $PUSH = 0 ]; then
  echo "dry run: nothing pushed; to push: scripts/push_code.sh --push [message]"
  echo "would run: kaggle datasets version -p $DS -m \"${1:-code @${C:0:7}}\" --dir-mode zip -q"
  exit 0
fi
rm -rf "$DS/casmi.old"; [ ! -d "$DS/casmi" ] || mv "$DS/casmi" "$DS/casmi.old"; mv "$S" "$DS/casmi"; rm -rf "$DS/casmi.old"
[ "$(ls -A "$DS" | tr '\n' ' ')" = "casmi dataset-metadata.json " ] || { echo "unexpected entries in $DS: $(ls -A "$DS")"; exit 1; }
kaggle datasets version -p "$DS" -m "${1:-code @${C:0:7}}" --dir-mode zip -q 2>&1 | grep -v Warning | tail -2
echo "pushed ${C:0:7}; record the new version number (kaggle datasets status bryantyu/casmi26-code, Data tab)"
