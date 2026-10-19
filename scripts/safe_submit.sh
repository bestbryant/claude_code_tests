#!/usr/bin/env bash
# Submit a kernel version only after checking its committed output is sane.
#   scripts/safe_submit.sh <kernel-slug> <version> "<message>"
# Refuses when fewer than 90% of rows carry real candidates (not the 'C' / 'CCO' fallback) or the log shows failures.
set -euo pipefail
SLUG=$1; VER=$2; MSG=$3
D=$(mktemp -d)
# only the submission and the log: some kernels also leave large library files in /kaggle/working
kaggle kernels output "bryantyu/$SLUG" -p "$D" --file-pattern '(submission\.csv|\.log)$' --page-size 200 > /dev/null 2>&1
python3 - "$D" <<'EOF'
import glob, sys
import pandas as pd
d = sys.argv[1]
sub = pd.read_csv(f'{d}/submission.csv')
real = sub.smiles.map(lambda s: s not in ('C', 'CCO') and isinstance(s, str) and len(s) > 3).mean()
logs = ''.join(open(f).read() for f in glob.glob(f'{d}/*.log'))
fails = logs.count(' fail ')
print(f'rows {len(sub)}  real-candidate rows {real:.3f}  failures in log {fails}')
if real < 0.9 or fails > len(sub) * 0.05:
    sys.exit('REFUSING TO SUBMIT: output looks broken')
EOF
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k "bryantyu/$SLUG" -v "$VER" -f submission.csv -m "$MSG" 2>&1 | tail -1
rm -rf "$D"
