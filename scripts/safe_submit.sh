#!/usr/bin/env bash
# Submit a kernel version only after checking its committed output is sane.
#   scripts/safe_submit.sh <kernel-slug> <version> "<message>"
# Refuses when fewer than 90% of rows carry real candidates (not the 'C' / 'CCO' fallback) or the log shows failures.
# When the output download fails (the output endpoint was rate-limited for hours on Oct 25), the check falls back to
# the execution log (`kaggle kernels logs`, another endpoint): 'rows with candidates N of M' with N / M >= 0.9 and no
# Traceback.
set -euo pipefail
SLUG=$1; VER=$2; MSG=$3
D=$(mktemp -d)
# only the submission and the log: some kernels also leave large library files in /kaggle/working
kaggle kernels output "bryantyu/$SLUG" -p "$D" --file-pattern '(submission\.csv|\.log)$' --page-size 200 > /dev/null 2>&1 || true
if [ ! -f "$D/submission.csv" ]; then
  echo "output download failed: checking the execution log instead"
  kaggle kernels logs "bryantyu/$SLUG" 2>/dev/null | grep -v Warning > "$D/kernel_logs.txt" || true
fi
python3 - "$D" <<'EOF'
import glob, os, re, sys
d = sys.argv[1]
if os.path.exists(f'{d}/submission.csv'):
    import pandas as pd
    sub = pd.read_csv(f'{d}/submission.csv')
    real = sub.smiles.map(lambda s: s not in ('C', 'CCO') and isinstance(s, str) and len(s) > 3).mean()
    logs = ''.join(open(f).read() for f in glob.glob(f'{d}/*.log'))
    fails = logs.count(' fail ')
    print(f'rows {len(sub)}  real-candidate rows {real:.3f}  failures in log {fails}')
    if real < 0.9 or fails > len(sub) * 0.05:
        sys.exit('REFUSING TO SUBMIT: output looks broken')
else:
    logs = open(f'{d}/kernel_logs.txt').read() if os.path.exists(f'{d}/kernel_logs.txt') else ''
    m = re.findall(r'rows with candidates (\d+) of (\d+)', logs)
    if not m:
        sys.exit('REFUSING TO SUBMIT: neither the output nor a log line "rows with candidates" is available')
    n, tot = map(int, m[-1])
    print(f'log check: rows with candidates {n} of {tot}, Traceback in log: {"Traceback" in logs}')
    if n < 0.9 * tot or 'Traceback' in logs:
        sys.exit('REFUSING TO SUBMIT: log looks broken')
EOF
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k "bryantyu/$SLUG" -v "$VER" -f submission.csv -m "$MSG" 2>&1 | tail -1
rm -rf "$D"
