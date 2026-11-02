"""Embed scripts/build_e2_bank.py into kernels/casmi26-e2-bank/notebook.py (BANK_SRC + BANK_SHA256).

python scripts/make_e2_bank_kernel.py           rewrite the embedded copy and its sha256
python scripts/make_e2_bank_kernel.py --check   exit 1 if the embedded copy differs from the script
The copy sits in a raw triple-single-quoted string, so the script must be ASCII, contain no ''' and not end in a
backslash; the embedded text is then byte-identical to the file.
"""
import hashlib
import os
import re
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(R, 'scripts', 'build_e2_bank.py')
NB = os.path.join(R, 'kernels', 'casmi26-e2-bank', 'notebook.py')
BEGIN = "# >>> build_e2_bank.py (embedded by scripts/make_e2_bank_kernel.py; do not edit by hand)\nBANK_SRC = r'''"
END = "'''\n# <<< build_e2_bank.py\n"

src = open(SRC, 'rb').read().decode('ascii')
assert "'''" not in src and not src.rstrip('\n').endswith('\\'), 'build_e2_bank.py cannot sit in a raw string'
sha = hashlib.sha256(src.encode()).hexdigest()
nb = open(NB).read()
i, j = nb.index(BEGIN) + len(BEGIN), nb.index(END)
cur = nb[i:j]
m = re.search(r"^BANK_SHA256 = '([0-9a-f]{64})'$", nb, re.M)
in_sync = cur == src and m and m.group(1) == sha
if '--check' in sys.argv:
    print('in sync' if in_sync else f'OUT OF SYNC: embedded sha256 {hashlib.sha256(cur.encode()).hexdigest()[:12]}, '
          f'constant {m.group(1)[:12] if m else None}, script {sha[:12]}')
    sys.exit(0 if in_sync else 1)
new = nb[:i] + src + nb[j:]
new = re.sub(r"^BANK_SHA256 = '[0-9a-f]{64}'$", f"BANK_SHA256 = '{sha}'", new, count=1, flags=re.M)
open(NB, 'w').write(new)
print('embedded build_e2_bank.py', sha[:12], 'into', os.path.relpath(NB, R), '(unchanged)' if in_sync else '')
