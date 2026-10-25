"""Split keys for the stage-1 ranker and the FP / PF models (docs/reproduce.md 1.2, G6): val_keys.txt and
rank_keys.txt, one InChIKey first block per line in draw order, no trailing newline.

python scripts/make_split_keys.py <pool.parquet> <train_meta.parquet> <out_dir>
Eligible: key in train with a test adduct and 120 < pool mass < 1300 Da; COCONUT&train = in_train & in_coco,
train-only = in_train & not in_coco. Draws are np.random.default_rng(seed).choice(sorted keys, n, replace=False).
  val  (seed 42): every enveda-np-examples key with a test adduct (250, sorted) + 750 COCONUT&train (np-examples
                  excluded) + 500 train-only
  rank (seed 7) : 5,000 COCONUT&train + 3,000 train-only, val keys excluded
Same output for pool v1, pool2 and pool4 ($W/pool*/pool.parquet). Both files are checked against the md5 of the
Oct 17 lists before anything is written.
"""
import hashlib, os, sys
import numpy as np, pandas as pd

TEST_ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-',
            '[M-H2O-H]-', '[M-2H2O+H]+']
MD5 = {'val_keys.txt': 'e2c9664748b1ffa8603d658604cf7f18', 'rank_keys.txt': '0cf23d7508808fee1ccd9b275376ab08'}

pool, meta_p, out = sys.argv[1:4]
P = pd.read_parquet(pool, columns=['key', 'in_train', 'in_coco', 'mass'])
assert not P.key.duplicated().any(), 'duplicate pool keys'
m = pd.read_parquet(meta_p, columns=['inchikey14', 'ingest_lib', 'adduct'])
assert len(m) == 2539608, len(m)
ok = set(m.inchikey14[m.adduct.isin(TEST_ADD)])
npx = sorted(set(m.inchikey14[m.ingest_lib == 'enveda-np-examples']) & ok)
assert len(npx) == 250, len(npx)
elig = P.key.isin(ok) & (P.mass > 120) & (P.mass < 1300) & (P.in_train == 1)
coco, only = set(P.key[elig & (P.in_coco == 1)]), set(P.key[elig & (P.in_coco == 0)])
print('eligible COCONUT&train', len(coco), 'train-only', len(only))


def draw(seed, parts):
    rng = np.random.default_rng(seed)
    return [str(k) for keys, n in parts for k in rng.choice(sorted(keys), n, replace=False)]


val = npx + draw(42, [(coco - set(npx), 750), (only, 500)])   # train-only draw does not exclude np-examples
assert len(set(val)) == 1500
rank = draw(7, [(coco - set(val), 5000), (only - set(val), 3000)])
assert len(set(rank)) == 8000
txt = {'val_keys.txt': '\n'.join(val).encode(), 'rank_keys.txt': '\n'.join(rank).encode()}
for name, b in txt.items():                     # check both before writing either
    h = hashlib.md5(b).hexdigest()
    assert h == MD5[name], f'{name}: md5 {h} != {MD5[name]}'
os.makedirs(out, exist_ok=True)
for name, b in txt.items():
    open(f'{out}/{name}', 'wb').write(b)
    print(name, b.count(b'\n') + 1, hashlib.md5(b).hexdigest())
