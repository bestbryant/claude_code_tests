"""Conjugate engine on held-out PubChem-absent conjugates: rank of the true (acid, partner) pair and of the
true structure among joined products. Building blocks are counted with the held-out structure excluded.
Usage: python scripts/exp_conj_rank.py <n> <out.parquet>
"""
import sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.conjugate import BlockLib, candidate_pairs, join
from casmi.spectra import Store, clean_peaks

n, out = int(sys.argv[1]), sys.argv[2]
P = pd.read_parquet('/home/user/work/blocks_pieces.parquet')
c = pd.read_parquet('/home/user/work/conj_abs.parquet'); conj = c[c.n_known == 2]
ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet').set_index('inchikey14')
st = Store('/home/user/work/train_store')
meta = st.meta
TEST_ADD = {'[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+'}
rng = np.random.default_rng(0)
keys = rng.choice(conj.key.values, size=min(n, len(conj)), replace=False)
lib = BlockLib(P, min_count=1)
Pk = P.groupby('key')
rows = []; t0 = time.time()
for si, k in enumerate(keys):
    counts = lib.loo(Pk.get_group(k))   # pieces from all other structures (leave-one-out)
    mrows = meta.index[(meta.inchikey14 == k) & meta.adduct.isin(TEST_ADD)]
    if len(mrows) == 0:
        continue
    # prefer [M+H]+, merge up to 4 spectra by max intensity
    sub = meta.loc[mrows]
    pos = sub[sub.adduct == '[M+H]+']
    use = pos if len(pos) else sub
    positive = use.adduct.iloc[0].endswith('+')
    peaks = {}
    for i in use.index[:4]:
        mz, it = st.peaks(i)
        a, b = clean_peaks(mz, it, use.loc[i, 'precursor_mz'], rel_floor=0.005, top_k=64)
        for x, y in zip(np.round(a, 3), b):
            peaks[x] = max(peaks.get(x, 0), y)
    qmz = np.array(sorted(peaks)); qit = np.array([peaks[x] for x in qmz])
    M = ts.loc[k, 'mass']
    pairs = candidate_pairs(lib, M, qmz, qit, positive=positive, top=60, counts=counts)
    truth = c.set_index('key').loc[k]
    tp = {truth.a, truth.b}
    pr = next((r + 1 for r, p in enumerate(pairs) if {p[1], p[2]} == tp), 0)
    # structure rank: join top pairs in order
    srank, seen = 0, []
    for p in pairs[:30]:
        for s in join(p[1], p[2]):
            m = Chem.MolFromSmiles(s)
            kk = Chem.MolToInchiKey(m)[:14] if m is not None else None
            if kk in seen:
                continue
            seen.append(kk)
            if kk == k and not srank:
                srank = len(seen)
        if len(seen) > 100:
            break
    rows.append(dict(key=k, n_pairs=len(pairs), pair_rank=pr, struct_rank=srank, positive=positive))
    if si % 25 == 0:
        df = pd.DataFrame(rows)
        sr = df.struct_rank
        print(si, f'{time.time()-t0:.0f}s', 'pair@1', round((df.pair_rank == 1).mean(), 3), 'pair@10',
              round(((df.pair_rank > 0) & (df.pair_rank <= 10)).mean(), 3),
              'struct MRR', round(((1 / sr.replace(0, np.inf)).where(sr <= 25, 0)).mean(), 3), flush=True)
df = pd.DataFrame(rows); df.to_parquet(out)
sr = df.struct_rank
print('FINAL n', len(df), 'pair@1', (df.pair_rank == 1).mean(), 'pair found', (df.pair_rank > 0).mean(),
      'struct@1', (sr == 1).mean(), 'struct top25', ((sr > 0) & (sr <= 25)).mean(),
      'struct MRR', ((1 / sr.replace(0, np.inf)).where(sr <= 25, 0)).mean())
