"""Class-3 reachability: for held-out structures (all their spectra hidden), how often does the public
transform generator (derive.py) produce the truth from the top-K spectral analogs?

Usage: python scripts/exp_reach.py <n_structs> <out.parquet> [keyfile]
keyfile: optional text file of inchikey14 to sample from (e.g. PubChem-absent structures).
"""
import sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.analog import AnalogLib
from casmi import derive as derive  # local copy with the public transform table

n, out = int(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(0)
L = AnalogLib()
pool = pd.read_parquet('/home/user/pub/pool/pool_meta.parquet')
ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet')
lib_keys = pd.Series(L.key)
if len(sys.argv) > 3:
    allowed = set(open(sys.argv[3]).read().split())
    cand_keys = [k for k in ts.inchikey14 if k in allowed]
else:
    np_keys = set(pool.loc[pool.src == 3, 'key'])
    cand_keys = [k for k, mk in zip(ts.inchikey14, ts.key) if mk in np_keys]
ts = ts.set_index('inchikey14')
cand_keys = [k for k in cand_keys if 200 < ts.loc[k, 'mass'] < 900]
present = set(lib_keys.unique())
cand_keys = [k for k in cand_keys if k in present]
print('candidates', len(cand_keys), flush=True)
sample = rng.choice(cand_keys, size=min(n, len(cand_keys)), replace=False)

rows = []
t0 = time.time()
for si, k in enumerate(sample):
    truth_smi = ts.loc[k, 'smiles']; tmass = ts.loc[k, 'mass']
    qidx = np.where(L.key == k)[0][:3]
    agg = {}
    for q in qidx:
        a, b = L.off[q], L.off[q + 1]
        res = L.search(L.mz[a:b], L.it[a:b], L.nmass[q], int(L.pol[q]), exclude_keys={k}, top=40)
        for j, s, nm in res:
            kk = L.key[j]
            if kk not in agg or s > agg[kk][0]:
                agg[kk] = (s, j, nm)
    parents = sorted(agg.items(), key=lambda x: -x[1][0])[:20]
    hit_rank, n_prod, best_sim = 0, 0, (parents[0][1][0] if parents else 0)
    tik = Chem.MolToInchiKey(Chem.MolFromSmiles(truth_smi))[:14]
    for pr, (kk, (s, j, nm)) in enumerate(parents):
        psmi = L.smiles[j]
        d = tmass - L.nmass[j] + (L.nmass[j] - ts.loc[kk, 'mass'] if kk in ts.index else 0)
        pm = ts.loc[kk, 'mass'] if kk in ts.index else L.nmass[j]
        d = tmass - pm
        prods = derive.derive(psmi, d, tol=0.003, max_steps=2, max_per_analog=300)
        n_prod += len(prods)
        for p in prods:
            m = Chem.MolFromSmiles(p)
            if m is not None and Chem.MolToInchiKey(m)[:14] == tik:
                hit_rank = pr + 1
                break
        if hit_rank:
            break
    rows.append(dict(key=k, mass=tmass, best_sim=best_sim, n_par=len(parents), hit_parent_rank=hit_rank,
                     n_prod=n_prod, parent_sims=[p[1][0] for p in parents[:5]],
                     parent_deltas=[float(tmass - (ts.loc[p[0], 'mass'] if p[0] in ts.index else 0)) for p in parents[:5]]))
    if si % 10 == 0:
        df = pd.DataFrame(rows)
        print(si, f'{time.time()-t0:.0f}s', 'hit', (df.hit_parent_rank > 0).mean().round(3),
              'hit@5', ((df.hit_parent_rank > 0) & (df.hit_parent_rank <= 5)).mean().round(3), flush=True)
pd.DataFrame(rows).to_parquet(out)
df = pd.DataFrame(rows)
print('FINAL n', len(df), 'hit', (df.hit_parent_rank > 0).mean(), 'by parent rank',
      df.hit_parent_rank.value_counts().sort_index().to_dict())
