"""Rank generated derivatives: does modification-site localisation put the truth first?

For held-out structures (spectra hidden), take top-K spectral analogs as parents, generate products with site
tracking, and report the truth's rank under several orderings.
Usage: python scripts/exp_site.py <n> <out.parquet> [keyfile] [K]
"""
import sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.analog import AnalogLib
from casmi.gen import generate
from casmi.siteloc import ParentEvidence
from casmi.spectra import modcos

n, out = int(sys.argv[1]), sys.argv[2]
keyfile = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
rng = np.random.default_rng(0)
L = AnalogLib()
pool = pd.read_parquet('/home/user/pub/pool/pool_meta.parquet')
ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet')
if keyfile:
    allowed = set(open(keyfile).read().split()); cand_keys = [k for k in ts.inchikey14 if k in allowed]
else:
    np_keys = set(pool.loc[pool.src == 3, 'key']); cand_keys = [k for k, mk in zip(ts.inchikey14, ts.key) if mk in np_keys]
ts = ts.set_index('inchikey14')
present = set(L.key)
cand_keys = [k for k in cand_keys if 200 < ts.loc[k, 'mass'] < 900 and k in present]
sample = rng.choice(cand_keys, size=min(n, len(cand_keys)), replace=False)

def ik(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None

rows = []; t0 = time.time()
for si, k in enumerate(sample):
    tmass = ts.loc[k, 'mass']
    qidx = np.where(L.key == k)[0][:3]
    agg = {}
    for q in qidx:
        a, b = L.off[q], L.off[q + 1]
        for j, s, nm in L.search(L.mz[a:b], L.it[a:b], L.nmass[q], int(L.pol[q]), exclude_keys={k}, top=40):
            if L.key[j] not in agg or s > agg[L.key[j]][0]:
                agg[L.key[j]] = (s, j, q)
    parents = sorted(agg.items(), key=lambda x: -x[1][0])[:K]
    cands = []
    for pr, (kk, (s, j, q)) in enumerate(parents):
        pm = ts.loc[kk, 'mass'] if kk in ts.index else L.nmass[j]
        prods, pmol = generate(L.smiles[j], tmass - pm)
        if not prods:
            continue
        a, b = L.off[j], L.off[j + 1]; qa, qb = L.off[q], L.off[q + 1]
        ev = ParentEvidence(pmol, L.mz[a:b], L.it[a:b], L.mz[qa:qb], L.it[qa:qb], float(tmass - pm),
                            positive=(L.pol[j] == 0))
        for g, p in enumerate(prods):
            sc, tot = ev.score(p['sites'], p['removed'])
            cands.append(dict(par=pr, psim=s, gi=g, site=sc, n_steps=len(p['combo']), smiles=p['smiles']))
    if not cands:
        rows.append(dict(key=k, n_c=0, hit=False)); continue
    C = pd.DataFrame(cands)
    C['ik'] = [ik(s) for s in C.smiles]
    C = C.drop_duplicates('ik')
    hit = (C.ik == k).any()
    res = dict(key=k, n_c=len(C), hit=bool(hit), best_psim=parents[0][1][0])
    if hit:
        orders = {
            'gen': C.sort_values(['par', 'gi']),
            'site_in_par': C.sort_values(['par', 'site', 'gi'], ascending=[True, False, True]),
            'site_global': C.assign(z=C.site + 0.5 * C.psim - 0.15 * (C.n_steps - 1)).sort_values('z', ascending=False),
            'steps_site': C.assign(z=C.site * C.psim - 0.3 * (C.n_steps - 1)).sort_values('z', ascending=False),
        }
        for name, o in orders.items():
            res['r_' + name] = int(np.where(o.ik.values == k)[0][0]) + 1
        t = C[C.ik == k].iloc[0]
        res.update(t_par=int(t.par), t_site=float(t.site), t_steps=int(t.n_steps))
    rows.append(res)
    if si % 20 == 0:
        df = pd.DataFrame(rows); h = df[df.hit]
        msg = ' '.join(f"{c[2:]}:{(1/h[c]).where(h[c] <= 25, 0).sum()/len(df):.3f}" for c in h.columns if c.startswith('r_')) if len(h) else ''
        print(si, f'{time.time()-t0:.0f}s', 'hit', round(df.hit.mean(), 3), 'MRR', msg, flush=True)
df = pd.DataFrame(rows); df.to_parquet(out)
h = df[df.hit]
print('FINAL n', len(df), 'hit', df.hit.mean().round(3), 'median n_c', df.n_c.median())
for c in [c for c in h.columns if c.startswith('r_')]:
    r = h[c]
    print(c, 'MRR(all)', round(((1 / r).where(r <= 25, 0)).sum() / len(df), 4), 'top1', int((r == 1).sum()),
          'top5', int((r <= 5).sum()), 'top25', int((r <= 25).sum()))
