"""Class-2 isomer discrimination: rank the hidden truth among same-formula pool isomers.

Baseline (public analog channel): max over top spectral analogs a of sim(a)^4 * Tanimoto(C, a).
Site: for candidates that are transform-derivatives of an analog, modification-site consistency of the analog's
spectrum with the query. Usage: python scripts/exp_isomer.py <n> <out.parquet>
"""
import sys, time
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.analog import AnalogLib
from casmi.gen import generate
from casmi.siteloc import ParentEvidence

n, out = int(sys.argv[1]), sys.argv[2]
L = AnalogLib()
pool = pd.read_parquet('/home/user/pub/pool/pool_meta.parquet')
ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet')
tsi = ts.set_index('inchikey14')
by_formula = pool.groupby('formula').indices
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
np_keys = set(pool.loc[pool.src == 3, 'key'])
cands = sorted(k for k, mk in zip(ts.inchikey14, ts.key) if mk in np_keys and 200 < tsi.loc[k, 'mass'] < 900)
present = set(L.key)
cands = [k for k in cands if k in present]
rng = np.random.default_rng(0)
sample = rng.choice(cands, size=min(n, len(cands)), replace=False)


def fp(s):
    m = Chem.MolFromSmiles(s)
    return fpg.GetFingerprint(m) if m is not None else None


rows = []; t0 = time.time()
for si, k in enumerate(sample):
    tk = tsi.loc[k, 'key']; tform = tsi.loc[k, 'formula']; M = tsi.loc[k, 'mass']
    iso = pool.iloc[by_formula.get(tform, [])]
    if len(iso) < 2 or tk not in set(iso.key):
        continue
    iso = iso.drop_duplicates('key').head(400)
    qidx = np.where(L.key == k)[0][:3]
    agg = {}
    for q in qidx:
        a, b = L.off[q], L.off[q + 1]
        for j, s, nm in L.search(L.mz[a:b], L.it[a:b], L.nmass[q], int(L.pol[q]), exclude_keys={k}, top=40):
            if L.key[j] not in agg or s > agg[L.key[j]][0]:
                agg[L.key[j]] = (s, j, q)
    parents = sorted(agg.items(), key=lambda x: -x[1][0])[:10]
    if not parents:
        continue
    cfps = [fp(s) for s in iso.smiles]
    base = np.zeros(len(iso)); site = np.full(len(iso), np.nan)
    keyidx = {kk: i for i, kk in enumerate(iso.key)}
    for kk, (s, j, q) in parents:
        pf = fp(L.smiles[j])
        if pf is None:
            continue
        tan = np.array([DataStructs.TanimotoSimilarity(pf, c) if c is not None else 0 for c in cfps])
        base = np.maximum(base, s ** 4 * tan)
        pm = tsi.loc[kk, 'mass'] if kk in tsi.index else L.nmass[j]
        prods, pmol = generate(L.smiles[j], M - pm)
        if not prods:
            continue
        a, b = L.off[j], L.off[j + 1]; qa, qb = L.off[q], L.off[q + 1]
        ev = ParentEvidence(pmol, L.mz[a:b], L.it[a:b], L.mz[qa:qb], L.it[qa:qb], float(M - pm), positive=(L.pol[j] == 0),
                            cache_key=L.smiles[j])
        for p in prods:
            m = Chem.MolFromSmiles(p['smiles'])
            if m is None:
                continue
            from casmi.metric import key14
            ck = key14(p['smiles'])
            if ck in keyidx:
                sc, _ = ev.score(p['sites'], p['removed'])
                v = sc + 0.5 * s
                i = keyidx[ck]
                site[i] = v if np.isnan(site[i]) else max(site[i], v)
    t = keyidx[tk]
    def rank(score):
        return int((score > score[t]).sum() + 1)
    comb = base + np.nan_to_num(site, nan=-1.0) * 0.5 + 0.5
    rows.append(dict(key=k, n_iso=len(iso), r_base=rank(base), r_comb=rank(comb), t_has_site=not np.isnan(site[t]),
                     n_site=int((~np.isnan(site)).sum())))
    if si % 25 == 0:
        df = pd.DataFrame(rows)
        print(si, f'{time.time()-t0:.0f}s', 'MRR base', round((1 / df.r_base).mean(), 4), 'comb', round((1 / df.r_comb).mean(), 4),
              'truth derivable', round(df.t_has_site.mean(), 3), flush=True)
df = pd.DataFrame(rows); df.to_parquet(out)
print('FINAL n', len(df), 'median isomers', df.n_iso.median(), 'MRR base', round((1 / df.r_base).mean(), 4),
      'comb', round((1 / df.r_comb).mean(), 4), 'top1 base', int((df.r_base == 1).sum()), 'comb', int((df.r_comb == 1).sum()),
      'truth derivable', round(df.t_has_site.mean(), 3))
sub = df[df.t_has_site]
print('on derivable subset n', len(sub), 'MRR base', round((1 / sub.r_base).mean(), 4), 'comb', round((1 / sub.r_comb).mean(), 4))
