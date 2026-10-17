"""Evaluate NovelEngine (derivatives + conjugates) on held-out PubChem-absent structures (class-3 proxy).

Usage: python scripts/exp_novel.py <n> <out.parquet> [seed]
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.analog import AnalogLib
from casmi.conjugate import BlockLib
from casmi.keyset import KeySet, pack
from casmi.metric import key14
from casmi.novel import NovelEngine, interleave
from casmi.spectra import Store

n, out = int(sys.argv[1]), sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet')
L = AnalogLib()
P = pd.read_parquet('/home/user/work/blocks_pieces.parquet'); Pk = P.groupby('key')
B = BlockLib(P, min_count=1)
pc = KeySet.load('/home/user/ext/pc_keys_u64.npy')
known = KeySet(np.unique(pack(list(ts.inchikey14))))
E = NovelEngine(L, B, pc, known, dict(zip(ts.inchikey14, ts.mass)))
st = Store('/home/user/work/train_store'); meta = st.meta
TEST_ADD = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+CH2O2-H]-', '[M+K]+', '[M+Cl]-']
absent = open('/home/user/work/train_not_pc.txt').read().split()
tsi = ts.set_index('inchikey14')
absent = [k for k in absent if 150 < tsi.loc[k, 'mass'] < 1000]
rng = np.random.default_rng(seed)
keys = rng.choice(absent, size=min(n, len(absent)), replace=False)
grp = meta[meta.adduct.isin(TEST_ADD)].groupby('inchikey14').groups
rows = []; t0 = time.time()
for si, k in enumerate(keys):
    if k not in grp:
        continue
    idx = list(grp[k])
    rng.shuffle(idx)
    idx = idx[:int(rng.integers(1, 6))]          # test-like: 1-5 spectra per molecule
    spectra = []
    for i in idx:
        mz, it = st.peaks(i); r = meta.loc[i]
        spectra.append(dict(mz=mz, it=it, prec=r.precursor_mz, adduct=r.adduct, positive=r.adduct.endswith('+')))
    counts = B.loo(Pk.get_group(k)) if k in Pk.groups else None
    deriv, conj, M = E.predict(spectra, M=tsi.loc[k, 'mass'], exclude_keys={k}, loo_counts=counts, allow_keys={k})
    tk = key14(tsi.loc[k, 'smiles'])
    rk = lambda lst: next((r + 1 for r, x in enumerate(lst[:25]) if x[2] == tk), 0)
    res = dict(key=k, n_d=len(deriv), n_c=len(conj), r_d=rk(deriv), r_c=rk(conj),
               r_dc=rk(interleave(deriv, conj, first='d')), r_cd=rk(interleave(deriv, conj, first='c')),
               top_d=deriv[0][1] if deriv else np.nan, top_c=conj[0][1] if conj else np.nan)
    rows.append(res)
    if si % 25 == 0:
        df = pd.DataFrame(rows)
        f = lambda c: round(float((1 / df[c].replace(0, np.inf)).mean()), 4)
        print(si, f'{time.time()-t0:.0f}s', 'MRR d', f('r_d'), 'c', f('r_c'), 'dc', f('r_dc'), 'cd', f('r_cd'), flush=True)
df = pd.DataFrame(rows); df.to_parquet(out)
f = lambda c: round(float((1 / df[c].replace(0, np.inf)).mean()), 4)
print('FINAL n', len(df), 'MRR d', f('r_d'), 'c', f('r_c'), 'dc', f('r_dc'), 'cd', f('r_cd'),
      'top1 d', int((df.r_d == 1).sum()), 'c', int((df.r_c == 1).sum()), 'any top25', int(((df.r_d > 0) | (df.r_c > 0)).sum()))
