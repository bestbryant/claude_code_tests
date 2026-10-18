"""Validation of ICEBERG same-formula re-ranking on held-out enveda-180 queries (not in MassSpecGym -> no leakage).

python scripts/exp_ice.py <rows_fp.parquet> <pool_dir> <ranker.txt> <ice_pkg> <out.parquet> [n_queries] [top_n] [threads]
Scores the simulation rows with the ranker, keeps queries whose truth is a enveda-180 structure, has a positive
[M+H]+/[M+Na]+ query spectrum and sits in the ranker's top_n, predicts ICEBERG similarity for the top_n candidates and
writes one row per (query, candidate) with ranker score, ICE score, formula and label.
"""
import json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb

rows_p, pool_dir, rk_p, pkg, out = sys.argv[1:6]
nq = int(sys.argv[6]) if len(sys.argv) > 6 else 150
top_n = int(sys.argv[7]) if len(sys.argv) > 7 else 20
threads = sys.argv[8] if len(sys.argv) > 8 else '2'
sys.path.insert(0, pkg)
from fuse import build_ice_input, run_ice

feats = open(rk_p.replace('.txt', '_feats.txt')).read().split()
D = pd.read_parquet(rows_p)
D['s'] = lgb.Booster(model_file=rk_p).predict(D[feats].astype(np.float32))
D = D.sort_values(['qid', 's'], ascending=[True, False])
D['r'] = D.groupby('qid').cumcount() + 1
P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['smiles', 'formula'])
D['smiles'] = P.smiles.values[D.pid.values]; D['formula'] = P.formula.values[D.pid.values]
cols = ['ingest_lib', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct', 'ionization_mode',
        'instrument_type', 'collision_energy_ev', 'collision_energy_orig', 'collision_energy_orig_units']
Q = D.drop_duplicates('qid')[['qid', 'q_rows', 'cls']]
Q['rows'] = Q.q_rows.map(lambda s: [int(x) for x in s.split(',')])
need = sorted({i for rr in Q.rows for i in rr})
pf = pq.ParquetFile('/home/user/data/train.parquet')
parts, base = [], 0
for g in range(pf.metadata.num_row_groups):
    n = pf.metadata.row_group(g).num_rows
    loc = [i - base for i in need if base <= i < base + n]
    if loc:
        t = pf.read_row_group(g, columns=cols).take(loc).to_pandas(); t.index = [base + i for i in loc]; parts.append(t)
    base += n
T = pd.concat(parts)
Q['lib'] = Q.rows.map(lambda rr: T.ingest_lib[rr[0]])
Q['pos'] = Q.rows.map(lambda rr: any(T.ionization_mode[i] == 'positive' and T.adduct[i] in ('[M+H]+', '[M+Na]+') for i in rr))
tr = D[D.label == 1].set_index('qid').r
Q['truth_r'] = Q.qid.map(tr).fillna(9999)
print('queries', len(Q), 'enveda', (Q.lib == 'enveda-180').mean().round(3), 'pos', Q.pos.mean().round(3))
sel = Q[(Q.lib == 'enveda-180') & Q.pos]
print('enveda+pos', len(sel), 'MRR(ranker)', (1 / sel.truth_r).where(sel.truth_r <= 25, 0).mean().round(4),
      'truth in top', top_n, (sel.truth_r <= top_n).mean().round(3))
sel = sel.sample(min(nq, len(sel)), random_state=0)
top = D[D.qid.isin(sel.qid) & (D.r <= top_n)]
test_like = []
for q in sel.itertuples():
    for i in q.rows:
        r = T.loc[i]
        test_like.append(dict(molecule_id=str(q.qid), **{c: r[c] for c in cols if c != 'ingest_lib'}))
TL = pd.DataFrame(test_like)
cands = {str(q): list(g.smiles) for q, g in top.groupby('qid')}
items = build_ice_input(TL, cands)
ice = run_ice(pkg, items, workdir=out + '.work', device='cpu', budget_s=0, site='/home/user/ext/ice_site',
              extra_args=['--threads', threads])
top = top.copy()
top['ice'] = [ice.get(str(q), {}).get(s) for q, s in zip(top.qid, top.smiles)]
top[['qid', 'cls', 'pid', 'key', 'smiles', 'formula', 's', 'r', 'label', 'ice']].to_parquet(out)
print('scored', top.ice.notna().mean().round(3))
