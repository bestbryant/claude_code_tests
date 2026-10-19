"""Forward-model jobs for simulated queries: stage-1 top-k candidates + the query spectra, in runner input format.

python scripts/build_fwd_jobs.py <rows_fp.parquet> <ranker.txt | col:<score column>> <pool_dir> <k> <out_prefix> [n_shards] [--reach-pmid]
Writes <out_prefix>.top.parquet (qid, pid, key, smiles, formula, s1, s1_r + all row features) and
<out_prefix>.jobs_<i>.json (runner items, mid = str(qid)). With --reach-pmid only class-1 queries and class-2
queries whose truth is reachable without train.parquet are kept (the ranker-training filter).
--have=<glob> skips (query, smiles) pairs already scored in earlier runner outputs (incremental re-runs).
"""
import json, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.fwdfeat import top_k
from casmi.fwdrun import build_items

rows_p, rk_p, pool_dir, k, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
n_sh = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith('--') else 1
D = pd.read_parquet(rows_p)
if '--reach-pmid' in sys.argv:
    t = D[D.label == 1].set_index('qid')
    reach = (t.in_coco == 1) | (t.in_pcl == 1) | (t.pc_pmid > 0)
    D = D[(D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)]
if rk_p.startswith('col:'):            # precomputed stage-1 score column (e.g. out-of-fold scores)
    D['s_stage1'] = D[rk_p[4:]]
else:
    feats = open(rk_p.replace('.txt', '_feats.txt')).read().split()
    D['s_stage1'] = lgb.Booster(model_file=rk_p).predict(D[feats].astype(np.float32))
T = top_k(D, 's_stage1', k)
P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['smiles', 'formula'])
T['smiles'] = P.smiles.values[T.pid.values]; T['formula'] = P.formula.values[T.pid.values]
T.to_parquet(out + '.top.parquet')
print('queries', T.qid.nunique(), 'rows', len(T), 'truth in top', k, T.groupby('qid').label.max().mean().round(3))

cols = ['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct', 'ionization_mode', 'instrument_type',
        'collision_energy_ev', 'collision_energy_orig', 'collision_energy_orig_units']
Q = T.drop_duplicates('qid')[['qid', 'q_rows']]
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
S = pd.concat(parts)
TL = pd.DataFrame([dict(molecule_id=str(q.qid), **S.loc[i, cols].to_dict()) for q in Q.itertuples() for i in q.rows])
cands = {str(q): list(g.smiles) for q, g in T.groupby('qid', sort=False)}
have_pat = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--have=')), None)
if have_pat:
    import glob
    have = {}
    for f in glob.glob(have_pat):
        for q, d in json.load(open(f)).items():
            have.setdefault(q, set()).update(s for s, v in d.items() if v is not None)
    n0 = sum(len(v) for v in cands.values())
    cands = {q: [s for s in v if s not in have.get(q, ())] for q, v in cands.items()}
    cands = {q: v for q, v in cands.items() if v}
    print('incremental: candidates', n0, '->', sum(len(v) for v in cands.values()))
items = [it for it in build_items(TL, cands) if it['spectra']]
print('items with a covered spectrum', len(items), 'of', len(cands), 'candidates', sum(len(it['cands']) for it in items))
for i in range(n_sh):
    json.dump(items[i::n_sh], open(f'{out}.jobs_{i}.json', 'w'))
