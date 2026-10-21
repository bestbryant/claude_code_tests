"""Policy simulation for a tier proposal channel (Enamine tier, casmi/pcchannel.PubChemTier format) on validation queries.

python scripts/sim_enam_policy.py <rows_fp.parquet> <reach_c rows.parquet> <tier.npz> <pf checkpoint> <out.parquet> [procs]
Two populations, every query's tier window scored with the PeakFormer exactly as at test time (score = z(f.z) + z(cos)):
  present: validation queries whose answer is in the pool (reach_c rows: class 1 + reachable class 2); the channel
           only offers decoys (validation truths are training structures, absent from the tier);
  other:   class-2 queries whose truth is Enamine-deposited and outside COCONUT / PubChemLite / ChEBI-LMSD (unreachable
           through the pool), truth removed from the pool list and placed into its tier window.
Per query: main-list truth rank (v11l stage 1, present only), channel truth rank (other only), rel = best window f.z -
best pool f.z (pool top 30, same model), lib_max, top proposal score. Policies are evaluated by scripts/eval_enam_policy.py.
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from casmi.keyset import pack

rows_p, reach_p, tier_p, pf_p, out_p = sys.argv[1:6]
procs = int(sys.argv[6]) if len(sys.argv) > 6 else 4
W = '/home/user/work'
D = pd.read_parquet(rows_p)
D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
f = open(f'{W}/own_models/ret_ranker_v11l_feats.txt').read().split('\n')
D['s1'] = lgb.Booster(model_file=f'{W}/own_models/ret_ranker_v11l.txt').predict(D[f].astype(float).values)
present = set(pd.read_parquet(reach_p, columns=['qid']).qid.unique())
t = D[(D.label == 1) & (D.cls == 2)].drop_duplicates('qid')
enam = np.load('/home/user/ext/pubchem/enamine_keys.npy')
t = t[np.isin(pack(list(t.truth)), enam) & (t.in_coco == 0) & (t.in_pcl == 0) & (t.in_bio == 0)]
other = set(t.qid)
D = D[D.qid.isin(present | other)]
P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['smiles', 'mass'])
WT = pd.read_parquet(f'{W}/pool4/doc_w.parquet').set_index('key').w
G = {q: g for q, g in D.groupby('qid', sort=False)}
TIER = SC = ST = None


def init():
    global TIER, SC, ST
    import torch
    torch.set_num_threads(1)
    from casmi.pcchannel import PubChemTier
    from casmi.peakformer import PFEnsemble
    from casmi.spectra import Store
    TIER = PubChemTier(tier_p); SC = PFEnsemble([pf_p]); ST = Store(f'{W}/train_store')


def one(q):
    from casmi.prep_meta import inst_family
    g = G[q]
    kind = 'other' if q in other else 'present'
    tr = g[g.label == 1]
    meta = ST.meta
    spectra = []
    for i in map(int, g.q_rows.iloc[0].split(',')):
        mz, it = ST.peaks(i); r = meta.loc[i]; ce = r.collision_energy_ev
        spectra.append(dict(mz=mz, it=it, prec=float(r.precursor_mz), adduct=r.adduct,
                            ce=float(np.mean(ce)) if ce is not None and len(ce) else -1.0, inst=inst_family(r.instrument_type)))
    M = float(P.mass.values[tr.pid.iloc[0]]) if len(tr) else float(P.mass.values[g.pid.iloc[0]])
    lst = g if kind == 'present' else g[g.label == 0]
    lst = lst.sort_values('s1', ascending=False).head(30)
    rec = dict(qid=q, kind=kind, cls=int(g.cls.iloc[0]), lib_max=float(lst.lib.max()) if len(lst) else 0.0,
               pool_fz=float(lst.pf_bern.max()) if len(lst) else -1e9,
               w=float(np.clip(WT.get(tr.truth.iloc[0], 1.0), 0.02, 20)) if (len(tr) and int(g.cls.iloc[0]) == 2) else 1.0)
    lab = lst.label.values
    h = np.flatnonzero(lab == 1)
    rec['main_rank'] = int(h[0] + 1) if len(h) else 0
    idx = TIER.window(M, n_pop=50000)
    smi = [TIER.smiles(i) for i in idx]; ids = [-1 - int(i) for i in idx]
    if kind == 'other':
        smi.append(P.smiles.values[tr.pid.iloc[0]]); ids.append(int(tr.pid.iloc[0]))
    rec['n_win'] = len(smi)
    if not smi:
        rec.update(ch_rank=0, rel=np.nan, s_top=np.nan, s_truth=np.nan)
        return rec
    cos, fz = SC.score(spectra, ids, smi)
    zs = lambda v: (v - v.mean()) / (v.std() or 1.0)
    s = zs(fz) + zs(cos)
    rec['rel'] = float(fz.max() - rec['pool_fz'])
    rec['s_top'] = float(s.max())
    if kind == 'other':
        rec['ch_rank'] = int(1 + (s[:-1] > s[-1]).sum()); rec['s_truth'] = float(s[-1])
        rec['rel_wo'] = float(fz[:-1].max() - rec['pool_fz']) if len(fz) > 1 else np.nan
    else:
        rec['ch_rank'] = 0
    return rec


if __name__ == '__main__':
    qs = list(G)
    with Pool(procs, initializer=init) as pool:
        res = list(pool.imap_unordered(one, qs, chunksize=4))
    R = pd.DataFrame(res)
    R.to_parquet(out_p)
    print(R.groupby('kind').agg(n=('qid', 'size'), win=('n_win', 'median'), rel=('rel', 'median')))
    o = R[R.kind == 'other']
    print('other: channel MRR@25', round(float(np.where((o.ch_rank > 0) & (o.ch_rank <= 25), 1 / o.ch_rank.clip(lower=1), 0).mean()), 4),
          'top1', round(float((o.ch_rank == 1).mean()), 3))
