"""Training / evaluation rows for the channel re-ranker (casmi/pcchannel.rerank_feats) on a proposal tier.

python scripts/channel_rows.py <rows_fp.parquet> <tier.npz> <pf checkpoint> <out.parquet> [procs] [max_queries] [noreach|present:<reach_c rows>]
present:<rows>: the answer-present population instead (queries of the given reach_c rows: class 1 + reachable class 2),
windows around the precursor-derived mass without the truth (null proposals), for the policy simulation.
Queries: simulated class-2 queries whose truth is Enamine-deposited (the population the Enamine / purchasable tier
serves; noreach = only truths outside COCONUT / PubChemLite / ChEBI-LMSD). The truth (a training structure, absent from
the tier) is placed into its tier window, the window scored by the PeakFormer (score = z(f.z) + z(cos)), and the best
100 proposals get the re-ranker evidence; the analog search hides the truth's key (no public spectra). One row per
(query, proposal) with label = truth; queries whose truth misses the top 100 are kept as all-negative groups.
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from casmi.keyset import pack

rows_p, tier_p, pf_p, out_p = sys.argv[1:5]
procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
maxq = int(sys.argv[6]) if len(sys.argv) > 6 else 10 ** 9
noreach = len(sys.argv) > 7 and sys.argv[7] == 'noreach'
present = sys.argv[7].split(':', 1)[1] if len(sys.argv) > 7 and sys.argv[7].startswith('present:') else None
W = '/home/user/work'
ELECTRON = 0.00054858
D = pd.read_parquet(rows_p, columns=['qid', 'pid', 'label', 'cls', 'truth', 'q_rows', 'in_coco', 'in_pcl', 'in_bio'])
t = D[(D.label == 1) & (D.cls == 2)].drop_duplicates('qid')
t = t[np.isin(pack(list(t.truth)), np.load('/home/user/ext/pubchem/enamine_keys.npy'))]
if present:
    t = D[(D.label == 1) & D.qid.isin(pd.read_parquet(present, columns=['qid']).qid.unique())].drop_duplicates('qid')
t['noreach'] = (t.in_coco == 0) & (t.in_pcl == 0) & (t.in_bio == 0)
if noreach:
    t = t[t.noreach]
t = t.sample(min(maxq, len(t)), random_state=0)
P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['key', 'smiles', 'mass'])
TIER = SC = ST = L = FP = KI = None


def init():
    global TIER, SC, ST, L, FP, KI
    import torch
    torch.set_num_threads(1)
    from casmi.analog import AnalogLib
    from casmi.pcchannel import PubChemTier
    from casmi.peakformer import PFEnsemble
    from casmi.spectra import Store
    TIER = PubChemTier(tier_p); SC = PFEnsemble([pf_p]); ST = Store(f'{W}/train_store'); L = AnalogLib(f'{W}/analog_lib.npz')
    FP = np.load(f'{W}/pool4/pool_fp.npy', mmap_mode='r'); KI = {k: i for i, k in enumerate(P.key.values)}


def one(r):
    from casmi.pcchannel import rerank_feats
    from casmi.prep_meta import inst_family
    from casmi.spectra import clean_peaks
    try:
        meta = ST.meta
        raw, Q = [], []
        for i in map(int, r['q_rows'].split(',')):
            mz, it = ST.peaks(i); m = meta.loc[i]; ce = m.collision_energy_ev
            raw.append(dict(mz=mz, it=it, prec=float(m.precursor_mz), adduct=m.adduct,
                            ce=float(np.mean(ce)) if ce is not None and len(ce) else -1.0, inst=inst_family(m.instrument_type)))
            prec = float(m.precursor_mz) - (ELECTRON if (m.ingest_lib == 'enveda-180' and m.adduct.endswith('+')) else 0.0)
            a, b = clean_peaks(mz, it, prec, rel_floor=0.005, top_k=64)
            if len(a) >= 3:
                Q.append(dict(mz=a, it=b, pol=0 if m.adduct.endswith('+') else 1))
        M = float(P.mass.values[r['pid']])
        if present:                # test-like: window around the precursor-derived mass, no answer in the tier
            from casmi.spectra import neutral_mass
            M = float(np.nanmedian([neutral_mass(float(meta.loc[i].precursor_mz) - (ELECTRON if (meta.loc[i].ingest_lib == 'enveda-180'
                                    and meta.loc[i].adduct.endswith('+')) else 0.0), meta.loc[i].adduct) for i in map(int, r['q_rows'].split(','))]))
        idx = TIER.window(M, n_pop=50000)
        if present and len(idx) == 0:
            return None
        smi = [TIER.smiles(i) for i in idx] + ([] if present else [P.smiles.values[r['pid']]])
        cos, fz = SC.score(raw, [-1 - int(i) for i in idx] + ([] if present else [int(r['pid'])]), smi)
        zs = lambda v: (v - v.mean()) / (v.std() or 1.0)
        lab = np.zeros(len(idx)) if present else np.r_[np.zeros(len(idx)), 1]
        f = pd.DataFrame(dict(smiles=smi, pc_cos=cos, pc_fz=fz, label=lab.astype(np.int8)))
        f['pc_score'] = zs(f.pc_fz.values) + zs(f.pc_cos.values)
        f = f.sort_values('pc_score', ascending=False).reset_index(drop=True)
        rank0 = 0 if present else int(np.flatnonzero(f.label.values == 1)[0]) + 1
        # class 2: the answer has no library spectra (key hidden); present class 1: its other spectra stay in the
        # library as at test time (approximated by no hiding; the query's own spectra may then match themselves)
        g = rerank_feats(f, Q, M, L, FP, KI, hide=set() if (present and int(r['cls']) == 1) else {r['truth']})
        return g.drop(columns=['smiles']).assign(qid=int(r['qid']), rank0=rank0, noreach=bool(r['noreach']))
    except Exception as e:
        print('fail', r['qid'], repr(e), flush=True)
        return None


if __name__ == '__main__':
    res = []
    # workers recycled every 40 queries: the PeakFormer fingerprint cache grows with every window (OOM otherwise)
    with Pool(procs, initializer=init, maxtasksperchild=40) as pool:
        for n, x in enumerate(pool.imap_unordered(one, t.to_dict("records"), chunksize=1)):
            if x is not None:
                res.append(x)
            if n % 100 == 0:
                print('done', n, 'of', len(t), flush=True)
    R = pd.concat(res, ignore_index=True)
    R.to_parquet(out_p)
    q = R.drop_duplicates('qid')
    if present:
        print('present queries', len(q), 'rows', len(R)); sys.exit(0)
    print('queries', len(q), 'rows', len(R), '| PF-only MRR', round(float(np.where(q.rank0 <= 25, 1 / q.rank0, 0).mean()), 4),
          'top1', round(float((q.rank0 == 1).mean()), 3), 'truth in top 100', round(float((q.rank0 <= 100).mean()), 3))
