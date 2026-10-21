"""Offline check of the PubChem proposal channel (casmi/pcchannel.py): would it rank an answer that exists only in
PubChem? Simulated class-2 validation queries: the truth (a training structure, so absent from the tier) is placed into
its tier window with its own PubChem documentation; the window's n_pop most documented compounds are scored with the
PeakFormer and the truth's rank under score = z(f.z) + z(cos) + lam * pop is reported (0 when it misses the
documentation cut or ranks below 25). Documentation shift: training structures are better documented than typical
hidden answers, so results with lam > 0 are optimistic; the weighted line uses the doc-shift weights.

python scripts/eval_pcchannel.py <rows_fp.parquet> <tier.npz> <pf checkpoints, comma-separated> [n_queries]
PC_TRUTH_KEYS=<packed keys .npy>: only truths in this set (e.g. Enamine-deposited structures for the Enamine tier);
PC_NOREACH=1: only truths outside COCONUT / PubChemLite / ChEBI-LMSD (unreachable through the pool's other sources).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.keyset import pack
from casmi.pcchannel import PubChemTier
from casmi.peakformer import PFEnsemble
from casmi.prep_meta import inst_family
from casmi.spectra import Store

rows, tier_p, pf = sys.argv[1:4]
nq = int(sys.argv[4]) if len(sys.argv) > 4 else 300
D = pd.read_parquet(rows, columns=['qid', 'pid', 'label', 'cls', 'truth', 'q_rows', 'in_coco', 'in_pcl', 'in_bio'])
t = D[(D.label == 1) & (D.cls == 2)].drop_duplicates('qid')
if os.environ.get('PC_TRUTH_KEYS'):
    t = t[np.isin(pack(list(t.truth)), np.load(os.environ['PC_TRUTH_KEYS']))]
if os.environ.get('PC_NOREACH') == '1':
    t = t[(t.in_coco == 0) & (t.in_pcl == 0) & (t.in_bio == 0)]
t = t.sample(min(nq, len(t)), random_state=0)
P = pd.read_parquet('/home/user/work/pool4/pool.parquet', columns=['smiles', 'mass'])
need = pack(list(t.truth))
KP = pd.read_parquet('/home/user/ext/pubchem/key_pop.parquet', filters=[('key', 'in', list(map(int, need)))]).set_index('key')
W = pd.read_parquet('/home/user/work/pool4/doc_w.parquet').set_index('key').w
tier = PubChemTier(tier_p)
sc = PFEnsemble(pf.split(','))
st = Store('/home/user/work/train_store'); meta = st.meta
LAMS = tuple(float(x) for x in os.environ.get('PC_LAMS', '0,0.1,0.25,0.5').split(',')); NPOP = tuple(int(x) for x in os.environ.get('PC_NPOPS', '500,2000').split(','))
res = []
for q in t.itertuples():
    spectra = []
    for i in map(int, q.q_rows.split(',')):
        mz, it = st.peaks(i); r = meta.loc[i]; ce = r.collision_energy_ev
        spectra.append(dict(mz=mz, it=it, prec=float(r.precursor_mz), adduct=r.adduct,
                            ce=float(np.mean(ce)) if ce is not None and len(ce) else -1.0, inst=inst_family(r.instrument_type)))
    M = float(P.mass.values[q.pid]); smi_t = P.smiles.values[q.pid]
    k = int(pack([q.truth])[0])
    tp = (float(np.log1p(KP.sid[k]) + np.log1p(KP.pmid[k]) + 0.5 * np.log1p(KP.patent[k])) if k in KP.index else 0.0)
    idx = tier.window(M, n_pop=max(NPOP))
    smi = [tier.smiles(i) for i in idx] + [smi_t]
    pop = np.r_[tier.pop[idx], tp]
    cos, fz = sc.score(spectra, [-1 - int(i) for i in idx] + [int(q.pid)], smi)
    rec = dict(qid=q.qid, n_win=len(idx), w=float(np.clip(W.get(q.truth, 1.0), 0.02, 20)))
    for n in NPOP:
        top = np.argsort(-pop[:-1], kind='stable')[:n]
        keep = np.r_[top, len(pop) - 1] if (len(top) < n or tp >= pop[top[-1]]) else top
        if keep[-1] != len(pop) - 1:
            for lam in LAMS:
                rec[f'rr_n{n}_l{lam}'] = 0.0
            continue
        zs = lambda v: (v - v.mean()) / (v.std() or 1.0)
        for lam in LAMS:
            s = zs(fz[keep]) + zs(cos[keep]) + lam * pop[keep]
            r = 1 + int((s[:-1] > s[-1]).sum())
            rec[f'rr_n{n}_l{lam}'] = 1.0 / r if r <= 25 else 0.0
    res.append(rec)
R = pd.DataFrame(res)
print('queries', len(R), 'median window', int(R.n_win.median()))
for c in [c for c in R.columns if c.startswith('rr_')]:
    print(f'{c:16s} MRR {R[c].mean():.4f} weighted {np.average(R[c], weights=R.w):.4f} top1 {(R[c] == 1).mean():.3f}')
