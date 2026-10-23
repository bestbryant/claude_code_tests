"""PeakFormer f.z MRR of the truth in its pool window under the true and forced instrument tokens (1 timsTOF, 3 TOF,
0 unknown) for validation queries on Enveda NP examples (npx) and Enveda-180 (Enamine) structures.

python scripts/eval_inst_token.py
"""
import sys, numpy as np, pandas as pd, torch
torch.set_num_threads(4)
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.peakformer import PFEnsemble
from casmi.prep_meta import inst_family
from casmi.spectra import Store
W = '/home/user/work'
D = pd.read_parquet(f'{W}/ret10pl_val_fp.parquet', columns=['qid', 'pid', 'label', 'cls', 'truth', 'q_rows'])
m = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['inchikey14', 'ingest_lib'])
npx = set(m.inchikey14[m.ingest_lib == 'enveda-np-examples']); e180 = set(m.inchikey14[m.ingest_lib == 'enveda-180'])
P = pd.read_parquet(f'{W}/pool4/pool.parquet', columns=['smiles'])
st = Store(f'{W}/train_store'); meta = st.meta
sc = PFEnsemble([f'{W}/pf_local/pf_model_d_final.pt'])
t = D[D.label == 1].drop_duplicates('qid')
groups = {'npx': t[t.truth.isin(npx)], 'e180': t[t.truth.isin(e180)].sample(120, random_state=0)}
for name, tq in groups.items():
    res = {k: [] for k in ('true', 1, 3, 0)}; insts = []
    for q in tq.itertuples():
        g = D[D.qid == q.qid]
        rows = list(map(int, q.q_rows.split(',')))
        base = []
        for i in rows:
            mz, it = st.peaks(i); r = meta.loc[i]; ce = r.collision_energy_ev
            base.append(dict(mz=mz, it=it, prec=float(r.precursor_mz), adduct=r.adduct,
                             ce=float(np.mean(ce)) if ce is not None and len(ce) else -1.0, inst=inst_family(r.instrument_type)))
        insts.append(base[0]['inst'])
        for tok in res:
            sp = [dict(s, inst=s['inst'] if tok == 'true' else tok) for s in base]
            _, fz = sc.score(sp, g.pid.values, P.smiles.values[g.pid.values])
            lab = g.label.values
            r_ = 1 + int((fz[lab == 0] > fz[lab == 1].max()).sum())
            res[tok].append(1 / r_ if r_ <= 25 else 0)
    print(name, 'queries', len(tq), 'query inst tokens', pd.Series(insts).value_counts().to_dict(),
          ' '.join(f'tok {k}: f.z MRR {np.mean(v):.3f}' for k, v in res.items()), flush=True)
