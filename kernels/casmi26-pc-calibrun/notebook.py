"""Calibration of the full-PubChem channel's promotion thresholds on 400 simulated class-2 validation queries.
For each query the full-tier window is scored exactly as at test time (casmi.pcchannel.propose_full, n1 5000, pop
union 200, lam 0.25) with the truth's key excluded, so the proposals are what the channel offers when the answer is
NOT in PubChem's window (null); the truth's own f.z / cosine give its rank among those proposals when the answer IS
there. Output calib_out.parquet: proposal max f.z, #1 score / pop, truth f.z / cos and rank."""
import glob, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-2026.3.3-*.whl', recursive=True) if tag in w]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('retrieval.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
sys.path.insert(0, '/kaggle/working/pkg')
import torch
torch.set_num_threads(1)
from multiprocessing import Pool
from casmi.pcchannel import FullTier, propose_full
from casmi.peakformer import PFEnsemble
Q = pd.read_parquet(first('calib_queries.parquet'))
TIER = FullTier(os.path.dirname(first('tier_full/mass.npy')), first('pop.npy'))
PFS = PFEnsemble([os.path.join(os.path.dirname(first('ret_ranker_v11l.txt')), 'pf_local_final.pt')])


def one(i):
    r = Q.iloc[i]
    sp = json.loads(r.spectra)
    for s in sp:
        s['mz'] = np.asarray(s['mz']); s['it'] = np.asarray(s['it'])
    try:
        pr = propose_full(TIER, PFS, float(r.mass), sp, n1=5000, exclude={r.truth_std}, lam=0.25, pop_union=200)
        m0 = PFS.ms[0]
        tc, tf = m0.score(sp, [10 ** 9], [r.truth_smiles])
        allf = np.r_[pr.pc_fz.values, tf]; allc = np.r_[pr.pc_cos.values, tc]
        zs = lambda v: (v - v.mean()) / (v.std() or 1.0)
        comb = zs(allf) + zs(allc)
        return dict(qid=int(r.qid), n_prop=len(pr), pc_max_fz=float(pr.pc_fz.max()) if len(pr) else np.nan,
                    s_top=float(pr.pc_score.iloc[0]) if len(pr) else np.nan, pop_top=float(pr['pop'].iloc[0]) if len(pr) else np.nan,
                    truth_fz_tier=float(tf[0]), truth_cos=float(tc[0]),
                    truth_rank_fz=int(1 + (allf[:-1] > allf[-1]).sum()), truth_rank_comb=int(1 + (comb[:-1] > comb[-1]).sum()))
    except Exception as e:
        return dict(qid=int(r.qid), error=repr(e))


t0 = time.time()
with Pool(4) as p:
    out = list(p.imap_unordered(one, range(len(Q)), chunksize=2))
O = Q.drop(columns='spectra').merge(pd.DataFrame(out), on='qid')
O.to_parquet('/kaggle/working/calib_out.parquet')
print('done', len(O), f'{time.time() - t0:.0f}s', O.get('error', pd.Series(dtype=str)).notna().sum(), 'errors')
