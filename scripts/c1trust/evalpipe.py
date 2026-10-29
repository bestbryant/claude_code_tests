"""Full-window v18c validation pipeline (stage 1 -> top 30 -> GLACIER features -> stage 2) on modified rows, and the
per-query full / source-restricted RR decomposition. Reproduces fwd_v18c/val.top.parquet exactly on unmodified rows."""
import glob, json, os, sys
sys.path.insert(0, '/home/user/claude_code_tests')
import numpy as np, pandas as pd, lightgbm as lgb
from casmi.fwdfeat import top_k, add_features
W = '/home/user/work'
S1 = W + '/own_models/ret_ranker_v18c.txt'; S2 = W + '/own_models/stage2_v18c.txt'
F1 = open(S1.replace('.txt', '_feats.txt')).read().split(); F2 = open(S2.replace('.txt', '_feats.txt')).read().split()
B1 = lgb.Booster(model_file=S1); B2 = lgb.Booster(model_file=S2)
_P = None; _SC = None

def pool():
    global _P
    if _P is None:
        _P = pd.read_parquet(W + '/pool_v18c/pool.parquet', columns=['smiles', 'formula'])
    return _P

def gl_scores():
    global _SC
    if _SC is None:
        _SC = {}
        for f in sorted(glob.glob(W + '/fwd_v18c/val.jobs_*.gl.json')):
            for q, d in json.load(open(f)).items():
                cur = _SC.setdefault(q, {}); cur.update({k: v for k, v in d.items() if v is not None or k not in cur})
    return _SC

def load_rows():
    return pd.read_parquet(W + '/ret18c_val_reach_c.parquet').drop(columns=['__index_level_0__'], errors='ignore')

def recompute_lib_rank(D, qids=None):
    """lib_r / lib_gap over the full candidate window (as casmi.retrieval.features)."""
    m = D.qid.isin(qids) if qids is not None else np.ones(len(D), bool)
    g = D[m].groupby('qid')
    D.loc[m, 'lib_r'] = g.lib.rank(ascending=False, method='min')
    D.loc[m, 'lib_gap'] = D.loc[m, 'lib'] - g.lib.transform('max')
    return D

def run(D, s1_fixed=None):
    """D: full-window rows. Returns the stage-2 top-30 frame with s2. s1_fixed: optional top-30 frame whose s1 (stage-1
    score) is kept (stage-2-only substitution)."""
    D = D.copy()
    D['s_stage1'] = B1.predict(D[F1].astype(np.float32), num_threads=2)
    if s1_fixed is not None:
        T = s1_fixed[['qid', 'pid', 's1', 's1_r', 's_stage1']].merge(D.drop(columns=['s_stage1']), on=['qid', 'pid'])
    else:
        T = top_k(D, 's_stage1', 30)
    P = pool(); T['smiles'] = P.smiles.values[T.pid.values]; T['formula'] = P.formula.values[T.pid.values]
    T = add_features(T, gl_scores()).sort_values(['qid', 's1_r']).reset_index(drop=True)
    T['s2'] = B2.predict(T[F2].astype(np.float32), num_threads=2)
    T['src'] = np.where(T.in_train == 1, 'train', np.where(T.in_coco == 1, 'coco', 'other'))
    return T

def decompose(T, col='s2', all_qids=None, cls_of=None):
    """Per query: rr (full list, MRR@25), rr_res (truth's own-source list, re-ranked, @25), intruders above."""
    out = []
    for q, g in T.groupby('qid', sort=False):
        g = g.sort_values(col, ascending=False, kind='mergesort')
        lab = g.label.values; src = g.src.values
        hit = np.flatnonzero(lab > 0)
        if not len(hit):
            out.append(dict(qid=q, cls=g.cls.iat[0], tsrc=None, rr=0.0, rr_res=0.0, n_tr=0, n_co=0, n_ot=0, intop=0)); continue
        r = hit[0]; ts = src[r]; ab = src[:r]
        rres = 1 + (ab == ts).sum()
        out.append(dict(qid=q, cls=g.cls.iat[0], tsrc=ts, rr=1.0 / (r + 1) if r < 25 else 0.0,
                        rr_res=1.0 / rres if rres <= 25 else 0.0, n_tr=int((ab == 'train').sum()),
                        n_co=int((ab == 'coco').sum()), n_ot=int((ab == 'other').sum()), intop=1))
    R = pd.DataFrame(out)
    if all_qids is not None:            # queries whose truth left the window / top 30 count 0
        R = R.set_index('qid').reindex(all_qids).reset_index()
        R['cls'] = R.qid.map(cls_of); R[['rr', 'rr_res', 'intop']] = R[['rr', 'rr_res', 'intop']].fillna(0)
    R['loss'] = R.rr_res - R.rr
    return R

def summ(R, tag=''):
    c1 = R[R.cls == 1]; c2 = R[R.cls == 2]
    return dict(tag=tag, n1=len(c1), c1_mrr=c1.rr.mean(), c1_res=c1.rr_res.mean(), c1_loss=c1.loss.mean(),
                c1_intop=c1.intop.mean(), c1_top1=(c1.rr == 1).mean(), c2_mrr=c2.rr.mean(), c2_loss=c2.loss.mean())

def boot_delta(a, b, n=2000, seed=0):
    """paired bootstrap SE of mean(b - a) over aligned per-query arrays."""
    d = np.asarray(b) - np.asarray(a); rng = np.random.default_rng(seed)
    return d.mean(), np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
