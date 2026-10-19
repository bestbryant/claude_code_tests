"""5-fold CV of stage-1 configurations over all simulated queries (train + val rows), with the documentation-shift
weighted class-2 MRR and its bootstrap standard error. python scripts/cv_stage1.py [config ...]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.docw import query_weights
from casmi.own import FREL_FEATS
W = '/home/user/work'
base = open(f'{W}/own_models/ret_ranker_v5w_feats.txt').read().split()
def load(p, off):
    D = pd.read_parquet(p).sort_values(['qid', 'pid']).reset_index(drop=True)
    D['qid'] = D.qid + off
    D.loc[(D.cls == 2) & (D.label == 1), 'in_train'] = 0
    t = D[D.label == 1].set_index('qid')
    reach = (t.in_coco == 1) | (t.in_pcl == 1) | (t.in_bio == 1)
    keep = (D.cls == 1) | D.qid.map(reach).fillna(False).astype(bool)
    keep &= ~D.qid.isin(t.index[(t.cls == 1) & (t.lib_n == 0)])
    return D[keep].reset_index(drop=True)
D = pd.concat([load(f'{W}/ret7_train_fp.parquet', 0), load(f'{W}/ret7_val_fp.parquet', 100_000_000)], ignore_index=True)
q = D.drop_duplicates('qid').set_index('qid')
t = D[D.label == 1].drop_duplicates('qid').set_index('qid')
RW = pd.read_parquet(f'{W}/pool4/doc_w.parquet').set_index('key').w
qw = t.truth.map(RW).fillna(1).clip(0.02, 20).where(t.cls == 2)
print('queries', len(q), 'cls2', int((q.cls == 2).sum()), 'ESS weighted cls2', round(qw.sum() ** 2 / (qw ** 2).sum(), 1), flush=True)
fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(q)), index=q.index)
D['fold'] = D.qid.map(fold)
params = dict(objective='lambdarank', metric='map', eval_at=[25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=25, verbose=-1, num_threads=4)
def rr(Dx, col):
    o = Dx.sort_values(['qid', col], ascending=[True, False]); o['r'] = o.groupby('qid').cumcount() + 1
    r = o[(o.label == 1) & (o.r <= 25)].groupby('qid').r.min().reindex(Dx.qid.unique())
    return (1 / r).fillna(0)
configs = {'v4c': (base, 0.0), 'v5w': (base, 1.0), 'v4c+frel': (base + FREL_FEATS, 0.0), 'v7w': (base + FREL_FEATS, 1.0),
           'a0.5': (base, 0.5), 'a0.5+frel': (base + FREL_FEATS, 0.5)}
sel = sys.argv[1:] or list(configs)
for name in sel:
    feats, alpha = configs[name]
    D['p'] = np.nan
    for k in range(5):
        tr = D[D.fold != k]; c1 = tr[tr.cls == 1]
        T = pd.concat([tr, c1.assign(qid=c1.qid + 1_000_000_000)], ignore_index=True).sort_values(['qid', 'pid']).reset_index(drop=True)
        w = query_weights(T, f'{W}/pool4/doc_w.parquet', alpha) if alpha > 0 else None
        m = lgb.train(params, lgb.Dataset(T[feats].astype(float).values, T.label.values, weight=w, group=T.groupby('qid', sort=False).size().values), 500)
        te = D.fold == k
        D.loc[te, 'p'] = m.predict(D.loc[te, feats].astype(float).values)
    r = rr(D, 'p').reindex(q.index)
    c2 = q.index[q.cls == 2]
    wm = np.average(r[c2], weights=qw[c2]); 
    # bootstrap SE of weighted cls2
    rng = np.random.default_rng(1); bs = []
    for _ in range(200):
        i = rng.integers(0, len(c2), len(c2)); bs.append(np.average(r[c2].values[i], weights=qw[c2].values[i]))
    print(f'{name:10s} all {r.mean():.4f} cls1 {r[q.cls == 1].mean():.4f} cls2 {r[c2].mean():.4f} cls2w {wm:.4f} (se {np.std(bs):.4f})', flush=True)
    D[f'p_{name}'] = D.p
D[['qid', 'pid', 'label', 'cls'] + [f'p_{n}' for n in sel]].to_parquet('cv5_preds.parquet')
