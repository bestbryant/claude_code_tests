"""Stage-2 ranker: stage-1 features + forward-model (GLACIER) features on the stage-1 top-k.

python scripts/train_stage2.py cv <prefix> <stage1_feats.txt>              5-fold CV by query on one set
python scripts/train_stage2.py fit <train_prefix> <stage1_feats.txt> <out.txt> [<val_prefix>]
<prefix>.top.parquet from build_fwd_jobs.py and <prefix>.jobs_*.gl.json from the GLACIER workers.
MRR counts the truth's rank inside the top-k list (0 when the truth is outside it); class-1 queries weigh x2 in fit.
"""
import glob, json, sys
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.fwdfeat import FWD_FEATS, add_features

PARAMS = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 5], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=30,
              verbose=-1, seed=0)


def load(prefix):
    """<prefix>.top.parquet + GLACIER scores from <prefix>.jobs_*.gl.json and any extra comma-separated globs in
    $GL_GLOB (earlier runs of the same queries; later files fill pairs the earlier ones lack)."""
    import os
    T = pd.read_parquet(prefix + '.top.parquet')
    sc = {}
    pats = [prefix + '.jobs_*.gl.json'] + [p for p in os.environ.get('GL_GLOB', '').split(',') if p]
    for f in sorted({f for p in pats for f in glob.glob(p)}):
        for q, d in json.load(open(f)).items():
            cur = sc.setdefault(q, {})
            cur.update({k: v for k, v in d.items() if v is not None or k not in cur})
    T = add_features(T, sc)
    print(prefix, 'queries', T.qid.nunique(), 'with GLACIER', T.groupby('qid').gl.apply(lambda v: v.notna().any()).mean().round(3))
    return T.sort_values(['qid', 's1_r']).reset_index(drop=True)


def mrr(T, col):
    o = T.sort_values(['qid', col], ascending=[True, False])
    o['r'] = o.groupby('qid').cumcount() + 1
    r = o[o.label == 1].set_index('qid').r.reindex(T.qid.unique())
    rr = (1 / r).fillna(0)
    cls = T.drop_duplicates('qid').set_index('qid').cls.reindex(rr.index)
    return round(rr.mean(), 4), {int(c): round(rr[cls == c].mean(), 4) for c in sorted(cls.unique())}


def fit(T, feats, rounds=400):
    c1 = T[T.cls == 1]           # class-1 queries twice (as for the stage-1 ranker)
    T = pd.concat([T, c1.assign(qid=c1.qid + 10_000_000)], ignore_index=True).sort_values(['qid', 's1_r'])
    ds = lgb.Dataset(T[feats].astype(np.float32), T.label, group=T.groupby('qid', sort=True).size().values)
    return lgb.train(PARAMS, ds, rounds)


if __name__ == '__main__':
    mode = sys.argv[1]
    s1_feats = open(sys.argv[3]).read().split()
    feats = s1_feats + FWD_FEATS
    if mode == 'cv':
        T = load(sys.argv[2])
        q = T.qid.unique(); rng = np.random.default_rng(0); fold = dict(zip(q, rng.integers(0, 5, len(q))))
        T['fold'] = T.qid.map(fold)
        T['s2'] = np.nan; T['s2_nofwd'] = np.nan
        for k in range(5):
            tr, te = T[T.fold != k], T[T.fold == k]
            T.loc[te.index, 's2'] = fit(tr, feats).predict(te[feats].astype(np.float32))
            base = s1_feats + ['s1', 's1_r']
            T.loc[te.index, 's2_nofwd'] = fit(tr, base).predict(te[base].astype(np.float32))
        print('stage-1            ', mrr(T, 's1'))
        print('stage-2 without fwd', mrr(T, 's2_nofwd'))
        print('stage-2 with GLACIER', mrr(T, 's2'))
        g = T.groupby('qid').gl.apply(lambda v: v.notna().any())
        Tg = T[T.qid.isin(g.index[g])]
        print('  covered queries: stage-1', mrr(Tg, 's1'), 'stage-2', mrr(Tg, 's2'))
    else:
        T = load(sys.argv[2])
        m = fit(T, feats)
        m.save_model(sys.argv[4])
        open(sys.argv[4].replace('.txt', '_feats.txt'), 'w').write('\n'.join(feats))
        if len(sys.argv) > 5:
            V = load(sys.argv[5])
            V['s2'] = m.predict(V[feats].astype(np.float32))
            print('VAL stage-1', mrr(V, 's1'), 'stage-2', mrr(V, 's2'))
