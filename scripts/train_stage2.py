"""Stage-2 ranker: stage-1 features + forward-model (GLACIER) features on the stage-1 top-k.

python scripts/train_stage2.py cv <prefix> <stage1_feats.txt>              5-fold CV by query on one set
python scripts/train_stage2.py fit <train_prefix> <stage1_feats.txt> <out.txt> [<val_prefix>]
<prefix>.top.parquet from build_fwd_jobs.py and <prefix>.jobs_*.gl.json from the GLACIER workers.
MRR counts the truth's rank inside the top-k list (0 when the truth is outside it); class-1 queries weigh x2 in fit.
S2_PARAMS: JSON overrides of PARAMS (e.g. a seed replicate). DOC_W / DOC_ALPHA: documentation-shift weights for class-2 queries, as in train_ret_ranker.py. S2_EXTRA=frag2 adds
the fragment evidence v2 columns (casmi/fragscore.py).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.fwdfeat import FWD_FEATS, add_features

PARAMS = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 5], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=30,
              verbose=-1, seed=0, num_threads=2, deterministic=True, force_row_wise=True)   # G3: row-wise reproduces stage2_v17b
PARAMS.update(json.loads(os.environ.get('S2_PARAMS', '{}')))      # e.g. '{"seed": 1}' (seed replicate)


def load(prefix, env='GL_GLOB'):
    """<prefix>.top.parquet + GLACIER scores from <prefix>.jobs_*.gl.json and any extra comma-separated globs in
    $GL_GLOB (earlier runs of the same queries; later files fill pairs the earlier ones lack)."""
    import os
    T = pd.read_parquet(prefix + '.top.parquet')
    sc = {}
    pats = [prefix + '.jobs_*.gl.json'] + [p for p in os.environ.get(env, '').split(',') if p]
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


def wmrr(T, col, path=os.environ.get('EVAL_W', '/home/user/work/pool4/doc_w.parquet')):
    """Class-2 MRR under the documentation-shift weights (casmi/docw.py) and its bootstrap standard error."""
    if not os.path.exists(path):
        return None
    o = T.sort_values(['qid', col], ascending=[True, False])
    o['r'] = o.groupby('qid').cumcount() + 1
    t = T.drop_duplicates('qid').set_index('qid')        # every class-2 query (truth outside the top k counts 0)
    t = t[t.cls == 2]
    r = (1 / o[o.label == 1].groupby('qid').r.min()).reindex(t.index).fillna(0).values
    w = t.truth.map(pd.read_parquet(path).set_index('key').w).fillna(1).clip(0.02, 20).values
    rng = np.random.default_rng(1)
    bs = [np.average(r[i], weights=w[i]) for i in (rng.integers(0, len(r), len(r)) for _ in range(200))]
    return 'cls2-weighted %.4f (se %.4f)' % (np.average(r, weights=w), np.std(bs))


def fit(T, feats, rounds=400):
    c1 = T[T.cls == 1]           # class-1 queries twice (as for the stage-1 ranker; C1_DUP copies, default 2)
    T = pd.concat([T] + [c1.assign(qid=c1.qid + j * 10_000_000) for j in range(1, int(os.environ.get('C1_DUP', '2')))],
                  ignore_index=True).sort_values(['qid', 's1_r'])
    wts = None
    if os.environ.get('DOC_W'):        # documentation-shift weights for class-2 queries (casmi/docw.py)
        from casmi.docw import query_weights
        wts = query_weights(T, os.environ['DOC_W'], float(os.environ.get('DOC_ALPHA', '1')))
    ds = lgb.Dataset(T[feats].astype(np.float32), T.label, weight=wts, group=T.groupby('qid', sort=True).size().values)
    return lgb.train(PARAMS, ds, rounds)


if __name__ == '__main__':
    mode = sys.argv[1]
    s1_feats = open(sys.argv[3]).read().split()
    feats = s1_feats + FWD_FEATS
    if os.environ.get('S2_EXTRA') == 'frag2':         # fragment evidence v2 columns (scripts/add_frag2.py)
        from casmi.fragscore import FRAG2_FEATS
        feats = feats + FRAG2_FEATS
    if mode == 'cv':
        T = load(sys.argv[2])
        q = T.qid.unique(); rng = np.random.default_rng(int(os.environ.get('S2_FOLD_SEED', '0')))
        fold = dict(zip(q, rng.integers(0, 5, len(q))))
        T['fold'] = T.qid.map(fold)
        T['s2'] = np.nan; T['s2_nofwd'] = np.nan
        for k in range(5):
            tr, te = T[T.fold != k], T[T.fold == k]
            T.loc[te.index, 's2'] = fit(tr, feats).predict(te[feats].astype(np.float32))
            base = s1_feats + ['s1', 's1_r']
            if os.environ.get('S2_CV_NOFWD', '1') == '1':
                T.loc[te.index, 's2_nofwd'] = fit(tr, base).predict(te[base].astype(np.float32))
        print('stage-1            ', mrr(T, 's1'), wmrr(T, 's1'))
        print('stage-2 without fwd', mrr(T, 's2_nofwd'), wmrr(T, 's2_nofwd'))
        print('stage-2 with GLACIER', mrr(T, 's2'), wmrr(T, 's2'))
        g = T.groupby('qid').gl.apply(lambda v: v.notna().any())
        Tg = T[T.qid.isin(g.index[g])]
        print('  covered queries: stage-1', mrr(Tg, 's1'), 'stage-2', mrr(Tg, 's2'))
    else:
        T = load(sys.argv[2])
        m = fit(T, feats)
        m.save_model(sys.argv[4])
        open(sys.argv[4].replace('.txt', '_feats.txt'), 'w').write('\n'.join(feats))
        if len(sys.argv) > 5:
            V = load(sys.argv[5], 'GL_GLOB_VAL')       # query ids of the two sets overlap: separate score files
            V['s2'] = m.predict(V[feats].astype(np.float32))
            print('VAL stage-1', mrr(V, 's1'), 'stage-2', mrr(V, 's2'))
