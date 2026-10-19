"""Documentation-shift weights for simulated class-2 queries.

A simulated class-2 truth is a training structure: it has public spectra, so it is far better documented (PubChem
SIDs / PMIDs / patents, PubChemLite annotations, ChEBI / LIPID MAPS membership) than the reachable structures WITHOUT
public spectra that hidden class-2 answers are drawn from (a classifier separates the two populations with AUC 0.93).
The ranker then learns "the best-documented candidate is the answer" too strongly. Importance weights
w = p(no spectra | doc) / p(spectra | doc), from a classifier over all reachable pool structures, reweight the
class-2 queries towards the undocumented population; alpha tempers them (0 = off, 1 = full correction).

python -m casmi.docw <pool_dir> <out.parquet>        builds the weight table (key, p, w)
"""
import numpy as np
import pandas as pd

DOC = ['pc_sid', 'pc_pmid', 'pc_patent', 'pc_ncid', 'in_pcl', 'in_bio', 'in_coco', 'pcl_pubmed', 'pcl_patent',
       'pcl_anno', 'pcl_bio', 'pcl_food', 'mass']


def build(pool_dir, out):
    import lightgbm as lgb
    P = pd.read_parquet(f'{pool_dir}/pool.parquet').drop_duplicates('key')
    R = P[(P.in_coco == 1) | (P.in_pcl == 1) | (P.in_bio == 1)]
    m = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=31, verbose=-1, num_threads=4, seed=0),
                  lgb.Dataset(R[DOC].astype(float), R.in_train), 200)
    R = R.assign(p=m.predict(R[DOC].astype(float)))
    pi = R.in_train.mean()
    R['w'] = (1 - R.p) / R.p * pi / (1 - pi)
    R[['key', 'p', 'w']].to_parquet(out)
    print('reachable', len(R), 'with spectra', round(pi, 4), 'weights of training structures',
          R[R.in_train == 1].w.describe().round(3).to_dict())


def query_weights(D, path, alpha, lo=0.02, hi=20.0):
    """Per-row weights for rows D (qid, label, truth, cls): class-2 queries get (clipped w)^alpha normalised to mean 1
    over class-2 queries, class-1 queries 1."""
    W = pd.read_parquet(path).set_index('key').w
    t = D[D.label == 1].drop_duplicates('qid').set_index('qid')
    w = t.truth.map(W).fillna(1.0).clip(lo, hi) ** alpha
    w = w.where(t.cls == 2, np.nan)
    w = (w / w.mean()).fillna(1.0)
    return D.qid.map(w).fillna(1.0).values


if __name__ == '__main__':
    import sys
    build(sys.argv[1], sys.argv[2])
