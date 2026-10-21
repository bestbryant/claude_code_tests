"""Documentation-shift weights for simulated class-2 queries.

A simulated class-2 truth is a training structure: it has public spectra, so it is far better documented (PubChem
SIDs / PMIDs / patents, PubChemLite annotations, ChEBI / LIPID MAPS membership) than the reachable structures WITHOUT
public spectra that hidden class-2 answers are drawn from (a classifier separates the two populations with AUC 0.93).
The ranker then learns "the best-documented candidate is the answer" too strongly. Importance weights
w = p(no spectra | doc) / p(spectra | doc), from a classifier over all reachable pool structures, reweight the
class-2 queries towards the undocumented population; alpha tempers them (0 = off, 1 = full correction).

python -m casmi.docw <pool_dir> <out.parquet> [purch]  builds the weight table (key, p, w)

purch: target population = reachable non-training structures that are PURCHASABLE (Enamine or a bioactive /
natural-product vendor deposit, pool_dir/vendor.parquet). The test molecules are physical samples measured by Enveda:
98.8% of enveda-np-examples and 92.3% of enveda-180 structures carry such a deposit, against 6.1% of COCONUT-not-train.
The classifier then separates training structures from purchasable reachable ones (documentation + depositor columns).
"""
import os
import numpy as np
import pandas as pd
from .retrieval import LT_COLS, PRIOR_COLS

DOC = ['pc_sid', 'pc_pmid', 'pc_patent', 'pc_ncid', 'in_pcl', 'in_bio', 'in_coco', 'pcl_pubmed', 'pcl_patent',
       'pcl_anno', 'pcl_bio', 'pcl_food', 'mass']


def build(pool_dir, out, target='nospectra'):
    import lightgbm as lgb
    P = pd.read_parquet(f'{pool_dir}/pool.parquet')
    doc = list(DOC)
    if target == 'purch':
        from .retrieval import VEND_COLS
        V = pd.read_parquet(f'{pool_dir}/vendor.parquet')
        P = pd.concat([P, V[VEND_COLS]], axis=1)
        doc += VEND_COLS
    P = P.drop_duplicates('key')
    R = P[(P.in_coco == 1) | (P.in_pcl == 1) | (P.in_bio == 1)]
    if target == 'purch':            # sim population (training structures) vs purchasable structures without spectra
        R = R[(R.in_train == 1) | (R.v_bioact > 0) | (R.v_enamine == 1)]
    doc += [c for c in LT_COLS if c in R.columns]          # LOTUS occurrences are documentation too (pool v5)
    if os.path.exists(f'{pool_dir}/coconut_prior.parquet') and LT_COLS[0] in R.columns:   # COCONUT records (pool v5)
        pr = pd.read_parquet(f'{pool_dir}/coconut_prior.parquet').drop_duplicates('ik14').set_index('ik14')
        cols = [c for c in PRIOR_COLS if c != 'npl']
        R = R.join(pr[cols], on='key')
        R[cols] = R[cols].fillna(0).astype(float)
        doc += cols
    m = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=31, verbose=-1, num_threads=4, seed=0),
                  lgb.Dataset(R[doc].astype(float), R.in_train), 200)
    R = R.assign(p=m.predict(R[doc].astype(float)))
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
    build(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ['nospectra']))
