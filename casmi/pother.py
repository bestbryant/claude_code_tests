"""List confidence: features of a stage-1 top-30 list for p(the answer is not in the pool) (scripts/train_pother.py),
used to place PubChem-channel proposals (casmi/run_own.py PC_POTHER)."""
import numpy as np

# the full-data ranker's scores have the same scale as the out-of-fold models' used for training (validation rows:
# top-1 mean 3.46 vs 3.58, within-list sd 2.71 vs 2.76, corr 0.96), so the absolute top score is a feature
FEATS = ['p1', 'z1', 'g12', 'g15', 'g110', 'lib1', 'libmax', 'an1', 'anmax', 'antan1', 'frag1', 'fp1', 'fpmax', 'tr1',
         'co1', 'pcl1', 'bio1', 'sid1', 'pmid1', 'n_cand', 'n_form', 'z_tr', 'z_co']
TOP = 30


def list_feats(g, score='p'):
    """g: candidate frame of one query with a stage-1 score column -> dict of FEATS over its top 30."""
    g = g.sort_values(score, ascending=False).head(TOP)
    p = g[score].values; t = g.iloc[0]
    sd = float(p.std()) or 1.0
    at = lambda i: p[min(i, len(p) - 1)]
    get = lambda c: float(t[c]) if c in g and t[c] == t[c] else 0.0
    ztr = (g[g.in_train == 1][score].max() - p[0]) / sd if (g.in_train == 1).any() else -10
    zco = (g[g.in_coco == 1][score].max() - p[0]) / sd if (g.in_coco == 1).any() else -10
    return dict(p1=p[0], z1=(p[0] - p.mean()) / sd, g12=(p[0] - at(1)) / sd, g15=(p[0] - at(4)) / sd,
                g110=(p[0] - at(9)) / sd, lib1=get('lib'), libmax=float(g.lib.max()), an1=get('an_best'),
                anmax=float(g.an_best.max()), antan1=get('an_tan_best'), frag1=get('frag'), fp1=get('fp_cos'),
                fpmax=float(g.fp_cos.max()) if 'fp_cos' in g else 0.0, tr1=get('in_train'), co1=get('in_coco'),
                pcl1=get('in_pcl'), bio1=get('in_bio'), sid1=get('pc_sid'), pmid1=get('pc_pmid'),
                n_cand=get('n_cand'), n_form=get('n_form'), z_tr=ztr, z_co=zco)
