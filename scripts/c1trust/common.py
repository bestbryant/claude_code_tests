"""Shared loaders for d3_hiddenlike (read-only on /home/user/work and the repo)."""
import os, sys, time
sys.path.insert(0, '/home/user/claude_code_tests')
os.environ.setdefault('C1_DUP', '2')
os.environ.setdefault('DOC_W', '/home/user/work/pool_v18c/doc_w.parquet'); os.environ.setdefault('DOC_ALPHA', '1')
import numpy as np, pandas as pd, lightgbm as lgb
import importlib.util
sys.argv = ['x']
spec = importlib.util.spec_from_file_location('ts2', '/home/user/claude_code_tests/scripts/train_stage2.py')
ts2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts2)
ts2.PARAMS['num_threads'] = 2
OUT = '/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/c1trust/d3_hiddenlike'
MODEL = '/home/user/work/own_models/stage2_v18c.txt'

def load_sets(which=('val', 'train')):
    out = {}
    for w in which:
        T = ts2.load('/home/user/work/fwd_v18c/' + w, 'GL_GLOB' if w == 'train' else 'GL_GLOB_VAL')
        T['src'] = np.where(T.in_train == 1, 'train', np.where(T.in_coco == 1, 'coco', 'other'))
        out[w] = T
    return out

def s2(T, model=MODEL):
    m = lgb.Booster(model_file=model)
    feats = open(model.replace('.txt', '_feats.txt')).read().split()
    return m.predict(T[feats].astype(np.float32), num_threads=2)

def meta():
    return pd.read_parquet('/home/user/work/train_store/spec_meta.parquet',
                           columns=['inchikey14', 'ingest_lib', 'adduct', 'ionization_mode', 'instrument_type'])

def inst_class(m):
    """coarse instrument class per spectrum: tims / orbi / qtof / other."""
    it = m.instrument_type.fillna('').str.lower()
    c = np.where(it.str.contains('tims'), 'tims',
        np.where(it.str.contains('orbi|qft|q-exactive|qexactive|itft|hybrid ft|ft'), 'orbi',
        np.where(it.str.contains('tof'), 'qtof', 'other')))
    return pd.Series(c, index=m.index)
