"""CLI: own pipeline predictions for every test molecule -> JSON {molecule_id: [[smiles, score, key14], ...]}.

python -m casmi.run_own <test.parquet> <train.parquet> <res_dir> <out.json> [workdir] [procs]
res_dir holds pool.parquet, pool_fp.npy, coconut_prior.parquet, fp_model.pt, ret_ranker.txt (+ feats.txt).
"""
import json, os, sys, time
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .analog import AnalogLib
from .build import build_analog_lib
from .own import predict
from .retrieval import Retrieval

R = FPS = RK = FEATS = TEST = None


def _one(mid):
    try:
        return mid, [list(x) for x in predict(R, FPS, RK, FEATS, TEST[TEST.molecule_id == mid])]
    except Exception as e:
        print('own fail', mid, repr(e), flush=True)
        return mid, []


def main(argv):
    global R, FPS, RK, FEATS, TEST
    test_p, train_p, res, out = argv[:4]
    work = argv[4] if len(argv) > 4 else '/kaggle/working/own'
    procs = int(argv[5]) if len(argv) > 5 else 4
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    build_analog_lib(train_p, f'{work}/analog_lib.npz', per_struct=3)
    build_analog_lib(train_p, f'{work}/lib10.npz', per_struct=10)
    print('own libraries', f'{time.time()-t0:.0f}s', flush=True)
    import lightgbm as lgb, torch
    torch.set_num_threads(1)
    from .fpscore import FPScorer
    R = Retrieval(res, AnalogLib(f'{work}/analog_lib.npz'), f'{res}/coconut_prior.parquet',
                  lib_full=AnalogLib(f'{work}/lib10.npz'))
    fpw = f'{res}/fp_model.pt'
    FPS = FPScorer(fpw, np.load(f'{res}/pool_fp.npy', mmap_mode='r')) if os.path.exists(fpw) else None
    RK = lgb.Booster(model_file=f'{res}/ret_ranker.txt')
    FEATS = open(f'{res}/feats.txt').read().split()
    TEST = pd.read_parquet(test_p)
    TEST['molecule_id'] = TEST.molecule_id.astype(str)
    mids = list(dict.fromkeys(TEST.molecule_id))
    with Pool(procs) as pool:
        res_ = dict(pool.imap_unordered(_one, mids, chunksize=2))
    json.dump(res_, open(out, 'w'))
    print('own lists', sum(bool(v) for v in res_.values()), 'of', len(mids), f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
