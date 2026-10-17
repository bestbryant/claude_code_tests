"""CLI: novel-structure candidates for every test molecule -> JSON {molecule_id: [[smiles, score, src, key14], ...]}.

python -m casmi.run_novel <test.parquet> <train.parquet> <pc_keys_u64.npy> <out.json> [model.txt] [workdir] [procs]
Runs in its own process so the main pipeline's memory is untouched.
"""
import json, os, sys, time
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .analog import AnalogLib
from .build import build_analog_lib, structures, hydrolyse_all
from .conjugate import BlockLib
from .keyset import KeySet, pack
from .novel import NovelEngine

E = None
TEST = None


def _one(mid):
    g = TEST[TEST.molecule_id == mid]
    spectra = [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float),
                    prec=float(r.precursor_mz), adduct=r.adduct, positive=(r.ionization_mode == 'positive'))
               for r in g.itertuples()]
    try:
        df = E.rank(E.candidates(spectra), n=25)
        return mid, [[s, float(z), src, k] for s, z, src, k in zip(df.smiles, df.z, df.src, df.key)]
    except Exception as e:
        print('novel fail', mid, repr(e), flush=True)
        return mid, []


def main(argv):
    global E, TEST
    test_p, train_p, pck, out = argv[:4]
    model_p = argv[4] if len(argv) > 4 and argv[4] not in ('', '-') else None
    work = argv[5] if len(argv) > 5 else '/kaggle/working/novel'
    procs = int(argv[6]) if len(argv) > 6 else 4
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    S = structures(train_p)
    build_analog_lib(train_p, f'{work}/analog_lib.npz')
    P = hydrolyse_all(S, procs=procs)
    extra_pieces = os.path.join(os.path.dirname(pck), 'blocks_pieces_coconut.parquet')
    if os.path.exists(extra_pieces):           # natural-product building blocks mined from COCONUT
        P = pd.concat([P, pd.read_parquet(extra_pieces)], ignore_index=True)
    print('novel resources', f'{time.time()-t0:.0f}s', flush=True)
    model = None
    if model_p:
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_p)
    known = [KeySet(np.unique(pack(list(S.inchikey14))))]
    coco = os.path.join(os.path.dirname(pck), 'coconut_keys_u64.npy')
    if os.path.exists(coco):
        known.append(KeySet.load(coco))
    E = NovelEngine(AnalogLib(f'{work}/analog_lib.npz'), BlockLib(P, min_count=1), KeySet.load(pck),
                    known, dict(zip(S.inchikey14, S.mass)), model=model, mode=os.environ.get('CASMI_NOVEL_MODE', 'novel'))
    TEST = pd.read_parquet(test_p)
    mids = list(dict.fromkeys(TEST.molecule_id.astype(str)))
    TEST['molecule_id'] = TEST.molecule_id.astype(str)
    with Pool(procs) as pool:
        res = dict(pool.imap_unordered(_one, mids, chunksize=2))
    json.dump(res, open(out, 'w'))
    print('novel lists', sum(bool(v) for v in res.values()), 'of', len(mids), f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
