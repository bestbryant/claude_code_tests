"""CLI: own pipeline predictions for every test molecule -> JSON {molecule_id: [[smiles, score, key14], ...]}.

python -m casmi.run_own <test.parquet> <train.parquet> <pool_dir> <model_dir> <out.json> [workdir] [procs]
pool_dir holds pool.parquet, pool_fp.npy, coconut_prior.parquet; model_dir holds ret_ranker.txt, feats.txt and
optionally fp_model.pt.
"""
import json, os, sys, time
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .analog import AnalogLib
from .build import build_analog_lib
from .own import predict, stage1
from .retrieval import Retrieval

R = FPS = RK = FEATS = TEST = DREAMS = None


STAGE2 = os.environ.get('OWN_STAGE2')     # stage-2 ranker name (GLACIER features) in the models dataset, or unset


def _one(mid):
    try:
        if STAGE2:
            return mid, stage1(R, FPS, RK, FEATS, TEST[TEST.molecule_id == mid], k=30, dreams=DREAMS)
        return mid, [list(x) for x in predict(R, FPS, RK, FEATS, TEST[TEST.molecule_id == mid])]
    except Exception as e:
        print('own fail', mid, repr(e), flush=True)
        return mid, [] if not STAGE2 else None


def stage2(frames, mdir, work):
    """GLACIER (own runner, casmi.fwdrun) on every stage-1 top-30 -> forward-model features -> stage-2 ranker.
    Any GLACIER failure leaves the features NaN (the ranker handles missing values) rather than dropping molecules."""
    import json as _json, subprocess as _sp
    import lightgbm as lgb
    from .fwdfeat import add_features
    g = os.environ['GLACIER_DIR']               # bryantyu/casmi26-glacier: mspred/, ckpt/, wheels/, fwdshim/
    scores = {}
    try:
        items = []
        for m, f in frames.items():
            sp = []
            for r in TEST[TEST.molecule_id == m].itertuples():
                ce = r.collision_energy_ev
                sp.append(dict(mz=[float(x) for x in r.ms2_mzs], it=[float(x) for x in r.ms2_normalized_intensities],
                               prec=float(r.precursor_mz), adduct=r.adduct, mode=r.ionization_mode,
                               instrument=r.instrument_type, ce_ev=[float(x) for x in ce] if ce is not None else []))
            items.append(dict(mid=m, spectra=sp, cands=list(f.smiles)))
        fin, fout = f'{work}/gl_in.json', f'{work}/gl_out.json'
        _json.dump(items, open(fin, 'w'))
        r = _sp.run([sys.executable, '-m', 'casmi.fwdrun', fin, fout, '--src', f'{g}/mspred', '--ckpt', f'{g}/ckpt/best.ckpt',
                     '--site', f'{work}/fwd_site', '--wheels', f'{g}/wheels', '--shim', f'{g}/fwdshim',
                     '--threads', os.environ.get('GL_THREADS', '4'), '--budget', os.environ.get('GL_BUDGET', '14400')],
                    capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-800:], flush=True)
        scores = _json.load(open(fout))
    except Exception as e:
        print('GLACIER failed', repr(e), flush=True)
    rk2 = lgb.Booster(model_file=f'{mdir}/{STAGE2}.txt')
    f2 = open(f'{mdir}/{STAGE2}_feats.txt').read().split()
    out = {}
    for m, f in frames.items():
        f = add_features(f.assign(qid=m), scores)
        f['z'] = rk2.predict(f[f2].astype(float).values)
        f = f.sort_values('z', ascending=False)
        out[m] = [[s, float(z), k] for s, z, k in zip(f.smiles, f.z, f.key14)][:25]
    print('stage 2: molecules with GLACIER scores', sum(1 for v in scores.values() if any(x is not None for x in v.values())),
          'of', len(frames), flush=True)
    return out


def main(argv):
    global R, FPS, RK, FEATS, TEST, DREAMS
    test_p, train_p, res, mdir, out = argv[:5]
    work = argv[5] if len(argv) > 5 else '/kaggle/working/own'
    procs = int(argv[6]) if len(argv) > 6 else 4
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
    fpw = f'{mdir}/fp_model.pt'
    FPS = FPScorer(fpw, np.load(f'{res}/pool_fp.npy', mmap_mode='r')) if os.path.exists(fpw) else None
    rk = os.environ.get('OWN_RANKER', 'ret_ranker')        # which ranker variant in the models dataset
    RK = lgb.Booster(model_file=f'{mdir}/{rk}.txt')
    FEATS = open(f'{mdir}/{rk}_feats.txt').read().split()
    TEST = pd.read_parquet(test_p)
    TEST['molecule_id'] = TEST.molecule_id.astype(str)
    mids = list(dict.fromkeys(TEST.molecule_id))
    if os.environ.get('DREAMS_DIR') and 'dr_best' in FEATS:     # DreaMS analog channel (casmi/dreams_feats.py)
        import glob as _glob, pyarrow.parquet as _pq
        from .dreams_emb import DreamsEmbedder
        from .dreams_feats import DreamsLib
        d = os.environ['DREAMS_DIR']
        shards = sorted(_glob.glob(f'{d}/**/dreams_lib_*.npz', recursive=True))
        meta = _pq.read_table(train_p, columns=['inchikey14', 'ionization_mode']).to_pandas()
        rows = np.concatenate([np.load(p)['row'] for p in shards])
        DL = DreamsLib(shards, dict(zip(rows.tolist(), meta.inchikey14.values[rows])),
                       dict(zip(rows.tolist(), (meta.ionization_mode.values[rows] != 'positive').astype(np.int8))))
        em = DreamsEmbedder(_glob.glob(f'{d}/**/DreaMS_embedding_model_torchscript.pt', recursive=True)[0], procs)
        E = em.embed(list(zip(TEST.ms2_mzs, TEST.ms2_normalized_intensities, TEST.precursor_mz)))
        DREAMS = (DL, dict(zip(TEST.index.tolist(), E)))
        torch.set_num_threads(1)
        print('DreaMS: library', len(DL.row), 'test spectra embedded', len(E), f'{time.time()-t0:.0f}s', flush=True)
    with Pool(procs) as pool:
        res_ = dict(pool.imap_unordered(_one, mids, chunksize=2))
    if STAGE2:
        frames = {m: f for m, f in res_.items() if f is not None and len(f)}
        print('stage 1 done', len(frames), f'{time.time()-t0:.0f}s', flush=True)
        res_ = {m: [] for m in mids}
        res_.update(stage2(frames, mdir, work))
    json.dump(res_, open(out, 'w'))
    print('own lists', sum(bool(v) for v in res_.values()), 'of', len(mids), f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
