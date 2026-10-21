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

R = FPS = PFS = RK = FEATS = TEST = DREAMS = TIER = FRAMES = PC_EXCL = SECOND = None


STAGE2 = os.environ.get('OWN_STAGE2')     # stage-2 ranker name (GLACIER features) in the models dataset, or unset


def _one(mid):
    try:
        if STAGE2:
            return mid, stage1(R, FPS, RK, FEATS, TEST[TEST.molecule_id == mid], k=30, dreams=DREAMS, pfs=PFS, second=SECOND)
        return mid, [list(x) for x in predict(R, FPS, RK, FEATS, TEST[TEST.molecule_id == mid], pfs=PFS)]
    except Exception as e:
        print('own fail', mid, repr(e), flush=True)
        return mid, [] if not STAGE2 else None


def source_gate(f, top, margin):
    """LB experiment (SRC_GATE=top,margin): candidates outside training structures and COCONUT may not take the first
    `top` ranks unless their stage-2 score beats the best training/COCONUT candidate by `margin` per-molecule SDs.
    f must be sorted by z (descending); returns the reordered frame."""
    core = ((f.in_train == 1) | (f.in_coco == 1)).values
    if core.all() or not core.any():
        return f
    sd = float(f.z.std()) or 1.0
    best_core = float(f.z[core].max())
    A, B = f[core], f[~core]
    allowed = B.z.values > best_core + margin * sd          # strong enough to compete for the top ranks
    head = pd.concat([A, B[allowed]]).sort_values('z', ascending=False).iloc[:int(top)]
    rest = f.drop(head.index)
    return pd.concat([head, rest])


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
    demote = float(os.environ.get('SRC_DEMOTE', '0'))   # LB experiment: push PubChemLite-only candidates down
    gate = tuple(float(x) for x in os.environ['SRC_GATE'].split(',')) if os.environ.get('SRC_GATE') else None
    pop = float(os.environ.get('POP_BOOST', '0'))
    frag2 = 'ps_mix_mean' in f2
    r2, n_r2 = int(os.environ.get('OWN_R2_SLOT', '2')) if SECOND is not None else 0, 0
    if frag2:
        from .fragscore import candidate_scores, frag2_feats
        from .spectra import clean_peaks
    for m, f in frames.items():
        f = add_features(f.assign(qid=m), scores)
        if frag2:            # fragment evidence v2 (casmi/fragscore.py) on the stage-1 top 30
            sp = []
            for r in TEST[TEST.molecule_id == m].itertuples():
                a, b = clean_peaks(np.asarray(r.ms2_mzs, np.float64), np.asarray(r.ms2_normalized_intensities, np.float64),
                                   float(r.precursor_mz), rel_floor=0.005, top_k=64)
                if len(a) >= 3:
                    sp.append((a.astype(np.float64), b.astype(np.float64), float(r.precursor_mz), r.adduct))
            for c, v in candidate_scores(list(f.smiles), sp).items():
                f[c] = v
            frag2_feats(f)
        f['z'] = rk2.predict(f[f2].astype(float).values)
        if pop and len(f) > 1:          # public-lineage style: z(score) + lambda * pop, pop = log1p SIDs + log1p PMIDs
            f['z'] = (f.z - f.z.mean()) / (float(f.z.std()) or 1.0) + pop * (f.pc_sid.fillna(0) + f.pc_pmid.fillna(0))
        if demote and len(f) > 1:
            pcl_only = ((f.in_pcl == 1) & (f.in_train == 0) & (f.in_coco == 0)
                        & (f.get('in_bio', 0) == 0)).values
            f.loc[pcl_only, 'z'] -= demote * float(f.z.std())
        f = f.sort_values('z', ascending=False)
        if gate:
            f = source_gate(f, *gate)
        if r2 and 'r2_top' in f and f.r2_top.any():      # second ranker's top-1 moved up to rank r2
            j = int(np.flatnonzero(f.r2_top.values == 1)[0])
            if j >= r2:
                o = list(range(len(f))); o.insert(r2 - 1, o.pop(j))
                f = f.iloc[o]; n_r2 += 1
        out[m] = [[s, float(z), k] for s, z, k in zip(f.smiles, f.z, f.key14)][:25]
    print('stage 2: molecules with GLACIER scores', sum(1 for v in scores.values() if any(x is not None for x in v.values())),
          'of', len(frames), '| second-ranker insertions', n_r2, flush=True)
    return out


def _pc_one(args):
    """PubChem proposals for one molecule (casmi/pcchannel.py)."""
    from .own import query_from_rows
    from .pcchannel import FullTier, propose, propose_full
    mid, n_pop, lam = args
    try:
        _, raw, M = query_from_rows(TEST[TEST.molecule_id == mid])
        if not np.isfinite(M):
            return mid, None
        if isinstance(TIER, FullTier):
            pr = propose_full(TIER, PFS, M, raw, n1=int(os.environ.get('PC_N1', '2000')), exclude=PC_EXCL, lam=lam,
                              pop_union=int(os.environ.get('PC_POP_UNION', '0')))
            f = (FRAMES or {}).get(mid)
            # rel: best proposal f.z minus best stage-1 pool f.z (single PeakFormer: both raw); the proposals' own
            # pc_fz, not a rescoring under other ids (the per-worker fingerprint cache is keyed by id)
            if (len(pr) and f is not None and len(f) and 'pf_bern' in f and len(getattr(PFS, 'ms', [PFS])) == 1):
                pr = pr.assign(rel=float(pr.pc_fz.max() - f.pf_bern.max()))
            return mid, pr
        from .pcchannel import finalize          # distinct metric identities (stereo / tautomer duplicates)
        pr = propose(TIER, PFS, M, raw, n_pop=n_pop, lam=lam)
        f = (FRAMES or {}).get(mid)
        # rel: best window f.z minus best stage-1 pool f.z (single PeakFormer: both raw, as in scripts/sim_enam_policy.py)
        rel = (float(pr.pc_fz.max() - f.pf_bern.max()) if (len(pr) and f is not None and len(f) and 'pf_bern' in f
                                                           and len(getattr(PFS, 'ms', [PFS])) == 1) else None)
        pr = finalize(pr, PC_EXCL or None)
        return mid, (pr.assign(rel=rel) if rel is not None and len(pr) else pr)
    except Exception as e:
        print('pc fail', mid, repr(e), flush=True)
        return mid, None


def pc_channel(lists, frames, procs):
    global FRAMES
    FRAMES = frames
    """PC_TIER=<tier.npz>: insert the PubChem channel's best proposals at PC_SLOTS (default 4,8,12,16,20) into every
    list whose best library similarity is below PC_GATE (default 0.9). PC_NPOP / PC_LAM: window size by documentation
    and the documentation weight of the proposal score; PC_ONLY=1 replaces every list by the proposals (LB probe);
    PC_PROMO=<f.z margin>: with the full tier, PC #1 goes to rank 1 when its f.z exceeds the best pool candidate's by
    more than the margin (same PeakFormer member); PC_REL_AGG=<margin>: slots 2,4,6,8,10 above it; PC_S_PROMO /
    PC_POP_PROMO: also promote when the #1 score exceeds PC_S_PROMO and its documentation prior is >= PC_POP_PROMO.
    Full tier: PC_POP=<pop.npy>, PC_N1 (pass-1 survivors), PC_POP_UNION (best-documented rows added to pass 2)."""
    from .metric import key14
    from .pcchannel import insert
    slots = tuple(int(x) for x in os.environ.get('PC_SLOTS', '4,8,12,16,20').split(','))
    hi = None                 # PC_POTHER=<model>: lists likely missing the answer get the proposals at PC_SLOTS_HI
    if os.environ.get('PC_POTHER'):
        import lightgbm as lgb
        from .pother import FEATS as PO_FEATS, list_feats
        po = lgb.Booster(model_file=os.path.join(os.environ['OWN_MODELS'], os.environ['PC_POTHER']))
        th = float(os.environ.get('PC_TH', '0.7'))
        hi_slots = tuple(int(x) for x in os.environ.get('PC_SLOTS_HI', '1,3,5,7,9').split(','))
        hi = {m: float(po.predict(np.array([[list_feats(f, 's1')[c] for c in PO_FEATS]], float))[0]) for m, f in frames.items()}
    gate = float(os.environ.get('PC_GATE', '0.9'))
    n_pop, lam = int(os.environ.get('PC_NPOP', '2000')), float(os.environ.get('PC_LAM', '0.25'))
    if os.environ.get('PC_ONLY') == '1':
        todo = list(lists)
    else:
        # molecules without a stage-1 frame (empty pool window, unusable spectra) are proposed for too
        todo = [m for m in lists if m not in frames or float(frames[m].lib.max()) < gate]
    with Pool(procs) as pool:
        props = dict(pool.imap_unordered(_pc_one, [(m, n_pop, lam) for m in todo], chunksize=2))
    n = n_promo = 0
    env_f = lambda k: float(os.environ[k]) if os.environ.get(k) else None
    promo, agg = env_f('PC_PROMO'), env_f('PC_REL_AGG')        # f.z margins over the pool: promotion / aggressive slots
    s_promo, pop_promo = env_f('PC_S_PROMO'), env_f('PC_POP_PROMO') or 5.0    # or a high score of a documented #1
    only = os.environ.get('PC_ONLY') == '1'          # LB probe: the channel's own top 25 replaces every list
    for m, pr in props.items():
        if pr is not None and len(pr):
            if only:
                lists[m] = [[s, float(v), key14(s)] for s, v in zip(pr.smiles, pr.pc_score)][:25]
            else:
                sl = hi_slots if (hi is not None and hi.get(m, 0.0) > th) else slots
                rel = float(pr.rel.iloc[0]) if 'rel' in pr else None
                if agg is not None and rel is not None and rel > agg:
                    sl = (2, 4, 6, 8, 10)                            # strong out-of-pool fit: aggressive slots
                if ((promo is not None and rel is not None and rel > promo)
                        or (s_promo is not None and float(pr.pc_score.iloc[0]) > s_promo and float(pr['pop'].iloc[0]) >= pop_promo)):
                    sl = (1,) + tuple(x for x in sl if x > 1)       # PC #1 promoted to rank 1
                    n_promo += 1
                lists[m] = insert(lists[m], pr, sl, len(sl), key14)
            n += 1
    print('PubChem channel: gated', len(todo), 'of', len(lists), 'lists with proposals', n,
          'early slots', 0 if hi is None else sum(v > th for v in hi.values()), 'promoted', n_promo, flush=True)
    return lists


def main(argv):
    global R, FPS, PFS, RK, FEATS, TEST, DREAMS, TIER, PC_EXCL, SECOND
    test_p, train_p, res, mdir, out = argv[:5]
    work = argv[5] if len(argv) > 5 else '/kaggle/working/own'
    procs = int(argv[6]) if len(argv) > 6 else 4
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    build_analog_lib(train_p, f'{work}/analog_lib.npz', per_struct=3)
    cap = int(os.environ.get('LIB_CAP', '10'))    # spectra per (structure, polarity) for direct library matching
    build_analog_lib(train_p, f'{work}/lib10.npz', per_struct=cap)
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
    if os.environ.get('OWN_R2'):   # second (source-blind) stage-1 ranker: its top-1 goes to rank OWN_R2_SLOT (default 2)
        SECOND = (lgb.Booster(model_file=f"{mdir}/{os.environ['OWN_R2']}.txt"),
                  open(f"{mdir}/{os.environ['OWN_R2']}_feats.txt").read().split())
    if 'pf_cos' in FEATS:          # PeakFormer checkpoints (OWN_PF: comma-separated names in the models dataset)
        from .peakformer import PFEnsemble
        PFS = PFEnsemble([f'{mdir}/{n}' for n in os.environ.get('OWN_PF', 'pf_model.pt').split(',')])
        print('PeakFormer members', len(PFS.ms), flush=True)
    TEST = pd.read_parquet(test_p)
    TEST['molecule_id'] = TEST.molecule_id.astype(str)
    mids = list(dict.fromkeys(TEST.molecule_id))
    if os.environ.get('DREAMS_DIR') and 'dr_best' in FEATS:     # DreaMS analog channel (casmi/dreams_feats.py)
        import glob as _glob, pyarrow.parquet as _pq
        from .dreams_emb import DreamsEmbedder
        from .dreams_feats import DreamsLib
        d = os.environ['DREAMS_DIR']
        shards = sorted(_glob.glob(f'{d}/**/dreams_lib_*.npz', recursive=True))
        lr = _glob.glob(f'{d}/**/lib_rows.npy', recursive=True)
        n_lib = sum(len(np.load(p, mmap_mode='r')['row']) for p in shards) if shards else 0
        assert shards and (not lr or n_lib == len(np.load(lr[0]))), ('DreaMS library incomplete', shards, n_lib)
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
    else:
        frames = {}
    if os.environ.get('PC_TIER'):
        from .pcchannel import FullTier, PubChemTier
        TIER = (FullTier(os.environ['PC_TIER'], os.environ.get('PC_POP')) if os.path.isdir(os.environ['PC_TIER'])
                else PubChemTier(os.environ['PC_TIER']))
        if PFS is None:
            from .peakformer import PFEnsemble
            PFS = PFEnsemble([f'{mdir}/{n}' for n in os.environ.get('OWN_PF', 'pf_model.pt').split(',')])
        # standard InChIKey14 never proposed: the pool (scored by stage 1/2) and/or probe source sets
        # (PC_EXCLUDE=<srckeys parquet>, PC_EXCLUDE_SRC=train,coco)
        PC_EXCL = set()
        if os.environ.get('PC_EXCL_POOL') == '1':
            PC_EXCL |= set(R.key_idx)
        if os.environ.get('PC_EXCLUDE'):
            K = pd.read_parquet(os.environ['PC_EXCLUDE'])
            PC_EXCL |= set(K.key[K.src.isin(os.environ.get('PC_EXCLUDE_SRC', 'train,coco').split(','))])
        print('PubChem channel excludes', len(PC_EXCL), 'keys', flush=True)
        os.environ['OWN_MODELS'] = mdir
        res_ = pc_channel(res_, frames, procs)
    if os.environ.get('ZEN_PATH'):          # Zenodo-extra library channel (casmi/zen.py)
        from .metric import key14
        from .zen import ZenLib, insert_zen
        ZL = ZenLib(os.environ['ZEN_PATH'])
        t1, t2 = float(os.environ.get('ZEN_T1', '0.7')), float(os.environ.get('ZEN_T2', '0.5'))
        n = 0
        for m in mids:
            mt = ZL.match(TEST[TEST.molecule_id == m])
            if mt and mt[0][0] >= t2 and res_.get(m) is not None:
                res_[m] = insert_zen(res_[m], mt, t1, t2, key14); n += 1
        print('Zenodo channel: inserted into', n, 'of', len(mids), 'lists', flush=True)
    json.dump(res_, open(out, 'w'))
    print('own lists', sum(bool(v) for v in res_.values()), 'of', len(mids), f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
