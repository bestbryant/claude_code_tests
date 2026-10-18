"""GLACIER predicted-spectrum similarity for candidate structures (own runner).

GLACIER (Goldman et al., ms-pred, MIT licence; MassSpecGym-trained weights published by the authors) predicts each
candidate's MS/MS; the score is the weighted spectral-entropy similarity between the prediction and the query,
averaged over the molecule's covered query spectra (positive mode, [M+H]+ / [M+Na]+).

Runs in its own interpreter because ms-pred needs RDKit 2025.03 (installed offline from wheels into --site) while
the main pipeline uses RDKit 2026.03:
  python -m casmi.fwdrun --src <ms_pred src dir> --ckpt <best.ckpt> --site <dir> [--wheels <dir>] in.json out.json
in.json: [{"mid": str, "spectra": [{"mz": [...], "it": [...], "prec": float, "adduct": str, "mode": str,
           "instrument": str, "ce_ev": [...]}], "cands": [smiles, ...]}]
out.json: {mid: {smiles: score | null}} (rewritten after every batch; a budget stops cleanly).
"""
import argparse, glob, importlib.abc, importlib.machinery, json, os, subprocess, sys, time, types

COVERED = ('[M+H]+', '[M+Na]+')
OPTIONAL = ('multiprocess', 'pygmtools', 'h5py', 'matplotlib', 'seaborn', 'cairosvg', 'ray', 'wandb')  # imported by ms-pred at module level, unused for inference


class _AnyMeta(type):
    def __getattr__(cls, k):
        return _Any


class _Any(metaclass=_AnyMeta):
    """Placeholder usable as a value, a callable, a decorator or a base class."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, k):
        return _Any


class _StubModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return _Any


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in OPTIONAL:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


def install_site(wheels, site):
    """pip-install the wheels (*.whl, or *.whl.ice = renamed so that other wheel globs skip them) for this
    interpreter into `site`, once (marker file)."""
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    src = sorted(glob.glob(os.path.join(wheels, '*.whl')) + glob.glob(os.path.join(wheels, '*.whl.ice')))
    src = [w for w in src if '-cp3' not in os.path.basename(w) or f'-{tag}-' in os.path.basename(w)]
    mark = os.path.join(site, '.casmi_site_ok')
    want = '\n'.join(sorted(os.path.basename(w) for w in src))
    if os.path.exists(mark) and open(mark).read() == want:
        return
    os.makedirs(os.path.join(site, '.w'), exist_ok=True)
    whl = []
    for w in src:
        d = os.path.join(site, '.w', os.path.basename(w).replace('.whl.ice', '.whl'))
        with open(w, 'rb') as a, open(d, 'wb') as b:
            b.write(a.read())
        whl.append(d)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--no-compile',
                    '--disable-pip-version-check', '--target', site] + whl, check=True)
    open(mark, 'w').write(want)


def instrument_token(s):
    s = (s or '').lower()
    return 'Orbitrap' if ('orbitrap' in s or 'ft' in s or 'q exactive' in s) else 'QTOF'


def query_peaks(mz, it, prec, floor=0.001, topk=256):
    """Raw query peaks: <= precursor + 2 Da, >= floor x base peak, top-k by intensity, m/z sorted, base peak = 1."""
    import numpy as np
    keep = (mz <= prec + 2.0) & (mz > 0)
    mz, it = mz[keep], it[keep]
    if len(it) == 0 or it.max() <= 0:
        return mz[:0], it[:0]
    keep = it >= floor * it.max()
    mz, it = mz[keep], it[keep]
    if len(it) > topk:
        o = np.argsort(-it)[:topk]
        mz, it = mz[o], it[o]
    o = np.argsort(mz)
    return mz[o], it[o] / it.max()


def pred_peaks(mz, it, k=100):
    """Predicted peaks: positive only, identical m/z (4 dp) merged by max, top-k, m/z sorted, base peak = 1."""
    import numpy as np
    keep = (mz > 0) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return mz, it
    u, inv = np.unique(np.round(mz, 4), return_inverse=True)
    v = np.zeros(len(u)); np.maximum.at(v, inv, it)
    if len(u) > k:
        o = np.argsort(-v, kind='stable')[:k]
        u, v = u[o], v[o]
    o = np.argsort(u)
    return u[o], v[o] / v.max()


def spectrum_ce(sp):
    ce = sp.get('ce_ev')
    ce = [float(c) for c in ce] if ce else []
    return round(sum(ce) / len(ce) / 5.0) * 5.0 if ce else 30.0     # stepped CE -> mean, 5 eV buckets


def supervise(max_restarts=30):
    """Run this module as a child process until it finishes cleanly: a native crash (segfault) on some molecule
    loses only that batch (the child checkpoints its predictions and skips the batch it died in)."""
    env = {**os.environ, 'FWDRUN_CHILD': '1'}
    for attempt in range(max_restarts + 1):
        r = subprocess.run([sys.executable, '-m', 'casmi.fwdrun'] + sys.argv[1:], env=env)
        if r.returncode == 0:
            return 0
        print(f'fwdrun: child exited with {r.returncode}, restart {attempt + 1}', flush=True)
    return 1


def main():
    import faulthandler
    faulthandler.enable()          # a native crash (e.g. in torch sparse ops) still reports where it happened
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--src', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--site', required=True); ap.add_argument('--wheels', default=None)
    ap.add_argument('--shim', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fwdshim'))
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--budget', type=float, default=0.0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--dump-preds', default=None, help='pickle the predicted spectra here (for offline rescoring)')
    ap.add_argument('--batch-timeout', type=int, default=300, help='seconds before a batch is abandoned')
    a = ap.parse_args()
    t0 = time.time()
    if a.wheels:
        install_site(a.wheels, a.site)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [a.shim, a.src, a.site]
    sys.path.append(repo)
    sys.meta_path.insert(0, _StubFinder())
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from ms_pred import common
    from ms_pred.common import chem_utils
    from ms_pred.glacier import joint_model, dataset
    from casmi.entropy import entropy_sim

    model = joint_model.JointModel.load_from_checkpoint(a.ckpt, map_location='cpu', weights_only=False)
    model.eval(); model.freeze()
    tp = dataset.TreeProcessor(pe_embed_k=model.pe_embed_k, root_encode='graphormer',
                               embed_elem_group=model.embed_elem_group, multi_hop_max_dist=model.multi_hop_max_dist)
    items = json.load(open(a.inp))
    out = {it['mid']: {s: None for s in it['cands']} for it in items}
    json.dump(out, open(a.out, 'w'))

    def canon(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None or any(at.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM for at in m.GetAtoms()):
            return None
        m = Chem.RemoveHs(m)
        return Chem.MolToSmiles(m) if m.GetNumAtoms() <= 100 else None

    # queries and the unique prediction jobs (canonical smiles, adduct, CE bucket, instrument)
    queries, jobs = {}, {}
    for it in items:
        qs = []
        for sp in it['spectra']:
            if sp.get('mode') != 'positive' or sp.get('adduct') not in COVERED:
                continue
            mz, ii = query_peaks(np.asarray(sp['mz'], float), np.asarray(sp['it'], float), float(sp['prec']))
            if len(mz) >= 2:
                qs.append((mz, ii, (sp['adduct'], spectrum_ce(sp), instrument_token(sp.get('instrument')))))
        queries[it['mid']] = qs
        for s in it['cands']:
            c = canon(s)
            if c is None:
                continue
            for _, _, cond in qs:
                jobs.setdefault((c,) + cond, None)
    keys = sorted(jobs, key=lambda k: (len(k[0]), k))
    # crash resilience: finished predictions are checkpointed, the batch in progress is recorded; after a native
    # crash (the caller restarts us) that batch is skipped. A batch running over --batch-timeout is skipped too.
    import pickle, signal
    ck, cur = a.out + '.ckpt.pkl', a.out + '.cur'
    done_b, bad_b = set(), set()
    if os.path.exists(ck):
        saved = pickle.load(open(ck, 'rb'))
        jobs.update({k: v for k, v in saved['jobs'].items() if k in jobs})
        done_b, bad_b = saved['done'], saved['bad']
    if os.path.exists(cur):
        bad_b.add(int(open(cur).read()))
        print('fwdrun: skipping batch', open(cur).read(), 'after a crash', flush=True)

    def _alarm(signum, frame):
        raise TimeoutError('batch timeout')
    signal.signal(signal.SIGALRM, _alarm)
    print(f'fwdrun: {len(items)} molecules, {len(keys)} predictions, setup {time.time()-t0:.0f}s', flush=True)
    feat_cache = {}
    n_new = 0
    with torch.inference_mode():
        for bi, b0 in enumerate(range(0, len(keys), a.bs)):
            if bi in done_b or bi in bad_b:
                continue
            if a.budget and time.time() - t0 > a.budget:
                print('fwdrun: budget reached', flush=True)
                break
            open(cur, 'w').write(str(bi))
            signal.alarm(a.batch_timeout)
            try:
                batch = []
                for smi, adduct, ce, instr in keys[b0:b0 + a.bs]:
                    try:
                        f = feat_cache.get((smi, adduct))
                        if f is None:
                            f = tp.featurize_tree({'root_canonical_smiles': smi, 'adduct': adduct, 'collision_energy': ce},
                                                  include_inten_targs=False, include_frag_targs=False)
                            if len(feat_cache) > 64:          # jobs are sorted by molecule: a small cache suffices
                                feat_cache.clear()
                            feat_cache[(smi, adduct)] = f
                        item = dict(f)
                        item.update(name=smi, adduct=common.ion2onehot_pos[adduct], precursor=0.0, collision_energy=ce,
                                    instrument=common.instrument2onehot_pos.get(instr, common.instrument2onehot_pos.get('Orbitrap', 0)))
                        batch.append(((smi, adduct, ce, instr), item))
                    except TimeoutError:
                        raise
                    except Exception as e:
                        print('featurize fail', smi[:60], repr(e)[:120], flush=True)
                if batch:
                    col = dataset.IntenDataset.collate_fn([x[1] for x in batch])
                    res = model.predict_inten_frag_batch_sparse(col, 100)
                    ms, its, cnt = res['masses'], res['intens'], res['counts']
                    for i, (k, _) in enumerate(batch):
                        c = int(cnt[i])
                        jobs[k] = pred_peaks(np.asarray(ms[i, :c], float), np.asarray(its[i, :c], float))
            except Exception as e:
                print('batch fail', bi, repr(e)[:200], flush=True)
                bad_b.add(bi)
            finally:
                signal.alarm(0)
            done_b.add(bi); n_new += 1
            os.remove(cur)
            if n_new % 50 == 0:
                pickle.dump(dict(jobs={k: v for k, v in jobs.items() if v is not None}, done=done_b, bad=bad_b),
                            open(ck, 'wb'))
                print(f'fwdrun: {b0 + a.bs}/{len(keys)} {time.time()-t0:.0f}s', flush=True)
    if a.dump_preds:
        import pickle
        pickle.dump(jobs, open(a.dump_preds, 'wb'))
    # scores
    for it in items:
        qs = queries[it['mid']]
        for s in it['cands']:
            c = canon(s)
            if c is None or not qs:
                continue
            v = []
            for qmz, qit, cond in qs:
                p = jobs.get((c,) + cond)
                if p is None or len(p[0]) == 0:
                    continue
                v.append(float(entropy_sim(qmz, qit, p[0], p[1], 0.01)))
            out[it['mid']][s] = float(np.mean(v)) if v else None
    json.dump(out, open(a.out, 'w'))
    for f in (a.out + '.ckpt.pkl',):
        if os.path.exists(f):
            os.remove(f)
    print(f'fwdrun: done {time.time()-t0:.0f}s, scored {sum(v is not None for d in out.values() for v in d.values())}', flush=True)


if __name__ == '__main__':
    if os.environ.get('FWDRUN_CHILD'):
        main()
    else:
        sys.exit(supervise())
