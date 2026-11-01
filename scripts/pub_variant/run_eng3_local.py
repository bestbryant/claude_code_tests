"""Local end-to-end run of the ENG3 data path (ours): OUR stage-1 subprocess on the visible test's 12 commit molecules,
then the ENG3 merge with a synthetic Engine 2 list (and, if given, the original Engine 2's real lists of the casmi26-pubm
commit run, a data file). Only our code runs (eng2_merge.run_engine2 -> python -m casmi.run_eng2; eng3_merge).

python scripts/pub_variant/run_eng3_local.py <out_dir> [--procs 2] [--libs DIR] [--engine2 eng_lists.json]

Layout: <out_dir>/input mimics /kaggle/input (datasets/bryantyu/casmi26-elig-assets = symlinks to the repository's 16
casmi modules + the 6 model files of /home/user/ds/casmi26-models; datasets/bryantyu/casmi26-pool-v18c = symlinks to
/home/user/ds/casmi26-pool-v18c), <out_dir>/smoke/test.parquet = the 12 molecules frozen cell 30 keeps in a commit run
(sorted molecule ids, first 12). --libs: prebuilt analog_lib.npz / lib10.npz (casmi.run_eng2 --libs; disk is tight here,
the Kaggle run builds them from train.parquet). The run checks the files against elig1's pins (the bytes Kaggle mounts).
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


em, m3, e3 = _load('eng2_merge'), _load('eng3_merge'), _load('eng3')
MODELS = '/home/user/ds/casmi26-models'
POOL = '/home/user/ds/casmi26-pool-v18c'
DATA = '/home/user/data'


def key_of_factory():
    """The pipeline's chem.score_key semantics, implemented here in the main process (fresh enumerator): the first
    block (14 characters, connectivity only) of the InChIKey of RDKit's canonical tautomer (stereo is not stripped
    before canonicalisation; the block carries none), None when RDKit fails or gives an empty InChIKey. Independent
    of the subprocess's own keys."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    te = rdMolStandardize.TautomerEnumerator()

    def key_of(s):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            ik = Chem.MolToInchiKey(te.Canonicalize(m))
            return ik[:14] if ik else None
        except Exception:
            return None
    return key_of


def setup(out):
    inp = os.path.join(out, 'input', 'datasets', 'bryantyu')
    a, p = os.path.join(inp, 'casmi26-elig-assets'), os.path.join(inp, 'casmi26-pool-v18c')
    os.makedirs(a, exist_ok=True)
    os.makedirs(p, exist_ok=True)
    for name in e3.elig.ELIG_ASSETS_SHA256:
        src = os.path.join(REPO, 'casmi', name) if name.endswith('.py') else os.path.join(MODELS, name)
        dst = os.path.join(a, name)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    for name in e3.POOL_FILES:
        dst = os.path.join(p, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(POOL, name), dst)
    smoke = os.path.join(out, 'smoke')
    os.makedirs(smoke, exist_ok=True)
    full = pd.read_parquet(os.path.join(DATA, 'test.parquet'))
    keep = sorted(full.molecule_id.unique())[:12]                    # frozen cell 30's commit subset
    full[full.molecule_id.isin(keep)].to_parquet(os.path.join(smoke, 'test.parquet'))
    return os.path.join(out, 'input'), os.path.join(smoke, 'test.parquet'), [str(m) for m in keep]


def synthetic_engine2(ours, mids, key_of, seed=0, n=40):
    """Per molecule a 40-entry Engine 2 list: about half our identities at shuffled positions, written as DIFFERENT
    SMILES strings of the same structure (RDKit random SMILES; the key must still match), the rest structures ours does
    not list (other molecules' candidates). Keys = key_of, as Engine 2's canon_key writes them."""
    from rdkit import Chem
    rnd = random.Random(seed)
    other = [s for m in mids for s in (ours.get(m) or {}).get('smiles', [])]
    out, rewritten = {}, 0
    for m in mids:
        o = ours.get(m) or dict(smiles=[], keys=[])
        take = rnd.sample(range(len(o['smiles'])), k=min(len(o['smiles']), 20))
        sm, ks = [], []
        for i in take:
            s = o['smiles'][i]
            mol = Chem.MolFromSmiles(s)
            alt = Chem.MolToSmiles(mol, doRandom=True) if mol is not None else s
            if key_of(alt) == o['keys'][i]:
                rewritten += int(alt != s)
                s = alt
            sm.append(s)
            ks.append(key_of(s))
        own = set(o['keys'])
        for s in rnd.sample(other, k=len(other)):
            if len(sm) >= n:
                break
            k = key_of(s)
            if k and k not in own and k not in ks:
                sm.append(s)
                ks.append(k)
        order = list(range(len(sm)))
        rnd.shuffle(order)
        out[m] = dict(smiles=[sm[i] for i in order], keys=[ks[i] for i in order])
    return out, rewritten


def check_merged(merged, eng2, ours, mids, n=40):
    """Invariants of the merged lists every reader relies on."""
    assert set(mids) <= set(merged), 'coverage (frozen cell 48)'
    shared_smiles_kept = 0
    for m in mids:
        v = merged[m]
        real = [k for k in v['keys'] if k]
        assert v['smiles'] and len(v['smiles']) == len(v['keys']) <= n and len(set(real)) == len(real), m
        e = eng2.get(m) or {}
        e_smi = {k: s for s, k in zip(e.get('smiles', []), e.get('keys', [])) if k}
        e_keyless = {s for s, k in zip(e.get('smiles', []), e.get('keys', [])) if not k}
        o_keys = set((ours.get(m) or {}).get('keys', []))
        for s, k in zip(v['smiles'], v['keys']):
            if not k:
                assert s in e_keyless, (m, s)                     # an Engine 2 entry without a key, kept as written
                continue
            assert k in e_smi or k in o_keys, (m, k)
            if k in e_smi:
                assert s == e_smi[k], (m, k)                       # Engine 2's SMILES for identities it lists
                shared_smiles_kept += int(k in o_keys)
        if not (ours.get(m) or {}).get('smiles'):
            assert v == e, m                                      # no list of ours: unchanged
    return shared_smiles_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--libs', default='')
    ap.add_argument('--engine2', default='')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    input_root, test, mids = setup(out)
    cfg = dict(e3.ENG3_CFG, work=os.path.join(out, 'work'), procs=a.procs, omp_threads=1,
               extra_args=(['--libs', a.libs] if a.libs else []))
    t0 = time.time()
    ours_raw, rep = em.run_engine2(test, os.path.join(DATA, 'train.parquet'), os.path.join(out, 'eng3_ours.json'), cfg,
                                   input_root=input_root)
    em.write_json(rep, os.path.join(out, 'eng3_ours_report.json'))
    print(em.report_line(rep).replace('ENGINE 2 OURS subprocess', 'ENG3 OURS subprocess', 1), flush=True)
    if rep.get('log_tail'):
        print(rep['log_tail'])
    key_of = key_of_factory()
    t1 = time.time()
    ours, kc = em.verify_keys(ours_raw, key_of)                    # as frozen cell 39 does with chem.score_key
    res = dict(subprocess_seconds=rep['seconds'], ours_molecules=len(ours), key_check=kc,
               key_check_seconds=round(time.time() - t1, 1), files=rep.get('files'))
    engines = {}
    syn, rewritten = synthetic_engine2(ours, mids, key_of)
    engines['synthetic'] = syn
    res['synthetic_rewritten_smiles'] = rewritten
    if a.engine2 and os.path.isfile(a.engine2):
        real = json.load(open(a.engine2))
        engines['pubm_engine2'] = real
        res['pubm_engine2_key_check'] = m3.check_keys(real, key_of)  # Kaggle canon_key vs the local score_key semantics
        res['pubm_engine2_molecules_match'] = sorted(real) == sorted(mids)
    for name, eng2 in engines.items():
        for w in (1.0, 0.5):
            merged, st = m3.rrf_merge(eng2, ours, w, k=e3.K_RRF, n=e3.N_TOP)
            kept = check_merged(merged, eng2, ours, mids)
            print(f'[{name}] ' + m3.log_line(st, rep['seconds']), flush=True)
            print(f'[{name}] ' + m3.detail_line(st), flush=True)
            m3.write_lists(merged, os.path.join(out, f'merged_{name}_w{w}.json'))
            res[f'{name}_w{w}'] = {k: v for k, v in st.items() if k != 'per_molecule'}
            res[f'{name}_w{w}']['shared_smiles_from_engine2'] = kept
        merged0, st0 = m3.rrf_merge(eng2, {}, 1.0)                 # our subprocess failed: Engine 2 unchanged
        assert merged0 == {str(m): v for m, v in eng2.items()} and st0['merged'] == 0
    print(m3.keys_line(kc, res.get('pubm_engine2_key_check') or {}, time.time() - t1), flush=True)
    res['seconds_total'] = round(time.time() - t0, 1)
    em.write_json(res, os.path.join(out, 'eng3_local_report.json'))
    print(json.dumps({k: v for k, v in res.items() if k != 'files'}, indent=1, default=str))
    print('files:', json.dumps(rep.get('files'), default=str)[:600])
    return 0


if __name__ == '__main__':
    sys.exit(main())
