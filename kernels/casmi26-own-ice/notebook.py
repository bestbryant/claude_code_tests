"""CASMI26 own pipeline (all code and weights ours, reproducible from this repository + permitted data):
candidate pool (train + COCONUT [+ PubChemLite]) -> library / analog / fragment / prior / dual-encoder features ->
LambdaRank -> top 25 per molecule -> ICEBERG (ms-pred, MassSpecGym weights, MIT) re-scoring of same-formula
groups: z(ranker) + LAM * z(predicted-spectrum similarity)."""
import glob, json, os, shutil, subprocess, sys
import pandas as pd
OWN_RANKER = 'ret_ranker_v2'      # ranker variant shipped in bryantyu/casmi26-models
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-2026.3.3-*.whl', recursive=True) if tag in w]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('retrieval.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
COMP = os.path.dirname(first('test.parquet'))
POOL = os.path.dirname(first('pool.parquet'))
MODELS = os.path.dirname(first(OWN_RANKER + '.txt'))
r = subprocess.run([sys.executable, '-m', 'casmi.run_own', f'{COMP}/test.parquet', f'{COMP}/train.parquet', POOL, MODELS,
                    '/kaggle/working/own_lists.json', '/kaggle/working/own', '4'], capture_output=True, text=True,
                   cwd='/kaggle/working/pkg', env={**os.environ, 'PYTHONPATH': '/kaggle/working/pkg', 'NUMBA_NUM_THREADS': '1', 'OWN_RANKER': OWN_RANKER})
print(r.stdout[-3000:]); print(r.stderr[-3000:])
L = json.load(open('/kaggle/working/own_lists.json'))

# --- forward-model re-scoring (optional: any failure keeps the ranker order)
ICE_N, LAM, ICE_BUDGET = 20, 0.5, 4 * 3600
try:
    from rdkit.Chem import rdMolDescriptors
    from rdkit import Chem
    ICE_PKG = os.path.dirname(first('ice_runner.py'))
    sys.path.insert(0, ICE_PKG)
    from fuse import build_ice_input, ice_candidates, run_ice, rerank
    def formula(s):
        m = Chem.MolFromSmiles(s)
        return rdMolDescriptors.CalcMolFormula(m) if m is not None else s
    T = pd.read_parquet(f'{COMP}/test.parquet'); T['molecule_id'] = T.molecule_id.astype(str)
    F = {m: [formula(c[0]) for c in v[:ICE_N]] for m, v in L.items() if v}
    cands = {m: ice_candidates([c[0] for c in L[m][:ICE_N]], F[m], top_n=ICE_N) for m in F}
    cands = {m: c for m, c in cands.items() if c}
    items = build_ice_input(T, cands)
    ice = run_ice(ICE_PKG, items, workdir='/kaggle/working/ice_work', device='cpu', budget_s=ICE_BUDGET,
                  site='/kaggle/working/ice_site', extra_args=['--threads', '4'])
    moved = 0
    for m in F:
        v = L[m]
        order = rerank([c[0] for c in v], [c[2] for c in v], [c[1] for c in v], F[m] + [None] * (len(v) - len(F[m])),
                       ice.get(m, {}), lam=LAM, top_n=ICE_N)
        moved += order[0] != 0
        L[m] = [v[i] for i in order]
    print('ICE molecules scored', sum(any(x is not None for x in d.values()) for d in ice.values()), 'top1 changed', moved)
except Exception as e:
    print('ICE skipped:', repr(e))
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or [])][:25]) or 'C' for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('rows with candidates', sum(bool(L.get(str(m))) for m in sub.molecule_id), 'of', len(sub))
