"""CASMI26 own pipeline (all code and weights ours, reproducible from this repository + permitted data):
candidate pool (train + COCONUT + PubChemLite + ChEBI / LIPID MAPS) -> library / analog / fragment / prior / dual-encoder features ->
LambdaRank (stage 1) -> top 30 -> GLACIER predicted-spectrum similarity (own runner on ms-pred, MIT; MassSpecGym weights)
-> stage-2 LambdaRank with forward-model features -> top 25 per molecule."""
import glob, json, os, shutil, subprocess, sys
import pandas as pd
OWN_RANKER = 'ret_ranker_v5w'     # stage-1 ranker (documentation-shift weighted class 2) in bryantyu/casmi26-models
OWN_STAGE2 = 'stage2_v5w'        # stage-2 ranker (GLACIER features)
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
                   cwd='/kaggle/working/pkg', env={**os.environ, 'PYTHONPATH': '/kaggle/working/pkg', 'NUMBA_NUM_THREADS': '1', 'OWN_RANKER': OWN_RANKER,
                        'OWN_STAGE2': OWN_STAGE2, 'GLACIER_DIR': os.path.dirname(os.path.dirname(first('ckpt/best.ckpt'))),
                        'GL_THREADS': '4', 'GL_BUDGET': '14400'})
print(r.stdout[-3000:]); print(r.stderr[-3000:])
L = json.load(open('/kaggle/working/own_lists.json'))
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or [])][:25]) or 'C' for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('rows with candidates', sum(bool(L.get(str(m))) for m in sub.molecule_id), 'of', len(sub))

# --- source tiering ("olast"): within every list, candidates whose InChIKey14 is a training structure or a
# COCONUT-not-train structure keep their order and come first; all other candidates follow in their own order. Nothing
# is dropped (pure post-processing of the v5w lists; paired A/B against s2v5w 0.332).
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
K = pd.read_parquet(first('casmi_probe_srckeys.parquet'))
TC = set(K.key[K.src.isin(['train', 'coco'])])
def _tc(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None and Chem.MolToInchiKey(m)[:14] in TC
moved = 0
rows = []
for m in sub.molecule_id:
    c = [x[0] for x in (L.get(str(m)) or [])][:25]
    a = [s for s in c if _tc(s)]; b = [s for s in c if not _tc(s)]
    new = a + b
    assert sorted(new) == sorted(c)
    moved += new != c
    rows.append(';'.join(new) or 'C')
sub['smiles'] = rows
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('OLAST molecules reordered', moved, 'of', len(sub))
