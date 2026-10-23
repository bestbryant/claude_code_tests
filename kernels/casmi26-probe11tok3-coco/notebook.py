"""CASMI26 own pipeline (all code and weights ours, reproducible from this repository + permitted data):
candidate pool (train + COCONUT + PubChemLite + ChEBI / LIPID MAPS) -> library / analog / fragment / prior / dual-encoder features ->
LambdaRank (stage 1) -> top 30 -> GLACIER predicted-spectrum similarity (own runner on ms-pred, MIT; MassSpecGym weights)
-> stage-2 LambdaRank with forward-model features -> top 25 per molecule."""
import glob, json, os, shutil, subprocess, sys
import pandas as pd
OWN_RANKER = 'ret_ranker_v11l'   # stage-1 ranker v11l (PeakFormer features, doc-shift alpha 1)
OWN_STAGE2 = 'stage2_v11l'       # stage-2 ranker (GLACIER features)
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
                        'GL_THREADS': '4', 'GL_BUDGET': '14400', 'LIB_CAP': '100000', 'OWN_PF': 'pf_local_final.pt', 'OWN_PF_INST': '3'})
print(r.stdout[-3000:]); print(r.stderr[-3000:])
L = json.load(open('/kaggle/working/own_lists.json'))
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or [])][:25]) or 'C' for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('rows with candidates', sum(bool(L.get(str(m))) for m in sub.molecule_id), 'of', len(sub))

# --- LB decomposition probe: the same lists restricted to one candidate source (same key sets as the public-lineage
# probes kernels/casmi26-fprobe-*: standard InChIKey14 in train / COCONUT-not-train / other)
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
PROBE_SOURCE = 'coco'
K = pd.read_parquet(first('casmi_probe_srckeys.parquet'))
TR, CO = set(K.key[K.src == 'train']), set(K.key[K.src == 'coco'])
def _src(smi):
    m = Chem.MolFromSmiles(smi)
    k = Chem.MolToInchiKey(m)[:14] if m is not None else None
    return 'train' if k in TR else 'coco' if k in CO else 'other'
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or []) if _src(c[0]) == PROBE_SOURCE][:25]) or 'C'
                 for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('PROBE', PROBE_SOURCE, 'rows with candidates', int((sub.smiles != 'C').sum()), 'of', len(sub))
