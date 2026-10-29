"""CASMI26 own pipeline, kernel casmi26-own-s2v18cr1 (written by scripts/make_own_kernel.py from the s2v10 template).
Candidate pool (bryantyu/casmi26-pool-v18c: train + COCONUT + PubChemLite + ChEBI / LIPID MAPS)
-> features: library (LIB_CAP 10 spectra per structure and polarity; 100000 = all), analog, fragment, prior,
   FP dual encoder (fp_model_v1r.pt), PeakFormer (pf_model_b.pt; used if the ranker has pf_* features)
-> LambdaRank stage 1 (ret_ranker_v18cr1) -> top 30 -> GLACIER predicted-spectrum similarity (own runner on ms-pred, MIT;
   MassSpecGym checkpoint of the ms-pred authors) -> stage-2 LambdaRank (stage2_v18cr1) -> top 25 per molecule.
Run environment: the run_own call below.
Code is ours and the learned weights except GLACIER were trained by us on train.parquet + permitted data; how each
was trained (rows, doc-shift alpha) and the gaps that keep the chain from being fully reproducible are in
docs/reproduce.md; third-party data and models in docs/ATTRIBUTION.md."""
import glob, json, os, shutil, subprocess, sys
import pandas as pd
OWN_RANKER = 'ret_ranker_v18cr1'       # stage-1 ranker (library mode = LIB_CAP below; extra env at the end of the run_own call) in bryantyu/casmi26-models
OWN_STAGE2 = 'stage2_v18cr1'       # stage-2 ranker (GLACIER features)
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
                        'GL_THREADS': '4', 'GL_BUDGET': '14400', 'LIB_CAP': '100000', **{'OWN_PF': 'pf_model_b.pt', 'OWN_FP': 'fp_model_v1r.pt', 'LIB_CAP': '10'}})
print(r.stdout[-3000:]); print(r.stderr[-3000:])
L = json.load(open('/kaggle/working/own_lists.json'))
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or [])][:25]) or 'C' for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('rows with candidates', sum(bool(L.get(str(m))) for m in sub.molecule_id), 'of', len(sub))

# --- LB decomposition probe (scripts/make_probe_kernel.py, base kernel casmi26-own-s2v18cr1, mode filter): the same lists restricted
# to one candidate source (standard InChIKey14 in train / COCONUT-not-train / other)
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
PROBE_SOURCE = 'train'
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
