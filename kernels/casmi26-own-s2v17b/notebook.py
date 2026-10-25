"""CASMI26 own pipeline, kernel casmi26-own-s2v17b (LB 0.339; written by scripts/make_own_kernel.py, docs/reproduce.md
1.6-1.7). Candidate pool v4 (bryantyu/casmi26-pool4: train + COCONUT + PubChemLite + ChEBI / LIPID MAPS)
-> features: library (LIB_CAP 10 spectra per structure and polarity), analog, fragment, prior,
   FP dual encoder (fp_model.pt), PeakFormer (pf_model_b.pt)
-> LambdaRank stage 1 (ret_ranker_v17b, doc-shift weights alpha 1) -> top 30 -> GLACIER predicted-spectrum similarity
   (own runner on ms-pred, MIT; MassSpecGym checkpoint of the ms-pred authors) -> stage-2 LambdaRank (stage2_v17b)
   -> top 25 per molecule.
Code is ours and every learned weight except GLACIER was trained by us on train.parquet + permitted data, but the
chain is not fully reproducible yet: the open gaps (third-party ChEBI / LMSD set, merged stage-2 GLACIER scores, model
training stops, LightGBM determinism, ...) are in docs/reproduce.md section 3; third-party data and models in
docs/ATTRIBUTION.md. Docstring and comments corrected on 2026-10-25; the code is unchanged from the pushed (LB) version.
"""
import glob, json, os, shutil, subprocess, sys
import pandas as pd
OWN_RANKER = 'ret_ranker_v17b'       # stage-1 ranker (library LIB_CAP 10, doc-shift weights alpha 1) in bryantyu/casmi26-models
OWN_STAGE2 = 'stage2_v17b'       # stage-2 ranker (GLACIER features)
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
                        'GL_THREADS': '4', 'GL_BUDGET': '14400', 'LIB_CAP': '100000', **{'OWN_PF': 'pf_model_b.pt', 'LIB_CAP': '10'}})
print(r.stdout[-3000:]); print(r.stderr[-3000:])
L = json.load(open('/kaggle/working/own_lists.json'))
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or [])][:25]) or 'C' for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('rows with candidates', sum(bool(L.get(str(m))) for m in sub.molecule_id), 'of', len(sub))
