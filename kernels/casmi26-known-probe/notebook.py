"""CASMI26 probe: candidates from the analog-derivative / conjugate engine that ARE known structures
(PubChem or COCONUT), ranked by site-consistency + analog similarity. Measures class-2 power of the engine alone."""
import glob, json, os, shutil, subprocess, sys, time
import pandas as pd
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-2026.3.3-*.whl', recursive=True) if tag in w]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/novel.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
COMP = os.path.dirname(sorted(glob.glob('/kaggle/input/**/test.parquet', recursive=True), key=len)[0])
pck = sorted(glob.glob('/kaggle/input/**/pc_keys_u64.npy', recursive=True), key=len)[0]
r = subprocess.run([sys.executable, '-m', 'casmi.run_novel', f'{COMP}/test.parquet', f'{COMP}/train.parquet', pck,
                    '/kaggle/working/lists.json', '-', '/kaggle/working/novel', '4'], capture_output=True, text=True,
                   cwd='/kaggle/working/pkg', env={**os.environ, 'PYTHONPATH': '/kaggle/working/pkg', 'NUMBA_NUM_THREADS': '1',
                                                   'CASMI_NOVEL_MODE': os.environ.get('CASMI_NOVEL_MODE', 'known')})
print(r.stdout[-3000:]); print(r.stderr[-3000:])
L = json.load(open('/kaggle/working/lists.json'))
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or [])][:25]) or 'C' for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('rows with candidates', sum(bool(L.get(str(m))) for m in sub.molecule_id))
