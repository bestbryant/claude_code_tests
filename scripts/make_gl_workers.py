"""Kaggle CPU worker kernels that run our GLACIER runner (casmi.fwdrun) on prepared job files.

python scripts/make_gl_workers.py <worker_id>:<file>[,<file>...] [...]
e.g. 0:v2train.jobs_0.json,v2train.jobs_4.json 1:v2train.jobs_1.json
Creates kernels/casmi26-glw-<worker_id>/ reading the job files from bryantyu/casmi26-gl-jobs and writing
<file>.gl.json ({qid: {smiles: score}}) to the kernel output. Inputs: bryantyu/casmi26-code (casmi package),
bryantyu/casmi26-glacier (ms-pred source, GLACIER weights, wheels, DGL stand-in).
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = '''"""GLACIER worker {wid}: forward-model scores for simulated ranker-training queries."""
import glob, os, shutil, subprocess, sys
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{{pat}}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('fwdrun.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
G = os.path.dirname(os.path.dirname(first('ckpt/best.ckpt')))
for name in {files!r}:
    out = '/kaggle/working/' + name.replace('.json', '.gl.json')
    r = subprocess.run([sys.executable, '-m', 'casmi.fwdrun', first(name), out, '--src', f'{{G}}/mspred',
                        '--ckpt', f'{{G}}/ckpt/best.ckpt', '--site', '/tmp/fwd_site', '--wheels', f'{{G}}/wheels',
                        '--shim', f'{{G}}/fwdshim', '--threads', '4'], cwd='/kaggle/working/pkg',
                       env={{**os.environ, 'PYTHONPATH': '/kaggle/working/pkg'}}, capture_output=True, text=True)
    print(name, r.returncode, r.stdout[-2000:], r.stderr[-2000:], flush=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
'''
for spec in sys.argv[1:]:
    wid, files = spec.split(':')
    files = files.split(',')
    d = os.path.join(root, 'kernels', f'casmi26-glw-{wid}')
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'notebook.py'), 'w').write(NB.format(wid=wid, files=files))
    json.dump({'id': f'bryantyu/casmi26-glw-{wid}', 'title': f'casmi26-glw-{wid}', 'code_file': 'notebook.py',
               'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
               'enable_tpu': False, 'enable_internet': False,
               'dataset_sources': ['bryantyu/casmi26-code', 'bryantyu/casmi26-glacier', 'bryantyu/casmi26-gl-jobs'],
               'competition_sources': [], 'kernel_sources': [], 'model_sources': []},
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    print('wrote', d, files)
