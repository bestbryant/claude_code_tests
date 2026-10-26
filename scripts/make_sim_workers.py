"""Kaggle CPU worker kernels that SIMULATE spectra with our GLACIER runner (casmi.fwdrun --dump-preds) for the job files
of scripts/build_sim_jobs.py (dataset bryantyu/casmi26-sim-jobs).

python scripts/make_sim_workers.py <worker_id>:<file>[,<file>...] [...]
Env SIMW_DS (jobs dataset, default bryantyu/casmi26-sim-jobs), SIMW_PREFIX (kernel slug prefix, default casmi26-simw),
SIMW_DEADLINE (session seconds, default 38500 < the 12 h kernel limit): each file gets the time left minus 1,200 s as
its fwdrun budget (a file is skipped when < 20 min remain), and the log prints "predictions N of expected E" per file.
Creates kernels/<prefix>-<worker_id>/; each job file <f>.json yields <f>.sim.npz in the kernel output:
smiles (str), ce (int16, eV), ptr (int64 offsets), mz (float32, m/z sorted), it (float16, base peak 1) -- GLACIER's
top-100 predicted peaks per (structure, CE); the placeholder similarity scores are discarded.
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = '''"""GLACIER simulation worker {wid}: predicted [M+H]+ spectra at CE 20 / 40 / 60 eV for pool structures (PF-syn)."""
import glob, hashlib, json, os, pickle, shutil, subprocess, sys, time
import numpy as np
t0 = time.time()
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{{pat}}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('fwdrun.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
G = os.path.dirname(os.path.dirname(first('ckpt/best.ckpt')))
for name in {files!r}:
    base = '/kaggle/working/' + name.replace('.json', '')
    budget = int({deadline} - (time.time() - t0) - 1200)
    if budget < 1200:
        print(name, 'skipped: deadline', flush=True); continue
    expected = sum(3 * len(it['cands']) for it in json.load(open(first(name))))
    r = subprocess.run([sys.executable, '-m', 'casmi.fwdrun', first(name), base + '.gl.json', '--src', f'{{G}}/mspred',
                        '--ckpt', f'{{G}}/ckpt/best.ckpt', '--site', '/tmp/fwd_site', '--wheels', f'{{G}}/wheels',
                        '--shim', f'{{G}}/fwdshim', '--threads', '4', '--budget', str(budget), '--dump-preds', base + '.pkl'],
                       cwd='/kaggle/working/pkg', env={{**os.environ, 'PYTHONPATH': '/kaggle/working/pkg'}},
                       capture_output=True, text=True)
    print(name, r.returncode, r.stdout[-1500:], r.stderr[-1500:], flush=True)
    if not os.path.exists(base + '.pkl'):
        continue
    J = pickle.load(open(base + '.pkl', 'rb'))
    keys = sorted(k for k, v in J.items() if v is not None and len(v[0]))
    lens = np.array([len(J[k][0]) for k in keys], np.int64)
    np.savez_compressed(base + '.sim.npz', smiles=np.array([k[0] for k in keys]), ce=np.array([k[2] for k in keys], np.int16),
                        ptr=np.r_[0, np.cumsum(lens)], mz=np.concatenate([J[k][0] for k in keys]).astype(np.float32),
                        it=np.concatenate([J[k][1] for k in keys]).astype(np.float16))
    print(name, 'predictions', len(keys), 'of expected', expected, '(jobs', len(J), ') peaks', int(lens.sum()),
          'md5', hashlib.md5(open(base + '.sim.npz', 'rb').read()).hexdigest(), f'{{time.time() - t0:.0f}}s', flush=True)
    for f in (base + '.pkl', base + '.gl.json'):
        if os.path.exists(f):
            os.remove(f)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
'''
DS = os.environ.get('SIMW_DS', 'bryantyu/casmi26-sim-jobs'); PREFIX = os.environ.get('SIMW_PREFIX', 'casmi26-simw')
DEADLINE = int(os.environ.get('SIMW_DEADLINE', '38500'))
for spec in sys.argv[1:]:
    wid, files = spec.split(':')
    files = files.split(',')
    d = os.path.join(root, 'kernels', f'{PREFIX}-{wid}')
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'notebook.py'), 'w').write(NB.format(wid=wid, files=files, deadline=DEADLINE))
    json.dump({'id': f'bryantyu/{PREFIX}-{wid}', 'title': f'{PREFIX}-{wid}', 'code_file': 'notebook.py',
               'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
               'enable_tpu': False, 'enable_internet': False,
               'dataset_sources': ['bryantyu/casmi26-code', 'bryantyu/casmi26-glacier', DS],
               'competition_sources': [], 'kernel_sources': [], 'model_sources': []},
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    print('wrote', d, files)
