"""GLACIER simulation worker 3: predicted [M+H]+ spectra at CE 20 / 40 / 60 eV for pool structures (PF-syn)."""
import glob, hashlib, json, os, pickle, shutil, subprocess, sys, time
import numpy as np
t0 = time.time()
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('fwdrun.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
G = os.path.dirname(os.path.dirname(first('ckpt/best.ckpt')))
for name in ['sim2.jobs_3.json', 'sim2.jobs_27.json']:
    base = '/kaggle/working/' + name.replace('.json', '')
    budget = int(38500 - (time.time() - t0) - 1200)
    if budget < 1200:
        print(name, 'skipped: deadline', flush=True); continue
    expected = sum(3 * len(it['cands']) for it in json.load(open(first(name))))
    r = subprocess.run([sys.executable, '-m', 'casmi.fwdrun', first(name), base + '.gl.json', '--src', f'{G}/mspred',
                        '--ckpt', f'{G}/ckpt/best.ckpt', '--site', '/tmp/fwd_site', '--wheels', f'{G}/wheels',
                        '--shim', f'{G}/fwdshim', '--threads', '4', '--budget', str(budget), '--dump-preds', base + '.pkl'],
                       cwd='/kaggle/working/pkg', env={**os.environ, 'PYTHONPATH': '/kaggle/working/pkg'},
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
          'md5', hashlib.md5(open(base + '.sim.npz', 'rb').read()).hexdigest(), f'{time.time() - t0:.0f}s', flush=True)
    for f in (base + '.pkl', base + '.gl.json'):
        if os.path.exists(f):
            os.remove(f)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
