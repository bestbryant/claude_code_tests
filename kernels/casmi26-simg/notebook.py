"""GLACIER simulation on GPU (T4 x 2) for the PF-syn pilot: the job files of scripts/build_sim_jobs.py
(bryantyu/casmi26-sim-jobs, 238k pool structures x CE 20 / 40 / 60 eV) split over the two GPUs, casmi.fwdrun
--device cuda (the network forward pass is ~95% of the CPU time: ~1 prediction/s per 4-core CPU kernel). Output per
job file: <f>.sim.npz as in scripts/make_sim_workers.py. Hard wall-clock deadline DEADLINE_S."""
import glob, os, pickle, shutil, subprocess, sys, time
import numpy as np
t0 = time.time(); DEADLINE_S = 5400
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('fwdrun.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
G = os.path.dirname(os.path.dirname(first('ckpt/best.ckpt')))
subprocess.run(['nvidia-smi'])
files = sorted(glob.glob('/kaggle/input/**/sim1.jobs_*.json', recursive=True), key=lambda f: int(f.split('_')[-1].split('.')[0]))
SCRIPT = '''
import os, subprocess, sys, time
gpu, files, t_end = sys.argv[1], sys.argv[2].split(','), float(sys.argv[3])
for f in files:
    base = '/kaggle/working/' + os.path.basename(f).replace('.json', '')
    left = t_end - time.time()
    if left < 120:
        print('deadline: skipping', f, flush=True); continue
    r = subprocess.run([sys.executable, '-m', 'casmi.fwdrun', f, base + '.gl.json', '--src', '%(G)s/mspred', '--ckpt', '%(G)s/ckpt/best.ckpt',
                        '--site', '/tmp/fwd_site' + gpu, '--wheels', '%(G)s/wheels', '--shim', '%(G)s/fwdshim', '--threads', '2', '--bs', '128',
                        '--device', 'cuda', '--budget', str(int(left - 60)), '--dump-preds', base + '.pkl'],
                       cwd='/kaggle/working/pkg', env={**os.environ, 'PYTHONPATH': '/kaggle/working/pkg', 'CUDA_VISIBLE_DEVICES': gpu},
                       capture_output=True, text=True)
    print(f, r.returncode, r.stdout[-800:], r.stderr[-800:], flush=True)
''' % dict(G=G)
open('/kaggle/working/run_gpu.py', 'w').write(SCRIPT)
t_end = t0 + DEADLINE_S
procs = [subprocess.Popen([sys.executable, '/kaggle/working/run_gpu.py', str(g), ','.join(files[g::2]), str(t_end)]) for g in (0, 1)]
for p in procs:
    p.wait()
tot = 0
for f in files:
    base = '/kaggle/working/' + os.path.basename(f).replace('.json', '')
    if not os.path.exists(base + '.pkl'):
        print('missing', base, flush=True); continue
    J = pickle.load(open(base + '.pkl', 'rb'))
    keys = sorted(k for k, v in J.items() if v is not None and len(v[0]))
    lens = np.array([len(J[k][0]) for k in keys], np.int64)
    np.savez_compressed(base + '.sim.npz', smiles=np.array([k[0] for k in keys]), ce=np.array([k[2] for k in keys], np.int16),
                        ptr=np.r_[0, np.cumsum(lens)], mz=np.concatenate([J[k][0] for k in keys]).astype(np.float32),
                        it=np.concatenate([J[k][1] for k in keys]).astype(np.float16))
    print(os.path.basename(f), 'predictions', len(keys), 'of', len(J), flush=True); tot += len(keys)
    for x in (base + '.pkl', base + '.gl.json'):
        if os.path.exists(x):
            os.remove(x)
print('total predictions', tot, f'{time.time() - t0:.0f}s', flush=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
