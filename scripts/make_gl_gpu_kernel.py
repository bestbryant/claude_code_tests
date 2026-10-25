"""Kaggle GPU kernel (T4 x 2) that runs our GLACIER runner on every job file of a jobs dataset (casmi.fwdrun --device
cuda; one runner per GPU, files alternated) and writes <file>.gl.json scores to the output -- the one-pass GLACIER
scoring of scripts/build_v18c.sh stage E (docs/reproduce.md G2).

python scripts/make_gl_gpu_kernel.py <kernel_slug> <jobs_dataset_slug> [deadline_s]
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, ds = sys.argv[1], sys.argv[2]
deadline = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
NB = '''"""GLACIER scores on GPU for the job files of bryantyu/{ds} (scripts/make_gl_gpu_kernel.py; casmi.fwdrun --device cuda,
one runner per T4, batch 128, wall-clock deadline {deadline} s)."""
import glob, os, shutil, subprocess, sys, time
t0 = time.time()
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{{pat}}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('fwdrun.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
G = os.path.dirname(os.path.dirname(first('ckpt/best.ckpt')))
subprocess.run(['nvidia-smi'])
files = sorted(f for f in glob.glob('/kaggle/input/{ds}/**/*.jobs_*.json', recursive=True) if not f.endswith('.gl.json'))
print(len(files), 'job files', flush=True)
SCRIPT = """
import os, subprocess, sys, time
gpu, files, t_end = sys.argv[1], sys.argv[2].split(','), float(sys.argv[3])
for f in files:
    out = '/kaggle/working/' + os.path.basename(f).replace('.json', '.gl.json')
    left = t_end - time.time()
    if left < 120:
        print('deadline: skipping', f, flush=True); continue
    r = subprocess.run([sys.executable, '-m', 'casmi.fwdrun', f, out, '--src', '%(G)s/mspred', '--ckpt', '%(G)s/ckpt/best.ckpt',
                        '--site', '/tmp/fwd_site' + gpu, '--wheels', '%(G)s/wheels', '--shim', '%(G)s/fwdshim', '--threads', '2',
                        '--bs', '128', '--device', 'cuda', '--budget', str(int(left - 60))],
                       cwd='/kaggle/working/pkg', env={{**os.environ, 'PYTHONPATH': '/kaggle/working/pkg', 'CUDA_VISIBLE_DEVICES': gpu}},
                       capture_output=True, text=True)
    print(f, r.returncode, r.stdout[-600:], r.stderr[-600:], flush=True)
""" % dict(G=G)
open('/kaggle/working/run_gpu.py', 'w').write(SCRIPT)
t_end = t0 + {deadline}
procs = [subprocess.Popen([sys.executable, '/kaggle/working/run_gpu.py', str(g), ','.join(files[g::2]), str(t_end)]) for g in (0, 1)]
for p in procs:
    p.wait()
print('outputs', sorted(os.path.basename(f) for f in glob.glob('/kaggle/working/*.gl.json')), f'{{time.time() - t0:.0f}}s', flush=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
'''
d = os.path.join(root, 'kernels', slug)
os.makedirs(d, exist_ok=True)
open(os.path.join(d, 'notebook.py'), 'w').write(NB.format(ds=ds, deadline=deadline))
json.dump({'id': f'bryantyu/{slug}', 'title': slug, 'code_file': 'notebook.py', 'language': 'python', 'kernel_type': 'script',
           'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
           'dataset_sources': ['bryantyu/casmi26-code', 'bryantyu/casmi26-glacier', f'bryantyu/{ds}'],
           'competition_sources': [], 'kernel_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'},
          open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
print('wrote', d)
