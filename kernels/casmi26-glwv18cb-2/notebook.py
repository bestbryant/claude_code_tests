"""GLACIER worker 2: forward-model scores for simulated ranker-training queries."""
import glob, os, shutil, subprocess, sys
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('fwdrun.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
G = os.path.dirname(os.path.dirname(first('ckpt/best.ckpt')))
for name in ['train.jobs_2.json', 'val.jobs_0.json']:
    out = '/kaggle/working/' + name.replace('.json', '.gl.json')
    r = subprocess.run([sys.executable, '-m', 'casmi.fwdrun', first(name), out, '--src', f'{G}/mspred',
                        '--ckpt', f'{G}/ckpt/best.ckpt', '--site', '/tmp/fwd_site', '--wheels', f'{G}/wheels',
                        '--shim', f'{G}/fwdshim', '--threads', '4'], cwd='/kaggle/working/pkg',
                       env={**os.environ, 'PYTHONPATH': '/kaggle/working/pkg'}, capture_output=True, text=True)
    print(name, r.returncode, r.stdout[-2000:], r.stderr[-2000:], flush=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
