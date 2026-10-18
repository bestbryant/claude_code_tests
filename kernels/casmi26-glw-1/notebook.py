"""GLACIER worker: forward-model scores for simulated ranker-training queries (shard 1 of val)."""
import glob, os, subprocess, sys
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
ICE = os.path.dirname(first('ice_runner.py')); GL = os.path.dirname(first('gl_runner.py'))
sys.path.insert(0, ICE)
import ice_runner
SITE = '/tmp/ice_site'          # outside /kaggle/working: its thousands of files would bloat the output listing
ice_runner.install_site(ICE, SITE, print)
for f in sorted(glob.glob('/kaggle/input/**/val.jobs_1.json', recursive=True)):
    out = '/kaggle/working/' + os.path.basename(f).replace('.json', '.gl.json')
    r = subprocess.run([sys.executable, f'{GL}/gl_runner.py', GL, f, out, '--device', 'cpu', '--site', SITE,
                        '--threads', '4'], capture_output=True, text=True)
    print(f, r.stdout[-1500:], r.stderr[-1500:])
