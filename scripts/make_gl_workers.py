"""Kaggle CPU worker kernels that run GLACIER on prepared job shards (bryantyu/casmi26-gl-jobs).

python scripts/make_gl_workers.py <prefix> <shard> [<shard> ...]
Creates kernels/casmi26-glw-<shard>/ processing every /kaggle/input/**/<prefix>.jobs_<shard>.json -> output
<prefix>.jobs_<shard>.gl.json (runner output {qid: {smiles: score}}).
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prefix, shards = sys.argv[1], sys.argv[2:]
NB = '''"""GLACIER worker: forward-model scores for simulated ranker-training queries (shard {shard} of {prefix})."""
import glob, os, subprocess, sys
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{{pat}}', recursive=True), key=len)[0]
ICE = os.path.dirname(first('ice_runner.py')); GL = os.path.dirname(first('gl_runner.py'))
sys.path.insert(0, ICE)
import ice_runner
SITE = '/tmp/ice_site'          # outside /kaggle/working: its thousands of files would bloat the output listing
ice_runner.install_site(ICE, SITE, print)
for f in sorted(glob.glob('/kaggle/input/**/{prefix}.jobs_{shard}.json', recursive=True)):
    out = '/kaggle/working/' + os.path.basename(f).replace('.json', '.gl.json')
    r = subprocess.run([sys.executable, f'{{GL}}/gl_runner.py', GL, f, out, '--device', 'cpu', '--site', SITE,
                        '--threads', '4'], capture_output=True, text=True)
    print(f, r.stdout[-1500:], r.stderr[-1500:])
'''
for sh in shards:
    d = os.path.join(root, 'kernels', f'casmi26-glw-{sh}')
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'notebook.py'), 'w').write(NB.format(prefix=prefix, shard=sh))
    json.dump({'id': f'bryantyu/casmi26-glw-{sh}', 'title': f'casmi26-glw-{sh}', 'code_file': 'notebook.py',
               'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
               'enable_tpu': False, 'enable_internet': False,
               'dataset_sources': ['bryantyu/casmi26-fwd', 'bryantyu/casmi26-gl-jobs'],
               'competition_sources': [], 'kernel_sources': [], 'model_sources': []},
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    print('wrote', d)
