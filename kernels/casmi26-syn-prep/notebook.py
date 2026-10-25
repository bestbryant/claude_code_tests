"""PF-syn data build (scripts/prep_syn_data.py) from the GLACIER simulation workers kernels/casmi26-simw-0..3:
synthetic [M+H]+ spectra (CE 20 / 40 / 60 eV) of 238k pool structures -> /kaggle/working/syn.npz."""
import glob, os, shutil, subprocess, sys
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-2026.3.3-*.whl', recursive=True) if tag in w]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
shutil.copytree(os.path.dirname(first('peakformer.py')), '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
sims = sorted(glob.glob('/kaggle/input/**/*.sim.npz', recursive=True))
print(len(sims), 'simulation files', flush=True)
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/prep_syn_data_script.py', first('sim1_structs.parquet'), first('fpx.npz'),
                    '/kaggle/working/syn.npz'] + sims, env={**os.environ, 'POOL': first('pool.parquet'), 'RICHFP_CACHE': '/none',
                    'PREP_PROCS': '4', 'PYTHONPATH': '/kaggle/working/pkg'}, text=True)
print('exit', r.returncode)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
