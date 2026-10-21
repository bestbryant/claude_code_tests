"""PeakFormer training on CPU, variant f (code: bryantyu/casmi26-code; data: casmi26-fp-data + casmi26-pf-data)."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
fpx = sorted(glob.glob('/kaggle/input/**/fpx.npz', recursive=True), key=len)[0]
env = dict(os.environ, PF_HOURS='11', PF_THREADS=str(os.cpu_count()), PF_VAL_N='1000', **{'PF_D': '192', 'PF_LAYERS': '4', 'PF_HEADS': '6', 'PF_SEED': '5', 'PF_MERGE': '0.35', 'PF_SBAL': '1.0', 'PF_DMOL': '256', 'PF_DROP': '0', 'PF_LOG': '200'})
print('cpus', os.cpu_count(), flush=True)
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_pf_script.py', data, fpx, '/kaggle/working/pf_model.pt',
                    '40', '1500', '256', '23'], env=env, text=True)
print('exit', r.returncode)
