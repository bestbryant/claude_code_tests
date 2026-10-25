"""Recorded retrain of the deployed binned-spectrum FP model fp_model.pt (docs/reproduce.md G4): the original was the
epoch-2 save (step 9,000 of an 8 x 3,000 OneCycle schedule) of a local CPU run that died. Same command with
FP_STOP_EPOCH=3: scripts/train_fp.py <casmi26-fp-data> fp_model.pt 8 3000 512 15 3. Expected validation MRR at epochs
0 / 1 / 2: .5118 / .5346 / .5461 (CPU, 3 threads; bit-identity is not expected across machines)."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/fpmodel.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_fp_script.py', data, '/kaggle/working/fp_model.pt',
                    '8', '3000', '512', '15', '3'], env=dict(os.environ, FP_STOP_EPOCH='3'), text=True)
print('exit', r.returncode)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
