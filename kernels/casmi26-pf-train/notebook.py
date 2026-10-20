"""Train the PeakFormer spectrum<->structure encoder on GPU (code: bryantyu/casmi26-code, data: bryantyu/casmi26-fp-data)."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
subprocess.run(['nvidia-smi'])
E, S, B, NNEG = os.environ.get('E', '12'), os.environ.get('S', '4000'), '512', '31'
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_pf_script.py', data, '/kaggle/working/pf_model.pt',
                    E, S, B, NNEG], text=True)
print('exit', r.returncode)
