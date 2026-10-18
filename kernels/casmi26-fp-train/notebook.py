"""Train the spectrum<->structure dual encoder on GPU (code: bryantyu/casmi26-code, data: bryantyu/casmi26-fp-data)."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/fpmodel.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
EPOCHS, STEPS, BATCH, NNEG = os.environ.get('E', '10'), os.environ.get('S', '3000'), '1024', '31'
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_fp_script.py', data, '/kaggle/working/fp_model.pt',
                    EPOCHS, STEPS, BATCH, NNEG], text=True)
print('exit', r.returncode)
