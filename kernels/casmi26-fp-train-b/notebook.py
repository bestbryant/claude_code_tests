"""Train the spectrum<->structure dual encoder on GPU (code: bryantyu/casmi26-code, data: bryantyu/casmi26-fp-data).
Variant b: instrument-free (token always dropped), no timsTOF up-weighting, window negatives restricted to training
structures (bryantyu/casmi26-pf-negtrain) -- no "COCONUT-not-train is never the answer" prior."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/fpmodel.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
EPOCHS, STEPS, BATCH, NNEG = os.environ.get('E', '10'), os.environ.get('S', '3000'), '1024', '31'
negtrain = sorted(glob.glob('/kaggle/input/**/negtrain.npy', recursive=True), key=len)[0]
env = dict(os.environ, FP_NEG_TRAIN=negtrain, FP_INST_DROP='1.0', FP_TIMS_W='1.0')
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_fp_script.py', data, '/kaggle/working/fp_model.pt',
                    EPOCHS, STEPS, BATCH, NNEG], env=env, text=True)
print('exit', r.returncode)
