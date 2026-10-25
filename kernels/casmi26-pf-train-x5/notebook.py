"""PeakFormer training on a Kaggle TPU v5e-8 (scripts/train_pf_xla.py: static shapes, host-side batches, data parallel
over the 8 chips). Variant b data (instrument-free, window negatives restricted to training POSITIVES, drawn without
replacement, no timsTOF up-weighting; see kernels/casmi26-pf-train-b) with a larger model than the deployed b
(d=384 x 6 layers, 8 heads, molecule embedding 512 vs 192 x 4 / 256): the b run's validation plateaued after ~7 of
its 20 epochs, i.e. the d=192 x 4 model is capacity-bound. The TPU quota (20 h / week) is separate from the GPU
quota shared with the owner."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
fpx = sorted(glob.glob('/kaggle/input/**/fpx.npz', recursive=True), key=len)[0]
negtrain = sorted(glob.glob('/kaggle/input/**/negtrain.npy', recursive=True), key=len)[0]
subprocess.run('python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"; nproc; free -g', shell=True)
env = dict(os.environ, PF_HOURS='8', PF_VAL_N='800', PF_D='384', PF_LAYERS='6', PF_HEADS='8', PF_DMOL='512',
           PF_DROP='0.1', PF_SBAL='0.5', PF_MERGE='0.35', PF_SEED='21', PF_LR='5e-4', PF_LOG='200', PF_INST_DROP='1.0',
           PF_TIMS_W='1.0', PF_NEG_TRAIN=negtrain, PF_NEG_NOREP='1', PF_CAL='200')
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_pf_xla_script.py', data, fpx, '/kaggle/working/pf_model.pt',
                    '40', '8000', '256', '63'], env=env, text=True)
print('exit', r.returncode)
