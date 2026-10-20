"""PeakFormer training on GPU (code: bryantyu/casmi26-code; data: casmi26-fp-data + casmi26-pf-data). Larger model
than the CPU variants: d=384 x 6 layers, molecule embedding 512, 63 window negatives, batch 512, structure-balanced
half batches, ~8.5 h."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
fpx = sorted(glob.glob('/kaggle/input/**/fpx.npz', recursive=True), key=len)[0]
subprocess.run(['nvidia-smi'])
env = dict(os.environ, PF_HOURS='8.5', PF_VAL_N='1500', PF_D='384', PF_LAYERS='6', PF_HEADS='8', PF_DMOL='512',
           PF_DROP='0.1', PF_SBAL='0.5', PF_MERGE='0.35', PF_SEED='10', PF_LR='4e-4', PF_LOG='500')
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_pf_script.py', data, fpx, '/kaggle/working/pf_model.pt',
                    '60', '3000', '512', '63'], env=env, text=True)
print('exit', r.returncode)
