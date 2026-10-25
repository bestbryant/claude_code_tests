"""PF-syn pilot (synthetic): fine-tune PeakFormer b (same config: d=192 x 4, 6 heads, d_mol 256, no dropout, variant-b data)
for 4 h on a TPU v3-8 (scripts/train_pf_xla.py, batch 256 per core x 8, 63 negatives) with 25% GLACIER-simulated rows (PF_SYN, kernels/casmi26-syn-prep). Decision (docs/plan.md):
syn vs ctl on the 620-query neutral metric -- pf_bern / pf_cos win rate vs same-formula training decoys >= +0.03,
win vs COCONUT decoys >= -0.01, excess <= PF-b's +0.044."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
data = os.path.dirname(first('spectra.npz'))
subprocess.run('python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"; nproc; free -g', shell=True)
env = dict(os.environ, PF_HOURS='4', PF_VAL_N='800', PF_D='192', PF_LAYERS='4', PF_HEADS='6', PF_DMOL='256',
           PF_DROP='0.0', PF_SBAL='0.5', PF_MERGE='0.35', PF_SEED='31', PF_LR='2e-4', PF_LOG='200', PF_INST_DROP='1.0',
           PF_TIMS_W='1.0', PF_NEG_TRAIN=first('negtrain.npy'), PF_NEG_NOREP='1', PF_CAL='200', PF_INIT=first('pf_model_b.pt'), PF_SYN=first('syn.npz'), PF_SYN_P='0.25')
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_pf_xla_script.py', data, first('fpx.npz'), '/kaggle/working/pf_model.pt',
                    '40', '4000', '256', '63'], env=env, text=True)
print('exit', r.returncode)
