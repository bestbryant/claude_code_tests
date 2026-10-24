"""PeakFormer training on GPU (code: bryantyu/casmi26-code; data: casmi26-fp-data + casmi26-pf-data). Same size as
the deployed local model d (d=192 x 4 layers, 6 heads, molecule embedding 256, no dropout) so that b vs d isolates the
training-data change; 63 window negatives, batch 512, structure-balanced half batches, 5 h (the d=384 x 6 config ran out
of T4 memory at batch 512 on Oct 24; the GPU quota is shared with the owner).
Variant b: instrument-free (token always dropped), no timsTOF up-weighting, window negatives restricted to training
structures that are training POSITIVES (split 0; no "COCONUT-not-train is never the answer" prior, held-out truths
unseen like hidden answers), drawn without replacement; token columns tied at save (any token = token 0)."""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
data = os.path.dirname(sorted(glob.glob('/kaggle/input/**/spectra.npz', recursive=True), key=len)[0])
fpx = sorted(glob.glob('/kaggle/input/**/fpx.npz', recursive=True), key=len)[0]
negtrain = sorted(glob.glob('/kaggle/input/**/negtrain.npy', recursive=True), key=len)[0]   # bryantyu/casmi26-pf-negtrain
subprocess.run(['nvidia-smi'])
env = dict(os.environ, PF_HOURS='5', PF_VAL_N='1500', PF_D='192', PF_LAYERS='4', PF_HEADS='6', PF_DMOL='256',
           PF_DROP='0.0', PF_SBAL='0.5', PF_MERGE='0.35', PF_SEED='11', PF_LR='4e-4', PF_LOG='500', PF_INST_DROP='1.0', PF_TIMS_W='1.0', PF_NEG_TRAIN=negtrain,
           PF_NEG_NOREP='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_pf_script.py', data, fpx, '/kaggle/working/pf_model.pt',
                    '60', '3000', '512', '63'], env=env, text=True)
print('exit', r.returncode)
