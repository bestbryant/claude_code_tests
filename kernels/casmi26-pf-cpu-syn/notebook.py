"""CPU variant (scripts/make_pf_cpu_pilot.py): arms ['syn'], 7500 steps of batch 256 per arm on CPU, wall-clock guard 11.3 h per arm.
The TPU pilot's docstring follows.
PF-syn pilot, both arms in one TPU session (Kaggle allows one TPU session at a time, queued ones included):
matched control (real spectra only) then synthetic arm; each fine-tunes PeakFormer b (same config: d=192 x 4, 6 heads, d_mol 256, no dropout,
variant-b data) on a TPU v5e-8 (scripts/train_pf_xla.py, batch 256 per core x 8, 63 negatives, PF_STEPS fixed so that
both arms run the same number of steps) with 25% GLACIER-simulated rows (PF_SYN, kernels/casmi26-syn-prep):
evaluation windows sampled at full-pool density (PF_SYN_NEUTRAL_W 0.18) and half B of them held out
(PF_SYN_DROP, scripts/syn_ab_split.py).
Gate (pre-registered Oct 25, review wf_4f2803ec-ac0; scripts/eval_pf_pair.py ctl vs syn on the 620-query neutral set):
half A: pf_cos delta win vs same-formula training decoys >= +0.06 with the 95% paired-bootstrap interval above 0, delta
win vs COCONUT decoys >= -0.01, excess_syn <= excess_ctl and >= -0.02, and pf_cos_gl (z(pf_cos) + z(GLACIER)) delta
> 0 on GLACIER-covered A queries (PF-syn must add beyond the GLACIER stage 2 already has). Half B (never simulated)
is reported as the generalisation check. Reference: PF-d -> PF-b was +0.124 [+0.095, +0.154] on pf_bern = +0.005 LB."""
import glob, os, shutil, subprocess, sys, time
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
data = os.path.dirname(first('spectra.npz'))
subprocess.run('python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"; nproc; free -g', shell=True)
STEPS = '7500'
base = dict(os.environ, PF_DEV='cpu', PF_HOURS='11.3', PF_VAL_N='800', PF_D='192', PF_LAYERS='4', PF_HEADS='6', PF_DMOL='256',
            PF_DROP='0.0', PF_SBAL='0.5', PF_MERGE='0.35', PF_SEED='31', PF_LR='2e-4', PF_LOG='250', PF_INST_DROP='1.0',
            PF_TIMS_W='1.0', PF_NEG_TRAIN=first('negtrain.npy'), PF_NEG_NOREP='1', PF_CAL='200', PF_INIT=first('pf_model_b.pt'),
            PF_STEPS=STEPS)
arms = [('ctl', {}),
        ('syn', dict(PF_SYN=first('syn.npz'), PF_SYN_P='0.25', PF_SYN_NEUTRAL_W='0.18', PF_SYN_DROP=first('syn_drop_B.txt')))]
arms = [a for a in arms if a[0] in ['syn']]
for name, extra in arms:
    t = time.time()
    print('=== arm', name, flush=True)
    r = subprocess.run([sys.executable, '/kaggle/working/pkg/casmi/train_pf_xla_script.py', data, first('fpx.npz'),
                        f'/kaggle/working/pf_{name}.pt', '1', '7500', '256', '63'], env={**base, **extra}, text=True)
    print('=== arm', name, 'exit', r.returncode, f'{time.time() - t:.0f}s', flush=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
