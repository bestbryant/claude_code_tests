"""PeakFormer b fine-tune on the TPU v5e-8 (scripts/make_pf_tpu_kernel.py): synthetic arm, PF_SYN syn2.npz from casmi26-syn-prep2,
PF_SEED 43, session budget 8.3 h, PF_VAL_N 600. Strategy probe first (8 chips with PF_TPU_RESET=1, else one
chip with batch 1024), then the run with the step budget calibrated on the chosen strategy (steps 600-900)."""
import glob, os, shutil, signal, subprocess, sys, time
T0 = time.time()
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
data = os.path.dirname(first('spectra.npz'))
subprocess.run('python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"; nproc; free -g; env | grep -E "^(TPU|PJRT|XLA)" | sort', shell=True)
base = dict(os.environ, PF_D='192', PF_LAYERS='4', PF_HEADS='6', PF_DMOL='256', PF_DROP='0.0', PF_SBAL='0.5', PF_MERGE='0.35',
            PF_SEED='43', PF_LR='2e-4', PF_INST_DROP='1.0', PF_TIMS_W='1.0', PF_NEG_TRAIN=first('negtrain.npy'),
            PF_NEG_NOREP='1', PF_INIT=first('pf_model_b.pt'))
base.update(PF_SYN=first('syn2.npz'), PF_SYN_P='0.25')
script = '/kaggle/working/pkg/casmi/train_pf_xla_script.py'


def go(extra, args, out, timeout=None):
    t = time.time()
    p = subprocess.Popen([sys.executable, script, data, first('fpx.npz'), out] + args, env={**base, **extra}, text=True,
                         start_new_session=True)          # own process group: a timed-out probe takes its cores with it
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = 'timeout'
    print('=== run', extra, args, 'exit', rc, f'{time.time() - t:.0f}s', flush=True)
    return rc == 0 and os.path.exists(out)


strategies = [('8-chip', dict(PF_TPU_RESET='1'), '256'), ('1-chip', dict(PF_CORES='1'), '1024')]
chosen = None
for name, extra, bsz in strategies:
    if go(dict(extra, PF_HOURS='0.25', PF_CAL='40', PF_LOG='20', PF_VAL_N='100'), ['1', '120', bsz, '63'],
          f'/kaggle/working/probe_{name}.pt', timeout=1800):
        chosen = (name, extra, bsz)
        break
for f in glob.glob('/kaggle/working/probe_*.pt'):
    os.remove(f)
assert chosen, 'no TPU strategy completed its probe'
left = 8.3 * 3600 - (time.time() - T0) - 1200          # 20 min for the last validation, the save and the output upload
print('=== strategy', chosen[0], 'budget', round(left / 3600, 2), 'h', flush=True)
ok = go(dict(chosen[1], PF_HOURS=f'{left / 3600:.2f}', PF_CAL='300', PF_CAL_START='600', PF_LOG='500', PF_VAL_N='600', PF_STEPS='40000'),
        ['40', '10000', chosen[2], '63'], '/kaggle/working/pf_casmi26-pfs-tpu-f.pt')
print('=== done', ok, f'{time.time() - T0:.0f}s', flush=True)
subprocess.run('md5sum /kaggle/working/*.pt', shell=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
