"""PF-syn fine-tune of PeakFormer b on the Kaggle TPU v5e-8 with a strategy probe: the Oct 28 smoke kernel failed in
xmp.spawn ("TPU initialization failed: ... Expected 8 worker addresses, got 1"), and a TPU session waits ~18 h in the
queue, so one session must not be lost to start-up. The kernel first runs a short probe per strategy -- all 8 chips
with PF_TPU_RESET=1 (preset single-process TPU_* variables dropped before xmp.spawn), else one chip (PF_CORES=1, batch
1024) -- and then trains with the first strategy that completed its probe; the step budget is calibrated on the chosen
strategy (PF_CAL steps, no PF_STEPS) so that the cosine schedule ends inside the session.

python scripts/make_pf_tpu_kernel.py <slug> <syn source kernel | -> <syn npz name | -> <seed> [hours] [val_n] [log]
e.g. python scripts/make_pf_tpu_kernel.py casmi26-pfs-tpu-p casmi26-syn-prep2p syn2p.npz 41 8.3
'-' for the syn source = matched control (real spectra only). Model, data, LR, negatives and synthetic-row settings are
the TPU pilot's (kernels/casmi26-pf-pilot): PF_SYN_P 0.25, no neutral weighting, no dropped half.
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, syn_src, syn_name, seed = sys.argv[1:5]
hours = float(sys.argv[5]) if len(sys.argv) > 5 else 8.3
val_n = sys.argv[6] if len(sys.argv) > 6 else '600'
log = sys.argv[7] if len(sys.argv) > 7 else '500'
syn = syn_src != '-'
NB = f'''"""PeakFormer b fine-tune on the TPU v5e-8 (scripts/make_pf_tpu_kernel.py): {"synthetic arm, PF_SYN " + syn_name + " from " + syn_src if syn else "matched control (real spectra only)"},
PF_SEED {seed}, session budget {hours} h, PF_VAL_N {val_n}. Strategy probe first (8 chips with PF_TPU_RESET=1, else one
chip with batch 1024), then the run with the step budget calibrated on the chosen strategy (PF_CAL 200)."""
import glob, os, shutil, signal, subprocess, sys, time
T0 = time.time()
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{{pat}}', recursive=True), key=len)[0]
data = os.path.dirname(first('spectra.npz'))
subprocess.run('python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"; nproc; free -g; env | grep -E "^(TPU|PJRT|XLA)" | sort', shell=True)
base = dict(os.environ, PF_D='192', PF_LAYERS='4', PF_HEADS='6', PF_DMOL='256', PF_DROP='0.0', PF_SBAL='0.5', PF_MERGE='0.35',
            PF_SEED='{seed}', PF_LR='2e-4', PF_INST_DROP='1.0', PF_TIMS_W='1.0', PF_NEG_TRAIN=first('negtrain.npy'),
            PF_NEG_NOREP='1', PF_INIT=first('pf_model_b.pt'))
{"base.update(PF_SYN=first(" + repr(syn_name) + "), PF_SYN_P='0.25')" if syn else "# matched control: no PF_SYN"}
script = '/kaggle/working/pkg/casmi/train_pf_xla_script.py'


def go(extra, args, out, timeout=None):
    t = time.time()
    p = subprocess.Popen([sys.executable, script, data, first('fpx.npz'), out] + args, env={{**base, **extra}}, text=True,
                         start_new_session=True)          # own process group: a timed-out probe takes its cores with it
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = 'timeout'
    print('=== run', extra, args, 'exit', rc, f'{{time.time() - t:.0f}}s', flush=True)
    return rc == 0 and os.path.exists(out)


strategies = [('8-chip', dict(PF_TPU_RESET='1'), '256'), ('1-chip', dict(PF_CORES='1'), '1024')]
chosen = None
for name, extra, bsz in strategies:
    if go(dict(extra, PF_HOURS='0.25', PF_CAL='40', PF_LOG='20', PF_VAL_N='100'), ['1', '120', bsz, '63'],
          f'/kaggle/working/probe_{{name}}.pt', timeout=1800):
        chosen = (name, extra, bsz)
        break
for f in glob.glob('/kaggle/working/probe_*.pt'):
    os.remove(f)
assert chosen, 'no TPU strategy completed its probe'
left = {hours} * 3600 - (time.time() - T0) - 1200          # 20 min for the last validation, the save and the output upload
print('=== strategy', chosen[0], 'budget', round(left / 3600, 2), 'h', flush=True)
ok = go(dict(chosen[1], PF_HOURS=f'{{left / 3600:.2f}}', PF_CAL='200', PF_LOG='{log}', PF_VAL_N='{val_n}'),
        ['16', '10000', chosen[2], '63'], '/kaggle/working/pf_{slug}.pt')
print('=== done', ok, f'{{time.time() - T0:.0f}}s', flush=True)
subprocess.run('md5sum /kaggle/working/*.pt', shell=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
'''
d = os.path.join(root, 'kernels', slug)
os.makedirs(d, exist_ok=True)
open(os.path.join(d, 'notebook.py'), 'w').write(NB)
json.dump({'id': f'bryantyu/{slug}', 'title': slug, 'code_file': 'notebook.py', 'language': 'python', 'kernel_type': 'script',
           'is_private': True, 'enable_gpu': False, 'enable_tpu': True, 'machine_shape': 'TpuV5E8', 'enable_internet': False,
           'dataset_sources': ['bryantyu/casmi26-code', 'bryantyu/casmi26-fp-data', 'bryantyu/casmi26-pf-data',
                               'bryantyu/casmi26-pf-negtrain', 'bryantyu/casmi26-models'],
           'competition_sources': [], 'kernel_sources': [f'bryantyu/{syn_src}'] if syn else [], 'model_sources': []},
          open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
print('wrote', d)
