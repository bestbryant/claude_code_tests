"""PF-x-syn on the TPU v5e-8 (roadmap Nov 1 - Dec 14, item S1): a larger PeakFormer (d=384 x 6 layers, 8 heads, d_mol 512,
dropout 0.1, ~25M parameters) trained FROM SCRATCH (no PF_INIT) on the tpu_f data recipe (real spectra + GLACIER-simulated
spectra of the full pool, syn2.npz from casmi26-syn-prep2, PF_SYN_P 0.25), meant as a decorrelated ensemble member next to
pf_tpu_f (same-start fine-tunes correlate 0.98 within a query and do not ensemble).
8 chips only (PF_TPU_RESET=1; a 1-chip run could not finish the steps). A 120-step probe checks the config (memory,
compile) only: its step times are warm-up dominated (3.29 / 0.64 s/step in the tpu_p / tpu_f probes vs ~0.25 s steady
state; steps 500-1000 of tpu_f still ran at 0.8 s). The main run has a FIXED PF_STEPS of 60,000 (x 2,048 = 123M samples;
~5 h if the larger model stays host-bound near 0.3 s/step); the PF_HOURS guard only stops a pathological run (VOID by the
roadmap rule). Versions go to pfx_env.json."""
import glob, json, os, re, shutil, signal, subprocess, sys, time
T0 = time.time()
src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/peakformer.py', recursive=True), key=len)[0])
shutil.copytree(src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
data = os.path.dirname(first('spectra.npz'))
subprocess.run('python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"; nproc; free -g; env | grep -E "^(TPU|PJRT|XLA)" | sort', shell=True)
base = dict(os.environ, PF_D='384', PF_LAYERS='6', PF_HEADS='8', PF_DMOL='512', PF_DROP='0.1', PF_SBAL='0.5', PF_MERGE='0.35',
            PF_SEED='51', PF_LR='5e-4', PF_INST_DROP='1.0', PF_TIMS_W='1.0', PF_NEG_TRAIN=first('negtrain.npy'),
            PF_NEG_NOREP='1', PF_TPU_RESET='1', PF_SYN=first('syn2.npz'), PF_SYN_P='0.25')
script = '/kaggle/working/pkg/casmi/train_pf_xla_script.py'


def go(extra, args, out, timeout=None):
    t = time.time(); lines = []
    p = subprocess.Popen([sys.executable, script, data, first('fpx.npz'), out] + args, env={**base, **extra}, text=True,
                         start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    try:
        for line in p.stdout:
            print(line, end='', flush=True); lines.append(line)
            if timeout and time.time() - t > timeout:
                raise subprocess.TimeoutExpired(p.args, timeout)
        rc = p.wait()
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = 'timeout'
    print('=== run', extra, args, 'exit', rc, f'{time.time() - t:.0f}s', flush=True)
    return rc == 0 and os.path.exists(out), ''.join(lines)


ok, log = go(dict(PF_HOURS='0.4', PF_CAL='40', PF_LOG='20', PF_VAL_N='100'), ['1', '120', '256', '63'],
             '/kaggle/working/probe.pt', timeout=2400)
for f in glob.glob('/kaggle/working/probe*.pt'):
    os.remove(f)
assert ok, '8-chip probe failed'
m = re.findall(r'([0-9.]+) s/step after compile', log)
per = float(m[-1]) if m else None
steps = 60000
left = 8.3 * 3600 - (time.time() - T0) - 1200
print('=== probe', per, 's/step (warm-up) | PF_STEPS', steps, '| guard', round(left / 3600, 2), 'h', flush=True)
ok, _ = go(dict(PF_HOURS=f'{left / 3600:.2f}', PF_LOG='500', PF_VAL_N='600', PF_STEPS=str(steps)),
           ['40', '10000', '256', '63'], '/kaggle/working/pf_pfx_tpu.pt')
try:
    import torch, torch_xla
    VER = dict(torch=torch.__version__, torch_xla=torch_xla.__version__)
except Exception as e:
    VER = dict(error=repr(e))
json.dump(dict(steps=steps, probe_s_per_step=per, **VER,
               seconds=round(time.time() - T0)), open('/kaggle/working/pfx_env.json', 'w'))
print('=== done', ok, f'{time.time() - T0:.0f}s', flush=True)
subprocess.run('md5sum /kaggle/working/*.pt; cat /kaggle/working/pfx_env.json', shell=True)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
