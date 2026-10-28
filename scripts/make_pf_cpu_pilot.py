"""PF-syn pilot on Kaggle CPU kernels (no TPU slot needed): kernels/casmi26-pf-pilot with PF_DEV=cpu, one process with
batch 256, the chosen arms, a fixed step count (both arms step-matched) and a wall-clock guard.

python scripts/make_pf_cpu_pilot.py <slug> <arms: ctl,syn | ctl | syn | synP> <steps> <hours guard per arm> [val_n] [log_every] [seed]
Env STOP_AT=<step>: PF_STOP_AT (stop there, LR schedule still spanning <steps>).
synP = production synthetic arm: PF_SYN = SYN_NAME (default syn2.npz) from the kernel SYN_SRC (default
bryantyu/casmi26-syn-prep2, scripts/make_syn_prep.py), PF_SYN_P 0.25, no neutral weighting and no dropped half (every
pool structure is simulated, as at test time); it replaces the pilot's syn-prep kernel source.
The model, data, learning rate, negatives and synthetic-row settings are the TPU pilot's (same notebook text); only
the device, the per-step batch (256 instead of 8 x 256) and the step count differ, identically for both arms.
Env DEV=tpu: the same arms on the TPU v5e-8 instead (the pilot's device and batch 8 x 256; only the step count, guard, val_n,
log interval, seed and arms change) -- e.g. a short smoke run of the never-yet-run XLA path.
Env DEV=gpu: one T4 (PF_DEV=cuda, fp16 autocast), otherwise exactly the CPU variant (batch 256, PF_STEPS = <steps>).
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, arms, steps, hours = sys.argv[1], sys.argv[2].split(','), int(sys.argv[3]), float(sys.argv[4])
val_n = sys.argv[5] if len(sys.argv) > 5 else '800'
log = sys.argv[6] if len(sys.argv) > 6 else '200'
seed = sys.argv[7] if len(sys.argv) > 7 else '31'          # the TPU pilot's PF_SEED
assert set(arms) <= {'ctl', 'syn'} or arms == ['synP'], arms
src = os.path.join(root, 'kernels', 'casmi26-pf-pilot')
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))
TPU, GPU = os.environ.get('DEV') == 'tpu', os.environ.get('DEV') == 'gpu'
meta.update(id=f'bryantyu/{slug}', title=slug, enable_tpu=TPU, enable_gpu=GPU)
if GPU:
    meta['machine_shape'] = 'NvidiaTeslaT4'
elif not TPU:
    meta.pop('machine_shape', None)
DEVN = 'cuda' if GPU else 'cpu'
nb = open(os.path.join(src, 'notebook.py')).read()
reps = [("STEPS = os.environ.get('PILOT_STEPS', '20000')", f"STEPS = '{steps}'"),
        ("base = dict(os.environ, PF_HOURS='3.5', PF_VAL_N='800',",
         f"base = dict(os.environ, {'' if TPU else f"PF_DEV='{DEVN}', "}PF_HOURS='{hours}', PF_VAL_N='{val_n}',"),
        ("PF_LOG='200'", f"PF_LOG='{log}'"),
        ("PF_SEED='31'", f"PF_SEED='{seed}'"),
        ("for name, extra in arms:", f"arms = [a for a in arms if a[0] in {arms!r}]\nfor name, extra in arms:"),
        ("'40', '4000', '256', '63']", "'40', '4000', '256', '63']" if TPU else f"'1', '{steps}', '256', '63']")]
if arms == ['synP']:
    old = ("arms = [('ctl', {}),\n        ('syn', dict(PF_SYN=first('syn.npz'), PF_SYN_P='0.25', PF_SYN_NEUTRAL_W='0.18', "
           "PF_SYN_DROP=first('syn_drop_B.txt')))]")
    syn_name = os.environ.get('SYN_NAME', 'syn2.npz')
    reps.append((old, f"arms = [('synP', dict(PF_SYN=first({syn_name!r}), PF_SYN_P='0.25'))]"))
    meta['kernel_sources'] = [os.environ.get('SYN_SRC', 'bryantyu/casmi26-syn-prep2')]
if os.environ.get('STOP_AT'):          # PF_STOP_AT: matched partner of a guard-cut arm (train_pf_xla.py)
    reps.append(("PF_STEPS=STEPS)", f"PF_STEPS=STEPS, PF_STOP_AT='{os.environ['STOP_AT']}')"))
for a, b in reps:
    assert nb.count(a) == 1, a
    nb = nb.replace(a, b)
nb = (f'"""{"TPU" if TPU else DEVN.upper()} variant (scripts/make_pf_cpu_pilot.py): arms {arms}, {steps} steps of batch {"8 x " if TPU else ""}256 per arm on {"TPU v5e-8" if TPU else "one T4" if GPU else "CPU"}, '
      f'wall-clock guard {hours} h per arm, PF_SEED {seed}.\nThe TPU pilot\'s docstring follows.\n' + nb[3:])
d = os.path.join(root, 'kernels', slug); os.makedirs(d, exist_ok=True)
open(os.path.join(d, 'notebook.py'), 'w').write(nb)
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
print('wrote', d)
