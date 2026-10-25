"""PF-syn pilot on Kaggle CPU kernels (no TPU slot needed): kernels/casmi26-pf-pilot with PF_DEV=cpu, one process with
batch 256, the chosen arms, a fixed step count (both arms step-matched) and a wall-clock guard.

python scripts/make_pf_cpu_pilot.py <slug> <arms: ctl,syn | ctl | syn> <steps> <hours guard per arm> [val_n] [log_every]
The model, data, learning rate, negatives and synthetic-row settings are the TPU pilot's (same notebook text); only
the device, the per-step batch (256 instead of 8 x 256) and the step count differ, identically for both arms.
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, arms, steps, hours = sys.argv[1], sys.argv[2].split(','), int(sys.argv[3]), float(sys.argv[4])
val_n = sys.argv[5] if len(sys.argv) > 5 else '800'
log = sys.argv[6] if len(sys.argv) > 6 else '200'
assert set(arms) <= {'ctl', 'syn'}
src = os.path.join(root, 'kernels', 'casmi26-pf-pilot')
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))
meta.update(id=f'bryantyu/{slug}', title=slug, enable_tpu=False, enable_gpu=False)
meta.pop('machine_shape', None)
nb = open(os.path.join(src, 'notebook.py')).read()
reps = [("STEPS = os.environ.get('PILOT_STEPS', '20000')", f"STEPS = '{steps}'"),
        ("base = dict(os.environ, PF_HOURS='3.5', PF_VAL_N='800',", f"base = dict(os.environ, PF_DEV='cpu', PF_HOURS='{hours}', PF_VAL_N='{val_n}',"),
        ("PF_LOG='200'", f"PF_LOG='{log}'"),
        ("for name, extra in arms:", f"arms = [a for a in arms if a[0] in {arms!r}]\nfor name, extra in arms:"),
        ("'40', '4000', '256', '63']", f"'1', '{steps}', '256', '63']")]
for a, b in reps:
    assert nb.count(a) == 1, a
    nb = nb.replace(a, b)
nb = (f'"""CPU variant (scripts/make_pf_cpu_pilot.py): arms {arms}, {steps} steps of batch 256 per arm on CPU, '
      f'wall-clock guard {hours} h per arm.\nThe TPU pilot\'s docstring follows.\n' + nb[3:])
d = os.path.join(root, 'kernels', slug); os.makedirs(d, exist_ok=True)
open(os.path.join(d, 'notebook.py'), 'w').write(nb)
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
print('wrote', d)
