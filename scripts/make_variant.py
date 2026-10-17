"""Build a patched variant of the forked s2s notebook.

Usage: python scripts/make_variant.py <slug> [--npeaks] [--eundo]
  --npeaks  engine-1 feature q_npeaks counted on cleaned peaks (0.001 floor, top 512, <= prec+2 Da) like the
            offline simulations, instead of raw peaks (train/serve mismatch)
  --eundo   hidden rerun only: add the electron mass back to positive-mode precursor and fragment m/z so the test
            matches the (uncorrected) enveda-180 data every model and ranker was fitted on
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug = sys.argv[1]; opts = set(sys.argv[2:])
src = os.path.join(root, 'kernels', 'casmi26-s2s-fork')
dst = os.path.join(root, 'kernels', slug); os.makedirs(dst, exist_ok=True)
nb = json.load(open(os.path.join(src, 'notebook.ipynb')))
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))

NPEAKS = r'''# variant: q_npeaks on cleaned peaks (matches the offline simulation spectra)
_eng_path = os.path.join(V4CODE, 'casmi', 'engine.py')
_txt = open(_eng_path).read()
_old = "X[:, col['q_npeaks']] = math.log1p(float(np.mean([len(s['mz']) for s in spectra]))) if spectra else 0.0"
_new = ("X[:, col['q_npeaks']] = math.log1p(float(np.mean([min(512, int(((np.asarray(s['mz']) <= float(s['prec']) + 2.0) & "
        "(np.asarray(s['it']) >= 0.001 * max(float(np.max(s['it'])) if len(s['it']) else 0.0, 1e-12))).sum())) "
        "for s in spectra]))) if spectra else 0.0")
assert _txt.count(_old) == 1, 'q_npeaks line not found'
os.remove(_eng_path) if os.path.islink(_eng_path) else None
open(_eng_path, 'w').write(_txt.replace(_old, _new))
print('VARIANT npeaks patch applied')
'''

EUNDO = r'''# variant: undo the hidden-test electron-mass correction for positive mode (models were fitted on uncorrected data)
if IS_RERUN:
    _E = 0.00054858
    _t = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _pos = _t.ionization_mode.eq('positive').values
    _t.loc[_pos, 'precursor_mz'] = _t.loc[_pos, 'precursor_mz'] + _E
    _t['ms2_mzs'] = [list(np.asarray(m, float) + _E) if p else m for m, p in zip(_t.ms2_mzs, _pos)]
    _EC = '/kaggle/working/ecomp'; os.makedirs(_EC, exist_ok=True)
    _t.to_parquet(os.path.join(_EC, 'test.parquet'))
    for _f in ['train.parquet', 'sample_submission.csv']:
        if not os.path.exists(os.path.join(_EC, _f)):
            os.symlink(os.path.join(COMP, _f), os.path.join(_EC, _f))
    COMP = _EC
    print('VARIANT eundo applied to', int(_pos.sum()), 'positive spectra')
'''

out = []
for c in nb['cells']:
    out.append(c)
    s = ''.join(c['source']) if c['cell_type'] == 'code' else ''
    if '--npeaks' in opts and 'V43_DETERMINISM = _v43_patch_engine' in s:
        out.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=NPEAKS))
    if '--eundo' in opts and "SMOKE switch" in s and 'IS_RERUN' in s:
        out.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=EUNDO))
added = len(out) - len(nb['cells'])
assert added == len([o for o in opts if o in ('--npeaks', '--eundo')]), added
nb['cells'] = out
json.dump(nb, open(os.path.join(dst, 'notebook.ipynb'), 'w'))
meta['id'] = f'bryantyu/{slug}'; meta['title'] = slug
json.dump(meta, open(os.path.join(dst, 'kernel-metadata.json'), 'w'), indent=2)
print('wrote', dst, 'cells added', added)
