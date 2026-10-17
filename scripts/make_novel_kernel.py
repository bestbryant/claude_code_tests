"""Build kernels/<slug>: the forked s2s notebook + novel-structure overlay cells.

Usage: python scripts/make_novel_kernel.py <slug> '<overlay kwargs as JSON>' [model_dataset_file]
"""
import json, os, shutil, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, kw = sys.argv[1], sys.argv[2]
model_file = sys.argv[3] if len(sys.argv) > 3 else ''
src = os.path.join(root, 'kernels', 'casmi26-s2s-fork')
dst = os.path.join(root, 'kernels', slug)
os.makedirs(dst, exist_ok=True)
nb = json.load(open(os.path.join(src, 'notebook.ipynb')))
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))

CELL_RUN = '''# ---- Novel-structure engine (class 3): analog derivatives + building-block conjugates absent from PubChem ----
import glob, json, os, shutil, subprocess, sys, time
_t0 = time.time()
_src = os.path.dirname(sorted(glob.glob('/kaggle/input/**/novel.py', recursive=True), key=len)[0])
os.makedirs('/kaggle/working/pkg', exist_ok=True)
shutil.copytree(_src, '/kaggle/working/pkg/casmi', dirs_exist_ok=True)
_pck = sorted(glob.glob('/kaggle/input/**/pc_keys_u64.npy', recursive=True), key=len)[0]
_model = sorted(glob.glob('/kaggle/input/**/MODEL_FILE', recursive=True), key=len)
_model = _model[0] if _model and 'MODEL_FILE' else ''
NOVEL = {}
try:
    _r = subprocess.run([sys.executable, '-m', 'casmi.run_novel', os.path.join(COMP, 'test.parquet'),
                         os.path.join(COMP, 'train.parquet'), _pck, '/kaggle/working/novel_lists.json', _model,
                         '/kaggle/working/novel', '4'], capture_output=True, text=True, timeout=3 * 3600,
                        cwd='/kaggle/working/pkg', env={**os.environ, 'PYTHONPATH': '/kaggle/working/pkg', 'NUMBA_NUM_THREADS': '1'})
    print(_r.stdout[-2000:]); print(_r.stderr[-2000:])
    NOVEL = json.load(open('/kaggle/working/novel_lists.json'))
except Exception as _e:
    print('NOVEL ENGINE FAILED -> submission unchanged:', repr(_e))
print('novel lists', len(NOVEL), f'{time.time()-_t0:.0f}s', flush=True)
'''.replace('MODEL_FILE', model_file or '__none__')

CELL_MERGE = '''# ---- Overlay: insert novel candidates where the main pipeline's library evidence is weak ----
import pandas as pd
sys.path.insert(0, '/kaggle/working/pkg')
from casmi import overlay as _ov
OVERLAY_KW = json.loads(%r)
try:
    _sub = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _lib = {str(m): float(v[2]) for m, v in BASE.items()}
    _new, _stats = _ov.apply(_sub, NOVEL, _lib, **OVERLAY_KW)
    assert _new.molecule_id.tolist() == _sub.molecule_id.tolist()
    assert _new.smiles.map(lambda s: 1 <= len(s.split(';')) <= 25).all()
    _sub.to_csv('submission_main.csv', index=False)
    _new.to_csv('submission.csv', index=False)
    print('NOVEL OVERLAY', OVERLAY_KW, _stats, flush=True)
except Exception as _e:
    print('NOVEL OVERLAY FAILED -> main submission kept:', repr(_e), flush=True)
''' % kw

for c in (CELL_RUN, CELL_MERGE):
    nb['cells'].append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=c))
json.dump(nb, open(os.path.join(dst, 'notebook.ipynb'), 'w'))
meta['id'] = f'bryantyu/{slug}'; meta['title'] = slug
for d in ('bryantyu/casmi26-code', 'bryantyu/casmi26-pubchem-keys'):
    if d not in meta['dataset_sources']:
        meta['dataset_sources'].append(d)
json.dump(meta, open(os.path.join(dst, 'kernel-metadata.json'), 'w'), indent=2)
print('wrote', dst, meta['dataset_sources'])
