"""LB source-decomposition probe kernel from an own-pipeline kernel: the same run (kernels/<base>/notebook.py unchanged),
then the 25-lists restricted to one candidate source (standard InChIKey14 in train / COCONUT-not-train / other, key
sets of bryantyu/casmi26-srckeys, as kernels/casmi26-probe17b-coco). Its LB score is the MRR carried by that source.

python scripts/make_probe_kernel.py <base kernel slug> <probe slug> <coco|train|other>
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base, slug, source = sys.argv[1:4]
assert source in ('coco', 'train', 'other'), source
src = os.path.join(root, 'kernels', base)
dst = os.path.join(root, 'kernels', slug); os.makedirs(dst, exist_ok=True)
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))
meta['id'] = f'bryantyu/{slug}'; meta['title'] = slug
meta['dataset_sources'] = sorted(set(meta['dataset_sources'] + ['bryantyu/casmi26-srckeys']))
json.dump(meta, open(os.path.join(dst, 'kernel-metadata.json'), 'w'), indent=2)
nb = open(os.path.join(src, 'notebook.py')).read()
assert "sub.to_csv('/kaggle/working/submission.csv', index=False)" in nb and 'PROBE_SOURCE' not in nb
nb = nb.rstrip('\n') + f'''

# --- LB decomposition probe (scripts/make_probe_kernel.py, base kernel {base}): the same lists restricted to one
# candidate source (standard InChIKey14 in train / COCONUT-not-train / other)
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
PROBE_SOURCE = {source!r}
K = pd.read_parquet(first('casmi_probe_srckeys.parquet'))
TR, CO = set(K.key[K.src == 'train']), set(K.key[K.src == 'coco'])
def _src(smi):
    m = Chem.MolFromSmiles(smi)
    k = Chem.MolToInchiKey(m)[:14] if m is not None else None
    return 'train' if k in TR else 'coco' if k in CO else 'other'
sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or []) if _src(c[0]) == PROBE_SOURCE][:25]) or 'C'
                 for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('PROBE', PROBE_SOURCE, 'rows with candidates', int((sub.smiles != 'C').sum()), 'of', len(sub))
'''
open(os.path.join(dst, 'notebook.py'), 'w').write(nb)
print('wrote', dst)
