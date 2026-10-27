"""LB source-decomposition probe kernel from an own-pipeline kernel: the same run (kernels/<base>/notebook.py unchanged),
then the 25-lists restricted to one candidate source (standard InChIKey14 in train / COCONUT-not-train / other, key
sets of bryantyu/casmi26-srckeys, as kernels/casmi26-probe17b-coco). Its LB score is the MRR carried by that source.

python scripts/make_probe_kernel.py <base kernel slug> <probe slug> <coco|train|other> [filter|mask]
mask: keep the base's final 25-lists and replace every off-source candidate IN PLACE by a distinct n-alkane placeholder
(C1..C25: valid, never an answer), so positions are preserved and the score is exactly the full-list RR carried by that
source's answers (docs/interleaving_oct27.md E1); filter (default) drops off-source candidates and re-ranks.
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base, slug, source = sys.argv[1:4]
assert source in ('coco', 'train', 'other'), source
mode = sys.argv[4] if len(sys.argv) > 4 else 'filter'
assert mode in ('filter', 'mask'), mode
src = os.path.join(root, 'kernels', base)
dst = os.path.join(root, 'kernels', slug); os.makedirs(dst, exist_ok=True)
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))
meta['id'] = f'bryantyu/{slug}'; meta['title'] = slug
meta['dataset_sources'] = sorted(set(meta['dataset_sources'] + ['bryantyu/casmi26-srckeys']))
json.dump(meta, open(os.path.join(dst, 'kernel-metadata.json'), 'w'), indent=2)
nb = open(os.path.join(src, 'notebook.py')).read()
assert "sub.to_csv('/kaggle/working/submission.csv', index=False)" in nb and 'PROBE_SOURCE' not in nb
FILTER = """sub['smiles'] = [';'.join([c[0] for c in (L.get(str(m)) or []) if _src(c[0]) == PROBE_SOURCE][:25]) or 'C'
                 for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('PROBE', PROBE_SOURCE, 'rows with candidates', int((sub.smiles != 'C').sum()), 'of', len(sub))
"""
MASK = """PH = ['C' * (i + 1) for i in range(25)]   # n-alkanes C1..C25: valid, distinct, never answers
kept = masked = 0
def _mask(c):
    global kept, masked
    c = [x[0] for x in c][:25]
    out = [s if _src(s) == PROBE_SOURCE else PH[i] for i, s in enumerate(c)]
    kept += sum(o == s for o, s in zip(out, c)); masked += sum(o != s for o, s in zip(out, c))
    return ';'.join(out) or 'C'
sub['smiles'] = [_mask(L.get(str(m)) or []) for m in sub.molecule_id]
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('MASKED PROBE', PROBE_SOURCE, 'kept slots', kept, 'masked slots', masked, 'rows', len(sub))
"""
nb = nb.rstrip('\n') + f'''

# --- LB decomposition probe (scripts/make_probe_kernel.py, base kernel {base}, mode {mode}): the same lists restricted
# to one candidate source (standard InChIKey14 in train / COCONUT-not-train / other)
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
PROBE_SOURCE = {source!r}
K = pd.read_parquet(first('casmi_probe_srckeys.parquet'))
TR, CO = set(K.key[K.src == 'train']), set(K.key[K.src == 'coco'])
def _src(smi):
    m = Chem.MolFromSmiles(smi)
    k = Chem.MolToInchiKey(m)[:14] if m is not None else None
    return 'train' if k in TR else 'coco' if k in CO else 'other'
''' + (FILTER if mode == 'filter' else MASK)
open(os.path.join(dst, 'notebook.py'), 'w').write(nb)
print('wrote', dst)
