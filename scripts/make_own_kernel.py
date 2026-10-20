"""Write an own-pipeline inference kernel (kernels/<slug>/) from the s2v10 template.

python scripts/make_own_kernel.py <slug> <stage1_ranker> <stage2_ranker> [KEY=VALUE ...]
KEY=VALUE pairs become environment variables of casmi.run_own (e.g. OWN_PF=pf_a.pt,pf_b.pt  PC_SLOTS=4,8,12,16,20).
PC_TIER=1 mounts bryantyu/casmi26-pc-tier and points PC_TIER at its tier.npz.
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, rk1, rk2 = sys.argv[1:4]
env = dict(a.split('=', 1) for a in sys.argv[4:])
src = os.path.join(root, 'kernels', 'casmi26-own-s2v10')
dst = os.path.join(root, 'kernels', slug); os.makedirs(dst, exist_ok=True)
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))
meta['id'] = f'bryantyu/{slug}'; meta['title'] = slug
if env.get('PC_TIER'):
    meta['dataset_sources'] = meta['dataset_sources'] + ['bryantyu/casmi26-pc-tier']
json.dump(meta, open(os.path.join(dst, 'kernel-metadata.json'), 'w'), indent=2)
nb = open(os.path.join(src, 'notebook.py')).read()
nb = nb.replace("OWN_RANKER = 'ret_ranker_v10'", f"OWN_RANKER = {rk1!r}").replace("OWN_STAGE2 = 'stage2_v10'", f"OWN_STAGE2 = {rk2!r}")
extra = {k: v for k, v in env.items() if k != 'PC_TIER'}
pc = "'PC_TIER': first('tier.npz'), " if env.get('PC_TIER') else ''
old = "'GL_THREADS': '4', 'GL_BUDGET': '14400', 'LIB_CAP': '100000'}"
assert nb.count(old) == 1
nb = nb.replace(old, "'GL_THREADS': '4', 'GL_BUDGET': '14400', 'LIB_CAP': '100000', " + pc + f"**{extra!r}}}")
open(os.path.join(dst, 'notebook.py'), 'w').write(nb)
print('wrote', dst, extra, 'PC tier' if pc else '')
