"""Write an own-pipeline inference kernel (kernels/<slug>/) from the s2v10 template.

python scripts/make_own_kernel.py <slug> <stage1_ranker> <stage2_ranker> [KEY=VALUE ...]
KEY=VALUE pairs become environment variables of casmi.run_own (e.g. OWN_PF=pf_a.pt,pf_b.pt  PC_SLOTS=4,8,12,16,20).
PC_TIER=1 mounts bryantyu/casmi26-pc-tier and points PC_TIER at its tier.npz; PC_TIER=full mounts the output of
kernels/casmi26-pc-full-build (full PubChem tier, two-pass channel); PC_TIER=enam the Enamine tier
(bryantyu/casmi26-pc-enam, tier_enam.npz); PC_TIER=purch the purchasable tier (Enamine + bioactive vendors,
bryantyu/casmi26-pc-purch); ZEN=1 mounts bryantyu/casmi26-zen-extra (ZEN_PATH); PC_POP=1 the full-tier documentation
prior; PC_EXCLUDE=1 the probe source key sets (bryantyu/casmi26-srckeys); VEND=1 the depositor columns
(bryantyu/casmi26-vendor, VENDOR_PATH).
"""
import json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, rk1, rk2 = sys.argv[1:4]
env = dict(a.split('=', 1) for a in sys.argv[4:])
src = os.path.join(root, 'kernels', 'casmi26-own-s2v10')
dst = os.path.join(root, 'kernels', slug); os.makedirs(dst, exist_ok=True)
meta = json.load(open(os.path.join(src, 'kernel-metadata.json')))
meta['id'] = f'bryantyu/{slug}'; meta['title'] = slug
if env.get('PC_TIER') == 'full':             # full PubChem tier = output of kernels/casmi26-pc-full-build
    meta['kernel_sources'] = meta.get('kernel_sources', []) + ['bryantyu/casmi26-pc-full-build']
elif env.get('PC_TIER') == 'enam':           # Enamine tier (Enamine-deposited PubChem compounds outside train + COCONUT)
    meta['dataset_sources'] = meta['dataset_sources'] + ['bryantyu/casmi26-pc-enam']
elif env.get('PC_TIER') == 'purch':          # purchasable tier: Enamine + bioactive / NP vendor deposits
    meta['dataset_sources'] = meta['dataset_sources'] + ['bryantyu/casmi26-pc-purch']
elif env.get('PC_TIER'):
    meta['dataset_sources'] = meta['dataset_sources'] + ['bryantyu/casmi26-pc-tier']
if env.get('ZEN'):                           # Zenodo-extra library channel (casmi/zen.py)
    meta['dataset_sources'] = meta['dataset_sources'] + ['bryantyu/casmi26-zen-extra']
if env.get('PC_POP'):                        # documentation prior aligned to the full tier
    meta['dataset_sources'] = meta['dataset_sources'] + ['bryantyu/casmi26-pc-full-pop']
if env.get('VEND'):                          # depositor columns per pool structure (scripts/build_vendor_feats.py)
    meta['dataset_sources'] = sorted(set(meta['dataset_sources'] + ['bryantyu/casmi26-vendor']))
if env.get('PC_EXCLUDE'):                    # probe source key sets (train / coco)
    meta['dataset_sources'] = sorted(set(meta['dataset_sources'] + ['bryantyu/casmi26-srckeys']))
json.dump(meta, open(os.path.join(dst, 'kernel-metadata.json'), 'w'), indent=2)
nb = open(os.path.join(src, 'notebook.py')).read()
nb = nb.replace("OWN_RANKER = 'ret_ranker_v10'", f"OWN_RANKER = {rk1!r}").replace("OWN_STAGE2 = 'stage2_v10'", f"OWN_STAGE2 = {rk2!r}")
extra = {k: v for k, v in env.items() if k not in ('PC_TIER', 'ZEN', 'PC_POP', 'PC_EXCLUDE', 'VEND')}
pc = ("'PC_TIER': os.path.dirname(first('tier_full/mass.npy')), " if env.get('PC_TIER') == 'full'
      else "'PC_TIER': first('tier_enam.npz'), " if env.get('PC_TIER') == 'enam'
      else "'PC_TIER': first('tier_purch.npz'), " if env.get('PC_TIER') == 'purch'
      else "'PC_TIER': first('tier.npz'), " if env.get('PC_TIER') else '')
pc += "'ZEN_PATH': first('zen_extra.parquet'), " if env.get('ZEN') else ''
pc += "'PC_POP': first('pop.npy'), " if env.get('PC_POP') else ''
pc += "'PC_EXCLUDE': first('casmi_probe_srckeys.parquet'), " if env.get('PC_EXCLUDE') else ''
pc += "'VENDOR_PATH': first('vendor.parquet'), " if env.get('VEND') else ''
old = "'GL_THREADS': '4', 'GL_BUDGET': '14400', 'LIB_CAP': '100000'}"
assert nb.count(old) == 1
nb = nb.replace(old, "'GL_THREADS': '4', 'GL_BUDGET': '14400', 'LIB_CAP': '100000', " + pc + f"**{extra!r}}}")
open(os.path.join(dst, 'notebook.py'), 'w').write(nb)
print('wrote', dst, extra, 'PC tier' if pc else '')
