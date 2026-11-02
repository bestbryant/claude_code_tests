"""Checks of the NE480=1 builds (scripts/pub_variant/ne480.py; ours). python scripts/pub_variant/test_ne480_build.py
1. every committed NE480 kernel regenerates byte-identically (notebook + kernel-metadata);
2. the driver cell of the pubm-based NE480 build equals bgunshui/casmi26-480-neural-ensemble's driver cell except our
   guarded PF-syn call (a no-op with OURS = False), when the upstream copy is present locally (read as text only);
3. NE480 adds exactly one kernel source and touches only the driver cell, cell 0 and (ELIG=1) cell 8 of the base build;
4. the option refuses SENS / OURS / ELIG=0.
Nothing of the notebooks is executed."""
import difflib
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(REPO, 'scripts', 'make_pub_variant.py')
PUBM = ['OURS=False', 'PF_MID_W=0', 'ISO_ALPHA=0']
NE = {'casmi26-pube1-ne': ['ELIG=1', 'NE480=1'] + PUBM}
UPSTREAM = '/home/user/kern/bgunshui_casmi26-480-neural-ensemble/casmi26-480-neural-ensemble.ipynb'
PF_CALL = ['try:', '    PF_REPORT = ours_pf_blend()          # PF-syn joins the neural term; saner-only on any failure',
           'except Exception as _pf_exc:',
           "    print('WARNING PF-SYN wrapper error, saner-only neural term kept:', repr(_pf_exc)[:300], flush=True)",
           "    PF_REPORT = {'status': 'wrapper_error', 'error': repr(_pf_exc)[:300]}"]


def gen(slug, args, out):
    r = subprocess.run([sys.executable, GEN, slug] + args + [f'--out-dir={out}'], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def src(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']


def load(d, slug):
    return (json.load(open(os.path.join(d, slug, 'notebook.ipynb'), encoding='utf-8')),
            json.load(open(os.path.join(d, slug, 'kernel-metadata.json'), encoding='utf-8')))


def main():
    with tempfile.TemporaryDirectory() as d:
        for slug, args in NE.items():                                       # 1
            rc, log = gen(slug, args, d)
            assert rc == 0, log
            for f in ('notebook.ipynb', 'kernel-metadata.json'):
                assert open(os.path.join(d, slug, f), 'rb').read() == \
                    open(os.path.join(REPO, 'kernels', slug, f), 'rb').read(), (slug, f)
        for slug, args, base, base_args, elig in (
                ('zz-pubm-ne', ['NE480=1'] + PUBM, 'zz-pubm', PUBM, False),
                ('zz-pube1-ne', ['ELIG=1', 'NE480=1'] + PUBM, 'zz-pube1', ['ELIG=1'] + PUBM, True)):
            for s_, a_ in ((slug, args), (base, base_args)):
                rc, log = gen(s_, a_, d)
                assert rc == 0, log
            nb, meta = load(d, slug)
            nb0, meta0 = load(d, base)
            changed = [i for i, (x, y) in enumerate(zip(nb['cells'], nb0['cells'])) if src(x) != src(y)]
            drv = [c.get('id') for c in nb['cells']].index('67514abd')
            want = sorted([0, drv] + ([[c.get('id') for c in nb['cells']].index('3e2e781f')] if elig else []))
            assert len(nb['cells']) == len(nb0['cells']) and changed == want, (slug, changed, want)   # 3
            assert meta['kernel_sources'] == ['bgunshui/casmi26-scorer-train-public'] and meta0['kernel_sources'] == []
            assert {k: v for k, v in meta.items() if k not in ('id', 'title', 'kernel_sources')} == \
                {k: v for k, v in meta0.items() if k not in ('id', 'title', 'kernel_sources')}
            if not elig and os.path.isfile(UPSTREAM):                       # 2
                up = json.load(open(UPSTREAM, encoding='utf-8'))
                d_ = [l for l in difflib.unified_diff(src(up['cells'][12]).splitlines(), src(nb['cells'][drv]).splitlines(),
                                                      lineterm='', n=0) if not l.startswith(('---', '+++', '@@'))]
                assert d_ == ['-'] + ['+' + l for l in PF_CALL], d_
                print('driver cell = upstream + our PF-syn no-op call')
        for bad in (['NE480=1', 'SENS=2', 'SENS_MODE=swap', 'SENS_SEEDS=[20262023]'] + PUBM, ['NE480=1', 'OURS=True'],
                    ['ELIG=0', 'NE480=1'] + PUBM):                         # 4
            rc, log = gen('zz-bad', bad, d)
            assert rc != 0, bad
    print('ne480 build checks OK')


if __name__ == '__main__':
    main()
