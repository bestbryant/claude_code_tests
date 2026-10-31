"""Prepare the slim dataset bryantyu/casmi26-elig-assets for the eligible fork elig1 (fix F2b of docs/fork_eligibility.md):
exactly the files Engine 2 OURS (python -m casmi.run_eng2, stage 1 of kernels/casmi26-own-s2v18cue) reads, so the fork
no longer mounts bryantyu/casmi26-code / casmi26-models (which also hold G1-open pre-v18c models). Ours, MIT.

python scripts/pub_variant/make_elig_assets.py <out_dir>        (nothing is uploaded; `kaggle datasets create -p <out_dir>`
                                                                is the owner's step after setting the licence)
Layout: FLAT (Kaggle flattened our casmi26-code upload: its modules mount at the dataset root): the 16 casmi modules (the
import closure of casmi.run_eng2's stage 1, checked by running it from this directory), the FP, PeakFormer and the two
rankers + feature lists, README.md (provenance), MANIFEST.sha256 (sha256sum -c format), dataset-metadata.json (licence
left as a TODO for the owner). Engine 2 copies the *.py files into a package dir named casmi at run time. Large files are hard links of the published local copies
(/home/user/ds/casmi26-models, /home/user/work/own_models), code files hard links of /home/user/ds/casmi26-code/casmi when
byte-identical to the repository, else copies of the repository files. The pool is NOT included: bryantyu/casmi26-pool-v18c
(pool.parquet, pool_fp.npy, coconut_prior.parquet; CC BY 4.0 on Kaggle) is eligible as published and is mounted as is.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = ['__init__.py', 'run_eng2.py', 'run_own.py', 'own.py', 'metric.py', 'prep_meta.py', 'spectra.py', 'analog.py',
        'build.py', 'retrieval.py', 'entropy.py', 'frag.py', 'fpscore.py', 'fpmodel.py', 'peakformer.py', 'richfp.py']
MODELS = {   # name: (md5 recorded in docs/reproduce.md section 6 / docs/plan.md, local sources in order of preference)
    'fp_model_v1r.pt': ('a9748a4a3978564ebe2772b2e6e3317d', ['/home/user/ds/casmi26-models', '/home/user/work/own_models']),
    'pf_tpu_f.pt': ('3f594bf07d8632029cf99b6111a82f65', ['/home/user/ds/casmi26-models', '/home/user/work/pfs']),
    'ret_ranker_v18cu.txt': ('1bfde65fc31c06b4d8ecf289894f3614', ['/home/user/ds/casmi26-models', '/home/user/work/own_models']),
    'ret_ranker_v18cu_feats.txt': (None, ['/home/user/ds/casmi26-models', '/home/user/work/own_models']),
    'ret_ranker_v18cur1.txt': ('d3b10b6bfbedb30e8f1d8d6b76b83f4d', ['/home/user/ds/casmi26-models', '/home/user/work/own_models']),
    'ret_ranker_v18cur1_feats.txt': (None, ['/home/user/ds/casmi26-models', '/home/user/work/own_models'])}
POOL_DS = 'bryantyu/casmi26-pool-v18c'
POOL_MD5 = {'pool.parquet': '8afda40aa9324130b800523c1e5f267b', 'pool_fp.npy': 'cef50fc519f985ddbf878b5ff5889ada',
            'coconut_prior.parquet': 'f8b83ce5512deabd27431a9ae8b48c87'}
LICENCE_TODO = ('TODO-OWNER: set before `kaggle datasets create` -- MIT for code; data per docs/ATTRIBUTION.md; models '
                'per docs/plan.md roadmap section 7 (trained on the CC BY-NC 4.0 competition data)')


def digest(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def link_or_copy(src, dst, link=True):
    if os.path.exists(dst):
        os.remove(dst)
    if link:
        try:
            os.link(src, dst)
            return 'hard link'
        except OSError:
            pass
    shutil.copy2(src, dst)
    return 'copy'


def git_head():
    try:
        return subprocess.run(['git', '-C', REPO, 'rev-parse', '--short=12', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return 'unknown'


def git_dirty(paths):
    try:
        r = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--'] + paths, capture_output=True, text=True)
        return [l[3:] for l in r.stdout.splitlines()]
    except Exception:
        return []


README = """# casmi26-elig-assets

Our assets for **Engine 2 OURS** of the eligible public-lineage fork (kernels/casmi26-pube1, fix F2b of
docs/fork_eligibility.md in our repository): stage 1 of our own pipeline (kernels/casmi26-own-s2v18cue) run as a
subprocess, `python -m casmi.run_eng2 <test.parquet> <train.parquet> <pool_dir> <models_dir> <out.json>`, writing the
top 40 candidates per molecule in the fork's Engine 2 schema `{{molecule_id: {{smiles, keys}}}}`. It replaces
bryantyu/casmi26-code and bryantyu/casmi26-models in the eligible fork (those also hold older models with the open gap
G1 of docs/reproduce.md). The candidate pool is **not** in this dataset: the fork mounts {pool} as published
(pool.parquet md5 8afda40a, pool_fp.npy cef50fc5, coconut_prior.parquet f8b83ce5; built by scripts/build_v18c.sh stage A
from our COCONUT 10-2026 build, our ChEBI r255 + LMSD build, PubChemLite v3.3.0 and PubChem counts; CC BY 4.0 on Kaggle).

All files sit at the dataset root (Kaggle mounted our earlier code upload flat): the casmi modules (*.py), the models
(*.pt, *.txt), this README and MANIFEST.sha256, which lists every file with its SHA-256 (`sha256sum -c MANIFEST.sha256`;
the fork's Engine 2 cell checks them at run time, log only). Engine 2 copies the *.py files into a package directory
named casmi before it runs `python -m casmi.run_eng2`. Upload with `kaggle datasets create -p <this dir>` (no
subdirectories) after setting the licence in dataset-metadata.json.

## Files and provenance (docs/reproduce.md section 6; docs/plan.md "PF-syn production" and roadmap section 7)

| file | built by | inputs | md5 | gap status |
|---|---|---|---|---|
{rows}

Code: the *.py files are the import closure of `casmi.run_eng2` (stage 1 only: no stage 2, no PubChem channel,
no DreaMS), taken from our repository at commit {head}{dirty}. run_eng2.py is new for this dataset (Engine 2
entry point; the other modules are unchanged from bryantyu/casmi26-code at 1e31261, the code the s2v18cue kernel ran).
Stage 1 = Retrieval features (library match against train.parquet spectra rebuilt at run time with
casmi.build.build_analog_lib, LIB_CAP 10; analog search; fragments; pool priors) + FP dual encoder + PeakFormer, ranked
by the mean of the two LightGBM rankers (seed ensemble). Keys are the fork pipeline's chem.score_key semantics
(tautomer-canonical InChIKey14 with stereo kept), so Engine 2's identities match Engine 1's in the fusion.

Third-party data behind these models (docs/ATTRIBUTION.md): competition train.parquet (CC BY-NC 4.0, tolerated by the
host), COCONUT (CC0), ChEBI (CC BY 4.0), LIPID MAPS LMSD, PubChemLite for Exposomics (CC BY 4.0), PubChem
(NCBI policy); the PF-syn training spectra are GLACIER predictions (ms-pred, MIT; MassSpecGym checkpoint of its authors,
host-allowed). No NIST data, no NIST-trained model, no non-commercial input other than train.parquet.

Licence: {licence}

Made by scripts/pub_variant/make_elig_assets.py.
"""

PROV = {
    '*.py (casmi package)': ('repository casmi/ (as scripts/push_code.sh ships it)', 'source code', '-', 'G19 licence (owner)'),
    'fp_model_v1r.pt': ('kernels/casmi26-fp-train-v1: scripts/train_fp.py, FP_STOP_EPOCH=3 (Kaggle CPU)',
                        'casmi26-fp-data (scripts/prep_fp_data.py)', 'a9748a4a', 'G4 closed (equivalent retrain)'),
    'pf_tpu_f.pt': ('kernels/casmi26-pfs-tpu-f: scripts/train_pf_xla.py, PF-b init, full syn2, PF_STEPS 40,000 x 2,048, '
                    'seed 43, 8 TPU chips', 'pf_model_b.pt (kernels/casmi26-pf-train-b); casmi26-pf-data; GLACIER-simulated '
                    'pool spectra (build_sim_jobs.py -> simw2 workers -> syn-prep2 946badfb)', '3f594bf0',
                    'G5 open (PF-b init: 5 h wall-clock stop); PF-syn chain to be written in reproduce.md section 7'),
    'ret_ranker_v18cu.txt': ('scripts/build_pfswap_pair.sh pf_tpu_f.pt v18cu (= BASE=v18c REUSE_C1=0 build_v18c.sh, '
                             'stages B-D; LightGBM deterministic, seed 0)', 'ret18c rows (pool v18c, rank / val keys), '
                             'fp_model_v1r.pt, pf_tpu_f.pt', '1bfde65f', 'deterministic (re-run check: plan.md roadmap 7.7c)'),
    'ret_ranker_v18cur1.txt': ('same script, LightGBM seed 1 on the same rows (RANKER_PARAMS seed 1)', 'same rows',
                               'd3b10b6b', 'deterministic'),
    'ret_ranker_v18cu_feats.txt': ('scripts/train_ret_ranker.py (feature list of the ranker)', '-', '-', '-'),
    'ret_ranker_v18cur1_feats.txt': ('scripts/train_ret_ranker.py (same list)', '-', '-', '-')}


def main(out):
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    for sub in ('casmi', 'models'):                      # an earlier nested layout
        if os.path.isdir(os.path.join(out, sub)):
            shutil.rmtree(os.path.join(out, sub))
    how = {}
    pub = '/home/user/ds/casmi26-code/casmi'
    for name in CODE:
        src = os.path.join(REPO, 'casmi', name)
        p = os.path.join(pub, name)
        same = os.path.isfile(p) and digest(p) == digest(src)
        how[name] = link_or_copy(p if same else src, os.path.join(out, name), link=same)
    for name, (md5, dirs) in MODELS.items():
        src = next((os.path.join(d, name) for d in dirs if os.path.isfile(os.path.join(d, name))), None)
        assert src, f'{name}: no local copy in {dirs}'
        if md5:
            assert digest(src, 'md5') == md5, f'{name}: md5 differs from the recorded {md5[:8]}'
        how[name] = link_or_copy(src, os.path.join(out, name))
    feats = [open(os.path.join(out, f'{r}_feats.txt')).read().split() for r in ('ret_ranker_v18cu', 'ret_ranker_v18cur1')]
    assert feats[0] == feats[1] and 'pf_cos' in feats[0] and 'fp_cos' in feats[0], 'seed replicates share one feature list'
    files = sorted(how)
    for stale in ('README.md', 'MANIFEST.sha256', 'dataset-metadata.json'):
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))
    extra = sorted(os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs
                   if os.path.relpath(os.path.join(d, f), out) not in how)
    assert not extra, f'unexpected files in {out}: {extra}'
    head = git_head()
    dirty = git_dirty(['casmi/' + n for n in CODE])
    assert len(set(how)) == len(how) == len(CODE) + len(MODELS), 'file names must be unique in the flat layout'
    rows = []
    for key, (by, inputs, md5, gap) in PROV.items():
        rows.append(f'| {key} | {by} | {inputs} | {md5} | {gap} |')
    readme = README.format(pool=POOL_DS, rows='\n'.join(rows), head=head,
                           dirty=(' plus uncommitted ' + ', '.join(dirty)) if dirty else '', licence=LICENCE_TODO)
    with open(os.path.join(out, 'README.md'), 'w') as f:
        f.write(readme)
    meta = {'title': 'casmi26-elig-assets', 'id': 'bryantyu/casmi26-elig-assets',
            'subtitle': 'Engine 2 OURS of the eligible CASMI26 fork: our stage-1 code and models',
            'description': readme, 'isPrivate': True, 'licenses': [{'name': LICENCE_TODO}],
            'keywords': [], 'resources': []}
    with open(os.path.join(out, 'dataset-metadata.json'), 'w') as f:
        f.write(json.dumps(meta, indent=2) + '\n')
    listed = files + ['README.md']
    lines = [f'{digest(os.path.join(out, f))}  {f}' for f in listed]
    with open(os.path.join(out, 'MANIFEST.sha256'), 'w') as f:
        f.write('# bryantyu/casmi26-elig-assets, made by scripts/pub_variant/make_elig_assets.py at repository commit '
                f'{head}; check: sha256sum -c MANIFEST.sha256\n' + '\n'.join(lines) + '\n')
    size = sum(os.path.getsize(os.path.join(out, f)) for f in listed)
    print(json.dumps(dict(out=out, files=len(listed) + 2, bytes=size, how=how, head=head, uncommitted=dirty), indent=1))
    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         '/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/elig/ds_casmi26_elig')
