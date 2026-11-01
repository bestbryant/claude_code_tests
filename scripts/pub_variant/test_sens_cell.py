"""Tests of the SENS cell (scripts/pub_variant/sens_cell.py; ours) on synthetic data.

python scripts/pub_variant/test_sens_cell.py

Only our own code runs: the cell source (definitions only) is exec'd into a namespace with stand-in globals, and the
driver test replaces the pinned public neural_scores.py by a stub scorer written here (same command-line interface and
output layout: rows grouped by molecule, sorted by baseline_rank; deterministic float32 scores from the checkpoint bytes).
No public / third-party code is executed.
  A. sens_ensemble_frames: raw / z values (vs the holdout evaluation's own combine() / zscore() of
     kernels/casmi26-saner-seeds/notebook.py, extracted with ast), zero-std members, finite / neural_ok / processed
     handling (molecule-level for the final lists), outside-candidate mapping (same per-molecule maps, published-alone
     fallback, unchanged molecules), audit columns and dtypes, stats (changed top1, rank corr, per seed), alignment
     refusals, a non-default index.
  B. ours_seed_ensemble end to end with the stub: applied (raw, z), command line / cwd / environment as cell 9, sha256
     checks, temporary bundles removed, backups, report and log lines (NOTE when the seeds decision's k / mode / members
     differ from the build); fallbacks with the published files byte-identical:
     sha256 mismatch, missing checkpoint, scorer exit 1, failure between the two replaces, budget, wrong pin of
     neural_scores.py, a missing evidence file; no outsider features file (final lists only); input failures of one
     member keep that molecule published.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CELL = (HERE / 'sens_cell.py').read_text(encoding='utf-8')
T, W = 0.07, 0.5                     # stand-ins for config temperature / fp_score_weight


def cell_ns(**g):
    ns = dict(g)
    exec(compile(CELL, 'sens_cell.py', 'exec'), ns)        # definitions only (our code)
    return ns


def seeds_eval_funcs():
    """zscore / combine of the holdout evaluation (kernels/casmi26-saner-seeds/notebook.py, ours), via ast."""
    src = (REPO / 'kernels' / 'casmi26-saner-seeds' / 'notebook.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('zscore', 'combine')]
    assert len(fns) == 2
    ns = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), 'seeds_eval', 'exec'), ns)
    return ns['zscore'], ns['combine']


# ---------------------------------------------------------------------------------------------------------- A. pure
def frames(rng, mols, k_extra, outs_per=3):
    """Published + k_extra member frames (final lists and outsiders), float32 like the scorer."""
    rows = []
    for m, n in mols:
        for j in range(n):
            rows.append(dict(molecule_id=m, smiles=f'C{m}{j}', metric_key=f'K{m}{j}', baseline_rank=j + 1,
                             processed=True, library_max=0.5))
    base = pd.DataFrame(rows)
    orows = []
    for m, _ in mols:
        for j in range(outs_per):
            orows.append(dict(molecule_id=m, smiles=f'O{m}{j}', metric_key=f'Q{m}{j}', baseline_rank=26 + j,
                              processed=True))
    obase = pd.DataFrame(orows)

    def member(b):
        f = b.copy()
        cos = rng.normal(0.3, 0.1, len(f)).astype(np.float32)
        fp = rng.normal(0.2, 0.05, len(f)).astype(np.float32)
        f['neural_cos'] = cos
        f['neural_fp'] = fp
        f['neural_score'] = (cos / np.float32(T) + np.float32(W) * fp).astype(np.float32)
        f['neural_std'] = np.float32(0.1)
        f['n_spectra'] = 3.0
        f['neural_ok'] = True
        return f
    return [member(base) for _ in range(1 + k_extra)], [member(obase) for _ in range(1 + k_extra)]


def test_pure():
    zscore, combine = seeds_eval_funcs()
    ns = cell_ns()
    ens = ns['sens_ensemble_frames']
    rng = np.random.default_rng(0)
    mols = [('m0', 6), ('m1', 5), ('m2', 4), ('m3', 7), ('m4', 3), ('m5', 2), ('m6', 5)]
    M, O = frames(rng, mols, 3)
    seeds = [20262023, 20263023, 20264023]
    # m1: an extra member is not neural_ok (the scorer's failure path: zeros, neural_ok False) -> kept
    M[2].loc[M[2].molecule_id == 'm1', ['neural_ok']] = False
    M[2].loc[M[2].molecule_id == 'm1', ['neural_cos', 'neural_fp', 'neural_score']] = 0.0
    # m2: one member non-finite neural_fp in one row -> kept
    i = M[3].index[M[3].molecule_id == 'm2'][1]
    M[3].loc[i, 'neural_fp'] = np.nan
    # m3: published not processed (zeros, neural_ok False in every member) -> inactive
    for f in M:
        f.loc[f.molecule_id == 'm3', 'processed'] = False
        f.loc[f.molecule_id == 'm3', 'neural_ok'] = False
        f.loc[f.molecule_id == 'm3', ['neural_cos', 'neural_fp', 'neural_score']] = 0.0
    # m4: published neural_ok False only -> inactive
    M[0].loc[M[0].molecule_id == 'm4', 'neural_ok'] = False
    # m6: one member constant on the molecule (std 0)
    M[1].loc[M[1].molecule_id == 'm6', 'neural_score'] = np.float32(1.5)
    # outsiders: m0 row 1 has member 2 not ok (-> published alone), m0 row 2 published not ok (-> unchanged)
    o0 = O[0].index[O[0].molecule_id == 'm0']
    O[2].loc[o0[1], 'neural_ok'] = False
    O[2].loc[o0[1], 'neural_score'] = np.nan
    O[0].loc[o0[2], 'neural_ok'] = False
    # an outsider molecule absent from the final lists
    extra = [f.iloc[:2].assign(molecule_id='zz', smiles=['Oz0', 'Oz1']) for f in O]
    O = [pd.concat([f, e], ignore_index=True) for f, e in zip(O, extra)]
    pub_main, pub_outs = M[0].copy(), O[0].copy()
    enabled = ('m0', 'm5', 'm6')
    for mode in ('raw', 'z'):
        nm, no, st = ens(M[0], O[0], M[1:], O[1:], mode, seeds)
        assert M[0].equals(pub_main) and O[0].equals(pub_outs), 'inputs modified'
        K = 4
        assert st['k'] == K and st['molecules'] == len(enabled) and st['kept_not_ok'] == 2 and st['inactive'] == 2, st
        assert st['molecules_total'] == len(mols)
        for c in ('neural_score', 'neural_cos', 'neural_fp'):
            assert nm[c].dtype == np.float64 and no[c].dtype == np.float64
            assert np.array_equal(nm[f'{c}_published'].to_numpy(), M[0][c].to_numpy(np.float64), equal_nan=True)
        for k, s in enumerate(seeds, 1):
            assert np.array_equal(nm[f'neural_score_seed{s}'].to_numpy(), M[k].neural_score.to_numpy(np.float64),
                                  equal_nan=True)
        keep = [c for c in M[0].columns if c not in ('neural_score', 'neural_cos', 'neural_fp')]
        okeep = [c for c in O[0].columns if c not in ('neural_score', 'neural_cos', 'neural_fp')]
        assert nm[keep].equals(M[0][keep]) and no[okeep].equals(O[0][okeep])
        sc = {m: [f.neural_score[f.molecule_id == m].to_numpy(np.float64) for f in M] for m, _ in mols}
        want = combine([{m: sc[m][k] for m in enabled} for k in range(K)], list(enabled), mode)
        for m, _ in mols:
            sel = (nm.molecule_id == m).to_numpy()
            got = nm.neural_score.to_numpy()[sel]
            if m in enabled:
                assert np.allclose(got, want[m], rtol=0, atol=1e-12), (mode, m)
                assert (nm.sens_members[sel] == K).all()
                for c in ('neural_cos', 'neural_fp'):
                    mean = np.mean([f[c][f.molecule_id == m].to_numpy(np.float64) for f in M], 0)
                    assert np.allclose(nm[c].to_numpy()[sel], mean, rtol=0, atol=1e-12), (mode, m, c)
                if mode == 'raw' and m != 'm6':   # the mean score is still cos / T + w fp of the mean cos / fp
                    lin = nm.neural_cos.to_numpy()[sel] / T + W * nm.neural_fp.to_numpy()[sel]
                    assert np.allclose(got, lin, atol=1e-4), m
            else:
                assert np.array_equal(got, M[0].neural_score[M[0].molecule_id == m].to_numpy(np.float64))
                assert (nm.sens_members[sel] == 0).all()
                for c in ('neural_cos', 'neural_fp'):
                    assert np.array_equal(nm[c].to_numpy()[sel], M[0][c][M[0].molecule_id == m].to_numpy(np.float64),
                                          equal_nan=True)
        if mode == 'z':                     # m6: the constant member adds zeros (std 0 -> 1, as zscore())
            s6 = sc['m6']
            assert np.allclose(nm.neural_score[nm.molecule_id == 'm6'], np.mean([zscore(a) for a in s6], 0))
            assert np.allclose(zscore(s6[1]), 0)
        # outsiders
        mm = {m: [f.neural_score[f.molecule_id == m].to_numpy(np.float64) for f in M] for m in enabled}
        for m in ('m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'zz'):
            sel = (no.molecule_id == m).to_numpy()
            os_ = [f.neural_score.to_numpy(np.float64)[sel] for f in O]
            got = no.neural_score.to_numpy()[sel]
            nmem = no.sens_members.to_numpy()[sel]
            if m not in enabled:
                assert np.array_equal(got, os_[0], equal_nan=True) and (nmem == 0).all(), (mode, m)
                continue
            for j in range(sel.sum()):
                okj = [bool(f.neural_ok.to_numpy()[sel][j]) and np.isfinite(os_[k][j]) for k, f in enumerate(O)]
                if mode == 'raw':
                    full, alone = np.mean([os_[k][j] for k in range(K)]), os_[0][j]
                else:
                    zs = [(os_[k][j] - mm[m][k].mean()) / (mm[m][k].std() if mm[m][k].std() > 0 else 1.0)
                          for k in range(K)]
                    full, alone = float(np.mean(zs)), zs[0]
                if all(okj):
                    assert np.isclose(got[j], full, rtol=0, atol=1e-12) and nmem[j] == K, (mode, m, j)
                    cf = np.mean([f.neural_cos.to_numpy(np.float64)[sel][j] for f in O])
                    assert np.isclose(no.neural_cos.to_numpy()[sel][j], cf, rtol=0, atol=1e-12)
                elif okj[0]:
                    assert np.isclose(got[j], alone, rtol=0, atol=1e-12) and nmem[j] == 1, (mode, m, j)
                    assert no.neural_cos.to_numpy()[sel][j] == O[0].neural_cos.to_numpy(np.float64)[sel][j]
                else:
                    assert got[j] == os_[0][j] and nmem[j] == 0, (mode, m, j)
        assert st['outsiders'] == 3 * len(enabled) - 2 and st['outsiders_published_only'] == 1, st
        assert st['outsiders_kept'] == 3 * (len(mols) - len(enabled)) + 2 + 1, st
        assert st['rows'] == st['rows_main'] + st['outsiders'] and st['rows_main'] == 6 + 2 + 5
        assert set(st['per_seed']) == set(map(str, seeds)) and all(d['molecules'] == 3 for d in st['per_seed'].values())
    # stats on a constructed case: m0 leader flips, m1 does not
    base = pd.DataFrame(dict(molecule_id=['a'] * 3 + ['b'] * 3, smiles=list('xyzuvw'), baseline_rank=[1, 2, 3] * 2,
                             processed=True, neural_ok=True, neural_cos=0.1, neural_fp=0.1,
                             neural_score=np.array([3, 2, 1, 3, 2, 1], np.float32)))
    other = base.assign(neural_score=np.array([0, 9, 1, 3, 1, 2], np.float32))
    nm, _, st = ens(base, None, [other], None, 'raw', ['s'])
    assert st['top1_changed'] == 1 and st['molecules'] == 2
    assert nm.neural_score.tolist() == [1.5, 5.5, 1.0, 3.0, 1.5, 1.5]
    assert st['per_seed']['s']['top1_agree'] == 0.5 and st['per_seed']['s']['rho'] == round((-0.5 + 0.5) / 2, 4)
    rho_a = pd.Series([3, 2, 1]).rank().corr(pd.Series([1.5, 5.5, 1.0]).rank())
    rho_b = pd.Series([3, 2, 1]).rank().corr(pd.Series([3.0, 1.5, 1.5]).rank())
    assert st['rho'] == round(float(np.mean([rho_a, rho_b])), 4), st['rho']
    # a non-default index (positions, not labels)
    shuf = base.set_index(pd.Index([50, 3, 9, 1, 7, 0]))
    nm2, _, _ = ens(shuf, None, [other.set_index(shuf.index)], None, 'raw', ['s'])
    assert nm2.neural_score.tolist() == nm.neural_score.tolist() and list(nm2.index) == list(shuf.index)
    # refusals
    bad = [(lambda: ens(base, None, [other.iloc[::-1]], None, 'raw', ['s']), 'differs in molecule_id'),
           (lambda: ens(base, None, [other.iloc[:5]], None, 'raw', ['s']), 'has 5 rows'),
           (lambda: ens(base, None, [other.drop(columns='neural_fp')], None, 'raw', ['s']), 'lacks'),
           (lambda: ens(base, None, [other], None, 'mean', ['s']), 'mode'),
           (lambda: ens(base, None, [other], None, 'raw', ['s', 't']), ''),
           (lambda: ens(base, None, [], None, 'raw', []), ''),
           (lambda: ens(nm, None, [other], None, 'raw', ['s']), 'already ran'),
           (lambda: ens(base, base, [other], [], 'raw', ['s']), 'one outside-candidate frame')]
    for f, why in bad:
        try:
            f()
        except AssertionError as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f'not refused: {why}')
    return dict(pure='ok')


# ------------------------------------------------------------------------------------------------- B. driver + stub
STUB = r'''"""Stub of the neural scorer CLI for the SENS driver test (ours): python stub.py BUNDLE CKPT EVIDENCE QUERIES OUT DEVICE"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

bundle, ckpt, evidence, queries, out, device = sys.argv[1:7]
B = Path(bundle)
for n in ('model.py', 'features.py', 'config.json', ckpt):
    assert (B / n).is_file(), n
raw = (B / ckpt).read_bytes()
Path(str(out) + '.call.json').write_text(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(),
    env={k: os.environ.get(k) for k in ('PYTHONHASHSEED', 'PYTHONUNBUFFERED', 'INC_PANEL')})))
if b'EXIT1' in raw:
    print('stub: asked to fail'); sys.exit(1)
fail = set(raw.split(b'fail=')[1].split(b';')[0].decode().split(',')) if b'fail=' in raw else set()
seed = int.from_bytes(hashlib.sha256(raw).digest()[:4], 'little')
frame = pd.read_parquet(evidence)
assert Path(queries).is_file()
parts, errors = [], []
for mid, g in frame.groupby('molecule_id', sort=False):
    g = g.sort_values('baseline_rank', kind='stable').copy()
    g['neural_ok'] = False
    if not g.processed.iloc[0] or mid in fail:
        if mid in fail:
            errors.append(dict(molecule_id=mid, error='stub failure'))
        for c in ['neural_cos', 'neural_fp', 'neural_score', 'neural_std', 'n_spectra', 'n_adducts']:
            g[c] = 0.
        parts.append(g); continue
    r = np.array([int.from_bytes(hashlib.sha256(f'{seed}|{s}'.encode()).digest()[:4], 'little') / 2**32
                  for s in g.smiles])
    cos = (0.2 + 0.3 * r).astype(np.float32); fp = (0.1 + 0.2 * r[::-1]).astype(np.float32)
    g['neural_cos'] = cos; g['neural_fp'] = fp; g['neural_score'] = cos / np.float32(0.07) + np.float32(0.5) * fp
    g['neural_std'] = np.float32(0.1); g['n_spectra'] = 2; g['n_adducts'] = 1; g['neural_ok'] = True
    parts.append(g)
res = pd.concat(parts, ignore_index=True)
res.to_parquet(out, index=False)
Path(str(out) + '.errors.json').write_text(json.dumps(errors))
print('NEURAL FEATURES', res.shape, ckpt, flush=True)
'''
SEEDS = [20262023, 20263023, 20264023, 20265023]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_tree(d, ckpt_extra=None, decision=None):
    d = Path(d)
    root, inp, code, bundle = d / 'working', d / 'input', d / 'working' / 'p2s_code', d / 'input' / 'bundle_v1'
    for p in (root, code, bundle):
        p.mkdir(parents=True)
    (code / 'neural_scores.py').write_text(STUB)
    for n, t in (('model.py', 'class M: pass\n'), ('features.py', 'X = 1\n'), ('config.json', '{"seed": 20261023}')):
        (bundle / n).write_text(t)
    (bundle / 'final_model.pt').write_bytes(b'published-weights')
    pins = {n: sha(bundle / n) for n in ('model.py', 'features.py', 'config.json', 'final_model.pt')}
    src = inp / 'notebooks' / 'bryantyu' / 'casmi26-saner-seeds'
    src.mkdir(parents=True)
    table = {}
    for s in [20261023] + SEEDS:
        p = src / f'final_model_seed{s}.pt'
        p.write_bytes(b'published-weights' if s == 20261023 else f'weights-{s};'.encode() + (ckpt_extra or {}).get(s, b''))
        (src / f'{p.name}.sha256').write_text(f'{sha(p)}  {p.name}\n')
        table[p.name] = dict(file=p.name, file_sha256=sha(p), seed=s, epochs=5)
    (src / 'final_models_sha256.json').write_text(json.dumps(table, indent=2))
    (src / 'seeds_report.json').write_text(json.dumps(dict(decision=decision or dict(verdict='GO', k=5, mode='raw'))))
    # a decoy folder elsewhere (no casmi26-saner-seeds in its path) that must not be chosen
    decoy = inp / 'other' / 'x'
    decoy.mkdir(parents=True)
    (decoy / 'final_models_sha256.json').write_text('{}')
    # evidence: 6 molecules, m5 not processed (the smoke commit's placeholders)
    rows, orows = [], []
    for i in range(6):
        for j in range(4 + i % 3):
            rows.append(dict(molecule_id=f'm{i}', smiles=f'C{i}_{j}', metric_key=f'K{i}{j}', baseline_rank=j + 1,
                             processed=i != 5, library_max=0.4))
        for j in range(3):
            orows.append(dict(molecule_id=f'm{i}', smiles=f'O{i}_{j}', metric_key=f'Q{i}{j}', baseline_rank=26 + j,
                              processed=i != 5))
    ev = pd.DataFrame(rows).sample(frac=1, random_state=1)             # unsorted: the scorer sorts per molecule
    pd.DataFrame(ev).to_parquet(root / 'frozen_candidate_evidence.parquet', index=False)
    pd.DataFrame(orows).to_parquet(root / 'outsider_evidence.parquet', index=False)
    pd.DataFrame(dict(molecule_id=['m0'], x=[1])).to_parquet(root / 'effective_query_spectra.parquet', index=False)
    import subprocess
    for evp, outp in (('frozen_candidate_evidence', 'neural_candidate_features'),
                      ('outsider_evidence', 'outsider_neural_features')):       # the "published" scoring
        subprocess.run([sys.executable, str(code / 'neural_scores.py'), str(bundle), 'final_model.pt',
                        str(root / f'{evp}.parquet'), str(root / 'effective_query_spectra.parquet'),
                        str(root / f'{outp}.parquet'), 'cuda'], cwd=root, check=True, capture_output=True)
    return dict(ROOT=root, INPUT=inp, CODE=code, BUNDLE=bundle, MODEL_PINS=pins, src=src,
                MODULE_SHA256={'neural_scores.py': sha(code / 'neural_scores.py')})


def run_driver(tree, **over):
    g = dict(ROOT=tree['ROOT'], INPUT=tree['INPUT'], CODE=tree['CODE'], BUNDLE=tree['BUNDLE'],
             MODEL_PINS=tree['MODEL_PINS'], MODULE_SHA256=tree['MODULE_SHA256'], NB_T0=time.time(), NEURAL_OK=True,
             OUTSIDERS_OK=True, SENS=5, SENS_MODE='raw', SENS_SEEDS=list(SEEDS), SENS_MAX_SECONDS=600)
    g.update({k: v for k, v in over.items() if not k.startswith('_')})
    ns = cell_ns(**g)
    if '_os_replace_fail_on' in over:          # make the n-th os.replace inside the cell fail
        real, calls = os, []

        class Shim:
            def __getattr__(self, name):
                return getattr(real, name)

            def replace(self, a, b):
                calls.append(a)
                if len(calls) == over['_os_replace_fail_on']:
                    raise OSError('injected replace failure')
                return real.replace(a, b)
        ns['os'] = Shim()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rep = ns['ours_seed_ensemble']()
    return rep, buf.getvalue(), ns


def published_bytes(tree):
    return {n: (tree['ROOT'] / n).read_bytes() for n in ('neural_candidate_features.parquet',
                                                          'outsider_neural_features.parquet')}


def test_driver():
    out = {}
    os.environ['INC_PANEL'] = 'set-by-test'          # cell 9 drops it from the scorer's environment
    with tempfile.TemporaryDirectory() as d:
        # ---- applied, raw
        tree = build_tree(Path(d) / 'raw', ckpt_extra={20263023: b'fail=m2'},
                          decision=dict(verdict='GO', k=5, mode='raw', members=[20261023] + SEEDS))
        before = published_bytes(tree)
        rep, log, ns = run_driver(tree, OUTSIDERS_OK=False)      # the features file, not the flag, decides
        assert rep['status'] == 'applied' and rep['outsiders_file'] is True, (rep, log)
        assert rep['k'] == 5 and rep['seeds'] == SEEDS and rep['failures'] == 2, rep      # m2 failed on main + outs
        assert rep['molecules'] == 4 and rep['kept_not_ok'] == 1 and rep['inactive'] == 1, rep
        assert rep['published_copy_matches_pin'] is True and rep['seeds_decision']['verdict'] == 'GO'
        assert all(v['ok'] for v in rep['checkpoints'].values()) and 'casmi26-saner-seeds' in rep['source']
        lines = [l for l in log.splitlines() if l.startswith('SENS ')]
        assert lines[0].startswith('SENS setup | K 5 (published + 20262023,20263023,20264023,20265023) | mode raw')
        assert 'sha256 vs final_models_sha256.json 4/4 ok (.sha256 files 4/4)' in lines[0] and 'NOTE' not in lines[0]
        assert lines[1].startswith('SENS seeds | vs published on the ensembled lists | 20262023 rho ')
        assert lines[2].startswith('SENS applied | K 5 | mode raw | molecules 4 | rows ') and \
            '| changed top1 (neural) ' in lines[2] and '| rank corr ' in lines[2] and lines[2].endswith(' s')
        root = tree['ROOT']
        assert not list((root / 'sens_work').glob('bundle_*')), 'temporary bundles left'
        assert {p.name for p in (root / 'sens_backup').iterdir()} == set(before)
        assert all((root / 'sens_backup' / n).read_bytes() == b for n, b in before.items())
        rj = json.loads((root / 'sens_report.json').read_text())
        assert rj['status'] == 'applied' and len(rj['runs']) == 8
        for s in SEEDS:                                  # command line, cwd and environment as cell 9
            for stem, ev in (('neural_candidate_features', 'frozen_candidate_evidence'),
                             ('outsider_neural_features', 'outsider_evidence')):
                call = json.loads((root / 'sens_work' / f'{stem}_seed{s}.parquet.call.json').read_text())
                assert call['argv'] == [str(root / 'sens_work' / f'bundle_{s}'), 'final_model.pt',
                                        str(root / f'{ev}.parquet'), str(root / 'effective_query_spectra.parquet'),
                                        str(root / 'sens_work' / f'{stem}_seed{s}.parquet'), 'cuda'], call
                assert Path(call['cwd']).resolve() == root.resolve()
                assert call['env'] == dict(PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', INC_PANEL=None), call
        nf = pd.read_parquet(root / 'neural_candidate_features.parquet')
        of = pd.read_parquet(root / 'outsider_neural_features.parquet')
        pub = pd.read_parquet(root / 'sens_backup' / 'neural_candidate_features.parquet')
        mem = [pd.read_parquet(root / 'sens_work' / f'neural_candidate_features_seed{s}.parquet') for s in SEEDS]
        on = nf.molecule_id.isin(['m0', 'm1', 'm3', 'm4']).to_numpy()
        want = np.mean([pub.neural_score.to_numpy(np.float64)] + [m.neural_score.to_numpy(np.float64) for m in mem], 0)
        assert np.allclose(nf.neural_score.to_numpy()[on], want[on], rtol=0, atol=1e-12)
        assert np.array_equal(nf.neural_score.to_numpy()[~on], pub.neural_score.to_numpy(np.float64)[~on])
        assert (nf.sens_members.to_numpy()[on] == 5).all() and (nf.sens_members.to_numpy()[~on] == 0).all()
        assert nf.drop(columns=[c for c in nf.columns if c.startswith(('neural_score', 'neural_cos', 'neural_fp',
                                                                        'sens_'))]).equals(
            pub.drop(columns=['neural_score', 'neural_cos', 'neural_fp']))
        assert (of.sens_members.to_numpy() == np.where(of.molecule_id.isin(['m0', 'm1', 'm3', 'm4']), 5, 0)).all()
        assert [c for c in nf.columns if c.startswith('neural_score_seed')] == [f'neural_score_seed{s}' for s in SEEDS]
        out['raw'] = {k: rep[k] for k in ('molecules', 'rows', 'top1_changed', 'rho', 'outsiders', 'kept_not_ok')}
        # ---- applied, z, K 3, seeds decision differs -> NOTE; no outsider features file -> final lists only
        tree = build_tree(Path(d) / 'z', decision=dict(verdict='LEAN_GO', k=4, mode='raw'))
        (tree['ROOT'] / 'outsider_neural_features.parquet').unlink()
        rep, log, _ = run_driver(tree, SENS=3, SENS_MODE='z', OUTSIDERS_OK=False)
        assert rep['status'] == 'applied' and rep['seeds'] == SEEDS[:2] and rep['outsiders'] == 0, rep
        assert rep['outsiders_file'] is False and len(rep['runs']) == 2
        assert 'NOTE: build K 3 mode z, seeds decision k 4 mode raw' in log
        root = tree['ROOT']
        assert not (root / 'outsider_neural_features.parquet').exists()
        nf = pd.read_parquet(root / 'neural_candidate_features.parquet')
        g = nf[nf.molecule_id == 'm0']
        cols = ['neural_score_published'] + [f'neural_score_seed{s}' for s in SEEDS[:2]]
        zs = [(g[c] - g[c].mean()) / g[c].std(ddof=0) for c in cols]
        assert np.allclose(g.neural_score, np.mean(zs, 0)) and abs(g.neural_score.mean()) < 1e-12
        out['z'] = {k: rep[k] for k in ('molecules', 'rows', 'top1_changed', 'rho')}
        # ---- applied; the seeds decision has the same k / mode but other members (a seed's holdout failed) -> NOTE
        alt = [20261023, 20262023, 20263023, 20265023, 20266023]
        tree = build_tree(Path(d) / 'members', decision=dict(verdict='GO', k=5, mode='raw', members=alt))
        rep, log, _ = run_driver(tree)
        assert rep['status'] == 'applied' and rep['seeds_decision']['members'] == alt, rep
        assert f'NOTE: build K 5 mode raw, seeds decision k 5 mode raw members {alt}' in log, log
        # ---- fallbacks: the published files stay (or come back) byte for byte
        cases = [('sha256 mismatch', dict(), 'sha256 differs'),
                 ('missing checkpoint', dict(SENS_SEEDS=[20262023, 20269999, 20264023, 20265023]), 'not in'),
                 ('scorer exit 1', dict(), 'neural_scores.py exit 1'),
                 ('second replace fails', dict(_os_replace_fail_on=2), 'injected replace failure'),
                 ('budget', dict(NB_T0=time.time() - 8 * 3600 + 60), 'TimeoutError'),
                 ('neural_scores.py pin', dict(MODULE_SHA256={'neural_scores.py': '0' * 64}), 'pin'),
                 ('bad mode', dict(SENS_MODE='mean'), 'SENS_MODE'),
                 ('too few seeds', dict(SENS_SEEDS=[20262023]), 'needs 4 distinct'),
                 ('published seed listed', dict(SENS_SEEDS=[20261023, 20262023, 20263023, 20264023]), 'distinct'),
                 ('outsider evidence missing', dict(), 'evidence file')]
        for i, (name, over, why) in enumerate(cases):
            tree = build_tree(Path(d) / f'f{i}', ckpt_extra={20264023: b'EXIT1'} if name == 'scorer exit 1' else None)
            if name == 'sha256 mismatch':
                t = json.loads((tree['src'] / 'final_models_sha256.json').read_text())
                t['final_model_seed20263023.pt']['file_sha256'] = '1' * 64
                (tree['src'] / 'final_models_sha256.json').write_text(json.dumps(t))
            if name == 'outsider evidence missing':
                (tree['ROOT'] / 'outsider_evidence.parquet').unlink()
            before = published_bytes(tree)
            rep, log, _ = run_driver(tree, **over)
            assert rep['status'] == 'fallback_published' and why in rep['error'], (name, rep.get('error'), log)
            assert published_bytes(tree) == before, name
            root = tree['ROOT']
            assert 'WARNING SENS failed, published saner scores kept' in log and \
                log.strip().splitlines()[-1].startswith('SENS fallback_published | K ') and \
                '| molecules 0 | rows 0 | changed top1 (neural) 0 |' in log.strip().splitlines()[-1], (name, log)
            assert not list(root.glob('*.sens.tmp')) and not (root / 'sens_work').is_dir() or \
                not list((root / 'sens_work').glob('bundle_*')), name
            assert json.loads((root / 'sens_report.json').read_text())['status'] == 'fallback_published'
            out[name] = rep['error'][:90]
        # SENS = 0: off, nothing touched
        tree = build_tree(Path(d) / 'off')
        before = published_bytes(tree)
        rep, log, _ = run_driver(tree, SENS=0)
        assert rep['status'] == 'off' and published_bytes(tree) == before and 'SENS off' in log
    os.environ.pop('INC_PANEL', None)
    return out


if __name__ == '__main__':
    res = dict(A=test_pure(), B=test_driver())
    print(json.dumps(res, indent=1, default=str))
    print('sens cell tests OK')
