"""GLACIER runner, drop-in for the fork's `gl_runner.py` (fix F5b, clean-room).

Written from the behavioural specification scratchpad f5b/glue_spec.md (sections 2, 4-10), from ms-pred @708148c2a8eb
(MIT; GLACIER `JointModel`, `TreeProcessor`, `IntenDataset.collate_fn`, `predict_smis_joint.prepare_entry`) and from
our own casmi/fwdrun.py / casmi/entropy.py. No code of the unlicensed runner glue was used.

Two ways to use it (the file is meant to be shipped as `<pkg>/gl_runner.py`, but works under any name):

1. Program (same CLI and files as the fork's runner):
     python gl_runner.py <pkg> <in.json> <out.json> [--device cuda] [--budget S] [--site DIR|none] [--wheels DIR]
                         [--ckpt FILE] [--bs 32] [--chunk-jobs 3000] [--threads N] [--memfrac F]
                         [--nondeterministic] [--any-rdkit] [--quiet] [--src DIR] [--shim DIR] [--dump-preds FILE]
   in.json : [{"mid", "cands": [smiles], "spectra": [{"mz", "it", "prec", "adduct", "mode", "instrument",
               "ce_ev", "ce_orig", "ce_units"} | {..., "ce"} | {..., "instr"}]}]
   out.json: {mid: {smiles: score | null}} (written all-null first, rewritten atomically after every chunk)
   out.json.meta.json: status (init / bad_input / ok / budget / error), counts, versions, timings, errors, log tail.
   score = mean over the molecule's covered spectra (positive [M+H]+ / [M+Na]+) of the weighted spectral-entropy
   similarity between the cleaned query and the cleaned GLACIER prediction at (adduct, instrument token, CE bucket).

2. Module (the N1 proxy driver `glb_driver.py` loads `<pkg>/gl_runner.py` and needs the 25-name API below, with the
   calling conventions of spec section 10). Importing this module only imports numpy (and numba if present): RDKit,
   torch and ms-pred are imported by `setup_env` / `load_model` / `Predictor`, after the wheel site is on sys.path.

Package layouts found automatically (`<pkg>` = the dataset dir): ms-pred source in `<pkg>/src` or `<pkg>/mspred`,
DGL / torch_scatter stand-ins in `<pkg>/shim` or `<pkg>/fwdshim`, weights `<pkg>/ckpt/glacier.pt` or
`<pkg>/ckpt/best.ckpt` (or $GL_CKPT), wheels `--wheels`, `<pkg>/wheels` or $GL_WHEELS.
"""
import argparse
import glob
import importlib.abc
import importlib.machinery
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import types
from collections import OrderedDict, deque

import numpy as np

_T_START = time.time()            # the budget is measured from process start (it includes site install and load)

try:
    import numba
    HAVE_NUMBA = True
except Exception:                 # pure-Python kernels without numba
    numba = None
    HAVE_NUMBA = False

# ------------------------------------------------------------------------------------------------ constants ------
COVERED = ('[M+H]+', '[M+Na]+')              # Q1: adducts that are scored (positive mode only)
POS_MODES = ('positive', 'pos', '+', '1')
SUM_CE = (20.0, 40.0, 60.0)                  # Q5: CEs predicted and summed when a spectrum has no usable CE
FLOOR, TOPK, POWER, ENTW = 0.002, 256, 1.0, True   # Q6 / S2: prep_query of query and prediction
CLEAN_FLOOR, CLEAN_TOPK, CLEAN_ABOVE = 0.001, 512, 2.0   # Q6: clean_peaks (relative floor, top-k, Da above prec)
TOL, PPM = 0.01, 20.0                        # S3: match tolerance max(TOL Da, PPM ppm of the query m/z)
PRED_TOPK = 100                              # P-GL: peaks kept per prediction
MAX_HEAVY = 100                              # G1: heavy-atom limit (= ms-pred prepare_entry)
BATCH_AREA = 1600                            # P-batch: batch <= bs * BATCH_AREA / N^2 (N = max heavy atoms)
FAIL_FAST_N = 64                             # P-fail: raise once this many attempts all failed
GUARD_N = 50                                 # C2: raise when this many SMILES were checked and all rejected
FEAT_LRU = 512                               # P-GL: featurisation cache entries per (canonical, adduct)
RDKIT_REQ = '2025.03'                        # E4
WHEEL_MARK = '.ice_site_ok'                  # E3 (same marker as the notebook's pre-seed writes)
OPTIONAL = ('pygmtools', 'h5py', 'multiprocess', 'platformdirs', 'matplotlib', 'seaborn', 'cairosvg', 'wandb')
PLACEHOLDERS = {'ms_pred.common.plot_utils': ('plot_mol_as_vector', 'plot_compare_ms', 'plot_ms'),
                'ms_pred.nn_utils.tune_utils': (), 'ms_pred.nn_utils.base_hyperopt': ()}
EVICT = ('rdkit', 'dgl', 'torch_scatter', 'ms_pred', 'LinSATNet')
RUNNER_ID = 'casmi-glrun-dropin 1'

_CANON = {}       # input SMILES -> (canonical SMILES | None, heavy atoms of the canonical form, reason)
_HEAVY = {}       # canonical SMILES -> heavy atoms


# ------------------------------------------------------------------------------------------------ utilities ------
class Log:
    """Callable logger: '[gl     12.3s] ...' lines on stderr; the last `keep` lines are kept for meta.log_tail."""

    def __init__(self, tag='gl', quiet=False, t0=None, keep=40):
        self.tag, self.quiet, self.t0 = tag, quiet, (_T_START if t0 is None else t0)
        self.lines = deque(maxlen=keep)

    def __call__(self, *x):
        s = f'[{self.tag:<4} {time.time() - self.t0:7.1f}s] ' + ' '.join(str(v) for v in x)
        self.lines.append(s)
        if not self.quiet:
            print(s, file=sys.stderr, flush=True)

    def tail(self):
        return list(self.lines)


def _nolog(*x):
    pass


def write_json(path, obj):
    """Atomic JSON write (temp file in the same directory, then os.replace); NaN / inf are refused."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.' + os.path.basename(path) + '.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


# ------------------------------------------------------------------------------------------- query metadata -----
def instr_family(s):
    """Q2: 0 timsTOF, 1 Orbitrap / FT family, 2 other TOF, 3 ion trap, 4 triple quad, 5 unknown (incl. None / NaN)."""
    if not isinstance(s, str):
        return 5
    s = s.lower()
    if 'timstof' in s:
        return 0
    if any(t in s for t in ('orbitrap', 'qft', 'ftms', 'hybrid ft', 'itft', 'exactive', 'ft-icr', 'fticr')):
        return 1
    if 'tof' in s:
        return 2
    if any(t in s for t in ('trap', 'qit', 'lit')):
        return 3
    if any(t in s for t in ('qq', 'quattro', 'triple')):
        return 4
    return 5


def instr_token(sp):
    """Q2: GLACIER instrument token of a spectrum dict: a pre-resolved 'instr' wins, else Orbitrap iff family 1."""
    t = sp.get('instr')
    if t in ('Orbitrap', 'QTOF'):
        return t
    return 'Orbitrap' if instr_family(sp.get('instrument')) == 1 else 'QTOF'


def is_covered(sp):
    """Q1: positive mode (1 or 'positive' / 'pos' / '+' / '1', any case) and adduct [M+H]+ or [M+Na]+."""
    m = sp.get('mode')
    if isinstance(m, str):
        pos = m.strip().lower() in POS_MODES
    elif isinstance(m, (int, float, np.integer, np.floating)) and not isinstance(m, bool):
        pos = m == 1
    else:
        pos = False
    return bool(pos) and sp.get('adduct') in COVERED


_NUM = re.compile(r'\d+(?:\.\d*)?|\.\d+')


def spectrum_ce(sp):
    """Q3: the spectrum's collision energy in eV (NaN when unknown).
    Raw metadata present (ce_units or ce_ev not None): NCE units -> mean of the numbers in ce_orig; otherwise the
    mean of ce_ev. Neither present: a pre-resolved 'ce'."""
    units, ev = sp.get('ce_units'), sp.get('ce_ev')
    if units is not None or ev is not None:
        if units == 'NCE':
            nums = [float(x) for x in _NUM.findall(str(sp.get('ce_orig') or ''))]
            return float(np.mean(nums)) if nums else float('nan')
        if ev is None:
            return float('nan')
        if isinstance(ev, (int, float, np.integer, np.floating)):
            ev = [ev]
        try:
            vals = [float(x) for x in ev]
        except (TypeError, ValueError):
            return float('nan')
        return float(np.mean(vals)) if vals else float('nan')
    ce = sp.get('ce')
    return float(ce) if _finite(ce) else float('nan')


def ce_bucket(ce):
    """Q4: round(CE / 5) * 5 (half to even), at least 5; NaN stays NaN."""
    if not _finite(ce):
        return float('nan')
    return max(5.0, float(np.round(float(ce) / 5.0)) * 5.0)


# ------------------------------------------------------------------------------------------------ peaks ---------
def _empty():
    return np.zeros(0, np.float64), np.zeros(0, np.float64)


def clean_peaks(mz, it, prec, floor=CLEAN_FLOOR, topk=CLEAN_TOPK, above=CLEAN_ABOVE):
    """Q6 step 1: peaks with 0 < m/z <= prec + above and intensity >= floor x base (base = the most intense peak
    with m/z <= prec + above), the `topk` most intense if more, intensities / base, m/z order, float32."""
    mz = np.asarray(mz, np.float64).ravel()
    it = np.asarray(it, np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it)
    mz, it = mz[ok], it[ok]
    e32 = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    if len(mz) == 0 or not _finite(prec):
        return e32
    win = mz <= float(prec) + above
    if not win.any():
        return e32
    base = it[win].max()
    if not base > 0:
        return e32
    keep = win & (mz > 0) & (it >= floor * base)
    mz, it = mz[keep], it[keep]
    if len(mz) > topk:
        o = np.argsort(-it, kind='stable')[:topk]
        mz, it = mz[o], it[o]
    o = np.argsort(mz, kind='stable')
    return mz[o].astype(np.float32), (it[o] / base).astype(np.float32)


def entropy_weights(it, power=POWER, entw=ENTW):
    """S2: p = I^power / sum; if entw and the entropy S (nats) < 3: p <- p^w / sum(p^w), w = 0.25 + 0.25 S."""
    it = np.asarray(it, np.float64)
    if len(it) == 0:
        return it
    w = it if power == 1 else it ** power
    s = w.sum()
    if not s > 0:
        return np.zeros_like(w)
    p = w / s
    if entw:
        nz = p[p > 0]
        h = float(-(nz * np.log(nz)).sum())
        if h < 3.0:
            p = p ** (0.25 + 0.25 * h)
            p = p / p.sum()
    return p


def prep_query(mz, it, floor=FLOOR, topk=TOPK, power=POWER, entw=ENTW):
    """Q6 step 3 / S1: peaks with intensity >= floor x max, the `topk` most intense if more (m/z order restored),
    then entropy weights. Returns (m/z float64, weights float64 summing to 1); empty in -> empty out."""
    mz = np.asarray(mz, np.float64).ravel()
    it = np.asarray(it, np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it)
    mz, it = mz[ok], it[ok]
    if len(mz) == 0:
        return _empty()
    mx = it.max()
    if not mx > 0:
        return _empty()
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > topk:
        o = np.argsort(-it, kind='stable')[:topk]
        mz, it = mz[o], it[o]
    o = np.argsort(mz, kind='stable')
    mz, it = mz[o], it[o]
    return mz, entropy_weights(it, power, entw)


def prep_spectrum(sp):
    """Q6: query spectrum dict -> (m/z, weights): clean_peaks twice (float32 round trip), then prep_query."""
    prec = sp.get('prec')
    prec = float(prec) if _finite(prec) else float('nan')
    m, i = clean_peaks(sp.get('mz') or [], sp.get('it') or [], prec)
    m, i = clean_peaks(m.astype(np.float64), i.astype(np.float64), prec)
    return prep_query(m.astype(np.float64), i.astype(np.float64), FLOOR, TOPK, POWER, ENTW)


def post(arr, k=PRED_TOPK):
    """P-GL step 3: raw prediction rows [n, 2] (m/z, intensity) -> rows with m/z > 0 and intensity > 0, identical
    m/z (4 dp) merged by maximum, the k most intense (stable), m/z order, / max; (m/z, intensity) float32 or None."""
    a = np.asarray(arr, np.float64)
    if a.size == 0:
        return None
    a = a.reshape(-1, 2)
    mz, it = a[:, 0], a[:, 1]
    keep = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0)
    if not keep.any():
        return None
    u, inv = np.unique(np.round(mz[keep], 4), return_inverse=True)
    v = np.zeros(len(u))
    np.maximum.at(v, inv, it[keep])
    if len(u) > k:
        o = np.argsort(-v, kind='stable')[:k]
        u, v = u[o], v[o]
    o = np.argsort(u, kind='stable')
    u, v = u[o], v[o]
    return u.astype(np.float32), (v / v.max()).astype(np.float32)


def merge(parts):
    """Q5: predictions at several CEs -> one: each part / its max, concatenated, intensities at identical m/z (4 dp)
    summed, / max (no top-k). Returns (m/z float64, intensity float64)."""
    ms, vs = [], []
    for p in parts:
        m = np.asarray(p[0], np.float64).ravel()
        v = np.asarray(p[1], np.float64).ravel()
        if len(v) and v.max() > 0:
            ms.append(m)
            vs.append(v / v.max())
    if not ms:
        return _empty()
    m, v = np.concatenate(ms), np.concatenate(vs)
    u, inv = np.unique(np.round(m, 4), return_inverse=True)
    s = np.zeros(len(u))
    np.add.at(s, inv, v)
    return u, s / s.max()


# ----------------------------------------------------------------------------------------------- similarity -----
def _sim_kernel(qm, qp, pm, pp, tol, ppm):
    na, nb = qm.shape[0], pm.shape[0]
    if na == 0 or nb == 0:
        return 0.0
    sa = 0.0
    for x in qp:
        if x > 0:
            sa -= x * math.log(x)
    sb = 0.0
    for x in pp:
        if x > 0:
            sb -= x * math.log(x)
    sab = 0.0
    i = 0
    j = 0
    while i < na and j < nb:
        t = max(tol, qm[i] * ppm * 1e-6)
        d = qm[i] - pm[j]
        if d < -t:                    # query peak below the prediction peak: unmatched
            v = qp[i] * 0.5
            i += 1
        elif d > t:                   # prediction peak below the query peak: unmatched
            v = pp[j] * 0.5
            j += 1
        else:                         # within tolerance: matched, both advance
            v = (qp[i] + pp[j]) * 0.5
            i += 1
            j += 1
        if v > 0:
            sab -= v * math.log(v)
    while i < na:
        v = qp[i] * 0.5
        if v > 0:
            sab -= v * math.log(v)
        i += 1
    while j < nb:
        v = pp[j] * 0.5
        if v > 0:
            sab -= v * math.log(v)
        j += 1
    s = 1.0 - (2.0 * sab - sa - sb) / math.log(4.0)
    if s != s:
        return s
    if s < 1e-9:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


if HAVE_NUMBA:
    try:
        _sim_kernel_jit = numba.njit(cache=False)(_sim_kernel)
    except Exception:
        _sim_kernel_jit = _sim_kernel
else:
    _sim_kernel_jit = _sim_kernel


def entropy_sim(qm, qp, pm, pp, tol=TOL, ppm=PPM):
    """S3: weighted spectral-entropy similarity of two prepped spectra (m/z ascending, weights summing to 1).
    Two-cursor walk with tolerance max(tol Da, ppm x query m/z); 1 - (2 S_AB - S_A - S_B) / ln 4, clipped to [0, 1]
    (< 1e-9 -> 0); 0 if either spectrum is empty."""
    a = [np.ascontiguousarray(x, dtype=np.float64) for x in (qm, qp, pm, pp)]
    return float(_sim_kernel_jit(a[0], a[1], a[2], a[3], float(tol), float(ppm)))


# ------------------------------------------------------------------------------------------- plans and jobs -----
def molecule_plan(spectra):
    """Covered spectra -> plan entries (query m/z, query weights, adduct, instrument token, CE bucket)."""
    plan = []
    for sp in spectra or []:
        if is_covered(sp):
            qm, qp = prep_spectrum(sp)
            plan.append((qm, qp, sp['adduct'], instr_token(sp), ce_bucket(spectrum_ce(sp))))
    return plan


def molecule_jobs(plan, cands):
    """Prediction jobs (candidate string, adduct, token, CE) of plan entries x candidates; a NaN bucket needs the
    SUM_CE jobs. Order-preserving, without duplicates."""
    jobs = {}
    for e in plan:
        ad, tok, b = e[2], e[3], e[4]
        ces = SUM_CE if not _finite(b) else (float(b),)
        for c in cands:
            for ce in ces:
                jobs[(c, ad, tok, float(ce))] = None
    return list(jobs)


def entry_prediction(entry, c, cache):
    """The prediction a plan entry is scored against: the job's, or the SUM_CE merge (None if any part missing)."""
    ad, tok, b = entry[2], entry[3], entry[4]
    if _finite(b):
        return cache.get((c, ad, tok, float(b)))
    parts = [cache.get((c, ad, tok, ce)) for ce in SUM_CE]
    if any(p is None for p in parts):
        return None
    return merge(parts)


def score_molecule(plan, cands, cache):
    """S1: {candidate: mean over plan entries with a prediction of entropy_sim(query, prep_query(prediction)) | None}."""
    res = {}
    for c in cands:
        vals = []
        for e in plan:
            p = entry_prediction(e, c, cache)
            if p is None:
                continue
            pm, pp = prep_query(p[0], p[1], FLOOR, TOPK, POWER, ENTW)
            s = entropy_sim(e[0], e[1], pm, pp, TOL, PPM)
            if math.isfinite(s):
                vals.append(s)
        res[c] = float(np.mean(vals)) if vals else None
    return res


# ------------------------------------------------------------------------------------------------ candidates -----
_VALID_ATOMS = None


def _valid_atoms():
    global _VALID_ATOMS
    if _VALID_ATOMS is None:
        from ms_pred.common import chem_utils
        _VALID_ATOMS = frozenset(int(a) for a in chem_utils.VALID_ATOM_NUM)
    return _VALID_ATOMS


def smiles_ok(smi):
    """G1: canonicalise like ms-pred prepare_entry (valid elements, explicit H removed, <= 100 heavy atoms, RDKit
    canonical SMILES with stereo, must re-parse with >= 1 atom). Fills _CANON[smi]; returns (ok, reason)."""
    if smi in _CANON:
        c, _, why = _CANON[smi]
        return c is not None, why
    canon, heavy, why = None, 0, ''
    try:
        from rdkit import Chem
        if not isinstance(smi, str) or not smi.strip():
            why = 'empty'
        else:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                why = 'unparseable'
            elif any(a.GetAtomicNum() not in _valid_atoms() for a in m.GetAtoms()):
                why = 'element'
            else:
                m = Chem.RemoveHs(m)
                if m.GetNumAtoms() > MAX_HEAVY:
                    why = 'too many atoms'
                else:
                    c = Chem.MolToSmiles(m)
                    m2 = Chem.MolFromSmiles(c)
                    if m2 is None or m2.GetNumAtoms() < 1:
                        why = 'canonical form does not re-parse'
                    else:
                        canon, heavy = c, m2.GetNumAtoms()
    except Exception as e:
        canon, why = None, f'error {e!r}'[:120]
    _CANON[smi] = (canon, heavy, why)
    if canon is not None:
        _HEAVY[canon] = heavy
    return canon is not None, why


# ------------------------------------------------------------------------------------------------ environment ----
class _AnyMeta(type):
    def __getattr__(cls, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return _Any


class _Any(metaclass=_AnyMeta):
    """Placeholder usable as a value, a callable, a decorator or a base class."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return _Any


def _cache_dir(*parts, **kw):
    d = os.path.join(tempfile.gettempdir(), 'gl_ms_pred_cache')
    return os.path.join(d, *[str(p) for p in parts]) if parts else d


class _StubModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        if k == 'user_cache_dir':
            return _cache_dir
        return _Any


class _FallbackFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Last entry of sys.meta_path: serves a placeholder module for OPTIONAL packages that no real finder has
    (ms-pred imports them at module level; GLACIER inference never calls them)."""

    def __init__(self, log=_nolog):
        self.log, self.served = log, set()

    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in OPTIONAL:
            if name.split('.')[0] not in self.served:
                self.served.add(name.split('.')[0])
                self.log('placeholder module for missing optional package', name.split('.')[0])
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


def _find_dir(pkg, names, marker=None):
    for n in names:
        d = os.path.join(pkg, n)
        if os.path.isdir(d) and (marker is None or os.path.exists(os.path.join(d, marker))):
            return d
    return None


def find_wheels(pkg, wheels=None):
    """E3: wheel dir = --wheels, else <pkg>/wheels, else $GL_WHEELS (the first that holds wheel files)."""
    for d in (wheels, os.path.join(pkg, 'wheels') if pkg else None, os.environ.get('GL_WHEELS')):
        if d and os.path.isdir(d) and (glob.glob(os.path.join(d, '*.whl')) or glob.glob(os.path.join(d, '*.whl.ice'))):
            return d
    return None


def _marked(site):
    return any(os.path.exists(os.path.join(site, m)) for m in (WHEEL_MARK, '.casmi_site_ok'))


def install_site(wheels, site, log=_nolog):
    """E3: offline pip install of the wheel dir into `site`, once. The marker <site>/.ice_site_ok holds the sorted
    newline-joined basenames of the dir's *.whl / *.whl.ice files; when it matches, nothing is installed. Wheels
    built for another CPython (-cp3XY-) are skipped; *.whl.ice are installed under their *.whl name."""
    src = sorted(glob.glob(os.path.join(wheels, '*.whl')) + glob.glob(os.path.join(wheels, '*.whl.ice')))
    if not src:
        raise RuntimeError(f'no wheels in {wheels}')
    want = '\n'.join(sorted(os.path.basename(w) for w in src))
    mark = os.path.join(site, WHEEL_MARK)
    if os.path.exists(mark):
        with open(mark) as f:
            if f.read() == want:
                log('wheel site ready (marker matches):', site)
                return False
    tag = f'-cp{sys.version_info.major}{sys.version_info.minor}-'
    use = [w for w in src if '-cp3' not in os.path.basename(w) or tag in os.path.basename(w)]
    wdir = os.path.join(site, '.wheels')
    os.makedirs(wdir, exist_ok=True)
    whl = []
    for w in use:
        b = os.path.basename(w)
        d = os.path.join(wdir, b[:-4] if b.endswith('.whl.ice') else b)
        with open(w, 'rb') as a, open(d, 'wb') as o:
            o.write(a.read())
        whl.append(d)
    log(f'installing {len(whl)} wheels into {site}')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--no-compile',
                        '--disable-pip-version-check', '--upgrade', '--target', site] + whl,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    if r.returncode != 0:
        raise RuntimeError('wheel install failed: ' + (r.stdout or '')[-2000:])
    with open(mark, 'w') as f:
        f.write(want)
    return True


def setup_env(ns, log=_nolog):
    """E1-E7: wheel site, import path (shim, ms-pred src, site first), module eviction, placeholders for unused
    ms-pred modules and missing optional packages, RDKit version check. `ns` needs .pkg; optional .site, .wheels,
    .any_rdkit, .src, .shim. Returns the RDKit version string."""
    pkg = os.path.abspath(ns.pkg)
    site = getattr(ns, 'site', None) or os.environ.get('ICE_SITE') or '/tmp/ice_site'
    shim = getattr(ns, 'shim', None) or _find_dir(pkg, ('shim', 'fwdshim'))
    src = getattr(ns, 'src', None) or _find_dir(pkg, ('src', 'mspred', os.path.join('ms-pred', 'src')), 'ms_pred')
    if src is None:
        raise RuntimeError(f'ms-pred source not found under {pkg} (src/ or mspred/)')
    paths = [p for p in (shim, src) if p]
    if str(site).lower() != 'none':
        os.makedirs(site, exist_ok=True)
        wd = find_wheels(pkg, getattr(ns, 'wheels', None))
        if wd:
            install_site(wd, site, log)
        elif _marked(site):
            log('no wheel dir; reusing the prepared site', site)
        else:
            raise RuntimeError(f'no wheel dir found and site {site} is not prepared')
        paths.append(site)
    for p in reversed(paths):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for name in list(sys.modules):
        if name.split('.')[0] in EVICT:
            del sys.modules[name]
    for name, funcs in PLACEHOLDERS.items():
        m = types.ModuleType(name)
        for f in funcs:
            setattr(m, f, lambda *a, **k: None)
        sys.modules[name] = m
    if not any(isinstance(f, _FallbackFinder) for f in sys.meta_path):
        sys.meta_path.append(_FallbackFinder(log))
    import rdkit
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    ver = rdkit.__version__
    if not ver.startswith(RDKIT_REQ) and not getattr(ns, 'any_rdkit', False):
        raise RuntimeError(f'RDKit {ver} found, {RDKIT_REQ}.x required (pass --any-rdkit to accept)')
    log(f'environment: rdkit {ver}, src {src}, shim {shim}, site {site}')
    return ver


def find_ckpt(pkg, ckpt=None):
    for c in (ckpt, os.environ.get('GL_CKPT'), os.path.join(pkg, 'ckpt', 'glacier.pt'),
              os.path.join(pkg, 'ckpt', 'best.ckpt')):
        if c and os.path.isfile(c):
            return c
    hits = sorted(glob.glob(os.path.join(pkg, 'ckpt', '*.ckpt')) + glob.glob(os.path.join(pkg, 'ckpt', '*.pt')))
    if hits:
        return hits[0]
    raise FileNotFoundError(f'no GLACIER checkpoint under {pkg}/ckpt (glacier.pt / best.ckpt) and no $GL_CKPT')


def _align_tables(sd, common, log):
    """P-load: the model's one-hot tables must match the checkpoint's embedders. A 2-slot instrument embedder (older
    weights) trims the instrument table to {Orbitrap, QTOF}; any other mismatch raises."""
    from ms_pred.common import chem_utils
    tab = chem_utils.instrument2onehot_pos
    if 'instrument_embedder' in sd:
        n = int(sd['instrument_embedder'].shape[0])
        have = len(set(tab.values()))
        if n != have:
            if n == 2:
                new = {'Orbitrap': 0, 'QTOF': 1}
                tab.clear()
                tab.update(new)
                if common.instrument2onehot_pos is not tab:
                    common.instrument2onehot_pos.clear()
                    common.instrument2onehot_pos.update(new)
                log('instrument table trimmed to', new)
            else:
                raise RuntimeError(f'checkpoint instrument embedder has {n} slots, ms-pred table {have}')
    if 'adduct_embedder' in sd:
        n = int(sd['adduct_embedder'].shape[0])
        have = len(set(chem_utils.ion2onehot_pos.values()))
        if n != have:
            raise RuntimeError(f'checkpoint adduct embedder has {n} slots, ms-pred table {have}')


def load_model(pkg, device='cpu', log=_nolog, ckpt=None):
    """P-load: GLACIER JointModel from {hyper_parameters, state_dict} (a Lightning best.ckpt or a re-save):
    hyper-parameters starting with '_' dropped, one-hot tables aligned, strict load, frozen, on `device`."""
    import torch
    from ms_pred import common
    from ms_pred.common import chem_utils
    from ms_pred.glacier import joint_model
    chem_utils.user_cache_dir = _cache_dir            # E6: never write into the home cache
    path = find_ckpt(pkg, ckpt)
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # Lightning checkpoints can carry non-tensor objects; this is the dataset's own (upstream) file
        obj = torch.load(path, map_location='cpu', weights_only=False)
    hp = {k: v for k, v in dict(obj['hyper_parameters']).items() if not str(k).startswith('_')}
    sd = obj['state_dict']
    _align_tables(sd, common, log)
    model = joint_model.JointModel(**hp)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.freeze()
    model._predict_prepared = True
    model.to(device)
    log(f'model {os.path.basename(path)} loaded on {device}')
    return model


# ------------------------------------------------------------------------------------------------ prediction -----
def _is_oom(e):
    s = repr(e).lower()
    return 'out of memory' in s or type(e).__name__ == 'OutOfMemoryError'


class Predictor:
    """P-GL / P-batch / P-fail: batched GLACIER predictions. predict(jobs, deadline) fills .cache[job] with
    (m/z float64, intensity float64) or None for jobs (candidate string, adduct, token, CE); jobs are de-duplicated
    on the canonical SMILES. Returns False when the deadline stopped it (checked before every batch)."""

    def __init__(self, model, device='cpu', bs=32, log=_nolog):
        import torch
        from ms_pred import common
        from ms_pred.glacier import dataset
        self.model, self.device, self.log = model, device, log
        self.bs = max(1, int(bs))
        self.cache = {}
        self.n_pred = self.n_fail = self.n_feat_fail = self.n_empty = 0
        self.t_pred = 0.0
        self._torch, self._common, self._ds = torch, common, dataset
        self._canon_cache = {}
        self._feat = OrderedDict()
        self._tp = dataset.TreeProcessor(pe_embed_k=model.pe_embed_k, root_encode='graphormer',
                                         embed_elem_group=model.embed_elem_group,
                                         multi_hop_max_dist=model.multi_hop_max_dist)
        self._n_logged = 0

    def _note(self, *x):
        if self._n_logged < 10:
            self._n_logged += 1
            self.log(*x)

    def _item(self, key):
        canon, adduct, tok, ce = key
        f = self._feat.get((canon, adduct))
        if f is None:
            f = self._tp.featurize_tree({'root_canonical_smiles': canon, 'adduct': adduct, 'collision_energy': ce},
                                        include_inten_targs=False, include_frag_targs=False)
            self._feat[(canon, adduct)] = f
            if len(self._feat) > FEAT_LRU:
                self._feat.popitem(last=False)
        else:
            self._feat.move_to_end((canon, adduct))
        pos = self._common.instrument2onehot_pos
        item = dict(f)
        item.update(name=canon, adduct=self._common.ion2onehot_pos[adduct], precursor=0.0, collision_energy=ce,
                    instrument=pos.get(tok, pos.get('Orbitrap', 0)))
        return item

    def _todev(self, x):
        t = self._torch
        if t.is_tensor(x):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, dict):
            return {k: self._todev(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._todev(v) for v in x)
        return x

    def _forward(self, items):
        col = self._ds.IntenDataset.collate_fn(items)
        if self.device != 'cpu':
            col = self._todev(col)
        with self._torch.inference_mode():
            out = self.model.predict_inten_frag_batch(col)
        return [post(s.detach().float().cpu().numpy()) for s in out['spec']]

    _FAILED = object()

    def _predict_items(self, items):
        """post() results per item (None = no positive peak), _FAILED for items whose prediction raised. A failing
        batch is retried in quarters down to single items; CUDA OOM also lowers the run-wide max batch."""
        try:
            return self._forward(items)
        except Exception as e:
            if _is_oom(e):
                try:
                    self._torch.cuda.empty_cache()
                except Exception:
                    pass
                if len(items) > 4:
                    self.bs = min(self.bs, max(4, len(items) // 2))
                    self._note(f'out of memory at batch {len(items)} -> max batch {self.bs}')
            if len(items) == 1:
                self._note('prediction failed', str(items[0].get('name'))[:80], repr(e)[:200])
                return [self._FAILED]
            q = -(-len(items) // 4)
            res = []
            for s in range(0, len(items), q):
                res += self._predict_items(items[s:s + q])
            return res

    def _run(self, keys):
        """-> [(prediction | None, status)] with status ok / empty / fail / feat (featurisation failed)."""
        res = [(None, 'feat')] * len(keys)
        items, idx = [], []
        for t, k in enumerate(keys):
            try:
                items.append(self._item(k))
                idx.append(t)
            except Exception as e:
                self._note('featurisation failed', k[0][:80], repr(e)[:200])
        if items:
            for t, p in zip(idx, self._predict_items(items)):
                if p is self._FAILED:
                    res[t] = (None, 'fail')
                elif p is None:
                    res[t] = (None, 'empty')
                else:
                    res[t] = ((p[0].astype(np.float64), p[1].astype(np.float64)), 'ok')
        return res

    def predict(self, jobs, deadline=float('inf')):
        t0 = time.time()
        try:
            todo = {}
            for j in sorted(set(jobs), key=repr):
                if j in self.cache:
                    continue
                s, ad, tok, ce = j
                if s not in _CANON:
                    smiles_ok(s)
                canon = _CANON[s][0]
                if canon is None:
                    self.cache[j] = None
                    continue
                k = (canon, ad, tok, float(ce))
                if k in self._canon_cache:
                    self.cache[j] = self._canon_cache[k]
                    continue
                todo.setdefault(k, []).append(j)
            keys = sorted(todo, key=lambda k: (_HEAVY.get(k[0], 0), k))
            i = 0
            while i < len(keys):
                if time.time() > deadline:
                    return False
                nmax = max(1, _HEAVY.get(keys[min(i + self.bs, len(keys)) - 1][0], 1))
                b = max(1, min(self.bs, (self.bs * BATCH_AREA) // (nmax * nmax)))
                batch = keys[i:i + b]
                for k, (r, st) in zip(batch, self._run(batch)):
                    if st == 'ok':
                        self.n_pred += 1
                    elif st == 'empty':
                        self.n_empty += 1           # predicted, but no positive peak survived post()
                    elif st == 'fail':
                        self.n_fail += 1
                    else:
                        self.n_feat_fail += 1
                    self._canon_cache[k] = r
                    for j in todo[k]:
                        self.cache[j] = r
                i += len(batch)
                tried = self.n_pred + self.n_fail + self.n_feat_fail + self.n_empty
                if tried >= FAIL_FAST_N and self.n_pred == 0:
                    raise RuntimeError(f'all {tried} GLACIER predictions failed (fail-fast)')
            return True
        finally:
            self.t_pred += time.time() - t0


# ------------------------------------------------------------------------------------------------ runner ---------
def parse(argv=None):
    ap = argparse.ArgumentParser(description='GLACIER runner (clean-room drop-in for gl_runner.py)')
    ap.add_argument('pkg')
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--budget', type=float, default=0.0, help='wall-clock seconds from process start (0 = none)')
    ap.add_argument('--site', default=os.environ.get('ICE_SITE') or '/tmp/ice_site')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--chunk-jobs', type=int, default=3000)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--memfrac', type=float, default=0.0)
    ap.add_argument('--nondeterministic', action='store_true')
    ap.add_argument('--any-rdkit', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--wheels', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--src', default=None, help='ms-pred src dir (default: <pkg>/src or <pkg>/mspred)')
    ap.add_argument('--shim', default=None, help='DGL / torch_scatter stand-ins (default: <pkg>/shim or <pkg>/fwdshim)')
    ap.add_argument('--dump-preds', default=None, help='pickle the canonical-level predictions here (diagnostics)')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if not a.nondeterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')     # before torch is imported
    deadline = _T_START + a.budget if a.budget > 0 else float('inf')
    log = Log('gl', quiet=a.quiet)
    meta = dict(model='glacier', runner=RUNNER_ID, status='init', n_mols=0, n_mols_covered=0, n_mols_scored=0,
                n_cands=0, n_scored=0, errors=[])
    out = {}
    meta_path = a.out + '.meta.json'

    def write(status=None):
        if status:
            meta['status'] = status
        meta['n_scored'] = sum(1 for d in out.values() for v in d.values() if v is not None)
        meta['seconds'] = round(time.time() - _T_START, 2)
        meta['log_tail'] = log.tail()
        write_json(a.out, out)
        write_json(meta_path, meta)

    try:
        with open(a.inp) as f:
            items = json.load(f)
        if not isinstance(items, list) or not all(isinstance(it, dict) and 'mid' in it for it in items):
            raise ValueError('input must be a list of {"mid", "cands", "spectra"} objects')
    except Exception as e:
        meta['errors'].append(f'bad input: {e!r}'[:500])
        log('bad input', repr(e)[:300])
        try:
            write('bad_input')
        except Exception:
            return 2
        return 0
    try:
        out = {}
        for it in items:
            out.setdefault(str(it['mid']), {}).update({str(c): None for c in it.get('cands') or []})
        meta['n_mols'] = len(items)
        meta['n_cands'] = sum(len(it.get('cands') or []) for it in items)
        write()
        meta['rdkit'] = setup_env(argparse.Namespace(pkg=a.pkg, site=a.site, wheels=a.wheels, any_rdkit=a.any_rdkit,
                                                     src=a.src, shim=a.shim), log)
        import torch
        meta['torch'] = torch.__version__
        dev = a.device
        if dev.startswith('cuda') and not torch.cuda.is_available():
            log('cuda unavailable -> cpu')
            dev = 'cpu'
        meta['device'] = dev
        if a.threads > 0:
            torch.set_num_threads(a.threads)
        if a.memfrac > 0 and dev.startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(a.memfrac)
        if not a.nondeterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        model = load_model(os.path.abspath(a.pkg), dev, log, a.ckpt)
        pred = Predictor(model, dev, a.bs, log)
        meta['t_load'] = round(time.time() - _T_START, 2)
        log(f'{len(items)} molecules, {meta["n_cands"]} candidates; numba {HAVE_NUMBA}')

        okc = {}
        chunk, chunk_jobs = [], set()

        def flush():
            nonlocal chunk, chunk_jobs
            if not chunk:
                return True
            t0 = time.time()
            if not pred.predict(chunk_jobs, deadline):
                return False
            for mid, cands, plan in chunk:
                try:
                    out[mid].update(score_molecule(plan, cands, pred.cache))
                except Exception as e:
                    meta['errors'].append(f'{mid}: score {e!r}'[:300])
            meta['n_mols_scored'] += len(chunk)
            write()
            log(f'chunk: {len(chunk)} molecules, {len(chunk_jobs)} jobs, {time.time() - t0:.1f}s '
                f'({meta["n_mols_scored"]} scored, {pred.n_pred} predictions)')
            chunk, chunk_jobs = [], set()
            return True

        stopped = False
        for it in items:
            mid = str(it['mid'])
            allc = list(dict.fromkeys(str(c) for c in it.get('cands') or []))
            for c in allc:
                if c not in okc:
                    okc[c], why = smiles_ok(c)
                    if not okc[c] and sum(1 for v in okc.values() if not v) <= 5:
                        meta['errors'].append(f'smiles rejected ({why}): {c[:100]}')
            if len(okc) >= GUARD_N and not any(okc.values()):
                raise RuntimeError(f'all {len(okc)} candidate SMILES rejected: broken RDKit / ms-pred environment')
            cands = [c for c in allc if okc[c]]
            plan = []
            for sp in it.get('spectra') or []:
                try:
                    if is_covered(sp):
                        qm, qp = prep_spectrum(sp)
                        plan.append((qm, qp, sp['adduct'], instr_token(sp), ce_bucket(spectrum_ce(sp))))
                except Exception as e:
                    meta['errors'].append(f'{mid}: plan {e!r}'[:300])
            meta['n_mols_covered'] += int(bool(plan))
            if not plan or not cands:
                continue
            chunk.append((mid, cands, plan))
            chunk_jobs |= set(molecule_jobs(plan, cands))
            if len(chunk_jobs) >= a.chunk_jobs and not flush():
                stopped = True
                break
        if not stopped and not flush():
            stopped = True
        if stopped:
            log('budget reached: unfinished molecules stay null')
        meta.update(n_pred=pred.n_pred, n_pred_fail=pred.n_fail, n_feat_fail=pred.n_feat_fail,
                    n_pred_empty=pred.n_empty, t_pred=round(pred.t_pred, 2),
                    n_bad_smiles=sum(1 for v in okc.values() if not v),
                    pred_per_s=round(pred.n_pred / max(pred.t_pred, 1e-9), 2), bs_final=pred.bs)
        if dev.startswith('cuda'):
            meta['max_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        if a.dump_preds:
            import pickle
            with open(a.dump_preds, 'wb') as f:
                pickle.dump(dict(canon=dict(_CANON), preds=pred._canon_cache), f)
        log('status', 'budget' if stopped else 'ok', f'{meta["n_mols_scored"]}/{meta["n_mols"]} molecules,',
            f'{pred.n_pred} predictions ({meta["pred_per_s"]}/s), {pred.n_fail} failed')
        write('budget' if stopped else 'ok')
        return 0
    except BaseException as e:                       # never crash the caller: status error, partial scores kept
        meta['errors'].append(traceback.format_exc()[-3000:])
        log('ERROR', repr(e)[:300])
        try:
            write('error')
        except Exception:
            return 2
        return 0


if __name__ == '__main__':
    sys.exit(main())
