"""Shared offline gate for class-1 library-trust fixes (docs/interleaving_oct27.md, "#3 Conditional on train-heavy").

    import sys; sys.path.insert(0, '<this dir>'); import gate_eval as ge
    T = ge.load_val()                                    # = train_stage2.load('fwd_v18c/val'), cwd-independent
    T['base'] = ge.stage2_scorer()(T)                    # shipped stage2_v18c
    T['new'] = ...                                       # your fix
    R = ge.gate(T, 'base', 'new', new_fn=my_fn)          # my_fn(frame) -> scores, needed for the transplant yardstick
    print(ge.report(R)); R['gate_pass']
    # a retrained stage 2:      ge.gate(T, 'base', ge.stage2_scorer('/path/stage2_x.txt'))
    # its own top-30 rows:      ge.gate(T, 'base', ge.stage2_scorer(s2), T_new=ge.load_val(tag='x'), tp_s1_new=s1)
    # a post-hoc rule f(frame): ge.gate(T, 'base', f)       (f may call ge.stage2_scorer()(F) inside)

Plain python, no repo edits; LightGBM predictions use NUM_THREADS = 2; one call takes ~10-15 s on val (16 transplant
frames are cached per frame object). Everything below is computed on the stage-2 row frame (top 30 of stage 1 per query;
fwd_v18c/val: 979 queries = group T 341, C 620 (3 with the truth outside the rows), O 18, N 0). split='train' works on
fwd_v18c/train but is meaningful only with out-of-fold stage-2 scores (stage2_v18c is in-sample there).

DEFINITIONS
-----------
Sources (candidate rows): T = in_train == 1; C = in_train == 0 and in_coco == 1 (COCONUT-not-train); O = the rest.
Query groups: T = class-1 queries (the truth is a training structure); C / O = class-2 queries whose truth is a
COCONUT-not-train / other structure (truth outside the rows: looked up in pool_v18c by in_coco; simulated class-2
truths are training structures with in_train hidden, so only in_coco is used); N = class 2, truth not in the pool.

Per query and score column:
  rr_full = 1 / rank of the truth in the list ordered by the score (ties: stage-1 row order), 0 beyond rank 25 or if
            the truth is not among the rows (LB full list / masked probe M_s: positions kept).
  rr_own  = probe analog P_s: truncate the full list to 25, keep only candidates of the truth's own source, 1 / rank of
            the truth among them (0 if the truth is outside the full top 25).  loss = rr_own - rr_full >= 0.
  kappa_s = 1 - mean(rr_full) / mean(rr_own) over group s  (share-free merge-loss ratio; LB: kappa_T = 1 - M_T / P_T =
            1 - 0.132/0.157 = 0.159, kappa_C = 1 - 0.199/0.215 = 0.074).  LB-scale losses L_T = 0.19 * mean loss over
            group T, L_C = 0.32 * mean loss over group C (LB: 0.025 / 0.016).

Yardsticks (both always computed):
  as-is      : the frame's own features (documentation of class-2 truths as simulated).
  transplant : documentation transplant (scripts/hlval.py, scratchpad/lcp/cw_tp.py): in each of N_TP = 16 draws
               (seeds 31..46) every class-2 truth row gets the documentation columns DOC_COLS (PRIOR_COLS + PCL_COLS +
               PC_COLS + in_bio: n_org n_coll n_doi n_syn has_cas has_name ann npl in_pcl pcl_* pc_sid pc_pmid
               pc_patent pc_ncid in_bio) of a random non-training decoy row of the same source (COCONUT-not-train
               donor for a COCONUT truth, other donor for an other truth), donors drawn from all queries of the frame;
               s1 is recomputed with ret_ranker_v18c (val: exact, max |diff| 0 against the stored s1; train split: the
               stored out-of-fold s1 plus the model's change), then s1_z / s1_r within the query's rows; the score
               columns are recomputed on each transplanted frame with the scorer callables (see gate()). Per-query RR
               is the mean over the 16 draws. Class-1 rows are untouched, so class-1 results are identical in both
               yardsticks. The truth is not re-selected into / out of the stage-1 top 30 (approximation).

Hardness matching (applied identically to base and new, constants calibrated once on v18c / fwd_v18c val by
calibrate.py and frozen below; gate() reports the base's achieved levels so drift is visible):
  class 1 (group T): query reweighting toward weak library matches of the truth (the hidden class-1 queries are
      timsTOF spectra whose library spectra of the truth are mostly cross-instrument, i.e. weaker than in our
      simulation). Truth-present class-1 queries get weight exp(-LAM_T * lib_truth) (lib = the truth row's library
      entropy similarity), normalised to their count; truth-absent class-1 queries (27 of 341; reach failures, RR 0
      under any re-ranking) keep weight 1. LAM_T is set so that the base's kappa_T equals the LB's 0.159 (share-free)
      -> it also lands own_T near the LB's 0.83 and L_T on the LB's 0.025 (see CALIB). No score change on class 1.
      (Score noise / truth shift matched to own_T 0.84 give kappa_T 0.10-0.12 and L_T 0.016-0.018: rejected.)
  class 2 (groups C, O, N): truth shift -- the truth's per-query z-score (z of each score column over the query's
      rows, sample SD) is lowered by D_C[yardstick] SDs, so that the base's own_C (within-source COCONUT MRR) equals
      the LB-implied 0.65. D_C is 0.344 under the transplant (which alone already takes own_C from 0.86 to 0.72) and
      0.99 as-is. Ranks are then taken on the z-scores (a monotone transform per query: no effect without the shift).

Mixed MRR at class-1 share w1 (class-1 share among train + COCONUT answers; 0.37 [0.31, 0.42]):
      mixed(w1) = w1 * wmean_T(rr_full) + (1 - w1) * mean_C(rr_full)   (O / N not in the mix; reported separately)
  LB projection: 0.19 * d_T + 0.32 * d_C + 0.08 * d_O (deltas of the hardness-matched per-query RR; hidden shares).
  Probe analogs: dP_T = 0.19 * d(own_T), dP_C = 0.32 * d(own_C); masked analogs dM_T = 0.19 * d_T, dM_C = 0.32 * d_C.
Masked-train cap (offline analog of mask-train >= parent - 0.006): d_T = change of the full-list RR carried by
  class-1 answers per class-1 query (tilt-weighted, = d(M_T) / 0.19) must be >= -0.006 / 0.19 = -0.0316.
Paired bootstrap: n_boot resamples of queries within group (T with its weights, C), the same index draws for both
  yardsticks; 95% percentile CI of each delta and of mixed(w1).
gate_pass = for BOTH yardsticks: mixed(0.37) point delta >= +0.005 AND the cap holds (d_T >= -0.0316).
  (CI lower bounds and w1 0.31 / 0.42 are reported, not gated.)

Scores under the transplant: a score COLUMN computed on T cannot be re-evaluated on transplanted frames, so gate()
needs, per side, one of: a callable fn(frame) -> scores (base_fn / new_fn, or the column argument itself is a
callable); precomputed columns '<col>@tp0' .. '@tp15' (made with transplant_frames()); for the base only, an automatic
check that the column equals the shipped stage2_v18c scores (then that model is used). If the new side has none of
these, the transplant scores are 'delta-carried': new_tp = base_tp + (new - base) row by row (exact only for post-hoc
fixes that do not read class-2 truth documentation or the score of the truth itself); flagged in R['tp_mode'].
(Checked: a post-hoc boost of training candidates with lib >= 0.8, callable vs delta-carry, transplant mix@0.37
-0.0021 vs -0.0022.)

CALIBRATION (calibrate.py -> calib.json / calib_table.csv; v18c, fwd_v18c val; LB values at shares 0.19 / 0.32)
                                     own_T  kappa_T  L_T     own_C  kappa_C  L_C
  LB (probes Oct 27 / Oct 29)        0.826  0.159   0.025   0.672  0.074   0.016
  raw as-is                          0.876  0.053   0.0088  0.864  0.031   0.0087
  raw transplant                     0.876  0.053   0.0088  0.720  0.062   0.0142
  matched as-is      (tilt + shift)  0.804  0.159   0.0243  0.651  0.095   0.0197
  matched transplant (tilt + shift)  0.804  0.159   0.0243  0.650  0.079   0.0164   <- closest to the LB
  rejected class-1 alternatives at own_T 0.84: noise on all rows kappa_T 0.122 / L_T 0.0194; truth shift 0.109 /
  0.0174; tilt fitted to own_T (LAM 2.36) 0.101 / 0.0161; hard stratum truth lib < 0.7 (n 73): 0.849 / 0.165 / 0.0266.
  L_T matches by construction (LAM_T is fitted to kappa_T); own_T (0.804 vs 0.826, the latter share-dependent) and the
  COCONUT kappa (0.079 vs 0.074 under the transplant; 0.095 as-is) are not fitted. Class-1 ESS under the tilt: 88 / 341.

VALIDATION (validate.py, validate2.py, validate3.py -> validate*.json, summary.csv; mix = mixed delta at w1 0.37,
LBp = LB projection; m = matched, r = raw; as-is / transplant)
  base vs base (columns; and a per-query affine transform via callables): every delta exactly 0.
  pair (LB dF)                      m mix as/tp       m LBp as/tp       r LBp as/tp       gate
  v18cw native rows    (-0.022)     -0.0020/+0.0003   +0.0020/+0.0028   +0.0056/+0.0015   fail
  v18cw s2-swap / s1+s2             +0.0000/+0.0020   +0.0008/+0.0041   +0.0058/+0.0027   fail
                                    -0.0051/-0.0006   +0.0006/+0.0025   +0.0055/+0.0023   fail
  v18c3 native rows    (+0.002)     +0.0010/+0.0056   +0.0023/+0.0028   +0.0069/+0.0026   fail (as-is < 0.005)
  v18c3 s2-swap / s1+s2             +0.0027/+0.0063   +0.0015/+0.0024   +0.0016/+0.0028   fail
                                    -0.0034/+0.0033   -0.0002/+0.0028   +0.0078/+0.0034   fail
  v17bpop on v17b rows (-0.016)     +0.0261/-0.0037   +0.0084/-0.0048   +0.0012/-0.0119   fail (transplant)
  v17bb native rows    (-0.017)     +0.0023/+0.0015   -0.0039/-0.0037   +0.0011/-0.0036   fail
  stage 1 -> stage2_v18c (+0.010 in the Oct 19 pool-v2 pipeline; positive control)
                                    +0.0283/+0.0359   +0.0227/+0.0296   +0.0099/+0.0172   PASS
  v18cr1 seed replicate (null; LB pending Oct 30)
           native rows              +0.0045/+0.0100   +0.0027/+0.0040   +0.0039/+0.0020   fail (as-is 0.0045)
           s2-swap                  -0.0021/+0.0052   -0.0007/+0.0028   +0.0019/+0.0007   fail
  probe analogs v18c3: dP_T matched +0.0027..+0.0044 vs LB +0.004 (raw +0.0002..+0.0016); dP_C ~0 vs LB -0.014 (miss).
  v18cw: dP_T -0.0012..+0.0051 vs LB -0.009 (miss); dP_C matched as-is -0.0022..-0.0044 vs LB -0.007 (raw: +0.002).
  Reading: the gate rejected every LB-flat/negative pair and passed the positive control, but it does NOT predict
  the size of LB losses (-0.016..-0.022 projected at -0.005..+0.003). The documentation-transplant yardstick is what
  rejects v17bpop (as-is says +0.026). SEED NOISE: a pure retrain with another seed moves the matched mix by up to
  +0.010 (transplant) and d_T by +0.018..+0.027, i.e. a single-seed retrain can clear the +0.005 bar by chance; the
  paired bootstrap covers query sampling only. For retrains, compare against SEED_NOISE_REF / use >= 2 seeds. Pairs on
  fwd_v18c rows with other pipelines' stage 2 (s2-swap, s1+s2) are approximations: those models were trained on other
  top-30 sets; 'native' uses their own rows (89% of (query, candidate) pairs shared with v18c). v17b-era rows need
  GLACIER scores from earlier runs (GL_EXTRA; without them only ~3% of rows have GLACIER and results are invalid).
"""
import os, sys
import numpy as np, pandas as pd

W = '/home/user/work'
REPO = '/home/user/claude_code_tests'
S1_MODEL = f'{W}/own_models/ret_ranker_v18c.txt'
S2_MODEL = f'{W}/own_models/stage2_v18c.txt'
POOL = f'{W}/pool_v18c/pool.parquet'
K = 25
NUM_THREADS = 2
PRIOR_COLS = ['n_org', 'n_coll', 'n_doi', 'n_syn', 'has_cas', 'has_name', 'ann', 'npl']        # casmi/retrieval.py
PCL_COLS = ['in_pcl', 'pcl_pubmed', 'pcl_patent', 'pcl_anno', 'pcl_bio', 'pcl_food']
PC_COLS = ['pc_sid', 'pc_pmid', 'pc_patent', 'pc_ncid']
DOC_COLS = PRIOR_COLS + PCL_COLS + PC_COLS + ['in_bio']                                      # scripts/hlval.py set
N_TP, TP_SEED = 16, 31
SHARES = dict(T=0.19, C=0.32, O=0.08)               # hidden public-124 shares (docs/interleaving_oct27.md section 2)
LB = dict(P_T=0.157, P_C=0.215, M_T=0.132, M_C=0.199, F=0.339)
LB_KAPPA_T = 1 - LB['M_T'] / LB['P_T']               # 0.159
LB_KAPPA_C = 1 - LB['M_C'] / LB['P_C']               # 0.074
TGT_OWN_T, TGT_OWN_C = 0.84, 0.65
W1, W1S = 0.37, (0.31, 0.37, 0.42)
MIN_GAIN = 0.005
CAP_T = -0.006 / SHARES['T']                          # -0.0316 per class-1 query
# frozen hardness constants (calibrate.py on stage2_v18c, fwd_v18c val; see CALIB)
LAM_T = 3.91
D_C = {'as_is': 0.99, 'transplant': 0.344}
# matched mix@0.37 of the v18cr1 seed replicate vs v18c (validate3.py): the retrain noise floor of this gate
SEED_NOISE_REF = {'v18cr1_native': dict(as_is=0.0045, transplant=0.0100, d_T=0.0266),
                  'v18cr1_s2swap': dict(as_is=-0.0021, transplant=0.0052, d_T=0.0177)}
CALIB = {}   # filled from calib.json next to this file when present (achieved base levels at the constants)
GRP = {0: 'T', 1: 'C', 2: 'O', 3: 'N'}
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, 'calib.json')):
    import json
    CALIB = json.load(open(os.path.join(_HERE, 'calib.json')))


# ----------------------------------------------------------------------------------------------- loading
def _ts2():
    import importlib.util
    argv = sys.argv
    sys.argv = ['x']
    try:
        spec = importlib.util.spec_from_file_location('ts2_gate', f'{REPO}/scripts/train_stage2.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    m.PARAMS['num_threads'] = NUM_THREADS
    return m


# builds that reused GLACIER scores of earlier runs (GL_HAVE): their own fwd dir holds only the new pairs (~3% of rows)
GL_EXTRA = {'v18cr1': 'fwd_v18c*', 'v17b': 'fwd*', 'v17bb': 'fwd*'}


def load_val(split='val', tag='v18c', gl_dirs=None):
    """fwd_<tag>/<split> stage-2 rows (train_stage2.load with cwd = /home/user/work), sorted by (qid, s1_r).
    gl_dirs: extra dir glob(s) whose <split>.jobs_*.gl.json fill missing GLACIER pairs (default GL_EXTRA[tag])."""
    cwd = os.getcwd()
    old = os.environ.get('GL_GLOB')
    extra = GL_EXTRA.get(tag) if gl_dirs is None else gl_dirs
    os.environ['GL_GLOB'] = ','.join(f'{W}/{d}/{split}.jobs_*.gl.json' for d in extra.split(',')) if extra else ''
    os.chdir(W)
    try:
        return _ts2().load(f'fwd_{tag}/{split}')
    finally:
        os.chdir(cwd)
        if old is None:
            os.environ.pop('GL_GLOB', None)
        else:
            os.environ['GL_GLOB'] = old


# ----------------------------------------------------------------------------------------------- index / ranking
_POOL_SRC = None
def _pool_coco():
    global _POOL_SRC
    if _POOL_SRC is None:
        P = pd.read_parquet(POOL, columns=['key', 'in_coco'])
        _POOL_SRC = dict(zip(P.key.values, np.where(P.in_coco.values == 1, 1, 2)))
    return _POOL_SRC


class Idx:
    """Row/query index of a stage-2 frame sorted by (qid, s1_r)."""

    def __init__(self, T):
        self.qid = T.qid.values
        self.uq, self.qstart = np.unique(self.qid, return_index=True)
        self.nq = len(self.uq)
        self.qi = np.repeat(np.arange(self.nq), np.diff(np.r_[self.qstart, len(T)]))
        self.lab = T.label.values == 1
        self.src = np.where(T.in_train.values == 1, 0, np.where(T.in_coco.values == 1, 1, 2))
        q = T.iloc[self.qstart]
        self.cls = q.cls.values.astype(int)
        self.trow = np.full(self.nq, -1)
        self.trow[self.qi[self.lab]] = np.flatnonzero(self.lab)
        self.has = self.trow >= 0
        g = np.where(self.has, self.src[np.maximum(self.trow, 0)], -1)
        miss = ~self.has & (self.cls == 2)
        if miss.any():
            P = _pool_coco()
            g[miss] = [P.get(k, 3) for k in q.truth.values[miss]]
        g[self.cls == 1] = 0
        self.grp = g
        self.tsrc_row = g[self.qi]
        self.cls_row = self.cls[self.qi]
        self.rowpos = np.arange(len(T)) - self.qstart[self.qi]
        self.tpos_row = np.where(self.has, self.trow - self.qstart, 10 ** 6)[self.qi]
        self.same_row = self.src == self.tsrc_row
        self.tlib = np.where(self.has, T.lib.fillna(0).values[np.maximum(self.trow, 0)], np.nan)

    def rr(self, Z):
        """Z: (n_rows,) or (n_rows, k). Returns (rr_full, rr_own), each (n_queries, k)."""
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        zt = Z[np.maximum(self.trow, 0)[self.qi]]
        above = (Z > zt) | ((Z == zt) & (self.rowpos < self.tpos_row)[:, None])
        above &= ~self.lab[:, None]
        n_all = np.add.reduceat(above.astype(np.int32), self.qstart, axis=0)
        n_own = np.add.reduceat((above & self.same_row[:, None]).astype(np.int32), self.qstart, axis=0)
        ok = self.has[:, None] & (n_all + 1 <= K)
        return np.where(ok, 1.0 / (n_all + 1), 0.0), np.where(ok, 1.0 / (n_own + 1), 0.0)

    def zscore(self, s):
        """Per-query z-score over the query's rows (sample SD; SD 0 or a single row -> SD 1)."""
        s = np.asarray(s, dtype=np.float64)
        n = np.diff(np.r_[self.qstart, len(s)])
        m = np.add.reduceat(s, self.qstart) / n
        d = s - m[self.qi]
        sd = np.sqrt(np.add.reduceat(d * d, self.qstart) / np.maximum(n - 1, 1))
        sd = np.where((sd > 0) & (n > 1), sd, 1.0)
        return d / sd[self.qi]

    def weights(self, lam=None):
        """Per-query hardness weights: class-1 truth-library tilt (truth-absent class 1 and class 2: weight 1)."""
        lam = LAM_T if lam is None else lam
        w = np.ones(self.nq)
        m = (self.grp == 0) & self.has
        e = np.exp(-lam * self.tlib[m])
        w[m] = e / e.sum() * m.sum()
        return w

    def degraded_rr(self, S, d):
        """Hardness-matched per-query (rr_full, rr_own), mean over the columns of S: z-score each column per query,
        lower class-2 truths by d SDs."""
        S = np.asarray(S, dtype=np.float64)
        if S.ndim == 1:
            S = S[:, None]
        Z = np.stack([self.zscore(S[:, j]) for j in range(S.shape[1])], 1)
        Z = Z - d * (self.lab & (self.cls_row == 2))[:, None]
        a, b = self.rr(Z)
        return a.mean(1), b.mean(1)


def row_normals(T, k, salt=0):
    """(n_rows, k) standard normals, a deterministic function of (qid, candidate key, draw, salt): the same candidate
    gets the same noise in every score column and in every frame (common random numbers; for sensitivity checks)."""
    from scipy.special import ndtri
    h = pd.util.hash_pandas_object(pd.DataFrame({'q': T.qid.values.astype(np.int64), 'k': T.key.astype(str).values}),
                                   index=False).values.astype(np.uint64)
    out = np.empty((len(h), k))
    with np.errstate(over='ignore'):
        for j in range(k):
            x = h ^ np.uint64(((salt * 1000003 + j + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
            out[:, j] = ndtri(((x >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53)
    return out


# ----------------------------------------------------------------------------------------------- scorers / transplant
_BOOST = {}
def _booster(path):
    import lightgbm as lgb
    if path not in _BOOST:
        _BOOST[path] = (lgb.Booster(model_file=path), open(path.replace('.txt', '_feats.txt')).read().split())
    return _BOOST[path]


def _gz_rows(qi, qstart, s):
    """casmi/fwdfeat._gz on whole queries: sample-SD z, NaN for single-row queries, 0 when the SD is 0."""
    n = np.diff(np.r_[qstart, len(s)])
    m = np.add.reduceat(s, qstart) / n
    d = s - m[qi]
    sd = np.sqrt(np.add.reduceat(d * d, qstart) / np.maximum(n - 1, 1))
    z = d / np.where(sd > 0, sd, 1.0)[qi]
    z = np.where((sd == 0)[qi], 0.0, z)
    return np.where((n >= 2)[qi], z, np.nan)


def restage1(F, s1_model=S1_MODEL, carry_from=None):
    """Recompute s1 with a stage-1 model on F's features, then s1_z / s1_r within each query's rows. carry_from: the
    untouched frame; s1 = its stored s1 + (model(F) - model(carry_from)) (keeps out-of-fold scores)."""
    b, f = _booster(s1_model)
    p = b.predict(F[f].astype(np.float32).values, num_threads=NUM_THREADS)
    if carry_from is not None:
        p0 = b.predict(carry_from[f].astype(np.float32).values, num_threads=NUM_THREADS)
        p = carry_from.s1.values + (p - p0)
    F = F.copy()
    uq, qstart = np.unique(F.qid.values, return_index=True)
    qi = np.repeat(np.arange(len(uq)), np.diff(np.r_[qstart, len(F)]))
    F['s1'] = p
    F['s1_z'] = _gz_rows(qi, qstart, p)
    F['s1_r'] = F.groupby('qid', sort=False).s1.rank(ascending=False, method='first').values
    return F


def stage2_scorer(s2_model=S2_MODEL, s1_model=None):
    """Callable frame -> stage-2 scores of a saved model. s1_model: first recompute s1 / s1_z / s1_r with that
    stage-1 model on the frame's rows (another pipeline's stage 1 applied to these top-30 rows)."""
    def fn(F):
        if s1_model is not None:
            F = restage1(F, s1_model)
        b, f = _booster(s2_model)
        return b.predict(F[f].astype(np.float32).values, num_threads=NUM_THREADS)
    fn.__name__ = os.path.basename(s2_model) + ('' if s1_model is None else '+' + os.path.basename(s1_model))
    return fn


_TP_CACHE = {}
def transplant_frames(T, n=N_TP, seed=TP_SEED, split='val', s1_model=S1_MODEL):
    """Documentation-transplant copies of T (see module docstring). Cached per (frame identity, columns, n, seed)."""
    key = (id(T), len(T), tuple(T.columns), n, seed, split, s1_model)
    if key in _TP_CACHE:
        return _TP_CACHE[key]
    dc = T[(T.label == 0) & (T.in_train == 0) & (T.in_coco == 1)][DOC_COLS].values
    do = T[(T.label == 0) & (T.in_train == 0) & (T.in_coco == 0)][DOC_COLS].values
    tr = np.flatnonzero(((T.label == 1) & (T.cls == 2)).values)
    co = T.in_coco.values[tr] == 1
    touched = np.isin(T.qid.values, T.qid.values[tr])
    out = []
    for k in range(n):
        rng = np.random.default_rng(seed + k)
        vals = np.where(co[:, None], dc[rng.integers(0, len(dc), len(tr))], do[rng.integers(0, len(do), len(tr))])
        F = T.copy()
        for i, c in enumerate(DOC_COLS):
            col = F[c].values.copy()
            col[tr] = vals[:, i]
            F[c] = col
        F2 = restage1(F, s1_model, carry_from=None if split == 'val' else T)
        for c in ('s1', 's1_z', 's1_r'):
            F[c] = np.where(touched, F2[c].values, T[c].values)
        out.append(F)
    if len(_TP_CACHE) > 4:
        _TP_CACHE.clear()
    _TP_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------------------------- gate
def _scores(T, spec, fn):
    if callable(spec):
        return np.asarray(spec(T), dtype=np.float64), spec
    return T[spec].values.astype(np.float64), fn


def _tp_scores(T, TPF, spec, fn, n, other=None, is_base=False):
    """Scores of one side on the transplanted frames -> ((n_rows, n) array, mode)."""
    if fn is not None:
        return np.stack([np.asarray(fn(F), dtype=np.float64) for F in TPF], 1), 'scorer'
    if isinstance(spec, str) and all(f'{spec}@tp{k}' in T.columns for k in range(n)):
        return np.stack([T[f'{spec}@tp{k}'].values for k in range(n)], 1).astype(np.float64), 'columns'
    if is_base:
        f2 = stage2_scorer()
        if np.allclose(f2(T), T[spec].values, rtol=0, atol=1e-6):
            return np.stack([f2(F) for F in TPF], 1), 'auto-stage2_v18c'
        raise ValueError(f'base column {spec!r} is not stage2_v18c: pass base_fn (callable) or {spec}@tp0..{n - 1}')
    return None, 'delta-carry'


def _boot_idx(ix, n_boot, seed):
    rng = np.random.default_rng(seed)
    return {g: (np.flatnonzero(ix.grp == g), rng.integers(0, (ix.grp == g).sum(), (n_boot, (ix.grp == g).sum())))
            for g in (0, 1, 2)}


def _evaluate(ix, w, rb, rn, BI, w1s):
    """rb / rn: dicts full / own per-query arrays (base / new). Returns stats dict."""
    out = {}
    m = {g: ix.grp == g for g in range(4)}
    def wm(x, g):
        return float(np.average(x[m[g]], weights=w[m[g]])) if m[g].any() else float('nan')
    lev = {}
    for side, r in (('base', rb), ('new', rn)):
        L = {}
        for g in (0, 1, 2):
            nm = GRP[g]
            L[f'own_{nm}'] = wm(r['own'], g); L[f'full_{nm}'] = wm(r['full'], g)
            L[f'kappa_{nm}'] = 1 - L[f'full_{nm}'] / L[f'own_{nm}'] if L[f'own_{nm}'] > 0 else float('nan')
        L['L_T_lb'] = SHARES['T'] * (L['own_T'] - L['full_T'])
        L['L_C_lb'] = SHARES['C'] * (L['own_C'] - L['full_C'])
        L['mix'] = {f'{a:.2f}': a * L['full_T'] + (1 - a) * L['full_C'] for a in w1s}
        lev[side] = L
    out['levels'] = lev
    d = {k: rn[k] - rb[k] for k in ('full', 'own')}
    dT, dC, dO = wm(d['full'], 0), wm(d['full'], 1), wm(d['full'], 2)
    c2 = ix.cls == 2
    out['d_T'] = dT                                   # = masked-train analog per class-1 query
    out['d_C'] = dC
    out['d_O'] = dO
    out['d_cls2_all'] = float(d['full'][c2].mean())
    out['d_own_T'] = wm(d['own'], 0); out['d_own_C'] = wm(d['own'], 1)
    out['lb_proj'] = SHARES['T'] * dT + SHARES['C'] * dC + SHARES['O'] * (0 if np.isnan(dO) else dO)
    out['dP_T_lb'] = SHARES['T'] * out['d_own_T']; out['dP_C_lb'] = SHARES['C'] * out['d_own_C']
    out['dM_T_lb'] = SHARES['T'] * dT; out['dM_C_lb'] = SHARES['C'] * dC
    out['cap_dT_min'] = CAP_T
    out['cap_ok'] = bool(dT >= CAP_T)
    # paired bootstrap (within group; class-1 weights travel with the query)
    bs = {}
    for g in (0, 1, 2):
        idx, J = BI[g]
        if not len(idx):
            continue
        x, ww = d['full'][idx][J], w[idx][J]
        bs[g] = (x * ww).sum(1) / ww.sum(1)
        xo = d['own'][idx][J]
        bs[(g, 'own')] = (xo * ww).sum(1) / ww.sum(1)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out['d_T_ci'] = ci(bs[0]); out['d_C_ci'] = ci(bs[1])
    if 2 in bs:
        out['d_O_ci'] = ci(bs[2])
    out['mix'] = {}
    for a in w1s:
        v = a * bs[0] + (1 - a) * bs[1]
        out['mix'][f'{a:.2f}'] = dict(delta=a * dT + (1 - a) * dC, ci=ci(v), se=float(v.std()), p_gt0=float((v > 0).mean()))
    lbv = SHARES['T'] * bs[0] + SHARES['C'] * bs[1] + (SHARES['O'] * bs[2] if 2 in bs else 0)
    out['lb_proj_ci'] = ci(lbv)
    out['dP_T_lb_ci'] = ci(SHARES['T'] * bs[(0, 'own')]); out['dP_C_lb_ci'] = ci(SHARES['C'] * bs[(1, 'own')])
    return out


def gate(T, base_col, new_col, split='val', n_boot=1000, base_fn=None, new_fn=None, T_new=None, n_tp=N_TP,
         tp_seed=TP_SEED, w1s=W1S, lam_T=None, d_C=None, min_gain=MIN_GAIN, boot_seed=0, return_pq=False,
         tp_s1=S1_MODEL, tp_s1_new=None):
    """Offline gate of `new_col` against `base_col` on the stage-2 row frame T (module docstring has every definition).

    base_col / new_col: column names of T, or callables frame -> scores.  base_fn / new_fn: callables used on the
    transplanted frames (default: the callable itself; for columns see the module docstring).  T_new: optional second
    frame (same queries, other rows, e.g. another pipeline's own top 30) on which new_col / new_fn is evaluated;
    tp_s1 / tp_s1_new: stage-1 model that re-scores the transplanted frames of T / T_new (the pipeline that made
    those rows; default ret_ranker_v18c; tp_s1_new defaults to tp_s1).
    lam_T / d_C: override the frozen hardness constants (d_C: dict as D_C).  Returns a dict (see report())."""
    if not T[['qid', 's1_r']].equals(T.sort_values(['qid', 's1_r'])[['qid', 's1_r']]):
        T = T.sort_values(['qid', 's1_r']).reset_index(drop=True)
    TN = T if T_new is None else T_new
    if T_new is not None and not TN[['qid', 's1_r']].equals(TN.sort_values(['qid', 's1_r'])[['qid', 's1_r']]):
        TN = TN.sort_values(['qid', 's1_r']).reset_index(drop=True)
    lam_T = LAM_T if lam_T is None else lam_T
    dC = dict(D_C if d_C is None else d_C)
    ix = Idx(T)
    ixn = ix if T_new is None else Idx(TN)
    if not np.array_equal(ix.uq, ixn.uq):
        raise ValueError('T and T_new must hold the same queries')
    w = ix.weights(lam_T)
    BI = _boot_idx(ix, n_boot, boot_seed)
    sb, base_fn = _scores(T, base_col, base_fn)
    sn, new_fn = _scores(TN, new_col, new_fn)
    TPF = transplant_frames(T, n_tp, tp_seed, split, s1_model=tp_s1)
    TPFn = TPF if T_new is None else transplant_frames(TN, n_tp, tp_seed, split, s1_model=tp_s1_new or tp_s1)
    sb_tp, mode_b = _tp_scores(T, TPF, base_col, base_fn, n_tp, is_base=True)
    sn_tp, mode_n = _tp_scores(TN, TPFn, new_col, new_fn, n_tp)
    if sn_tp is None:
        if T_new is not None:
            raise ValueError('delta-carry needs a single frame: pass new_fn for T_new')
        sn_tp = sb_tp + (sn - sb)[:, None]
    R = dict(split=split, n_queries=int(ix.nq), n_by_group={GRP[g]: int((ix.grp == g).sum()) for g in range(4)},
             ess_T=float(w[ix.grp == 0].sum() ** 2 / (w[ix.grp == 0] ** 2).sum()), lam_T=lam_T, d_C=dC,
             tp_mode=dict(base=mode_b, new=mode_n), n_tp=n_tp, n_boot=n_boot, w1s=list(w1s), min_gain=min_gain,
             cap_dT_min=CAP_T, seed_noise_ref=SEED_NOISE_REF, yardsticks={}, raw={})
    pq = {}
    for y, (Sb, Sn) in (('as_is', (sb, sn)), ('transplant', (sb_tp, sn_tp))):
        fb, ob = ix.degraded_rr(Sb, dC[y]); fn_, on = ixn.degraded_rr(Sn, dC[y])
        R['yardsticks'][y] = _evaluate(ix, w, dict(full=fb, own=ob), dict(full=fn_, own=on), BI, w1s)
        # raw view: no hardness matching (unweighted, no shift)
        fb0, ob0 = ix.degraded_rr(Sb, 0.0); fn0, on0 = ixn.degraded_rr(Sn, 0.0)
        R['raw'][y] = _evaluate(ix, np.ones(ix.nq), dict(full=fb0, own=ob0), dict(full=fn0, own=on0), BI, w1s)
        if return_pq:
            pq[y] = pd.DataFrame(dict(qid=ix.uq, cls=ix.cls, grp=[GRP[g] for g in ix.grp], w=w, tlib=ix.tlib,
                                      full_b=fb, full_n=fn_, own_b=ob, own_n=on, full_b_raw=fb0, full_n_raw=fn0))
    ys = R['yardsticks']
    R['gate_checks'] = {y: dict(mix_037=ys[y]['mix'][f'{W1:.2f}']['delta'] if f'{W1:.2f}' in ys[y]['mix'] else
                                ys[y]['mix'][list(ys[y]['mix'])[0]]['delta'],
                                gain_ok=False, cap_ok=ys[y]['cap_ok']) for y in ys}
    for y in ys:
        R['gate_checks'][y]['gain_ok'] = bool(R['gate_checks'][y]['mix_037'] >= min_gain)
    R['gate_pass'] = bool(all(c['gain_ok'] and c['cap_ok'] for c in R['gate_checks'].values()))
    if mode_n == 'delta-carry':
        R['warning'] = 'transplant scores of the new side are delta-carried (new_tp = base_tp + new - base)'
    if return_pq:
        R['per_query'] = pq
    return R


def report(R, title=''):
    """Compact text summary of a gate() result."""
    L = [f'== gate {title} | split {R["split"]} | n {R["n_by_group"]} | ESS(T) {R["ess_T"]:.0f} | lam_T {R["lam_T"]} '
         f'd_C {R["d_C"]} | tp {R["tp_mode"]}']
    for view in ('yardsticks', 'raw'):
        for y, s in R[view].items():
            b = s['levels']['base']
            mx = ' '.join(f'w{a}: {v["delta"]:+.4f} [{v["ci"][0]:+.4f},{v["ci"][1]:+.4f}]' for a, v in s['mix'].items())
            L.append(f'{"matched" if view == "yardsticks" else "raw":7s} {y:10s} base own_T {b["own_T"]:.3f} kT {b["kappa_T"]:.3f} '
                     f'own_C {b["own_C"]:.3f} kC {b["kappa_C"]:.3f} | d_T {s["d_T"]:+.4f} [{s["d_T_ci"][0]:+.4f},{s["d_T_ci"][1]:+.4f}] '
                     f'd_C {s["d_C"]:+.4f} [{s["d_C_ci"][0]:+.4f},{s["d_C_ci"][1]:+.4f}] d_O {s["d_O"]:+.4f} | mix {mx} | '
                     f'LB {s["lb_proj"]:+.4f} [{s["lb_proj_ci"][0]:+.4f},{s["lb_proj_ci"][1]:+.4f}] dP_T {s["dP_T_lb"]:+.4f} '
                     f'dP_C {s["dP_C_lb"]:+.4f} dM_T {s["dM_T_lb"]:+.4f} | cap {"ok" if s["cap_ok"] else "FAIL"}')
    L.append(f'gate_pass {R["gate_pass"]}  checks {R["gate_checks"]}' + (f'  WARNING {R["warning"]}' if 'warning' in R else ''))
    return '\n'.join(L)


def strip(R):
    """JSON-serialisable copy (drops per-query frames)."""
    return {k: v for k, v in R.items() if k != 'per_query'}
