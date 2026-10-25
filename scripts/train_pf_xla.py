"""PeakFormer training with static shapes for XLA devices (Kaggle TPU v3-8 via torch_xla), also runs on CUDA / CPU.
Same model, losses, sampling and augmentation as scripts/train_pf.py; the checkpoint loads in casmi.peakformer.PFScorer.

python scripts/train_pf_xla.py <fp_data_dir> <fpx.npz> <out.pt> [epochs] [steps_per_epoch] [batch per core] [n_neg]
Differences from train_pf.py (TPUs have no float64 and XLA recompiles for every new shape):
- batches are built on the host in numpy (merge augmentation, peak dropout, jitter) by a prefetch thread, including the
  float64 phase reduction of the m/z sinusoids (PeakFormer._sin); the device gets fixed-shape float32 phases;
- molecule bits live on the device as one packed bit matrix (structures, then negative rows); a batch's rows are
  gathered and unpacked into a dense multi-hot matrix: the EmbeddingBag sum becomes multi-hot @ bits.weight and f.z a
  batched matmul (same math, same parameters);
- data parallel over all cores (gradients all-reduced, in-batch negatives per core); the cosine LR schedule is step
  based (identical on every core): after PF_CAL steps the step budget that fits PF_HOURS is measured on core 0 and
  shared.
Env: as train_pf.py (PF_D, PF_LAYERS, PF_HEADS, PF_DMOL, PF_DROP, PF_LR, PF_MERGE, PF_BCE, PF_SBAL, PF_SEED, PF_HOURS,
PF_TIMS_W, PF_NEG_TRAIN, PF_NEG_NOREP, PF_INST_DROP, PF_VAL_N, PF_LOG, PF_INIT, FP_EXCLUDE) + PF_CORES (number of XLA
processes; default all), PF_CAL (calibration steps, default 300), PF_DEV (force 'cpu' / 'cuda').
PF-syn (PF_SYN = syn.npz of scripts/prep_syn_data.py, PF_SYN_P = share of each batch, default 0.25): GLACIER-simulated
[M+H]+ spectra of pool structures (training AND COCONUT-not-train, so that a synthetic look carries no source
information) as extra rows. Population balance: synthetic rows take their window negatives from the simulated set
only (every simulated population is positive and negative in the same proportions); real rows keep their negatives
(PF_NEG_TRAIN: training positives only); in-batch negatives never cross looks (a real query never sees a COCONUT
structure as a negative only). A few low noise peaks are added to synthetic spectra; no source token.
"""
import math, os, queue, sys, threading, time, traceback
os.environ.setdefault('XLA_NO_SPECIAL_SCALARS', '1')    # Adam's bias correction reaches 1.0: no constant -> no recompile
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, '/kaggle/working/pkg')
from casmi.peakformer import K, PeakFormer, csr_rows, embed_query, fz_scores, meta_matrix, ADD, N_ADD, N_INST

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA = os.environ.get('PF_DEV', 'xla') == 'xla'
except ImportError:
    XLA = False

data, fpx_p, out = sys.argv[1:4]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
B = int(sys.argv[6]) if len(sys.argv) > 6 else 256
NNEG = int(sys.argv[7]) if len(sys.argv) > 7 else 31
HOURS = float(os.environ.get('PF_HOURS', '1e9'))
SEED = int(os.environ.get('PF_SEED', '0'))
P_MERGE = float(os.environ.get('PF_MERGE', '0.35'))
INST_DROP = float(os.environ.get('PF_INST_DROP', '0'))
NOREP = os.environ.get('PF_NEG_NOREP') == '1'
W_BCE = float(os.environ.get('PF_BCE', '5'))
S_BAL = float(os.environ.get('PF_SBAL', '0'))
LR = float(os.environ.get('PF_LR', '5e-4'))
CAL = int(os.environ.get('PF_CAL', '300'))
LOG = int(os.environ.get('PF_LOG', '100'))
SYN = os.environ.get('PF_SYN')
B_SYN = int(round(float(os.environ.get('PF_SYN_P', '0.25')) * B)) if SYN else 0
N_FREQ = 64
FREQ = 1.0 / np.logspace(-2, 3, N_FREQ)          # = PeakFormer.freq
t0 = time.time()
G = {}
INIT_LOCK = threading.Lock()        # PJRT on TPU v2/v3 runs 2 cores as threads of one process sharing the CPU RNG
LAST = [time.time()]                # last training progress (watchdog)


def load():
    """Host arrays (module globals in G); loaded once, before the XLA processes fork (shared copy-on-write)."""
    if G:
        return
    S = np.load(f'{data}/spectra.npz'); T = np.load(f'{data}/structs.npz')
    off = S['off']; N = len(off) - 1; lens = np.diff(off)
    assert lens.max() <= K
    g = {k: S[k] for k in ('prec', 'ce', 'add_id', 'inst_id', 'pol', 'sidx', 'split')}
    row = np.repeat(np.arange(N), lens); pos = np.arange(off[-1]) - np.repeat(off[:-1], lens)
    PM = np.zeros((N, K), np.float32); PI = np.zeros((N, K), np.float16)
    PM[row, pos] = S['mz']; PI[row, pos] = S['it']
    del row, pos
    X = np.load(fpx_p)
    keep, s_ptr, s_idx, n_ptr, n_idx = X['keep'], X['s_ptr'], X['s_idx'], X['n_ptr'], X['n_idx']
    NB = len(keep)
    negs = np.load(f'{data}/negs.npy')
    if os.environ.get('PF_NEG_TRAIN'):      # see train_pf.py: window negatives restricted to training positives
        nt = np.load(os.environ['PF_NEG_TRAIN'])
        km = (negs >= 0) & nt[np.maximum(negs, 0)]
        o_ = np.argsort(~km, axis=1, kind='stable')
        negs = np.where(np.take_along_axis(km, o_, 1), np.take_along_axis(negs, o_, 1), -1).astype(negs.dtype)
        print('window negatives restricted to training structures: mean per structure', round(float((negs >= 0).sum(1).mean()), 1), flush=True)
    split, sidx = g['split'], g['sidx']
    if os.environ.get('FP_EXCLUDE'):
        ex = set(open(os.environ['FP_EXCLUDE']).read().split())
        split = split.copy(); split[(split == 0) & np.isin(sidx, np.where(np.isin(T['key'], list(ex)))[0])] = 2
    tr = np.where(split == 0)[0]
    print('train spectra', len(tr), 'val', int((split == 1).sum()), 'held out', int((split == 2).sum()), 'bits', NB,
          f'{time.time() - t0:.0f}s', flush=True)
    w = np.where(g['inst_id'][tr] == 1, float(os.environ.get('PF_TIMS_W', '2.0')), 1.0); cw = np.cumsum(w); cw /= cw[-1]
    so = tr[np.argsort(sidx[tr], kind='stable')]; ss = sidx[so]
    s_first = np.r_[0, np.where(np.diff(ss) != 0)[0] + 1]; s_len = np.diff(np.r_[s_first, len(ss)])
    gkey = sidx[tr].astype(np.int64) * 16 + g['add_id'][tr]
    o = np.argsort(gkey, kind='stable'); gs = gkey[o]
    first = np.r_[0, np.where(np.diff(gs) != 0)[0] + 1]; gsize = np.diff(np.r_[first, len(gs)])
    gid = np.repeat(np.arange(len(first)), gsize)
    grp_start = np.empty(len(tr), np.int64); grp_n = np.empty(len(tr), np.int64); rank_in = np.empty(len(tr), np.int64)
    grp_start[o] = first[gid]; grp_n[o] = gsize[gid]; rank_in[o] = np.arange(len(gs)) - first[gid]
    pos_in_tr = np.full(N, -1, np.int64); pos_in_tr[tr] = np.arange(len(tr))
    # packed bits of all molecule rows: structures 0..NS-1, then negative rows NS..; np.packbits bit order (MSB first)
    NS = len(s_ptr) - 1
    nbyte = (NB + 7) // 8
    PACK = np.zeros((NS + len(n_ptr) - 1, nbyte), np.uint8)
    for base, ptr, idx in ((0, s_ptr, s_idx), (NS, n_ptr, n_idx)):
        n = len(ptr) - 1
        for a in range(0, n, 20000):
            b_ = min(n, a + 20000)
            dense = np.zeros((b_ - a, nbyte * 8), bool)
            ln = np.diff(ptr[a:b_ + 1])
            dense[np.repeat(np.arange(b_ - a), ln), idx[ptr[a]:ptr[b_]].astype(np.int64)] = True
            PACK[base + a:base + b_] = np.packbits(dense, axis=1)
    if SYN:         # simulated structures' bits appended after the negative rows
        Z = np.load(SYN)
        y_ptr, y_idx = Z['y_ptr'], Z['y_idx']; ny = len(y_ptr) - 1
        YP = np.zeros((ny, nbyte), np.uint8)
        for a in range(0, ny, 20000):
            b_ = min(ny, a + 20000)
            dense = np.zeros((b_ - a, nbyte * 8), bool)
            ln = np.diff(y_ptr[a:b_ + 1])
            dense[np.repeat(np.arange(b_ - a), ln), y_idx[y_ptr[a]:y_ptr[b_]].astype(np.int64)] = True
            YP[a:b_] = np.packbits(dense, axis=1)
        g.update(NSY0=len(PACK), SPM=Z['PM'], SPI=Z['PI'], Sprec=Z['prec'], Sce=Z['ce'], Ssidx=Z['sidx'].astype(np.int64),
                 Swneg=Z['wneg'].astype(np.int64))
        PACK = np.concatenate([PACK, YP])
        print('synthetic spectra', len(Z['sidx']), 'structures', ny, 'per batch', B_SYN, 'window negatives per structure',
              round(float((Z['wneg'] >= 0).sum(1).mean()), 1), flush=True)
    print('packed bit matrix', PACK.shape, f'{PACK.nbytes / 1e9:.2f} GB', f'{time.time() - t0:.0f}s', flush=True)
    G.update(g, split=split, PM=PM, PI=PI, lens=lens, keep=keep, s_ptr=s_ptr, s_idx=s_idx, n_ptr=n_ptr, n_idx=n_idx,
             NB=NB, negs=negs, tr=tr, cw=cw, so=so, s_first=s_first, s_len=s_len, grp_start=grp_start, grp_n=grp_n,
             rank_in=rank_in, order_tr=tr[o], pos_in_tr=pos_in_tr, PACK=PACK, NS=NS)


def phases(x):
    """float64 masses (...) -> float32 fractional cycles (..., N_FREQ) (the float64 part of PeakFormer._sin)."""
    c = x[..., None] * FREQ
    return (c - np.floor(c)).astype(np.float32)


def pick_negs(rng, nl):
    """NNEG window negatives per row from left-packed lists nl (-1 padded): without replacement (NOREP) or uniform."""
    cnt = (nl >= 0).sum(1)
    if NOREP:
        pick_pos = np.argsort(rng.random(nl.shape) + (np.arange(nl.shape[1]) >= cnt[:, None]), 1)[:, :NNEG]
        pick = np.take_along_axis(nl, pick_pos, 1).astype(np.int64); pick[pick_pos >= cnt[:, None]] = -1
    else:
        pick_pos = np.minimum((rng.random((len(nl), NNEG)) * np.maximum(cnt, 1)[:, None]).astype(np.int64), nl.shape[1] - 1)
        pick = np.take_along_axis(nl, pick_pos, 1).astype(np.int64); pick[cnt == 0] = -1
    return pick


def augment(rng, mz, it, pr):
    """Peak dropout (base peak kept), intensity and ppm jitter, renormalisation (train_pf.py batch_inputs)."""
    n = len(mz)
    valid = (it > 0) & (mz > 0) & (mz <= pr[:, None] + 2.0)
    drop = rng.random(valid.shape) < 0.1
    drop[np.arange(n), np.where(valid, it, -1.0).argmax(1)] = False      # keep the base peak
    valid &= ~drop
    it = it * np.exp(0.15 * rng.standard_normal(it.shape)).astype(np.float32)
    cal = 1 + 3e-6 * rng.standard_normal((n, 1))
    mz = mz * cal * (1 + 2e-6 * rng.standard_normal(mz.shape)); pr = pr * cal[:, 0]
    it = np.where(valid, it, 0.0).astype(np.float32)
    it = it / np.maximum(it.max(1, keepdims=True), 1e-6)
    mz = np.where(valid, mz, 0.0)
    return mz, it, valid, pr


def syn_rows(rng, n):
    """n synthetic rows: uniform over simulated spectra (3 energies per structure), up to 4 low noise peaks replacing
    the weakest slots, window negatives inside the simulated set."""
    g = G
    j = rng.integers(0, len(g['Ssidx']), n)
    s = g['Ssidx'][j]
    mz = g['SPM'][j].astype(np.float64); it = g['SPI'][j].astype(np.float32); pr = g['Sprec'][j].astype(np.float64)
    k = rng.integers(0, 5, n)
    weak = np.argsort(it, 1, kind='stable')[:, :4]
    for c in range(4):
        r = np.where(k > c)[0]
        mz[r, weak[r, c]] = 40.0 + rng.random(len(r)) * np.maximum(pr[r] - 41.0, 1.0)
        it[r, weak[r, c]] = rng.uniform(0.005, 0.05, len(r)).astype(np.float32)
    mz, it, valid, pr2 = augment(rng, mz, it, pr)
    meta = meta_matrix(np.zeros(n, np.int64), np.zeros(n, np.int64), g['Sce'][j].astype(np.float32), np.zeros(n, np.int64),
                       pr, np.ones(n))
    return mz, it, valid, pr2, meta, s, pick_negs(rng, g['Swneg'][s])


def host_batch(rng):
    """One training batch as fixed-shape numpy arrays: B - B_SYN real rows (train_pf.py sampling + batch_inputs, in
    numpy) then B_SYN synthetic rows (PF_SYN)."""
    mz, it, valid, pr, meta, s, pick = real_rows(rng, B - B_SYN)
    rows_p = s; rows_n = G['NS'] + np.maximum(pick, 0)
    look = np.zeros(B, bool)
    if B_SYN:
        sm, si, sv, sp, smeta, ss, spick = syn_rows(rng, B_SYN)
        mz, it, valid, pr, meta = (np.concatenate(x) for x in ((mz, sm), (it, si), (valid, sv), (pr, sp), (meta, smeta)))
        rows_p = np.r_[rows_p, G['NSY0'] + ss]; rows_n = np.r_[rows_n, G['NSY0'] + np.maximum(spick, 0)]
        pick = np.r_[pick, spick]; s = np.r_[s, ss]; look[B - B_SYN:] = True
    # in-batch negatives: never the same structure, never across looks
    same = ((s[:, None] == s[None, :]) | (look[:, None] != look[None, :])) & ~np.eye(B, dtype=bool)
    return dict(ph_mz=phases(mz), ph_nl=phases(pr[:, None] - mz), it=it, valid=valid, ph_pr=phases(pr), meta=meta,
                rows_p=rows_p, rows_n=rows_n.reshape(-1), okn=pick >= 0, same=same)


def real_rows(rng, B):
    """B real training rows (train_pf.py sampling + batch_inputs, in numpy)."""
    g = G
    tr, cw = g['tr'], g['cw']
    idx = tr[np.minimum(np.searchsorted(cw, rng.random(B)), len(tr) - 1)]
    nb = int(round(S_BAL * B))
    if nb:
        j = rng.integers(0, len(g['s_first']), nb)
        idx[:nb] = g['so'][g['s_first'][j] + (rng.random(nb) * g['s_len'][j]).astype(np.int64)]
    s = g['sidx'][idx].astype(np.int64)
    pick = pick_negs(rng, g['negs'][s])
    PM, PI, ce, prec = g['PM'], g['PI'], g['ce'], g['prec']
    n = B
    mz = PM[idx].astype(np.float64); it = PI[idx].astype(np.float32); pr = prec[idx].astype(np.float64)
    nm = np.ones(n, np.int64); ce_b = ce[idx].astype(np.float32).copy()
    if P_MERGE > 0:
        p = g['pos_in_tr'][idx]
        gn = g['grp_n'][np.maximum(p, 0)]
        can = (p >= 0) & (gn >= 2) & (rng.random(n) < P_MERGE)
        parts = []
        for extra in range(3):
            sel = can & (nm == extra + 1) & (gn > extra + 1) & ((extra == 0) | (rng.random(n) < 0.5))
            if not sel.any():
                break
            pp = p[sel]
            k = (g['rank_in'][pp] + rng.integers(1, g['grp_n'][pp])) % g['grp_n'][pp]
            partner = np.full(n, -1, np.int64); partner[sel] = g['order_tr'][g['grp_start'][pp] + k]
            nm[sel] += 1
            cp = ce[partner[sel]]; rows_sel = np.where(sel)[0]
            both = (ce_b[sel] >= 0) & (cp >= 0)
            r = rows_sel[both]
            ce_b[r] = (ce_b[r] * (nm[r] - 1) + cp[both]) / nm[r]
            parts.append(partner)
        if parts:
            Pm = [mz]; Pi = [it / np.maximum(it.max(1, keepdims=True), 1e-6)]
            for partner in parts:
                use = (partner >= 0)[:, None]; q = np.maximum(partner, 0)
                i2 = PI[q].astype(np.float32); i2 = i2 / np.maximum(i2.max(1, keepdims=True), 1e-6)
                Pm.append(np.where(use, PM[q].astype(np.float64), 0.0)); Pi.append(np.where(use, i2, 0.0).astype(np.float32))
            Pm = np.concatenate(Pm, 1); Pi = np.concatenate(Pi, 1)
            top = np.argpartition(-Pi, K - 1, axis=1)[:, :K]      # top K by intensity (tokens are a set: order free)
            mz = np.take_along_axis(Pm, top, 1); it = np.take_along_axis(Pi, top, 1)
    mz, it, valid, pr = augment(rng, mz, it, pr)
    ib = g['inst_id'][idx]
    if INST_DROP > 0:
        ib = np.where(rng.random(n) < INST_DROP, 0, ib)
    meta = meta_matrix(g['add_id'][idx], g['pol'][idx], ce_b, ib, prec[idx].astype(np.float64), nm)
    return mz, it, valid, pr, meta, s, pick


class PeakFormerX(PeakFormer):
    """PeakFormer with host-computed phases and dense multi-hot molecule inputs (same parameters; the float64
    frequency buffer, unused here and non-persistent, is dropped so that nothing float64 reaches the device)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._buffers.pop('freq', None)

    @staticmethod
    def sin_ph(ph):
        a = (2 * math.pi) * ph
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def encode_spec_ph(self, ph_mz, ph_nl, it, mask, ph_pr, meta):
        ints = torch.stack([it.sqrt(), torch.log(it.clamp_min(1e-4)) / 4 + 1], -1)
        tok = self.mz_proj(self.sin_ph(ph_mz)) + self.nl_proj(self.sin_ph(ph_nl)) + self.int_proj(ints)
        g = self.meta_proj(meta) + self.prec_proj(self.sin_ph(ph_pr))
        x = torch.cat([g.unsqueeze(1), tok], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        h = self.norm(self.enc(x, src_key_padding_mask=pad)[:, 0])
        return F.normalize(self.spec_emb(h).float(), dim=-1), self.fp_head(h).float()

    def encode_mh(self, mh):
        return F.normalize(self.mol(mh @ self.bits.weight.to(mh.dtype)).float(), dim=-1)


def run(rank):
    load()
    g = G
    if not g.get('_watchdog'):
        g['_watchdog'] = True
        threading.Thread(target=watchdog, daemon=True).start()
    NB = g['NB']
    if XLA:
        dev = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()
        world = xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else 1
        try:
            import torch_xla.runtime as xr
            world = xr.world_size()
        except Exception:
            pass
        sync = (lambda: torch_xla.sync()) if hasattr(torch_xla, 'sync') else xm.mark_step
        mh_dtype = torch.bfloat16
    else:
        dev = os.environ.get('PF_DEV') or ('cuda' if torch.cuda.is_available() else 'cpu')
        world, sync = 1, (lambda: None)
        mh_dtype = torch.float16 if dev == 'cuda' else torch.float32
    master = rank == 0
    rng = np.random.default_rng(SEED * 1000 + rank)
    cfg = dict(n_bits=NB, d=int(os.environ.get('PF_D', '256')), layers=int(os.environ.get('PF_LAYERS', '6')),
               heads=int(os.environ.get('PF_HEADS', '8')), d_mol=int(os.environ.get('PF_DMOL', '512')),
               drop=float(os.environ.get('PF_DROP', '0.1')))
    with INIT_LOCK:                                 # identical initial weights on every core
        torch.manual_seed(SEED)
        model = PeakFormerX(**cfg)
    if os.environ.get('PF_INIT'):
        sd = torch.load(os.environ['PF_INIT'], map_location='cpu', weights_only=False)
        sd.pop('_cfg', None); sd.pop('_keep', None); sd.pop('freq', None); model.load_state_dict(sd)
    chk0 = float(sum(p.detach().double().sum() for p in model.parameters()))
    model = model.to(dev)
    if XLA and world > 1:
        xm.broadcast_master_param(model)            # core 0's weights everywhere (belt and braces with the lock)
        spread = xm.mesh_reduce('init_chk', chk0, lambda v: max(v) - min(v))
        if master:
            print('initial weight checksum spread across cores (before broadcast)', spread, flush=True)
    if XLA:
        xm.set_rng_state(SEED * 1000 + rank, device=dev)     # per-core dropout masks
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    if master:
        print('model params', sum(p.numel() for p in model.parameters()), cfg, 'device', dev, 'world', world,
              'batch per core', B, 'negatives', NNEG, flush=True)
    PACK = torch.from_numpy(g['PACK']).to(dev)
    SH = torch.arange(7, -1, -1, dtype=torch.uint8, device=dev)
    eye_ar = torch.arange(B, device=dev)
    amp = dict(device_type='xla', dtype=torch.bfloat16) if XLA else dict(device_type=dev, dtype=torch.float16, enabled=dev == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=(not XLA) and dev == 'cuda')

    def unpack(rows):
        p = PACK[rows]
        return ((p.unsqueeze(-1) >> SH) & 1).reshape(rows.shape[0], -1)[:, :NB].to(mh_dtype)

    q = queue.Queue(maxsize=4)

    def producer():
        r = np.random.default_rng(SEED * 1000 + rank + 7)
        while True:
            q.put(host_batch(r))
    threading.Thread(target=producer, daemon=True).start()

    def to_dev(b):
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        return {k: T(v) for k, v in b.items()}

    def train_step(bt):
        with torch.autocast(**amp):
            e_s, z = model.encode_spec_ph(bt['ph_mz'], bt['ph_nl'], bt['it'], bt['valid'], bt['ph_pr'], bt['meta'])
            mh_p = unpack(bt['rows_p']); mh_n = unpack(bt['rows_n'])
            e_pos = model.encode_mh(mh_p)
            e_neg = model.encode_mh(mh_n).view(B, NNEG, -1)
        z = z.float(); e_s = e_s.float()
        Ft = mh_p.float()
        same, okn = bt['same'], bt['okn']
        tau = model.log_tau.exp().clamp(0.01, 1.0); alpha = model.log_alpha.exp()
        sim_b = (e_s @ e_pos.T / tau).masked_fill(same, -1e4)
        s_neg = (torch.einsum('bd,bnd->bn', e_s, e_neg) / tau).masked_fill(~okn, -1e4)
        loss_cb = F.cross_entropy(sim_b, eye_ar)
        loss_ch = F.cross_entropy(torch.cat([sim_b.diagonal().unsqueeze(1), s_neg], 1), torch.zeros_like(eye_ar))
        fz_b = (alpha * (z @ Ft.T)).masked_fill(same, -1e4)
        f_neg = (alpha * torch.einsum('bk,bnk->bn', z, mh_n.view(B, NNEG, NB).float())).masked_fill(~okn, -1e4)
        loss_fb = F.cross_entropy(fz_b, eye_ar)
        loss_fh = F.cross_entropy(torch.cat([fz_b.diagonal().unsqueeze(1), f_neg], 1), torch.zeros_like(eye_ar))
        loss_bce = F.binary_cross_entropy_with_logits(z, Ft)
        loss = loss_cb + loss_ch + loss_fb + loss_fh + W_BCE * loss_bce
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if XLA and world > 1:
            xm.reduce_gradients(opt)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        return torch.stack([loss, loss_cb, loss_ch, loss_fb, loss_fh, loss_bce, tau, alpha]).detach()

    def set_lr(step, total):
        frac = min(1.0, step / max(total, 1))
        lr = LR * min(1.0, frac / 0.03) * 0.5 * (1 + np.cos(np.pi * frac)) if frac > 0 else LR * 1e-3
        for pg in opt.param_groups:
            pg['lr'] = max(lr, LR * 1e-3)

    def agree(name, v, fn):
        """Same value on every core (core 0's view of time decides the budget)."""
        if XLA and world > 1:
            return xm.mesh_reduce(name, v, fn)
        return v

    def save_and_validate(tag):
        if not master:
            return
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if INST_DROP >= 1:      # tie the instrument tokens (train_pf.py tie_inst) in the saved copy
            c = N_ADD + 4
            w_ = sd['meta_proj.weight'].clone(); w_[:, c + 1:c + N_INST] = w_[:, c:c + 1]; sd['meta_proj.weight'] = w_
        cpu = PeakFormer(**cfg); cpu.load_state_dict(sd); cpu.eval()
        res = validate(cpu)
        sd['_cfg'] = cfg; sd['_keep'] = torch.from_numpy(g['keep'].astype(np.int64))
        torch.save(sd, out)
        print(f'{tag} val {res} {time.time() - t0:.0f}s', flush=True)

    total = EPOCHS * STEPS
    t_start = None; step = 0
    set_lr(0, total)
    stats = None
    for ep in range(EPOCHS):
        for _ in range(STEPS):
            bt = to_dev(q.get())
            st = train_step(bt)
            sync()
            LAST[0] = time.time()
            stats = st if stats is None else stats + st
            step += 1
            set_lr(step, total)
            if step == 20:
                t_start = time.time()           # after compilation
            if step == 20 + CAL:
                per = (time.time() - t_start) / CAL
                left = HOURS * 3600 * 0.93 - (time.time() - t0)
                est = int(step + left / per)
                total = int(agree('total', min(total, est), min))
                if master:
                    print(f'{per:.3f} s/step after compile -> step budget {total} ({left / 3600:.2f} h left)', flush=True)
            if step % LOG == 0 and master:
                v = (stats / LOG).cpu().numpy(); stats = None
                print(f'step {step}/{total} loss {v[0]:.3f} (cos b {v[1]:.2f} h {v[2]:.2f} | fz b {v[3]:.2f} h {v[4]:.2f} | '
                      f'bce {v[5]:.4f}) tau {v[6]:.3f} alpha {v[7]:.3f} lr {opt.param_groups[0]["lr"]:.2e} '
                      f'{time.time() - t0:.0f}s', flush=True)
            elif step % LOG == 0:
                stats = None
            if step >= total:
                break
        save_and_validate(f'EPOCH {ep}')
        if XLA and world > 1:
            xm.rendezvous(f'ep{ep}')
        if step >= total:
            if master:
                print('step budget reached', flush=True)
            break


def validate(model, max_structs=int(os.environ.get('PF_VAL_N', '1500'))):
    """train_pf.py validate() on a CPU copy: split-1 structures, up to 3 spectra, truth vs its window negatives."""
    g = G
    split, sidx, inst_id, negs = g['split'], g['sidx'], g['inst_id'], g['negs']
    by = {}
    for i in np.where(split == 1)[0]:
        by.setdefault(int(sidx[i]), []).append(i)
    spectra = lambda ii: [dict(mz=g['PM'][i, :g['lens'][i]], it=g['PI'][i, :g['lens'][i]].astype(np.float32),
                               prec=float(g['prec'][i]), adduct=ADD[g['add_id'][i]], ce=float(g['ce'][i]),
                               inst=int(inst_id[i])) for i in ii]

    def cand(e, z, rows_s, rows_n):
        i1, o1, r1 = csr_rows(g['s_ptr'], g['s_idx'], rows_s); i2, o2, r2 = csr_rows(g['n_ptr'], g['n_idx'], rows_n)
        ids = torch.from_numpy(np.r_[i1, i2]); offs = torch.from_numpy(np.r_[o1, o2 + len(i1)])
        rid = torch.from_numpy(np.r_[r1, r2 + len(rows_s)]); n = len(rows_s) + len(rows_n)
        m = model.encode_mol(ids, offs)
        return ((e @ m.T).squeeze(0).numpy(),
                fz_scores(z.unsqueeze(0), ids, rid, torch.zeros(n, dtype=torch.long), n).numpy())
    res = {k: [] for k in ('cos', 'fz', 'sum', 'fz_tims')}
    rr = lambda sc: 1.0 / (1 + (sc[1:] > sc[0]).sum())
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    with torch.no_grad():
        for s in list(by)[:max_structs]:
            ng = negs[s][negs[s] >= 0].astype(np.int64)
            if len(ng) == 0:
                continue
            e, z = embed_query(model, spectra(by[s][:3]))
            cos, fz = cand(e, z, np.array([s]), ng)
            res['cos'].append(rr(cos)); res['fz'].append(rr(fz)); res['sum'].append(rr(zs(cos) + zs(fz)))
            ti = [i for i in by[s] if inst_id[i] == 1][:3]
            if ti:
                e, z = embed_query(model, spectra(ti))
                res['fz_tims'].append(rr(cand(e, z, np.array([s]), ng)[1]))
    return {k: (round(float(np.mean(v)), 4), round(float(np.mean(np.array(v) == 1)), 4), len(v)) for k, v in res.items()}


def _mp(index):
    try:
        run(index)
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)        # one failed core would leave the others waiting in a collective for the whole session


def watchdog(limit=float(os.environ.get('PF_STALL_MIN', '30')) * 60):
    """Kill the run when no training step finished for `limit` seconds (hung collective, dead prefetch thread):
    a hang would otherwise hold the TPU until the session limit."""
    while True:
        time.sleep(60)
        if time.time() - LAST[0] > limit:
            print(f'WATCHDOG: no progress for {limit / 60:.0f} min -> exit', flush=True)
            os._exit(2)


if __name__ == '__main__':
    load()
    LAST[0] = time.time() + 1800                     # compilation / first steps get extra time
    if XLA and os.environ.get('PF_CORES', '') != '1':
        import torch_xla.distributed.xla_multiprocessing as xmp
        n = int(os.environ['PF_CORES']) if os.environ.get('PF_CORES') else None
        try:        # fork: the cores share the host arrays loaded above (no XLA device was touched in this process)
            xmp.spawn(_mp, args=(), nprocs=n, start_method='fork')
        except TypeError:
            xmp.spawn(_mp, args=(), nprocs=n)
    else:
        run(0)
