"""Train the PeakFormer (casmi/peakformer.py). CUDA (AMP) or CPU.

python scripts/train_pf.py <fp_data_dir> <fpx.npz> <out.pt> [epochs] [steps_per_epoch] [batch] [n_neg]
fp_data_dir: spectra.npz / structs.npz / negs.npy of scripts/prep_fp_data.py; fpx.npz: multi-block fingerprints of
its structures and negatives (scripts/prep_pf_data.py).
Training spectra: split 0 only (split 1 = validation structures, split 2 = ranker hold-out; FP_EXCLUDE adds more).
Per step: weighted sampling (timsTOF x2), merge augmentation (p 0.35: 1-3 other spectra of the same structure and
adduct, top 64 of the union), peak dropout, intensity and ppm jitter.
Loss = f.z and cosine scores, each against in-batch and same-mass window negatives (negs.npy), + fingerprint BCE.
Validation (as scripts/train_fp.py): split-1 structures, up to 3 spectra, truth vs its window negatives -> MRR and
top-1 for f.z, cosine and their sum of within-query z-scores; also timsTOF-only.
Env: PF_SBAL, PF_D, PF_LAYERS, PF_HEADS, PF_DMOL, PF_DROP, PF_LR, PF_MERGE, PF_BCE, PF_THREADS, PF_VAL_N, PF_INIT, FP_EXCLUDE, PF_HOURS,
PF_STEPS (fixed total steps, step-based schedule), PF_TIMS_W (sampling weight of timsTOF spectra, default 2), PF_NEG_TRAIN (negatives restricted to training structures), PF_INST_DROP (share of training spectra whose instrument token is replaced by 0 = unknown: timsTOF occurs almost only
with Enamine chemistry, so the token became a chemistry prior that hurts timsTOF natural products).
"""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, '/kaggle/working/pkg')
from casmi.peakformer import K, PeakFormer, csr_rows, embed_query, fz_scores, meta_matrix, ADD, N_ADD, N_INST

data, fpx_p, out = sys.argv[1:4]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
B = int(sys.argv[6]) if len(sys.argv) > 6 else 256
NNEG = int(sys.argv[7]) if len(sys.argv) > 7 else 31
HOURS = float(os.environ.get('PF_HOURS', '1e9'))        # stop (after saving) once this much wall time is used
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
amp = dev == 'cuda'
if os.environ.get('PF_THREADS'):
    torch.set_num_threads(int(os.environ['PF_THREADS']))
SEED = int(os.environ.get('PF_SEED', '0')); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
t0 = time.time()

S = np.load(f'{data}/spectra.npz'); T = np.load(f'{data}/structs.npz')
off = S['off']; N = len(off) - 1; lens = np.diff(off)
assert lens.max() <= K
prec, ce, add_id, inst_id, pol, sidx, split = (S[k] for k in ('prec', 'ce', 'add_id', 'inst_id', 'pol', 'sidx', 'split'))
row = np.repeat(np.arange(N), lens); pos = np.arange(off[-1]) - np.repeat(off[:-1], lens)
PM = np.zeros((N, K), np.float32); PI = np.zeros((N, K), np.float16)
PM[row, pos] = S['mz']; PI[row, pos] = S['it']
del row, pos
X = np.load(fpx_p)
keep, s_ptr, s_idx, n_ptr, n_idx = X['keep'], X['s_ptr'], X['s_idx'], X['n_ptr'], X['n_idx']
NB = len(keep)
negs = np.load(f'{data}/negs.npy')
if os.environ.get('PF_NEG_TRAIN'):
    # window negatives restricted to training structures (PF_NEG_TRAIN = bool per negative row, scripts/
    # pf_negtrain.py): COCONUT-not-train structures are never positives, so as negatives they teach a population prior
    # "COCONUT-not-train is never the answer" -- exactly where hidden COCONUT answers come from. Kept negatives stay
    # left-aligned in their original order (same formula first); the batch sampler draws from the first cnt entries.
    nt = np.load(os.environ['PF_NEG_TRAIN'])
    km = (negs >= 0) & nt[np.maximum(negs, 0)]
    o_ = np.argsort(~km, axis=1, kind='stable')
    negs = np.where(np.take_along_axis(km, o_, 1), np.take_along_axis(negs, o_, 1), -1).astype(negs.dtype)
    print('window negatives restricted to training structures: mean per structure', round(float((negs >= 0).sum(1).mean()), 1), flush=True)
if os.environ.get('FP_EXCLUDE'):
    ex = set(open(os.environ['FP_EXCLUDE']).read().split())
    split = split.copy(); split[(split == 0) & np.isin(sidx, np.where(np.isin(T['key'], list(ex)))[0])] = 2
tr = np.where(split == 0)[0]
print('train spectra', len(tr), 'val', int((split == 1).sum()), 'held out', int((split == 2).sum()), 'bits', NB,
      f'{time.time() - t0:.0f}s', flush=True)
w = np.where(inst_id[tr] == 1, float(os.environ.get('PF_TIMS_W', '2.0')), 1.0); cw = np.cumsum(w); cw /= cw[-1]   # timsTOF spectra (99.9% enveda-180) up-weighted
S_BAL = float(os.environ.get('PF_SBAL', '0'))       # share of each batch drawn structure-uniformly
so = tr[np.argsort(sidx[tr], kind='stable')]; ss = sidx[so]
s_first = np.r_[0, np.where(np.diff(ss) != 0)[0] + 1]; s_len = np.diff(np.r_[s_first, len(ss)])

# merge partners: training spectra grouped by (structure, adduct)
gkey = sidx[tr].astype(np.int64) * 16 + add_id[tr]
o = np.argsort(gkey, kind='stable'); gs = gkey[o]
first = np.r_[0, np.where(np.diff(gs) != 0)[0] + 1]; gsize = np.diff(np.r_[first, len(gs)])
gid = np.repeat(np.arange(len(first)), gsize)
grp_start = np.empty(len(tr), np.int64); grp_n = np.empty(len(tr), np.int64); rank_in = np.empty(len(tr), np.int64)
grp_start[o] = first[gid]; grp_n[o] = gsize[gid]; rank_in[o] = np.arange(len(gs)) - first[gid]
order_tr = tr[o]
pos_in_tr = np.full(N, -1, np.int64); pos_in_tr[tr] = np.arange(len(tr))

D = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
gPM, gPI, g_prec = D(PM), D(PI), D(prec.astype(np.float64))
if dev == 'cuda':          # fingerprint CSR on the device: per-step gathers of ~30M bit ids stay off the CPU
    g_csr = {'s': (D(s_ptr), D(s_idx.astype(np.int32))), 'n': (D(n_ptr), D(n_idx.astype(np.int32)))}


def bits_of(which, rows):
    """On-bit ids, bag offsets and row ids (device tensors) of structure ('s') or negative ('n') rows."""
    if dev != 'cuda':
        i, o, r = csr_rows(*((s_ptr, s_idx) if which == 's' else (n_ptr, n_idx)), rows)
        return D(i), D(o), D(r)
    ptr, idx = g_csr[which]
    rows = torch.from_numpy(np.ascontiguousarray(rows)).to(dev)
    st = ptr[rows]; ln = ptr[rows + 1] - st
    off = torch.cumsum(ln, 0) - ln
    rid = torch.repeat_interleave(torch.arange(len(rows), device=dev), ln)
    pos = torch.repeat_interleave(st - off, ln) + torch.arange(int(ln.sum()), device=dev)
    return idx[pos].long(), off, rid
cfg = dict(n_bits=NB, d=int(os.environ.get('PF_D', '256')), layers=int(os.environ.get('PF_LAYERS', '6')),
           heads=int(os.environ.get('PF_HEADS', '8')), d_mol=int(os.environ.get('PF_DMOL', '512')),
           drop=float(os.environ.get('PF_DROP', '0.1')))
model = PeakFormer(**cfg).to(dev)
if os.environ.get('PF_INIT'):
    sd = torch.load(os.environ['PF_INIT'], map_location=dev, weights_only=False)
    sd.pop('_cfg', None); sd.pop('_keep', None); model.load_state_dict(sd)
LR = float(os.environ.get('PF_LR', '5e-4'))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total = EPOCHS * STEPS
FIXED = int(os.environ.get('PF_STEPS', '0'))       # fixed step count (docs/reproduce.md G5): step-based schedule, stop there
if FIXED:
    total = FIXED


def set_lr(step):
    """Warm-up then cosine decay over the larger of the step fraction and the wall-time fraction (PF_HOURS)."""
    frac = min(1.0, step / total if FIXED else max(step / total, (time.time() - t0) / (HOURS * 3600)))
    lr = LR * min(1.0, frac / 0.03) * 0.5 * (1 + np.cos(np.pi * frac)) if frac > 0 else LR * 1e-3
    for g in opt.param_groups:
        g['lr'] = max(lr, LR * 1e-3)
scaler = torch.amp.GradScaler('cuda', enabled=amp)
P_MERGE = float(os.environ.get('PF_MERGE', '0.35'))
INST_DROP = float(os.environ.get('PF_INST_DROP', '0'))
NOREP = os.environ.get('PF_NEG_NOREP') == '1'      # window negatives drawn without replacement (short lists: no repeats)
W_BCE = float(os.environ.get('PF_BCE', '5'))
print('model params', sum(p.numel() for p in model.parameters()), cfg, 'device', dev, 'threads', torch.get_num_threads(),
      flush=True)


def batch_inputs(idx, train=True):
    """Spectrum ids -> tensors (mz float64, it, mask, prec float64, meta) with augmentation when train."""
    n = len(idx)
    ii = torch.from_numpy(idx).to(dev)
    mz = gPM[ii].double(); it = gPI[ii].float(); pr = g_prec[ii]
    nm = np.ones(n, np.int64); ce_b = ce[idx].astype(np.float32).copy()
    if train and P_MERGE > 0:
        p = pos_in_tr[idx]
        gn = grp_n[np.maximum(p, 0)]
        can = (p >= 0) & (gn >= 2) & (rng.random(n) < P_MERGE)
        parts = []
        for extra in range(3):
            sel = can & (nm == extra + 1) & (gn > extra + 1) & ((extra == 0) | (rng.random(n) < 0.5))
            if not sel.any():
                break
            pp = p[sel]
            k = (rank_in[pp] + rng.integers(1, grp_n[pp])) % grp_n[pp]
            partner = np.full(n, -1, np.int64); partner[sel] = order_tr[grp_start[pp] + k]
            nm[sel] += 1
            cp = ce[partner[sel]]; rows_sel = np.where(sel)[0]
            both = (ce_b[sel] >= 0) & (cp >= 0)
            r = rows_sel[both]
            ce_b[r] = (ce_b[r] * (nm[r] - 1) + cp[both]) / nm[r]
            parts.append(partner)
        if parts:
            Pm = [mz]; Pi = [it / it.amax(1, keepdim=True).clamp_min(1e-6)]
            for partner in parts:
                pi_ = torch.from_numpy(np.maximum(partner, 0)).to(dev)
                use = torch.from_numpy(partner >= 0).to(dev).unsqueeze(1)
                m2 = gPM[pi_].double(); i2 = gPI[pi_].float()
                i2 = i2 / i2.amax(1, keepdim=True).clamp_min(1e-6)
                Pm.append(torch.where(use, m2, torch.zeros_like(m2))); Pi.append(torch.where(use, i2, torch.zeros_like(i2)))
            Pm = torch.cat(Pm, 1); Pi = torch.cat(Pi, 1)
            top = Pi.topk(K, dim=1).indices
            mz = Pm.gather(1, top); it = Pi.gather(1, top)
    valid = (it > 0) & (mz > 0) & (mz <= pr.unsqueeze(1) + 2.0)
    if train:
        drop = torch.rand(valid.shape, device=dev) < 0.1
        drop.scatter_(1, it.masked_fill(~valid, -1).argmax(1, keepdim=True), False)     # keep the base peak
        valid = valid & ~drop
        it = it * torch.exp(0.15 * torch.randn_like(it))
        cal = 1 + 3e-6 * torch.randn(n, 1, device=dev, dtype=torch.float64)
        mz = mz * cal * (1 + 2e-6 * torch.randn_like(mz)); pr = pr * cal.squeeze(1)
    it = torch.where(valid, it, torch.zeros_like(it))
    it = it / it.amax(1, keepdim=True).clamp_min(1e-6)
    mz = torch.where(valid, mz, torch.zeros_like(mz))
    ib = inst_id[idx]
    if train and INST_DROP > 0:
        ib = np.where(np.random.random(len(ib)) < INST_DROP, 0, ib)
    meta = meta_matrix(add_id[idx], pol[idx], ce_b, ib, prec[idx].astype(np.float64), nm)
    return mz, it.float(), valid, pr, D(meta)


def spectra_dicts(ii):
    return [dict(mz=PM[i, :lens[i]], it=PI[i, :lens[i]].astype(np.float32), prec=float(prec[i]), adduct=ADD[add_id[i]],
                 ce=float(ce[i]), inst=int(inst_id[i])) for i in ii]


def cand_scores(e, z, rows_s, rows_n):
    """Query embedding e (1, emb) and logits z (n_bits,) vs truth structure rows_s + negative rows rows_n."""
    i1, o1, r1 = csr_rows(s_ptr, s_idx, rows_s); i2, o2, r2 = csr_rows(n_ptr, n_idx, rows_n)
    ids = np.r_[i1, i2]; offs = np.r_[o1, o2 + len(i1)]; rid = np.r_[r1, r2 + len(rows_s)]
    n = len(rows_s) + len(rows_n)
    m = model.encode_mol(D(ids), D(offs))
    cos = (e.float() @ m.float().T).squeeze(0)
    fz = fz_scores(z.float().unsqueeze(0), D(ids), D(rid), torch.zeros(n, dtype=torch.long, device=dev), n)
    return cos.cpu().numpy(), fz.cpu().numpy()


def tie_inst():
    """INST_DROP >= 1: instrument tokens 1-5 never reach training -> copy token 0's weights so that every token behaves
    as token 0 at validation and inference."""
    if INST_DROP >= 1:
        with torch.no_grad():
            c = N_ADD + 4
            model.meta_proj.weight[:, c + 1:c + N_INST] = model.meta_proj.weight[:, c:c + 1]


def validate(max_structs=int(os.environ.get('PF_VAL_N', '1500'))):
    tie_inst()
    model.eval()
    by = {}
    for i in np.where(split == 1)[0]:
        by.setdefault(int(sidx[i]), []).append(i)
    res = {k: [] for k in ('cos', 'fz', 'sum', 'fz_tims')}
    rr = lambda sc: 1.0 / (1 + (sc[1:] > sc[0]).sum())
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    with torch.no_grad(), torch.autocast(dev, enabled=amp):
        for s in list(by)[:max_structs]:
            ng = negs[s][negs[s] >= 0].astype(np.int64)
            if len(ng) == 0:        # no window negatives left: nothing to rank against
                continue
            e, z = embed_query(model, spectra_dicts(by[s][:3]), dev)
            cos, fz = cand_scores(e, z, np.array([s]), ng)
            res['cos'].append(rr(cos)); res['fz'].append(rr(fz)); res['sum'].append(rr(zs(cos) + zs(fz)))
            ti = [i for i in by[s] if inst_id[i] == 1][:3]
            if ti:
                e, z = embed_query(model, spectra_dicts(ti), dev)
                res['fz_tims'].append(rr(cand_scores(e, z, np.array([s]), ng)[1]))
    model.train()
    return {k: (round(float(np.mean(v)), 4), round(float(np.mean(np.array(v) == 1)), 4), len(v)) for k, v in res.items()}


def save():
    tie_inst()
    sd = model.state_dict(); sd['_cfg'] = cfg; sd['_keep'] = torch.from_numpy(keep.astype(np.int64))
    torch.save(sd, out)


set_lr(0)
step = 0; eye = torch.eye(B, dtype=torch.bool, device=dev); ar = torch.arange(B, device=dev)
for ep in range(EPOCHS):
    for _ in range(STEPS):
        idx = tr[np.minimum(np.searchsorted(cw, rng.random(B)), len(tr) - 1)]
        nb = int(round(S_BAL * B))
        if nb:
            j = rng.integers(0, len(s_first), nb)
            idx[:nb] = so[s_first[j] + (rng.random(nb) * s_len[j]).astype(np.int64)]
        s = sidx[idx].astype(np.int64)
        nl = negs[s]; cnt = (nl >= 0).sum(1)
        if NOREP:       # random order of the valid prefix first, slots beyond the row's count masked
            pick_pos = np.argsort(rng.random(nl.shape) + (np.arange(nl.shape[1]) >= cnt[:, None]), 1)[:, :NNEG]
            pick = np.take_along_axis(nl, pick_pos, 1).astype(np.int64); pick[pick_pos >= cnt[:, None]] = -1
        else:
            pick_pos = np.minimum((rng.random((B, NNEG)) * np.maximum(cnt, 1)[:, None]).astype(np.int64), 63)
            pick = np.take_along_axis(nl, pick_pos, 1).astype(np.int64); pick[cnt == 0] = -1
        okn = pick >= 0
        owner = np.repeat(np.arange(B), NNEG)[okn.reshape(-1)]
        i1, o1, r1 = bits_of('s', s)
        i2, o2, r2 = bits_of('n', pick[okn])
        mz, it, valid, pr, meta = batch_inputs(idx)
        with torch.autocast(dev, enabled=amp):
            e_s, z = model.encode_spec(mz, it, valid, pr, meta)
            e_pos = model.encode_mol(i1, o1).float()
            e_neg = model.encode_mol(i2, o2).float()
        Ft = torch.zeros(B, NB, device=dev); Ft[r1, i1] = 1.0
        same = D(s[:, None] == s[None, :]) & ~eye
        tau = model.log_tau.exp().clamp(0.01, 1.0); alpha = model.log_alpha.exp()
        # cosine head
        sim_b = (e_s @ e_pos.T / tau).masked_fill(same, -1e4)
        s_neg = torch.full((B, NNEG), -1e4, device=dev)
        s_neg[D(okn)] = (e_s[D(owner)] * e_neg).sum(-1) / tau
        loss_cb = F.cross_entropy(sim_b, ar)
        loss_ch = F.cross_entropy(torch.cat([sim_b.diagonal().unsqueeze(1), s_neg], 1), torch.zeros(B, dtype=torch.long, device=dev))
        # f.z head
        fz_b = (alpha * (z @ Ft.T)).masked_fill(same, -1e4)
        f_neg = torch.full((B, NNEG), -1e4, device=dev)
        f_neg[D(okn)] = alpha * fz_scores(z, i2, r2, D(owner), len(owner))
        loss_fb = F.cross_entropy(fz_b, ar)
        loss_fh = F.cross_entropy(torch.cat([fz_b.diagonal().unsqueeze(1), f_neg], 1), torch.zeros(B, dtype=torch.long, device=dev))
        loss_bce = F.binary_cross_entropy_with_logits(z, Ft)
        loss = loss_cb + loss_ch + loss_fb + loss_fh + W_BCE * loss_bce
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1; set_lr(step)
        if step % int(os.environ.get('PF_LOG', '100')) == 0:
            print(f'step {step}/{total} loss {loss.item():.3f} (cos b {loss_cb.item():.2f} h {loss_ch.item():.2f} | '
                  f'fz b {loss_fb.item():.2f} h {loss_fh.item():.2f} | bce {loss_bce.item():.4f}) tau {tau.item():.3f} '
                  f'alpha {alpha.item():.3f} {time.time() - t0:.0f}s', flush=True)
        if time.time() - t0 > HOURS * 3600 or (FIXED and step >= FIXED):
            break
    print(f'EPOCH {ep} val {validate()} {time.time() - t0:.0f}s', flush=True)
    save()
    if time.time() - t0 > HOURS * 3600 or (FIXED and step >= FIXED):
        print(f'stopped at step {step} ({"PF_STEPS" if FIXED and step >= FIXED else "time budget reached"})', flush=True)
        break
