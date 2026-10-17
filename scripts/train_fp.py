"""Train the SparseDualEncoder on prepared arrays (scripts/prep_fp_data.py). Runs on CPU or CUDA.

Usage: python scripts/train_fp.py <data_dir> <out.pt> [epochs] [steps_per_epoch] [batch] [n_neg] [threads]
Validation: for held-out structures (split 1), mean spectrum embedding of up to 3 spectra vs the truth and its
window negatives -> MRR and top-1 (class-2-like retrieval among same-mass candidates).
"""
import sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/user/claude_code_tests'); sys.path.insert(0, '/kaggle/working/pkg')
from casmi.fpmodel import SparseDualEncoder, sparse_spec, sparse_bits, unpack_bits

data, out = sys.argv[1], sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
B = int(sys.argv[5]) if len(sys.argv) > 5 else 512
NNEG = int(sys.argv[6]) if len(sys.argv) > 6 else 15
if len(sys.argv) > 7:
    torch.set_num_threads(int(sys.argv[7]))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); rng = np.random.default_rng(0)
S = np.load(f'{data}/spectra.npz'); T = np.load(f'{data}/structs.npz')
negs = np.load(f'{data}/negs.npy'); negfp = np.load(f'{data}/negfp.npy')
mz, it, off = S['mz'], S['it'], S['off']
prec, ce, add_id, inst_id, pol, sidx, split = S['prec'], S['ce'], S['add_id'], S['inst_id'], S['pol'], S['sidx'], S['split']
sfp = T['fp']
tr = np.where(split == 0)[0]
w = np.where(inst_id[tr] == 1, 2.0, 1.0); w /= w.sum()          # timsTOF spectra sampled twice as often
model = SparseDualEncoder().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
total = epochs * steps
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.05)
T_ = lambda a: torch.from_numpy(a).to(dev)


def spec_inputs(idx):
    ids, bo, wts, meta = sparse_spec(mz, it, off, idx, prec, ce, add_id, inst_id, pol)
    return T_(ids), T_(bo), T_(wts), T_(meta)


def mol_inputs(packed):
    ids, bo = sparse_bits(packed)
    return T_(ids), T_(bo)


def validate(max_structs=1500):
    model.eval()
    vi = np.where(split == 1)[0]
    by = {}
    for i in vi:
        by.setdefault(sidx[i], []).append(i)
    keys = list(by)[:max_structs]
    rr = []
    with torch.no_grad():
        for s in keys:
            ii = np.array(by[s][:3])
            e, _ = model.encode_spec(*spec_inputs(ii))
            e = F.normalize(e.mean(0, keepdim=True), dim=-1)
            ng = negs[s][negs[s] >= 0]
            m = model.encode_mol(*mol_inputs(np.concatenate([sfp[s:s + 1], negfp[ng]], 0)))
            sc = (e @ m.T).squeeze(0).cpu().numpy()
            rr.append(1.0 / (1 + (sc[1:] > sc[0]).sum()))
    model.train()
    rr = np.array(rr)
    return rr.mean(), (rr == 1).mean(), len(rr)


t0 = time.time(); step = 0
for ep in range(epochs):
    for _ in range(steps):
        idx = rng.choice(tr, size=B, p=w)
        s = sidx[idx]
        nl = negs[s]
        cnt = (nl >= 0).sum(1)
        # sample NNEG negatives per row from the valid prefix (lists are left-packed)
        r = rng.random((B, NNEG))
        pick_pos = np.minimum((r * np.maximum(cnt, 1)[:, None]).astype(np.int64), 63)
        pick = np.take_along_axis(nl, pick_pos, 1).astype(np.int64)
        pick[cnt == 0] = -1
        e_s, logits = model.encode_spec(*spec_inputs(idx))
        e_pos = model.encode_mol(*mol_inputs(sfp[s]))
        pos_bits = T_(unpack_bits(sfp[s]))
        tau = model.log_tau.exp().clamp(0.01, 1.0)
        sim_b = e_s @ e_pos.T / tau
        same = T_(s[:, None] == s[None, :])
        sim_b = sim_b.masked_fill(same & ~torch.eye(B, dtype=torch.bool, device=dev), -1e4)
        loss_b = F.cross_entropy(sim_b, torch.arange(B, device=dev))
        valid = T_(pick >= 0)
        e_neg = model.encode_mol(*mol_inputs(negfp[np.maximum(pick, 0).reshape(-1)])).view(B, NNEG, -1)
        s_pos = (e_s * e_pos).sum(-1, keepdim=True) / tau
        s_neg = torch.einsum('bd,bnd->bn', e_s, e_neg) / tau
        s_neg = s_neg.masked_fill(~valid, -1e4)
        loss_h = F.cross_entropy(torch.cat([s_pos, s_neg], 1), torch.zeros(B, dtype=torch.long, device=dev))
        loss_fp = F.binary_cross_entropy_with_logits(logits, pos_bits)
        loss = loss_b + loss_h + 2.0 * loss_fp
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        if step % 200 == 0:
            print(f'step {step}/{total} loss {loss.item():.3f} (b {loss_b.item():.2f} h {loss_h.item():.2f} fp {loss_fp.item():.3f}) '
                  f'tau {tau.item():.3f} {time.time()-t0:.0f}s', flush=True)
    mrr, top1, n = validate()
    print(f'EPOCH {ep} val MRR {mrr:.4f} top1 {top1:.4f} (n={n}) {time.time()-t0:.0f}s', flush=True)
    torch.save(model.state_dict(), out)
