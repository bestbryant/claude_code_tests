"""Spectrum <-> structure dual encoder on frozen DreaMS embeddings (casmi/dreams_emb.py library shards).

python scripts/train_dreams_fp.py <pool_dir> <qemb_cache.npz> <val_rows_fp.parquet> <out.pt> <shard.npz> ...
Training spectra: the DreaMS library representatives of structures in FP-data split 0 only (simulation truths, split
1 / 2, are never trained on). Loss: in-batch contrastive (spectrum embedding vs Morgan-2048 structure embedding)
+ fingerprint BCE. Validation: fingerprint-only ranking of the truth among each validation query's candidates
(mean query embedding), compared with the binned-spectrum model's fp_cos.
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

pool_dir, qcache, val_rows, out = sys.argv[1:5]
shards = sys.argv[5:]
torch.manual_seed(0); rng = np.random.default_rng(0)
torch.set_num_threads(int(os.environ.get('THREADS', '4')))
NB = 2048


class DreamsDual(nn.Module):
    def __init__(self, d_in=1024, d=1024, emb=256, drop=0.1):
        super().__init__()
        self.spec = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.GELU())
        self.spec_emb = nn.Linear(d, emb)
        self.fp_head = nn.Linear(d, NB)
        self.mol = nn.Sequential(nn.Linear(NB, d), nn.GELU(), nn.Linear(d, emb))
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float32))

    def encode_spec(self, x):
        h = self.spec(x)
        return F.normalize(self.spec_emb(h), dim=-1), self.fp_head(h)

    def encode_mol(self, bits):
        return F.normalize(self.mol(bits), dim=-1)


if __name__ == '__main__':
    meta = pd.read_parquet('/home/user/data/train_meta.parquet', columns=['inchikey14'])
    rows = np.concatenate([np.load(p)['row'] for p in shards])
    X = np.concatenate([np.load(p)['emb'] for p in shards]).astype(np.float32)
    keys = meta.inchikey14.values[rows]
    T = np.load('/home/user/ds/casmi26-fp-data/structs.npz', allow_pickle=True)
    S = np.load('/home/user/ds/casmi26-fp-data/spectra.npz')
    excl = set(T['key'][np.unique(S['sidx'][S['split'] > 0])])
    P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['key'])
    key_idx = {k: i for i, k in enumerate(P.key.values)}
    fp = np.load(f'{pool_dir}/pool_fp.npy', mmap_mode='r')
    ok = np.array([k not in excl and k in key_idx for k in keys])
    X, keys = X[ok], keys[ok]
    kid = np.array([key_idx[k] for k in keys])
    bits = lambda ix: torch.from_numpy(np.unpackbits(np.asarray(fp[ix]), axis=-1)[:, :NB].astype(np.float32))
    print('training spectra', len(X), 'structures', len(set(keys)), flush=True)
    model = DreamsDual()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    B, EPOCHS = 1024, int(os.environ.get('EPOCHS', '15'))
    steps = EPOCHS * (len(X) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    t0 = time.time(); step = 0
    Xt = torch.from_numpy(X)
    for ep in range(EPOCHS):
        perm = rng.permutation(len(X))
        for i in range(0, len(perm) - B + 1, B):
            b = perm[i:i + B]
            e_s, logits = model.encode_spec(Xt[b])
            pb = bits(kid[b])
            e_m = model.encode_mol(pb)
            tau = model.log_tau.exp().clamp(0.01, 1.0)
            sim = e_s @ e_m.T / tau
            same = torch.from_numpy(kid[b][:, None] == kid[b][None, :])
            sim = sim.masked_fill(same & ~torch.eye(B, dtype=torch.bool), -1e4)
            loss = F.cross_entropy(sim, torch.arange(B)) + 2.0 * F.binary_cross_entropy_with_logits(logits, pb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        print(f'epoch {ep} loss {loss.item():.3f} tau {tau.item():.3f} {time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), out)

    # validation: fingerprint-only ranking among each validation query's candidates
    z = np.load(qcache); QE = dict(zip(z['row'].tolist(), z['emb']))
    V = pd.read_parquet(val_rows, columns=['qid', 'pid', 'label', 'cls', 'q_rows', 'fp_cos'])
    model.eval(); rr_d, rr_f = [], []
    with torch.no_grad():
        for q, g in V.groupby('qid'):
            if g.label.max() < 1:
                continue
            qr = [int(x) for x in g.q_rows.iloc[0].split(',')]
            e = torch.from_numpy(np.stack([QE[r] for r in qr]).astype(np.float32))
            es, _ = model.encode_spec(e)
            es = F.normalize(es.mean(0, keepdim=True), dim=-1)
            em = model.encode_mol(bits(g.pid.values))
            sc = (es @ em.T).squeeze(0).numpy()
            t = int(np.where(g.label.values == 1)[0][0])
            rr_d.append(1.0 / (1 + (sc > sc[t]).sum()))
            rr_f.append(1.0 / (1 + (g.fp_cos.values > g.fp_cos.values[t]).sum()))
    rr_d, rr_f = np.array(rr_d), np.array(rr_f)
    print(f'validation queries {len(rr_d)}: DreaMS-head MRR {rr_d.mean():.4f} top1 {(rr_d == 1).mean():.4f} | '
          f'binned fp_cos MRR {rr_f.mean():.4f} top1 {(rr_f == 1).mean():.4f}')
