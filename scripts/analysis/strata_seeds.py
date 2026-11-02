"""Split the saner seed-ensemble holdout gain by strata (e180 / coco / tims) - ours, data only."""
import sys, numpy as np, pandas as pd
S = sys.argv[1]
mol = pd.read_parquet(f'{S}/saner_kit/molecules.parquet', columns=['key']).key.values
st = pd.read_parquet(f'{S}/anaC/val1000_strata.parquet')
seeds = [20261023, 20262023, 20263023, 20264023, 20265023]
Z = {s: np.load(f'{S}/seeds_scores/saner_seeds/scores/holdout_{s}.npz') for s in seeds}
qs = sorted(Z[seeds[0]].files, key=int)
def rr(sc):   # gold = index 0; rank with ties pessimistic-mid as kit? use strict greater count + 1
    r = 1 + (sc[1:] > sc[0]).sum()
    return 1.0 / r if r <= 25 else 0.0
def z(x): return (x - x.mean()) / (x.std() + 1e-9)
rows = []
for q in qs:
    k = mol[int(q)]
    a = {s: Z[s][q].astype(np.float64) for s in seeds}
    row = dict(key=k, pub=rr(a[seeds[0]]), mean_member=np.mean([rr(a[s]) for s in seeds]),
               ens5_raw=rr(np.mean([a[s] for s in seeds], 0)), ens5_z=rr(np.mean([z(a[s]) for s in seeds], 0)),
               ens2_pub_s2=rr(z(a[seeds[0]]) + z(a[seeds[1]])))
    rows.append(row)
d = pd.DataFrame(rows).set_index('key').join(st[['e180', 'coco', 'tims_any', 'tims_only']], how='left')
print('queries', len(d), 'with strata', d.e180.notna().sum())
def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return round(x.mean(), 4), [round(np.percentile(m, 2.5), 4), round(np.percentile(m, 97.5), 4)]
for name, mask in [('all', d.index == d.index), ('e180', d.e180 == True), ('not e180', d.e180 == False),
                   ('coco', d.coco == True), ('tims_any', d.tims_any == True), ('not tims', d.tims_any == False)]:
    x = d[mask]
    print(f'{name:9s} n={len(x):4d} pub {x.pub.mean():.4f} | ens5_z - pub {boot(x.ens5_z - x.pub)} | ens5_z - mean member '
          f'{boot(x.ens5_z - x.mean_member)} | ens2(pub+s2) - pub {boot(x.ens2_pub_s2 - x.pub)}')
