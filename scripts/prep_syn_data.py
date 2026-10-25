"""Synthetic (GLACIER-simulated) training spectra for the PeakFormer (PF-syn), from the outputs of
scripts/make_sim_workers.py.

python scripts/prep_syn_data.py <structs.parquet> <fpx.npz> <out syn.npz> <sim.npz> [<sim.npz> ...]
structs.parquet: scripts/build_sim_jobs.py (pid, smiles, set). Predictions are matched back to pool ids by InChIKey14
(fwdrun canonicalised the SMILES under another RDKit version). Env: POOL (pool.parquet, default pool v4 in /home/user/work),
RICHFP_CACHE (optional scripts/cache_richfp.py npz, no default; must be stamped for POOL; missing structures are
computed), PREP_PROCS. Writes:
  PM (n, 64) float32 m/z / PI (n, 64) float16 intensity (top 64 predicted peaks <= precursor + 2, base 1),
  prec float64 ([M+H]+ = pool mass + 1.007276), ce float32 (eV), sidx int32 (row of the simulated structure),
  y_ptr / y_idx: on-bits of every simulated structure in the kept multi-block bits of fpx.npz (CSR, uint16),
  wneg (n_struct, 63) int32: same-mass window negatives (10 ppm or 2 mDa) WITHIN the simulated set, same formula first
  (as scripts/prep_fp_data.py), -1 padded -- the synthetic look sees every population as positive and as negative in
  the proportions in which it was simulated;
  pid, key, set (per structure).
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.richfp import N_RAW, raw_onbits

K, NNEG, H = 64, 63, 1.007276


def key14(s):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


if __name__ == '__main__':
    structs_p, fpx_p, out = sys.argv[1:4]
    S = pd.read_parquet(structs_p)
    pool_p = os.environ.get('POOL', '/home/user/work/pool4/pool.parquet')
    P = pd.read_parquet(pool_p, columns=['key', 'mass', 'formula'])
    procs = int(os.environ.get('PREP_PROCS', '4'))
    S['key'] = P.key.values[S.pid.values]
    pid_of = dict(zip(S.key, S.pid))
    keep = np.load(fpx_p)['keep']
    km = -np.ones(N_RAW, np.int64); km[keep] = np.arange(len(keep))
    sims = [np.load(f) for f in sys.argv[4:]]
    smiles = np.concatenate([z['smiles'] for z in sims])
    with Pool(procs) as pool:
        uk = dict(zip(np.unique(smiles), pool.map(key14, np.unique(smiles).tolist(), chunksize=2000)))
    rows = []          # (pid, ce, mz, it)
    for z in sims:
        ptr, mz, it = z['ptr'], z['mz'], z['it']
        for j, (s, ce) in enumerate(zip(z['smiles'], z['ce'])):
            p = pid_of.get(uk.get(s))
            if p is not None:
                rows.append((p, float(ce), mz[ptr[j]:ptr[j + 1]], it[ptr[j]:ptr[j + 1]]))
    print('predictions', len(smiles), 'matched to pool ids', len(rows), flush=True)
    pids = np.unique([r[0] for r in rows])
    row_of = {p: i for i, p in enumerate(pids)}
    mass = P.mass.values[pids]; form = P.formula.to_numpy(dtype=object)[pids]
    n = len(rows)
    PM = np.zeros((n, K), np.float32); PI = np.zeros((n, K), np.float16)
    prec = np.zeros(n); ce = np.zeros(n, np.float32); sidx = np.zeros(n, np.int32)
    for i, (p, c, a, b) in enumerate(rows):
        pr = P.mass.values[p] + H
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float32)
        ok = (a <= pr + 2.0) & (b > 0)
        a, b = a[ok], b[ok]
        if len(a) > K:
            o = np.argsort(-b, kind='stable')[:K]; a, b = a[o], b[o]
        if len(a):
            PM[i, :len(a)] = a; PI[i, :len(a)] = b / b.max()
        prec[i], ce[i], sidx[i] = pr, c, row_of[p]
    # fingerprints: richfp cache (RICHFP_CACHE, refused unless stamped for POOL) else computed
    cp = os.environ.get('RICHFP_CACHE')
    if cp and os.path.exists(cp):
        from casmi.richfp import load_fp_cache
        psmi = pd.read_parquet(pool_p, columns=['smiles']).smiles.values
        C = dict(zip(('pid', 'ptr', 'idx'), load_fp_cache(cp, psmi)))
    else:
        C = dict(pid=np.zeros(0, np.int64), ptr=np.zeros(1, np.int64), idx=np.zeros(0, np.int32))
    cpos = {p: i for i, p in enumerate(C['pid'].tolist())}
    need = [p for p in pids if p not in cpos]
    smi_of = dict(zip(S.pid, S.smiles))
    with Pool(procs) as pool:
        comp = dict(zip(need, pool.map(raw_onbits, [smi_of[p] for p in need], chunksize=500)))
    lists = []
    for p in pids:
        if p in cpos:
            j = cpos[p]; raw = C['idx'][C['ptr'][j]:C['ptr'][j + 1]].astype(np.int64)
        else:
            raw = np.asarray(comp[p], np.int64)
        k = km[raw]; lists.append(np.sort(k[k >= 0]).astype(np.uint16))
    y_ptr = np.r_[0, np.cumsum([len(x) for x in lists])].astype(np.int64)
    y_idx = np.concatenate(lists)
    print('structures', len(pids), f'(cache {len(pids) - len(need)}, computed {len(need)}), on-bits mean {np.diff(y_ptr).mean():.0f}', flush=True)
    # window negatives inside the simulated set
    o = np.argsort(mass, kind='stable'); ms = mass[o]
    wneg = -np.ones((len(pids), NNEG), np.int32)
    rng = np.random.default_rng(0)
    for i in range(len(pids)):
        M = mass[i]; tol = max(M * 1e-5, 0.002)
        c = o[np.searchsorted(ms, M - tol):np.searchsorted(ms, M + tol)]
        c = c[c != i]
        if len(c) == 0:
            continue
        same = c[form[c] == form[i]]; other = c[form[c] != form[i]]
        c = np.concatenate([rng.permutation(same), rng.permutation(other)])[:NNEG]
        wneg[i, :len(c)] = c
    cnt = (wneg >= 0).sum(1)
    sset = S.drop_duplicates('pid').set_index('pid')['set'].reindex(pids).values
    print('window negatives per structure: mean', round(float(cnt.mean()), 1), 'none', int((cnt == 0).sum()),
          '| by set', {s: round(float(cnt[sset == s].mean()), 1) for s in np.unique(sset)}, flush=True)
    np.savez(out, PM=PM, PI=PI, prec=prec, ce=ce, sidx=sidx, y_ptr=y_ptr, y_idx=y_idx, wneg=wneg, pid=pids,
             key=np.array(P.key.values[pids].tolist(), dtype='U'), set=np.array(list(sset), dtype='U'))
    print('wrote', out, 'spectra', n, flush=True)
