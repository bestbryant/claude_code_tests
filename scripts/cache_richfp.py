"""Raw multi-block fingerprints (casmi/richfp.py) of every pool row referenced by retrieval-row files, cached as CSR.

python scripts/cache_richfp.py <pool_dir> <out.npz> <rows.parquet> ...   -> pid (sorted), ptr, idx (uint16 raw ids),
    pool_n, pool_md5 (pool stamp: row count and md5 of the SMILES column, casmi.richfp.pool_stamp)
python scripts/cache_richfp.py --check <pool_dir> <cache.npz>             exit 0 iff the cache is stamped for this pool
A pure speed cache (scores are identical without it); readers (PFScorer via add_fp_feats.py RICHFP_CACHE,
prep_syn_data.py) refuse a cache without the stamp of their pool. PROCS (default 4).
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.richfp import check_fp_cache, pool_stamp, raw_onbits


def raw_u16(smi):
    return raw_onbits(smi).astype(np.uint16)


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        pool_dir, out = sys.argv[2:4]
        try:
            S = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['smiles']).smiles.values
            check_fp_cache(np.load(out), S, out)
        except Exception as e:          # missing / unreadable file, no stamp, other pool
            sys.exit(f'richfp cache not usable: {e}')
        print('richfp cache ok:', out, 'for', pool_dir)
        sys.exit(0)
    pool_dir, out = sys.argv[1:3]
    pids = np.unique(np.concatenate([pd.read_parquet(p, columns=['pid']).pid.values for p in sys.argv[3:]]))
    S = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['smiles']).smiles.values
    pool_n, pool_md5 = pool_stamp(S)
    smi = S[pids].tolist()
    ptrs, idxs, base = [np.zeros(1, np.int64)], [], 0
    with Pool(int(os.environ.get('PROCS', '4'))) as p:
        for c in range(0, len(smi), 20000):
            part = p.map(raw_u16, smi[c:c + 20000], chunksize=500)
            l = np.array([len(a) for a in part], np.int64)
            ptrs.append(base + np.cumsum(l)); base += int(l.sum()); idxs.append(np.concatenate(part))
            print(c + len(part), 'of', len(smi), flush=True)
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:        # file object: np.savez appends no suffix; replaced only when complete
        np.savez(f, pid=pids.astype(np.int64), ptr=np.concatenate(ptrs),
                 idx=np.concatenate(idxs) if idxs else np.zeros(0, np.uint16),
                 pool_n=np.int64(pool_n), pool_md5=np.str_(pool_md5))
    os.replace(tmp, out)
    print('done', len(pids), 'pool', pool_n, pool_md5[:8])
