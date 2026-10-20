"""Raw multi-block fingerprints (casmi/richfp.py) of every pool row referenced by retrieval-row files, cached as CSR.

python scripts/cache_richfp.py <pool_dir> <out.npz> <rows.parquet> ...     -> pid (sorted), ptr, idx (uint16 raw ids)
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.richfp import raw_onbits


def raw_u16(smi):
    return raw_onbits(smi).astype(np.uint16)


if __name__ == '__main__':
    pool_dir, out = sys.argv[1:3]
    pids = np.unique(np.concatenate([pd.read_parquet(p, columns=['pid']).pid.values for p in sys.argv[3:]]))
    smi = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['smiles']).smiles.values[pids].tolist()
    ptrs, idxs, base = [np.zeros(1, np.int64)], [], 0
    with Pool(int(os.environ.get('PROCS', '4'))) as p:
        for c in range(0, len(smi), 20000):
            part = p.map(raw_u16, smi[c:c + 20000], chunksize=500)
            l = np.array([len(a) for a in part], np.int64)
            ptrs.append(base + np.cumsum(l)); base += int(l.sum()); idxs.append(np.concatenate(part))
            print(c + len(part), 'of', len(smi), flush=True)
    np.savez(out, pid=pids.astype(np.int64), ptr=np.concatenate(ptrs), idx=np.concatenate(idxs))
    print('done', len(pids))
