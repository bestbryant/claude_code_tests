"""Vendor-deposit features per pool structure (pool_dir/vendor.parquet, rows aligned with pool.parquet).

python scripts/build_vendor_feats.py <pool_dir> <sidmap_dir> <cid_ik.npy>
The test molecules are physical samples measured by Enveda (100% timsTOF): Enamine's catalogue (enveda-180 structures:
94.8% Enamine-deposited in PubChem vs 4% of PubChem) and bioactive / natural-product vendor libraries (enveda-np-examples:
MedChemExpress 96%, TargetMol 94%, BioCrick 79%, Selleck 60% vs 0.1-4% of COCONUT-not-train). Columns: v_<vendor> = the
structure (any CID of its InChIKey14) has a deposit from that source; v_bioact = number of the bioactive / NP vendors;
v_nsrc = log1p number of distinct depositors. Depositors from scripts/pc_sid_sources.py (PubChem SID-Map).
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from casmi.keyset import pack
from casmi.retrieval import VENDORS, BIOACT

if __name__ == '__main__':
    pool_dir, sd, ikp = sys.argv[1:4]
    P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['key'])
    names = open(f'{sd}/sidmap_sources.txt').read().split('\n')
    allk = np.load(ikp, mmap_mode='r')
    pk = pack(list(P.key)); o = np.argsort(pk); spk = pk[o]
    row_of_cid = np.full(len(allk), -1, np.int32)                     # CID -> pool row or -1 (chunked: 200M CIDs)
    for a in range(0, len(allk), 20_000_000):
        b = np.asarray(allk[a:a + 20_000_000]); j = np.minimum(np.searchsorted(spk, b), len(spk) - 1)
        row_of_cid[a:a + len(b)] = np.where((spk[j] == b) & (b > 0), o[j], -1)
    cid = np.load(f'{sd}/sidmap_cid.npy', mmap_mode='r'); src = np.load(f'{sd}/sidmap_src.npy', mmap_mode='r')
    parts = []
    for a in range(0, len(cid), 20_000_000):
        c = np.asarray(cid[a:a + 20_000_000]).astype(np.int64); s_ = np.asarray(src[a:a + 20_000_000]).astype(np.int64)
        ok = c < len(row_of_cid); r = np.full(len(c), -1, np.int64); r[ok] = row_of_cid[c[ok]]
        sel = r >= 0
        parts.append(np.unique(r[sel] * 4096 + s_[sel]))
    pairs = np.unique(np.concatenate(parts))
    pr, ps = pairs // 4096, pairs % 4096
    V = pd.DataFrame(index=np.arange(len(P)))
    for c, n in VENDORS.items():
        f = np.zeros(len(P), np.int8)
        if n in names:
            f[pr[ps == names.index(n)]] = 1
        V[c] = f
    V['v_bioact'] = V[BIOACT].sum(1).astype(np.int8)
    V['v_nsrc'] = np.log1p(np.bincount(pr, minlength=len(P))).astype(np.float32)
    V.to_parquet(f'{pool_dir}/vendor.parquet')
    print('pool rows', len(P), 'with any deposit', int((V.v_nsrc > 0).sum()))
    print(V.mean().round(4).to_dict())
