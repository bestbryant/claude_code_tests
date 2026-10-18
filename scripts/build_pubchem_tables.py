"""PubChem documentation counts per connectivity key (InChIKey first block), streamed from the PubChem FTP.

Outputs in <out>:
  cid_ik.npy      packed InChIKey14 (first 13 letters, base 26, uint64) per CID (0 = no key), index = CID
  cnt_sid.npy, cnt_pmid.npy, cnt_patent.npy   uint32 counts per CID
  key_pop.parquet one row per packed key: max counts over its CIDs + n_cids
Usage: python scripts/build_pubchem_tables.py <out_dir> [steps: keys,sid,pmid,patent,agg]
"""
import gzip, os, subprocess, sys, time
import numpy as np, pandas as pd

FTP = 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras'
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
steps = sys.argv[2].split(',') if len(sys.argv) > 2 else ['keys', 'sid', 'pmid', 'patent', 'agg']
MAXCID = 200_000_000


def stream(name):
    """Yield decoded lines of a gzipped PubChem file via curl | gunzip."""
    p = subprocess.Popen(f'curl -s {FTP}/{name} | gunzip -c', shell=True, stdout=subprocess.PIPE, bufsize=1 << 24)
    return p


def pack13(arr):
    a = np.frombuffer(b''.join(arr), dtype=np.uint8).reshape(-1, 14)[:, :13].astype(np.uint64) - 65
    v = np.zeros(len(a), np.uint64)
    for c in range(13):
        v = v * np.uint64(26) + a[:, c]
    return v


t0 = time.time()
if 'keys' in steps:
    cid_ik = np.zeros(MAXCID, np.uint64)
    p = stream('CID-InChI-Key.gz')
    buf_c, buf_k = [], []
    n = 0
    for line in p.stdout:
        a = line.split(b'\t')
        if len(a) < 3:
            continue
        k = a[2][:14]
        if len(k) != 14 or not k.isalpha():
            continue
        buf_c.append(int(a[0])); buf_k.append(k)
        if len(buf_c) >= 2_000_000:
            c = np.array(buf_c); keep = c < MAXCID
            cid_ik[c[keep]] = pack13([x for x, kk in zip(buf_k, keep) if kk])
            n += len(buf_c); buf_c, buf_k = [], []
            print('keys', n, f'{time.time()-t0:.0f}s', flush=True)
    if buf_c:
        c = np.array(buf_c); keep = c < MAXCID
        cid_ik[c[keep]] = pack13([x for x, kk in zip(buf_k, keep) if kk])
    np.save(f'{out}/cid_ik.npy', cid_ik)
    print('keys done', n, f'{time.time()-t0:.0f}s', flush=True)

for step, name in (('sid', 'CID-SID.gz'), ('pmid', 'CID-PMID.gz'), ('patent', 'CID-Patent.gz')):
    if step not in steps:
        continue
    cnt = np.zeros(MAXCID, np.uint32)
    p = stream(name)
    buf = []
    n = 0
    for line in p.stdout:
        t = line.find(b'\t')
        if t <= 0:
            continue
        buf.append(int(line[:t]))
        if len(buf) >= 5_000_000:
            c = np.array(buf); c = c[c < MAXCID]
            cnt += np.bincount(c, minlength=MAXCID).astype(np.uint32)
            n += len(buf); buf = []
            print(step, n, f'{time.time()-t0:.0f}s', flush=True)
    if buf:
        c = np.array(buf); c = c[c < MAXCID]
        cnt += np.bincount(c, minlength=MAXCID).astype(np.uint32)
    np.save(f'{out}/cnt_{step}.npy', cnt)
    print(step, 'done', n, f'{time.time()-t0:.0f}s', flush=True)

if 'agg' in steps:
    cid_ik = np.load(f'{out}/cid_ik.npy', mmap_mode='r')
    has = np.nonzero(np.asarray(cid_ik))[0]
    keys = np.asarray(cid_ik)[has]
    df = pd.DataFrame(dict(key=keys))
    for step in ('sid', 'pmid', 'patent'):
        f = f'{out}/cnt_{step}.npy'
        if os.path.exists(f):
            df[step] = np.load(f, mmap_mode='r')[has]
    g = df.groupby('key')
    pop = g.max()
    pop['n_cids'] = g.size()
    pop.reset_index().to_parquet(f'{out}/key_pop.parquet')
    print('agg done', len(pop), f'{time.time()-t0:.0f}s', flush=True)
