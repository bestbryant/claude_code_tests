"""PubChem proposal tier: documented PubChem compounds outside the pool, mass-sorted, for a 10 ppm proposal channel.

python scripts/build_pubchem_tier.py <pubchem_dir> <pool_dir> <out_dir> [min_sid] [min_pmid]
pubchem_dir holds cid_ik.npy (CID -> packed InChIKey14) and cnt_sid / cnt_pmid / cnt_patent.npy (per-CID record
counts). A CID is kept when sid >= min_sid or pmid >= min_pmid, its key is new to the pool, its SMILES is a single
component with a net-neutral formula and its monoisotopic mass is in 80-1500 Da; one CID per key (the most documented). Masses come
from PubChem's CID-Mass, SMILES from CID-SMILES (both streamed from the PubChem FTP).
SEL_CIDS=<cids.npy>: keep exactly these CIDs (key new to the pool) instead of the documentation cut, e.g. the CIDs with
an Enamine deposit (scripts/pc_sid_sources.py) for the Enamine tier.
Writes <out_dir>/tier.npz: mass (float64, sorted), cid (uint32), key (uint64), pop (float32 = log1p sid + log1p pmid
+ 0.5 log1p patent), smi_off (int64), smi (uint8 concatenated SMILES).
"""
import os, subprocess, sys, time
from array import array
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casmi.keyset import pack

FTP = 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras'
pcd, pool_dir, out = sys.argv[1:4]
min_sid = int(sys.argv[4]) if len(sys.argv) > 4 else 5
min_pmid = int(sys.argv[5]) if len(sys.argv) > 5 else 1
os.makedirs(out, exist_ok=True)
t0 = time.time()
ik = np.load(f'{pcd}/cid_ik.npy', mmap_mode='r')
sid_m = np.load(f'{pcd}/cnt_sid.npy', mmap_mode='r'); pmid_m = np.load(f'{pcd}/cnt_pmid.npy', mmap_mode='r')
have = np.unique(pack(list(pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['key']).key)))
sel = np.zeros(len(ik), bool)
pick = None
if os.environ.get('SEL_CIDS'):
    c = np.load(os.environ['SEL_CIDS']).astype(np.int64)
    pick = np.zeros(len(ik), bool); pick[c[c < len(ik)]] = True
for a in range(0, len(ik), 10_000_000):             # chunked: the per-CID arrays are 200M long
    k = np.asarray(ik[a:a + 10_000_000])
    if pick is not None:
        m = pick[a:a + len(k)] & (k > 0)
    else:
        m = ((np.asarray(sid_m[a:a + 10_000_000]) >= min_sid) | (np.asarray(pmid_m[a:a + 10_000_000]) >= min_pmid)) & (k > 0)
    j = np.minimum(np.searchsorted(have, k), len(have) - 1)
    sel[a:a + len(k)] = m & (have[j] != k)
print('selected CIDs', int(sel.sum()), f'{time.time() - t0:.0f}s', flush=True)


def stream(name):
    p = subprocess.Popen(f'curl -s {FTP}/{name} | gunzip -c', shell=True, stdout=subprocess.PIPE, bufsize=1 << 24)
    for line in p.stdout:
        t = line.find(b'\t')
        if t <= 0:
            continue
        c = int(line[:t])
        if c < len(sel) and sel[c]:
            yield c, line[t + 1:]
    p.wait()


mono = np.full(len(sel), np.nan, np.float32)          # per-CID monoisotopic mass (selected CIDs; 0.1 ppm precision)
for c, rest in stream('CID-Mass.gz'):                    # CID \t formula \t monoisotopic mass \t exact mass
    f = rest.split(b'\t')
    if len(f) >= 2 and b'+' not in f[0] and b'-' not in f[0]:      # net-charged formulas (e.g. C9H18NO4+) dropped
        try:
            mono[c] = float(f[1])
        except ValueError:
            pass
print('masses', int(np.isfinite(mono).sum()), f'{time.time() - t0:.0f}s', flush=True)
cids, offs, buf = array('q'), array('q', [0]), bytearray()
for c, rest in stream('CID-SMILES.gz'):
    s = rest.strip()
    m = mono[c]
    if not 80 <= m <= 1500 or b'.' in s:              # NaN (no neutral mass) fails the range test too
        continue
    cids.append(c); buf += s; offs.append(len(buf))
print('smiles kept', len(cids), f'{time.time() - t0:.0f}s', flush=True)
cids = np.frombuffer(cids, np.int64); ms = mono[cids].astype(np.float64); del mono
offs = np.frombuffer(offs, np.int64); raw = np.frombuffer(bytes(buf), np.uint8); del buf
keys = np.asarray(ik[cids])
patent = np.asarray(np.load(f'{pcd}/cnt_patent.npy', mmap_mode='r')[cids])
pop = (np.log1p(np.asarray(sid_m[cids])) + np.log1p(np.asarray(pmid_m[cids])) + 0.5 * np.log1p(patent)).astype(np.float32)
o = np.lexsort((-pop, keys))                              # one CID per key: the most documented
first = np.r_[True, keys[o][1:] != keys[o][:-1]]
o = o[first]
o = o[np.argsort(ms[o], kind='stable')]
lens = (offs[1:] - offs[:-1])[o]
off = np.zeros(len(o) + 1, np.int64); off[1:] = np.cumsum(lens)
buf = np.empty(off[-1], np.uint8)
for a in range(0, len(o), 1_000_000):                     # reorder the SMILES bytes in chunks (bounded memory)
    oc, lc = o[a:a + 1_000_000], lens[a:a + 1_000_000]
    buf[off[a]:off[a + len(oc)]] = raw[np.repeat(offs[:-1][oc] - off[a:a + len(oc)], lc) + np.arange(off[a], off[a + len(oc)])]
np.savez(f'{out}/tier.npz', mass=ms[o], cid=cids[o].astype(np.uint32), key=keys[o], pop=pop[o], smi_off=off, smi=buf)
print('tier', len(o), 'smiles bytes', len(buf), f'{time.time() - t0:.0f}s', flush=True)
