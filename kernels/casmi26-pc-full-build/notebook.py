"""Full PubChem proposal tier (all single-component, net-neutral compounds of 80-1500 Da), built from the PubChem FTP
(CID-Mass, CID-SMILES; public domain) on a Kaggle kernel because the arrays (~7 GB) exceed the dev machine's disk.
Output /kaggle/working/tier_full/: mass.npy (float32, sorted), cid.npy (uint32), smi_off.npy (int64), smi.npy (uint8,
concatenated SMILES). Stereo variants keep separate CIDs (deduplicated by metric key at proposal time)."""
import os, subprocess, time
from array import array
import numpy as np
FTP = 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras'
OUT = '/kaggle/working/tier_full'; os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def stream(name):
    p = subprocess.Popen(f'curl -s {FTP}/{name} | gunzip -c', shell=True, stdout=subprocess.PIPE, bufsize=1 << 24)
    for line in p.stdout:
        t = line.find(b'\t')
        if t > 0:
            yield int(line[:t]), line[t + 1:]
    p.wait()


N = 200_000_000
mono = np.full(N, np.nan, np.float32)
for c, rest in stream('CID-Mass.gz'):                   # CID \t formula \t monoisotopic \t exact
    if c >= N:
        continue
    f = rest.split(b'\t')
    if len(f) >= 2 and b'+' not in f[0] and b'-' not in f[0]:
        try:
            m = float(f[1])
        except ValueError:
            continue
        if 80.0 <= m <= 1500.0:
            mono[c] = m
print('masses', int(np.isfinite(mono).sum()), f'{time.time() - t0:.0f}s', flush=True)
cids, offs, buf = array('I'), array('q', [0]), bytearray()
for c, rest in stream('CID-SMILES.gz'):
    if c >= N or not np.isfinite(mono[c]):
        continue
    s = rest.strip()
    if b'.' in s or b'[2H]' in s or b'[13C]' in s:
        continue
    cids.append(c); buf += s; offs.append(len(buf))
print('smiles kept', len(cids), 'bytes', len(buf), f'{time.time() - t0:.0f}s', flush=True)
cids = np.frombuffer(cids, np.uint32); offs = np.frombuffer(offs, np.int64); raw = np.frombuffer(bytes(buf), np.uint8)
del buf
ms = mono[cids]; del mono
o = np.argsort(ms, kind='stable')
lens = (offs[1:] - offs[:-1])[o]
off = np.zeros(len(o) + 1, np.int64); off[1:] = np.cumsum(lens)
out = np.lib.format.open_memmap(f'{OUT}/smi.npy', mode='w+', dtype=np.uint8, shape=(int(off[-1]),))
for a in range(0, len(o), 2_000_000):
    oc, lc = o[a:a + 2_000_000], lens[a:a + 2_000_000]
    out[off[a]:off[a + len(oc)]] = raw[np.repeat(offs[:-1][oc] - off[a:a + len(oc)], lc) + np.arange(off[a], off[a + len(oc)])]
out.flush(); del out
np.save(f'{OUT}/mass.npy', ms[o]); np.save(f'{OUT}/cid.npy', cids[o]); np.save(f'{OUT}/smi_off.npy', off)
print('tier', len(o), f'{time.time() - t0:.0f}s', flush=True)
