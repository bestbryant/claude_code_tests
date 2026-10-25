"""PubChem documentation counts per connectivity key (InChIKey first block), streamed from the PubChem FTP.

Outputs in <out>:
  cid_ik.npy      packed InChIKey14 (first 13 letters, base 26, uint64) per CID (0 = no key), index = CID
  cnt_sid.npy, cnt_pmid.npy, cnt_patent.npy   uint32 counts per CID
  key_pop.parquet one row per packed key: max counts over its CIDs + n_cids
  manifest.json   inputs: per Extras file URL, Last-Modified, ETag, bytes, md5 (checked against PubChem's <file>.md5
                  and the Content-Length), fetch time; outputs: bytes, md5 (+ rows) of every file written (G11: PubChem
                  regenerates the Extras in place, so this is the only record of the snapshot an output came from)
Usage: python scripts/build_pubchem_tables.py <out_dir> [steps: keys,sid,pmid,patent,agg]
PUBCHEM_FTP (mirror, or a file:// dir with the same names and .md5 files) and PUBCHEM_MAXCID (tests only) override.
The Oct 18 2026 snapshot gave key_pop.parquet md5 ce9c3671 (105,880,815 keys) and cid_ik.npy md5 0240a0bb;
CID-InChI-Key, CID-SID and CID-Patent were regenerated on Oct 24, so the live FTP gives other counts (G11).
"""
import hashlib, json, os, subprocess, sys, threading, time
import numpy as np, pandas as pd

FTP = os.environ.get('PUBCHEM_FTP', 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras')
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
steps = sys.argv[2].split(',') if len(sys.argv) > 2 else ['keys', 'sid', 'pmid', 'patent', 'agg']
MAXCID = int(os.environ.get('PUBCHEM_MAXCID', 200_000_000))
OCT18 = {'key_pop.parquet': 'ce9c3671d1d8f8521c5477b02f417aa7', 'cid_ik.npy': '0240a0bb3ad54c39ad1795abd64c2817'}


def md5file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def manifest(section, name, **rec):
    """Merge one entry into <out>/manifest.json (steps may run in separate invocations)."""
    path = f'{out}/manifest.json'
    m = json.load(open(path)) if os.path.exists(path) else {}
    m.setdefault(section, {})[name] = rec
    with open(path + '.tmp', 'w') as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(path + '.tmp', path)


def saved(name, **extra):
    """Record an output file in the manifest; note whether it matches the Oct 18 snapshot."""
    p = f'{out}/{name}'
    h = md5file(p)
    manifest('outputs', name, bytes=os.path.getsize(p), md5=h, **extra)
    if name in OCT18:
        print(name, 'md5', h, 'same as' if h == OCT18[name] else 'differs from', 'the Oct 18 snapshot', flush=True)


def stream(name):
    """Yield the decoded lines of a gzipped PubChem file (curl | gunzip -c, streamed). Size and md5 of the .gz are taken
    on the way and must match PubChem's <name>.md5 and the Content-Length (a truncated or replaced download fails)."""
    url = f'{FTP}/{name}'
    want = subprocess.run(['curl', '-sSfL', url + '.md5'], capture_output=True, check=True).stdout.split()[0].decode()
    hdr, t = f'{out}/.{name}.headers', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    curl = subprocess.Popen(['curl', '-sSfL', '-D', hdr, url], stdout=subprocess.PIPE)
    gz = subprocess.Popen(['gunzip', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 << 24)
    h, size = hashlib.md5(), [0]

    def pump():
        try:
            for b in iter(lambda: curl.stdout.read(1 << 22), b''):
                h.update(b); size[0] += len(b); gz.stdin.write(b)
        finally:
            gz.stdin.close(); curl.stdout.close()
    th = threading.Thread(target=pump, daemon=True); th.start()
    yield from gz.stdout
    th.join()
    rc = curl.wait(), gz.wait()
    assert rc == (0, 0), f'{name}: curl / gunzip exit {rc}'
    hd = {}
    for line in open(hdr, encoding='latin-1'):        # keep the last response (a proxy's CONNECT reply comes first)
        if line.startswith('HTTP/'):
            hd = {}
        elif ':' in line:
            k, v = line.split(':', 1); hd[k.strip().lower()] = v.strip()
    os.remove(hdr)
    md5 = h.hexdigest()
    assert md5 == want, f'{name}: md5 {md5} != published {want}'
    assert int(hd.get('content-length', size[0])) == size[0], f'{name}: {size[0]} B != Content-Length'
    manifest('inputs', name, url=url, bytes=size[0], md5=md5, last_modified=hd.get('last-modified'),
             etag=hd.get('etag'), fetched=t)
    print(name, size[0], 'B md5', md5, 'Last-Modified', hd.get('last-modified'), flush=True)


def pack13(arr):
    a = np.frombuffer(b''.join(arr), dtype=np.uint8).reshape(-1, 14)[:, :13].astype(np.uint64) - 65
    v = np.zeros(len(a), np.uint64)
    for c in range(13):
        v = v * np.uint64(26) + a[:, c]
    return v


t0 = time.time()
if 'keys' in steps:
    cid_ik = np.zeros(MAXCID, np.uint64)
    buf_c, buf_k = [], []
    n = 0
    for line in stream('CID-InChI-Key.gz'):
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
        n += len(buf_c)
    np.save(f'{out}/cid_ik.npy', cid_ik)
    saved('cid_ik.npy')
    print('keys done', n, f'{time.time()-t0:.0f}s', flush=True)

for step, name in (('sid', 'CID-SID.gz'), ('pmid', 'CID-PMID.gz'), ('patent', 'CID-Patent.gz')):
    if step not in steps:
        continue
    cnt = np.zeros(MAXCID, np.uint32)
    buf = []
    n = 0
    for line in stream(name):
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
        n += len(buf)
    np.save(f'{out}/cnt_{step}.npy', cnt)
    saved(f'cnt_{step}.npy')
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
    saved('key_pop.parquet', rows=len(pop), columns=list(pop.reset_index().columns))
    print('agg done', len(pop), f'{time.time()-t0:.0f}s', flush=True)
