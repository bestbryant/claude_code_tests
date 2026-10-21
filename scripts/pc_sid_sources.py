"""Per-CID depositor sources from PubChem's SID-Map (SID, source name, registry id, standardized CID), streamed from
the FTP. Writes <out_dir>/sidmap_cid.npy (uint32) and sidmap_src.npy (uint16) for every SID with a CID (one row per
deposit, duplicates kept) and sidmap_sources.txt (source names, index = src code).

python scripts/pc_sid_sources.py <out_dir>
"""
import subprocess, sys, time
import numpy as np, pandas as pd

out = sys.argv[1]
t0 = time.time()
p = subprocess.Popen("curl -s https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/Extras/SID-Map.gz | gunzip -c"
                     " | awk -F'\\t' 'NF >= 4 && $4 != \"\" {print $4 \"\\t\" $2}'", shell=True,
                     stdout=subprocess.PIPE, bufsize=1 << 24)
codes, cids, srcs = {}, [], []
for ch in pd.read_csv(p.stdout, sep='\t', header=None, names=['cid', 'src'], dtype={'src': str, 'cid': 'float64'},
                      chunksize=10_000_000, quoting=3, on_bad_lines='skip'):
    ch = ch[ch.cid.notna()]
    u = ch.src.unique()
    for s in u:
        if s not in codes:
            codes[s] = len(codes)
    cids.append(ch.cid.values.astype(np.uint32)); srcs.append(ch.src.map(codes).values.astype(np.uint16))
    print('rows', sum(map(len, cids)), 'sources', len(codes), f'{time.time() - t0:.0f}s', flush=True)
p.wait()
np.save(f'{out}/sidmap_cid.npy', np.concatenate(cids)); np.save(f'{out}/sidmap_src.npy', np.concatenate(srcs))
open(f'{out}/sidmap_sources.txt', 'w').write('\n'.join(sorted(codes, key=codes.get)))
print('done', f'{time.time() - t0:.0f}s')
