"""Engine 2 clone: locate and verify the eligible candidate bank (scripts/build_e2_bank.py on pool v18c; Kaggle kernel
output bryantyu/casmi26-e2-bank). Our own code.

The published Engine 2 finds its bank with find(), which takes the SHORTEST matching path under /kaggle/input
(E2:casmi_engine.py L41-45). In the eligible fork ahmedberatozer/casmi26-v2-pool also ships an fp_bits.npy, so the clone
never searches by file name: find_bank() picks the one directory whose MANIFEST.json says it is our bank built on pool
v18c, and check_bank() verifies every array against that manifest before the engine loads it (patch E5).
"""
import glob
import hashlib
import json
import os

import numpy as np

from . import config as C


def digest(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _manifest(path):
    try:
        with open(path) as f:
            m = json.load(f)
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


def find_bank(root, pool_md5=C.POOL_MD5):
    """The directory under root whose MANIFEST.json is an Engine 2 clone bank built on pool v18c. Exactly one must
    match; the choice never depends on path length."""
    hits = []
    for p in sorted(glob.glob(os.path.join(root, '**', 'MANIFEST.json'), recursive=True)):
        m = _manifest(p)
        if not m or not str(m.get('what', '')).startswith(C.BANK_WHAT):
            continue
        if (m.get('inputs') or {}).get('pool_md5') != pool_md5:
            continue
        hits.append(os.path.dirname(os.path.abspath(p)))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one Engine 2 clone bank (pool md5 {pool_md5}) under {root}, found {hits}')
    return hits[0]


def check_bank(d, expect=None, soft=None):
    """Hard checks of the bank directory d against its MANIFEST.json; returns nbits (selection.nbits).
    expect: dict(rows, nbits, bits_sha256) (default config.BANK_EXPECT); rows is hard, nbits / bits_sha256 are soft
    (the bank kernel's local expectation). soft: optional list that collects soft-check messages."""
    e = dict(C.BANK_EXPECT, **(expect or {}))
    m = _manifest(os.path.join(d, 'MANIFEST.json'))
    if not m:
        raise RuntimeError(f'{d}: no readable MANIFEST.json')
    bad = []
    if not str(m.get('what', '')).startswith(C.BANK_WHAT):
        bad.append(f"what {m.get('what')!r}")
    if (m.get('inputs') or {}).get('pool_md5') != C.POOL_MD5:
        bad.append(f"pool_md5 {(m.get('inputs') or {}).get('pool_md5')}")
    if m.get('rows') != e['rows']:
        bad.append(f"rows {m.get('rows')} != {e['rows']}")
    if m.get('limit') is not None:
        bad.append(f"limit {m.get('limit')} (a partial build)")
    order = m.get('order') or {}
    if order.get('sorted_here') is not False or order.get('pool_mass_ascending') is not True:
        bad.append(f'order {order} (bank row i must be pool row i)')
    failed = [k for k, g in (m.get('gates') or {}).items() if g.get('hard') and not g.get('ok')]
    if failed:
        bad.append(f'hard gates failed {failed}')
    outs = m.get('outputs') or {}
    for k in C.BANK_FILES:
        o = outs.get(k)
        if not o:
            bad.append(f'outputs[{k}] missing')
            continue
        p = os.path.join(d, o['file'])
        if not os.path.exists(p):
            bad.append(f'{o["file"]} missing')
            continue
        if os.path.getsize(p) != o['bytes']:
            bad.append(f'{o["file"]} bytes {os.path.getsize(p)} != {o["bytes"]}')
        elif digest(p) != o['sha256']:
            bad.append(f'{o["file"]} sha256 differs from MANIFEST')
    nbits = int((m.get('selection') or {}).get('nbits', -1))
    if nbits <= 0:
        bad.append(f'selection.nbits {nbits}')
    if bad:
        raise RuntimeError(f'bank {d} failed checks: ' + '; '.join(bad))
    msgs = []
    if nbits != e['nbits']:
        msgs.append(f'nbits {nbits} != expected {e["nbits"]}')
    if outs['bits']['sha256'] != e['bits_sha256']:
        msgs.append(f"fp_bits sha256 {outs['bits']['sha256'][:12]} != expected {e['bits_sha256'][:12]}")
    if soft is not None:
        soft.extend(msgs)
    for s in msgs:
        print(f'bank soft check: {s}', flush=True)
    return nbits


def check_bank_meta(d, pool_parquet):
    """bank_meta.parquet pool_row == arange(N) and key == pool.parquet key, row by row (hard). Returns N."""
    import pandas as pd
    bm = pd.read_parquet(os.path.join(d, 'bank_meta.parquet'), columns=['pool_row', 'key'])
    pk = pd.read_parquet(pool_parquet, columns=['key']).key.astype(str).values
    if len(bm) != len(pk):
        raise RuntimeError(f'bank_meta rows {len(bm)} != pool rows {len(pk)}')
    if not np.array_equal(bm.pool_row.values.astype(np.int64), np.arange(len(bm), dtype=np.int64)):
        raise RuntimeError('bank_meta.pool_row != arange(N): bank row i is not pool row i')
    if not np.array_equal(bm.key.astype(str).values, pk):
        raise RuntimeError('bank_meta.key differs from pool.parquet key')
    return len(bm)


def bank_summary(d):
    """Small dict for reports: nbits, fp / bits sha256, rows."""
    m = _manifest(os.path.join(d, 'MANIFEST.json')) or {}
    outs = m.get('outputs') or {}
    return dict(dir=d, rows=m.get('rows'), nbits=(m.get('selection') or {}).get('nbits'),
                fp_sha256=(outs.get('fp') or {}).get('sha256'), bits_sha256=(outs.get('bits') or {}).get('sha256'),
                failures=(m.get('failures') or {}).get('count'))
