"""casmi26-e2-bank: builds the Engine 2 clone candidate bank (docs/engine2_study.md option R step 1) on a Kaggle CPU.

The build is scripts/build_e2_bank.py, embedded below byte for byte (BANK_SRC, sha256 BANK_SHA256; refresh both with
`python scripts/make_e2_bank_kernel.py`, check with `--check`). The copy is sha256-checked, written to
/tmp/e2bank_code/build_e2_bank.py, run in a subprocess, and also kept in the outputs for provenance.
Inputs (no extra upload):
  bryantyu/casmi26-pool-v18c                pool.parquet (md5 8afda40a...; 1,160,133 rows, mass ascending)
  competition train.parquet                 only its inchikey14 column is read: 275,810 training keys, the same set as
                                            /home/user/data/train_meta.parquet (key-set sha256 23731cd9..., checked)
  metric/rdkit-2026-3-3-wheel               RDKit 2026.03.3, the local reference version (refused otherwise)
Outputs (/kaggle/working = the dataset bryantyu/casmi26-e2-bank):
  coco_fp.npy (uint8 N x ceil(nbits/8), ~1.0 GB), coco_mass.npy, coco_meta.pkl, fp_bits.npy, fp_freq.npy,
  bank_meta.parquet, MANIFEST.json, build_e2_bank.py, build_e2_bank.log, env.json.
  The 'coco_' names are those Engine 2's loader reads (E2:casmi_engine.py L147-150); the bank is all of pool v18c.
Temporary disk: N x 1,301 bytes (1.5 GB) under /tmp, deleted by the build. Expected wall time ~15-25 min on 4 CPUs.
Expected selection (local run over the same 275,810 training rows): 6,930 bits = 867 bytes per row, fp_bits.npy sha256
a61e372f...; a difference is reported (soft check), not fatal.
Local plumbing test: E2B_INPUT=<fake input tree> E2B_OUT=<dir> E2B_TMP=<dir> E2B_TEST=1 E2B_LIMIT=2000 python3 notebook.py
(TEST: no RDKit wheel needed; LIMIT: evenly spaced pool rows, full-size checks skipped).
"""
import glob, hashlib, json, os, shutil, subprocess, sys, time

T0 = time.time()
INPUT = os.environ.get('E2B_INPUT', '/kaggle/input')
OUT = os.environ.get('E2B_OUT', '/kaggle/working')
TMP = os.environ.get('E2B_TMP', '/tmp/e2bank_tmp')
TEST = os.environ.get('E2B_TEST') == '1'
LIMIT = int(os.environ.get('E2B_LIMIT', '0'))
PROCS = int(os.environ.get('E2B_PROCS', str(os.cpu_count() or 4)))
POOL_MD5 = '8afda40aa9324130b800523c1e5f267b'
POOL_ROWS = 1160133
TRAIN_KEYS = 275810
TRAIN_KEYS_SHA256 = '23731cd9702fd93fdcf2443ae742634b55274257768d86e6b1a03f6dadebb6d5'
RDKIT_REF = '2026.03.3'
# Local full selection (Nov 2; same functions over the 275,810 in_train rows, RDKit 2026.03.3, numpy 2.5.3): soft check
EXPECTED_NBITS = 6930
EXPECTED_FP_BITS_SHA256 = 'a61e372fc41091bf5c7467e08134081266262665077cb9e2c56d73e4887f3013'
problems, notes = [], []


def say(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def digest(p, algo='md5'):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


BANK_SHA256 = '88b6241d1d89d4da9edcceebf2277a5e1240da98c028d78401d5676449cce0a4'
# >>> build_e2_bank.py (embedded by scripts/make_e2_bank_kernel.py; do not edit by hand)
BANK_SRC = r'''"""Engine 2 clone, step 1 (docs/engine2_study.md section 5, option R): the eligible candidate bank.

Our own code. It re-implements the bank format that the published Engine 2 reads, from pool v18c and our own bit
selection. Nothing from Engine 2, cubres, prvsiyan or megayak is imported or executed; their sources were read as text
only. Citations below:
  E2:casmi_engine.py Lx = ENG_FILES casmi_engine.py written by the frozen 0.465 notebook (docs/engine2_study.md, citation
                          key; text copy in the eng2_study scratchpad), E2:eng_runner.py Lx likewise.
  cubres:<file> Lx      = github.com/cubres/enveda-casmi-2026 reproduction/legacy-v2/public_source/<file> (Apache-2.0).
  V43 hashes            = byte sizes of the fork's Engine 2 inputs, kernels/casmi26-pube1-ne/notebook.ipynb cell 8.

Usage:
  python scripts/build_e2_bank.py <pool_dir> <train.parquet | train_meta.parquet> <out_dir> [procs]
         [--limit N] [--chunk 2000] [--tmp-dir DIR] [--prefix coco] [--verify 1000] [--any-rdkit]

Inputs
  <pool_dir>/pool.parquet  pool v18c (1,160,133 rows; columns key, smiles, mass, in_train used). key = InChIKey14,
                           smiles = RDKit canonical stereo-free SMILES, mass = ExactMolWt (scripts/build_pool.py).
                           Mass ascending is asserted; if not, rows are stable-sorted and bank_meta.pool_row records it.
  <train parquet>          any parquet with an inchikey14 column: the competition train.parquet or its projection
                           train_meta.parquet (scripts/build_train_meta.py) give the same 275,810 keys (key-set sha256
                           23731cd9...). Used to check in_train row by row and that every training key is in the pool.

Descriptor (10,407 columns; E2:casmi_engine.py L111-113 and L124-127 = cubres:build_fps.py L9-11, L17-22 =
cubres:build_pool.py L11-13, L20-23), one uint8 0/1 column per bit, concatenated in this order:
  [0, 4096)       Morgan radius 2, 4096 bits  rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
  [4096, 8192)    Morgan radius 3, 4096 bits  GetMorganGenerator(radius=3, fpSize=4096)
  [8192, 10240)   RDKit path FP, 2048 bits    GetRDKitFPGenerator(fpSize=2048, maxPath=6)
  [10240, 10407)  MACCS, 167 bits             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8) (bit 0 is always 0)
  All other generator options are RDKit defaults, as in the sources: bits (no count simulation), no chirality, Morgan
  bond types and ring membership on; RDKit path minPath 1, branchedPaths, useHs, bond order on, 2 bits per path. The
  values used are read back from the generators into MANIFEST.json. The molecule is Chem.MolFromSmiles(smiles) with
  default sanitisation (E2:casmi_engine.py L121); GetFingerprintAsNumPy(...).astype(np.uint8) (L124-126).

Bit selection (cubres:train_fp.py L15-19; cubres:prepare_legacy_exports.py L47-49; cubres:build_fps.py L29-46):
  freq = per-column mean of the 10,407-column descriptor over the training structures, one row per InChIKey14, rows
  RDKit fails on removed; bits = np.where((freq >= 0.005) & (freq <= 0.995))[0] (int64, ascending). Our training
  structures: pool rows with in_train == 1 (one row per key by construction) that RDKit parses. The frequency is
  counts / n in float64, which equals numpy's mean of the 0/1 matrix bit for bit.

Packing (E2:casmi_engine.py L127-128; cubres:build_pool.py L44, L63): the selected columns are taken in fp_bits order,
  then np.packbits along the row with numpy's default bitorder 'big' (first selected bit = MSB of byte 0), the last byte
  zero-padded: uint8 (N, ceil(nbits / 8)). Readers unpack with np.unpackbits(fp, axis=1)[:, :nbits]. NOTE: Engine 2's
  pool_fps hard-codes [:, :6930] (E2:casmi_engine.py L193-194); the clone must use the bank's nbits instead.

Outputs (<out_dir>), named as Engine 2's bank loader reads them (E2:casmi_engine.py L147-150, L110; E2:eng_runner.py
L9-12 resolves fp_bits.npy next to coco_fp.npy). The prefix 'coco' is only the loader's file name: the bank is the whole
pool v18c, row i = pool row bank_meta.pool_row[i] (= i unless --limit or a re-sort):
  <prefix>_fp.npy     uint8 (N, ceil(nbits/8)), packed selected bits. Failed rows are all zero.
                      (Engine 2's coco_fp.npy is (436,389, 867) uint8: V43 hashes, 378,349,391 bytes = 436,389 x 867 + 128.)
  <prefix>_mass.npy   float64 (N,), ascending; pool.parquet mass (ExactMolWt, E2:casmi_engine.py L128). Agreement with
                      ExactMolWt recomputed from the SMILES is checked and recorded.
  <prefix>_meta.pkl   pickle protocol 4, {'keys': [InChIKey14 str], 'smiles': [str], 'nbits': int}: the loader reads
                      cm['keys'] and cm['smiles'] (E2:casmi_engine.py L148-150); the three-field primitive schema is the
                      one of cubres:prepare_legacy_exports.py L35-37, L51-52.
  fp_bits.npy         int64 (nbits,), ascending column indices into the 10,407-column descriptor, np.save
                      allow_pickle=False (cubres:prepare_legacy_exports.py L63-65). (Engine 2's: V43 hashes, 55,568 bytes
                      = 6,930 x 8 + 128.)
  fp_freq.npy         float64 (10407,), the training frequencies (cubres:build_fps.py L45-47 keeps the same array).
  bank_meta.parquet   pool_row, key, smiles, in_train, rdkit_ok, popcount (selected bits on), aligned with the arrays.
  MANIFEST.json       inputs (pool md5, training key-set sha256), descriptor options, selection (rule, counts, the bit
                      list), packing, failures, mass check, gates, md5 + sha256 + shape of every output, versions,
                      wall time and peak RSS.

Processing: worker processes compute the full descriptor for chunks of rows; the parent writes each chunk, bit-packed
(1,301 bytes per row), into a temporary .npy memmap in --tmp-dir and adds up the bit counts of training rows; after the
selection a second pass unpacks blocks of the memmap, takes the selected columns and writes <prefix>_fp.npy as a
memmap. Peak RSS stays near the pool's SMILES column plus one block; temporary disk = N x 1,301 bytes (1.5 GB for the
full pool), deleted at the end. Results do not depend on procs or chunk size.
--limit N keeps N evenly spaced pool rows (local tests); the nbits range gate is then reported but not enforced.
Gates (docs/engine2_study.md section 5 step 1): RDKit failures <= 0.5%; nbits in [6000, 8000]; masses finite and ascending;
--verify rows recomputed from SMILES give identical packed bits; in_train equals training-key membership row by row;
every training key is in the pool (full builds). The exit code is 1 when a hard gate fails (the outputs stay for
inspection).
"""
import argparse
import hashlib
import json
import os
import pickle
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RDKIT_REF = '2026.03.3'          # Kaggle's wheel (metric/rdkit-2026-3-3-wheel) and the local reference
LO, HI = 0.005, 0.995            # cubres:train_fp.py L10 (bit_lo, bit_hi) and L19
BLOCKS = (('morgan_r2', 4096), ('morgan_r3', 4096), ('rdkit_path', 2048), ('maccs', 167))
NCOL = sum(n for _, n in BLOCKS)  # 10407 (cubres:build_fps.py L22; cubres:prepare_legacy_exports.py L37)
NBYTE_FULL = (NCOL + 7) // 8      # 1301
NBITS_RANGE = (6000, 8000)        # docs/engine2_study.md step 1 gate 2 (legacy: 6,930)
MAX_FAIL = 0.005                  # gate 1
_G = {}


def _init():
    """Per-process generators (E2:casmi_engine.py L106-113)."""
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    _G['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _G['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _G['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)


def descriptor(smi):
    """10,407-column uint8 0/1 descriptor and ExactMolWt of one SMILES, or None if RDKit fails
    (E2:casmi_engine.py L116-130, without the bit selection, which is applied after the frequencies are known)."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    if not _G:
        _init()
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        fp = np.concatenate([_G['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _G['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _G['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])
        if fp.shape != (NCOL,):
            return None
        return fp, float(ExactMolWt(m))
    except Exception:
        return None


def _chunk(task):
    """Worker: one chunk of rows -> (start, packed full descriptors, ok, RDKit mass, bit counts over ok training rows)."""
    start, smiles, train = task
    n = len(smiles)
    full = np.zeros((n, NCOL), np.uint8)
    ok = np.zeros(n, bool)
    mass = np.full(n, np.nan)
    for i, s in enumerate(smiles):
        r = descriptor(s)
        if r is not None:
            full[i], mass[i], ok[i] = r[0], r[1], True
    counts = full[ok & train].sum(0, dtype=np.int64)
    return start, np.packbits(full, axis=1), ok, mass, counts


def select_bits(counts, n, lo=LO, hi=HI):
    """cubres:train_fp.py L18-19: frequency window over the training structures; int64 ascending indices."""
    freq = np.asarray(counts, np.float64) / float(n)
    return np.where((freq >= lo) & (freq <= hi))[0].astype(np.int64), freq


def pack_selected(full01, bits):
    """Selected columns in fp_bits order, np.packbits along rows (default bitorder 'big'), as E2:casmi_engine.py L127-128."""
    return np.packbits(np.asarray(full01, np.uint8)[:, bits], axis=1)


def unpack(packed, nbits):
    return np.unpackbits(np.asarray(packed, np.uint8), axis=1)[:, :nbits]


def file_digests(path):
    h5, h2 = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h5.update(b)
            h2.update(b)
    return h5.hexdigest(), h2.hexdigest()


def train_keys(path):
    col = pq.read_table(path, columns=['inchikey14']).column(0).to_pylist()
    keys = sorted({k for k in col if isinstance(k, str) and k})
    return keys, len(col), hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def generator_options():
    if not _G:
        _init()
    out = {}
    for name, g in (('morgan_r2', _G['m2']), ('morgan_r3', _G['m3']), ('rdkit_path', _G['rk'])):
        o = g.GetOptions()
        out[name] = {a: getattr(o, a) for a in dir(o) if not a.startswith('_') and not callable(getattr(o, a))}
    out['morgan_r2']['call'] = 'GetMorganGenerator(radius=2, fpSize=4096)'
    out['morgan_r3']['call'] = 'GetMorganGenerator(radius=3, fpSize=4096)'
    out['rdkit_path']['call'] = 'GetRDKitFPGenerator(fpSize=2048, maxPath=6)'
    out['maccs'] = {'call': 'np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)', 'bits': 167}
    return out


def log(t0, *a):
    print(f'[{time.time() - t0:7.1f}s]', *a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Engine 2 clone bank (option R step 1).')
    ap.add_argument('pool_dir')
    ap.add_argument('train_parquet')
    ap.add_argument('out_dir')
    ap.add_argument('procs', nargs='?', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='keep N evenly spaced pool rows (local tests)')
    ap.add_argument('--chunk', type=int, default=2000, help='rows per worker task')
    ap.add_argument('--tmp-dir', default=None, help='directory of the temporary full-descriptor memmap (default out_dir)')
    ap.add_argument('--prefix', default='coco', help="file prefix of the bank arrays (Engine 2's loader reads coco_*)")
    ap.add_argument('--verify', type=int, default=1000, help='rows recomputed from SMILES at the end (gate 4)')
    ap.add_argument('--any-rdkit', action='store_true', help=f'allow an RDKit other than {RDKIT_REF}')
    a = ap.parse_args(argv)
    t0 = time.time()
    import rdkit
    if rdkit.__version__ != RDKIT_REF and not a.any_rdkit:
        sys.exit(f'RDKit {rdkit.__version__} != {RDKIT_REF}; bits can differ between versions (use --any-rdkit to override)')
    os.makedirs(a.out_dir, exist_ok=True)
    tmp_dir = a.tmp_dir or a.out_dir
    os.makedirs(tmp_dir, exist_ok=True)
    pool_path = os.path.join(a.pool_dir, 'pool.parquet')
    pool_md5, pool_sha = file_digests(pool_path)
    P = pd.read_parquet(pool_path, columns=['key', 'smiles', 'mass', 'in_train'])
    n_pool = len(P)
    log(t0, f'pool {pool_path}: {n_pool:,} rows, md5 {pool_md5}; RDKit {rdkit.__version__}; procs {a.procs}')

    # ---- rows: optional evenly spaced subset, then mass order (E2:casmi_engine.py L185-187 sorts by mass, stable)
    rows = np.arange(n_pool, dtype=np.int64)
    if a.limit and a.limit < n_pool:
        rows = np.unique(np.linspace(0, n_pool - 1, num=a.limit).round().astype(np.int64))
    pmass = P['mass'].to_numpy(np.float64)
    if not np.isfinite(pmass[rows]).all():
        sys.exit(f'pool has {int((~np.isfinite(pmass[rows])).sum())} non-finite masses')
    ascending = bool(np.all(np.diff(pmass[rows]) >= 0))
    if not ascending:
        rows = rows[np.argsort(pmass[rows], kind='stable')]
        log(t0, 'pool rows are not mass-ascending: stable-sorted; bank_meta.pool_row records the permutation')
    n = len(rows)
    key = P['key'].to_numpy(dtype=object)[rows]
    smiles = P['smiles'].to_numpy(dtype=object)[rows]
    in_train = P['in_train'].to_numpy()[rows].astype(np.int8)
    mass = pmass[rows]
    all_keys = P['key'].to_numpy(dtype=object)
    del P

    # ---- training keys: in_train must equal training-key membership row by row
    tkeys, train_rows, tkeys_sha = train_keys(a.train_parquet)
    tset = set(tkeys)
    is_tk = np.fromiter((k in tset for k in key), bool, count=n)
    mism = np.where(is_tk != (in_train == 1))[0]
    missing = len(tset - set(all_keys)) if not a.limit else None
    del all_keys
    log(t0, f'training keys {len(tkeys):,} from {train_rows:,} rows of {a.train_parquet} (sha256 {tkeys_sha[:12]}); '
        f'bank rows {n:,}, in_train {int(in_train.sum()):,}, in_train/key mismatches {len(mism)}, '
        f'training keys missing from the pool {missing}')
    if len(mism):
        sys.exit(f'in_train disagrees with the training keys on {len(mism)} rows, e.g. pool rows {rows[mism[:10]].tolist()}')

    # ---- pass 1: full descriptors (bit-packed) into a temporary memmap; bit counts over training rows
    tmp_path = os.path.join(tmp_dir, f'.e2bank_full_{os.getpid()}.npy')
    full = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.uint8, shape=(n, NBYTE_FULL))
    ok = np.zeros(n, bool)
    rd_mass = np.full(n, np.nan)
    counts = np.zeros(NCOL, np.int64)
    tasks = [(s, list(smiles[s:s + a.chunk]), in_train[s:s + a.chunk] == 1) for s in range(0, n, a.chunk)]
    ntask = len(tasks)
    every = max(1, ntask // 20)
    t1 = time.time()

    def take(res, done):
        s, packed, okc, mc, cc = res
        full[s:s + len(okc)] = packed
        ok[s:s + len(okc)] = okc
        rd_mass[s:s + len(okc)] = mc
        counts[:] += cc
        if done % every == 0 or done == ntask:
            r = min(n, done * a.chunk) / max(time.time() - t1, 1e-9)
            log(t0, f'descriptors {done}/{ntask} chunks, {r:,.0f} rows/s, ETA {(n - min(n, done * a.chunk)) / r:,.0f}s')

    if a.procs <= 1:
        _init()
        for i, t in enumerate(tasks, 1):
            take(_chunk(t), i)
    else:
        from multiprocessing import Pool
        with Pool(a.procs, initializer=_init) as mp:
            for i, res in enumerate(mp.imap(_chunk, tasks, chunksize=1), 1):
                take(res, i)
    tasks = None
    full.flush()
    t_desc = time.time() - t1
    n_fail = int((~ok).sum())
    train_ok = (in_train == 1) & ok
    n_sel = int(train_ok.sum())
    log(t0, f'descriptors done in {t_desc:,.0f}s ({n / max(t_desc, 1e-9):,.0f} rows/s); RDKit failures {n_fail} '
        f'({n_fail / max(n, 1):.4%}); training rows used for frequencies {n_sel:,}')
    if n_sel == 0:
        sys.exit('no training rows parsed: cannot select bits')

    # ---- selection (cubres:train_fp.py L17-19)
    bits, freq = select_bits(counts, n_sel)
    nbits = int(len(bits))
    nbyte = (nbits + 7) // 8
    edges = np.cumsum([0] + [w for _, w in BLOCKS])
    per_block = {nm: int(((bits >= edges[i]) & (bits < edges[i + 1])).sum()) for i, (nm, _) in enumerate(BLOCKS)}
    log(t0, f'selected {nbits} bits with {LO} <= f <= {HI} ({per_block}); {nbyte} bytes per row')

    # ---- pass 2: selected columns, packed, straight into the output memmap
    names = {'fp': f'{a.prefix}_fp.npy', 'mass': f'{a.prefix}_mass.npy', 'meta': f'{a.prefix}_meta.pkl',
             'bits': 'fp_bits.npy', 'freq': 'fp_freq.npy', 'bank_meta': 'bank_meta.parquet'}
    out = {k: os.path.join(a.out_dir, v) for k, v in names.items()}
    fp = np.lib.format.open_memmap(out['fp'], mode='w+', dtype=np.uint8, shape=(n, nbyte))
    pop = np.zeros(n, np.int32)
    assert np.all(np.diff(bits) > 0)          # ascending and unique: compress(mask) == [:, bits]
    mask = np.zeros(NBYTE_FULL * 8, bool)     # over the padded width; the 5 pad columns are never selected
    mask[bits] = True
    blk = 1024                                # small blocks stay in cache (32k-row blocks were ~6x slower)
    for s in range(0, n, blk):
        sel = np.compress(mask, np.unpackbits(full[s:s + blk], axis=1), axis=1)
        fp[s:s + blk] = np.packbits(sel, axis=1)
        pop[s:s + blk] = sel.sum(1, dtype=np.int32)
    fp.flush()
    fp = full = None                          # close both memmaps before removing the temporary file
    os.remove(tmp_path)
    log(t0, f'packed {out["fp"]} ({n:,} x {nbyte}); mean popcount {pop[ok].mean() if ok.any() else 0:.1f}')

    # ---- the other arrays
    with open(out['bits'], 'wb') as f:
        np.save(f, bits, allow_pickle=False)
    with open(out['freq'], 'wb') as f:
        np.save(f, freq.astype(np.float64), allow_pickle=False)
    with open(out['mass'], 'wb') as f:
        np.save(f, np.ascontiguousarray(mass, np.float64), allow_pickle=False)
    meta = {'keys': [str(k) for k in key], 'smiles': [str(s) for s in smiles], 'nbits': nbits}
    with open(out['meta'], 'wb') as f:
        pickle.dump(meta, f, protocol=4)
    del meta
    pd.DataFrame({'pool_row': rows, 'key': pd.array(key, dtype='string'), 'smiles': pd.array(smiles, dtype='string'),
                  'in_train': in_train, 'rdkit_ok': ok.astype(np.int8), 'popcount': pop.astype(np.int16)}
                 ).to_parquet(out['bank_meta'], index=False)
    dm = np.abs(rd_mass[ok] - mass[ok])
    mass_check = {'source': 'pool.parquet mass (ExactMolWt at pool build)',
                  'max_abs_diff_vs_exactmolwt_of_smiles': float(dm.max()) if len(dm) else 0.0,
                  'n_diff_gt_1e-6_da': int((dm > 1e-6).sum())}

    # ---- gate 4: recompute random rows from SMILES, compare packed bits
    fpm = np.load(out['fp'], mmap_mode='r')
    rng = np.random.default_rng(0)
    vrows = np.sort(rng.choice(n, size=min(a.verify, n), replace=False)) if a.verify > 0 else np.zeros(0, np.int64)
    bad = []
    for i in vrows:
        r = descriptor(smiles[i])
        want = np.zeros(nbyte, np.uint8) if r is None else pack_selected(r[0][None, :], bits)[0]
        if not np.array_equal(want, fpm[i]):
            bad.append(int(rows[i]))
    del fpm
    log(t0, f'verify: {len(vrows)} rows recomputed from SMILES, {len(bad)} mismatches')

    # ---- gates, digests, manifest
    dmass = np.diff(mass)
    gates = {
        'rdkit_failures_le_0.5pct': dict(value=n_fail / max(n, 1), ok=n_fail <= MAX_FAIL * n, hard=True),
        'nbits_in_6000_8000': dict(value=nbits, ok=NBITS_RANGE[0] <= nbits <= NBITS_RANGE[1], hard=not a.limit),
        'mass_finite_ascending': dict(value=bool(np.isfinite(mass).all() and (dmass >= 0).all()),
                                      ok=bool(np.isfinite(mass).all() and (dmass >= 0).all()), hard=True),
        'verify_identical_bits': dict(value=f'{len(vrows) - len(bad)}/{len(vrows)}', ok=not bad, hard=True),
        'in_train_equals_training_keys': dict(value=int(len(mism)), ok=not len(mism), hard=True),
        'training_keys_missing_from_pool': dict(value=missing, ok=missing in (None, 0), hard=not a.limit),
        'mass_matches_exactmolwt_1e-6': dict(value=mass_check['n_diff_gt_1e-6_da'],
                                             ok=mass_check['n_diff_gt_1e-6_da'] == 0, hard=False),
    }
    outputs = {}
    for k, p in out.items():
        h5, h2 = file_digests(p)
        e = dict(file=os.path.basename(p), bytes=os.path.getsize(p), md5=h5, sha256=h2)
        if p.endswith('.npy'):
            arr = np.load(p, mmap_mode='r')
            e.update(dtype=str(arr.dtype), shape=list(arr.shape))
            del arr
        outputs[k] = e
    import numpy, pandas, pyarrow
    ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    fl = np.where(~ok)[0]
    manifest = {
        'what': 'Engine 2 clone candidate bank (docs/engine2_study.md option R step 1); pool v18c with our own bit selection',
        'script': dict(file=os.path.basename(__file__), sha256=file_digests(os.path.abspath(__file__))[1]),
        'inputs': dict(pool_parquet=os.path.abspath(pool_path), pool_md5=pool_md5, pool_sha256=pool_sha, pool_rows=n_pool,
                       train_parquet=os.path.abspath(a.train_parquet), train_rows=train_rows,
                       train_keys=len(tkeys), train_keys_sha256=tkeys_sha,
                       train_keys_sha256_note="sha256 of '\\n'.join(sorted unique inchikey14)"),
        'rows': n, 'limit': a.limit or None,
        'pool_rows_sha256': hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
        'order': dict(pool_mass_ascending=ascending, sorted_here=not ascending,
                      note='bank row i = pool row bank_meta.pool_row[i]'),
        'descriptor': dict(columns=NCOL, blocks=[dict(name=nm, start=int(edges[i]), width=w) for i, (nm, w) in enumerate(BLOCKS)],
                           generators=generator_options(), mol='Chem.MolFromSmiles(smiles), default sanitisation',
                           values='uint8 0/1 per bit (no counts)'),
        'selection': dict(rule=f'{LO} <= freq <= {HI}', lo=LO, hi=HI, freq='counts / n (float64)',
                          scope='bank rows with in_train == 1 that RDKit parses', n=n_sel,
                          in_train_rows=int((in_train == 1).sum()), in_train_rdkit_failures=int(((in_train == 1) & ~ok).sum()),
                          nbits=nbits, per_block=per_block, bits=bits.tolist()),
        'packing': dict(selected_order='fp_bits ascending', function='np.packbits(sel, axis=1)', bitorder='big',
                        bytes_per_row=nbyte, pad_bits=nbyte * 8 - nbits, unpack=f'np.unpackbits(fp, axis=1)[:, :{nbits}]'),
        'failures': dict(count=n_fail, fraction=n_fail / max(n, 1), pool_rows=rows[fl[:1000]].tolist(),
                         smiles=[str(smiles[i]) for i in fl[:50]], handling='all-zero bits, rdkit_ok 0, row kept'),
        'popcount': dict(mean=float(pop[ok].mean()) if ok.any() else 0.0, min=int(pop[ok].min()) if ok.any() else 0,
                         max=int(pop.max()) if n else 0, zero_rows_ok=int(((pop == 0) & ok).sum())),
        'mass': mass_check,
        'verify': dict(rows=int(len(vrows)), mismatched_pool_rows=bad[:100], seed=0),
        'gates': gates,
        'outputs': outputs,
        'versions': dict(python=sys.version.split()[0], rdkit=rdkit.__version__, numpy=numpy.__version__,
                         pandas=pandas.__version__, pyarrow=pyarrow.__version__),
        'timing': dict(seconds_total=round(time.time() - t0, 1), seconds_descriptors=round(t_desc, 1),
                       rows_per_second=round(n / max(t_desc, 1e-9), 1), procs=a.procs, chunk=a.chunk, cpus=os.cpu_count()),
        'peak_rss_kb': dict(parent=int(ru_self), largest_child=int(ru_child)),
    }
    with open(os.path.join(a.out_dir, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    failed = [k for k, g in gates.items() if g['hard'] and not g['ok']]
    for k, g in gates.items():
        log(t0, f"gate {k}: {g['value']} -> {'ok' if g['ok'] else ('FAIL' if g['hard'] else 'warn')}")
    log(t0, f'wrote {sorted(names.values()) + ["MANIFEST.json"]} to {a.out_dir}; peak RSS parent {ru_self / 1e6:.2f} GB, '
        f'child {ru_child / 1e6:.2f} GB')
    if failed:
        print('E2_BANK_FAILED', failed, flush=True)
        return 1
    print('E2_BANK_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''
# <<< build_e2_bank.py

# ---- 0. the embedded build script
got = hashlib.sha256(BANK_SRC.encode()).hexdigest()
if got != BANK_SHA256:
    raise SystemExit(f'embedded build_e2_bank.py sha256 {got} != {BANK_SHA256}')
os.makedirs(OUT, exist_ok=True)
code_dir = '/tmp/e2bank_code' if not TEST else os.path.join(TMP, 'code')
os.makedirs(code_dir, exist_ok=True)
script = os.path.join(code_dir, 'build_e2_bank.py')
with open(script, 'w') as f:
    f.write(BANK_SRC)
shutil.copy(script, os.path.join(OUT, 'build_e2_bank.py'))
say('build_e2_bank.py sha256', got[:12], '->', script)

# ---- 1. RDKit 2026.3.3 from the wheel dataset (as kernels/casmi26-v18cx-build)
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob(f'{INPUT}/**/rdkit-2026.3.3-*.whl', recursive=True) if tag in w]
if whl:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
elif not TEST:
    raise SystemExit(f'no rdkit-2026.3.3 {tag} wheel under {INPUT} (dataset metric/rdkit-2026-3-3-wheel)')
vers = subprocess.run([sys.executable, '-c', 'import sys, json, numpy, pandas, pyarrow, rdkit; print(json.dumps(dict('
                       'python=sys.version.split()[0], numpy=numpy.__version__, pandas=pandas.__version__, '
                       'pyarrow=pyarrow.__version__, rdkit=rdkit.__version__)))'], capture_output=True, text=True)
say('environment', vers.stdout.strip(), vers.stderr.strip()[-300:], 'cpus', os.cpu_count(), 'procs', PROCS)
v = json.loads(vers.stdout)
if v['rdkit'] != RDKIT_REF:
    raise SystemExit(f'RDKit {v["rdkit"]} != {RDKIT_REF}; refusing to build (bits can differ between versions)')

# ---- 2. inputs
pools = [p for p in sorted(glob.glob(f'{INPUT}/**/pool.parquet', recursive=True), key=len)]
pool_md5 = {p: digest(p) for p in pools}
good = [p for p in pools if pool_md5[p] == POOL_MD5]
say('pool.parquet candidates', {p: h[:8] for p, h in pool_md5.items()})
if not good:
    raise SystemExit(f'no pool.parquet with md5 {POOL_MD5} under {INPUT} (dataset bryantyu/casmi26-pool-v18c)')
pool_dir = os.path.dirname(good[0])
trains = [p for p in sorted(glob.glob(f'{INPUT}/**/train.parquet', recursive=True), key=len)
          if 'enveda-casmi26' in p.lower()]
if not trains:
    raise SystemExit(f'competition train.parquet not found under {INPUT} (competition_sources)')
train_p = trains[0]
say('pool', pool_dir, '| train', train_p, f'({os.path.getsize(train_p) / 1e9:.2f} GB; only inchikey14 is read)')

# ---- 3. build
cmd = [sys.executable, '-u', script, pool_dir, train_p, OUT, str(PROCS), '--tmp-dir', TMP]
if LIMIT:
    cmd += ['--limit', str(LIMIT)]
say('RUN', ' '.join(cmd))
t = time.time()
with open(os.path.join(OUT, 'build_e2_bank.log'), 'w') as fh:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
        fh.write(line)
    rc = p.wait()
build_s = round(time.time() - t)
say('build exit', rc, 'after', build_s, 's')
if rc:
    problems.append(f'build exit {rc}')

# ---- 4. checks on the manifest
man_p = os.path.join(OUT, 'MANIFEST.json')
man = json.load(open(man_p)) if os.path.exists(man_p) else {}
if not man:
    problems.append('no MANIFEST.json')
else:
    inp, sel = man['inputs'], man['selection']
    if inp['pool_md5'] != POOL_MD5:
        problems.append(f"manifest pool md5 {inp['pool_md5']}")
    if inp['train_keys'] != TRAIN_KEYS or inp['train_keys_sha256'] != TRAIN_KEYS_SHA256:
        problems.append(f"training keys {inp['train_keys']} sha256 {inp['train_keys_sha256'][:12]} != {TRAIN_KEYS} "
                        f"{TRAIN_KEYS_SHA256[:12]} (train_meta.parquet)")
    if not LIMIT:
        if man['rows'] != POOL_ROWS:
            problems.append(f"bank rows {man['rows']} != {POOL_ROWS}")
        if sel['in_train_rows'] != TRAIN_KEYS:
            problems.append(f"in_train rows {sel['in_train_rows']} != {TRAIN_KEYS}")
    bad = [k for k, g in man['gates'].items() if g['hard'] and not g['ok']]
    if bad:
        problems.append(f'hard gates failed: {bad}')
    for e in man['outputs'].values():
        if digest(os.path.join(OUT, e['file']), 'sha256') != e['sha256']:
            problems.append(f"{e['file']} sha256 differs from MANIFEST")
    say(f"bank {man['rows']:,} rows | nbits {sel['nbits']} {sel['per_block']} | frequency rows {sel['n']:,} | "
        f"failures {man['failures']['count']} | mean popcount {man['popcount']['mean']:.1f}")
    say('gates', {k: (g['value'], 'ok' if g['ok'] else 'FAIL') for k, g in man['gates'].items()})
    say('outputs', {e['file']: (e['bytes'], e.get('shape'), e['md5']) for e in man['outputs'].values()})
    say('timing', man['timing'], 'peak RSS kB', man['peak_rss_kb'])
    if not LIMIT:
        same = sel['nbits'] == EXPECTED_NBITS and man['outputs']['bits']['sha256'] == EXPECTED_FP_BITS_SHA256
        say('fp_bits.npy vs the local full selection:', 'MATCH' if same else
            f"DIFFERS (nbits {sel['nbits']} vs {EXPECTED_NBITS}, sha256 {man['outputs']['bits']['sha256'][:12]} vs "
            f"{EXPECTED_FP_BITS_SHA256[:12]})")
        if not same:
            notes.append('fp_bits.npy differs from the local full selection (environment difference?)')
mem = next((l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal')), '?')
json.dump(dict(versions=v, cpus=os.cpu_count(), procs=PROCS, mem_kb=mem, build_seconds=build_s,
               seconds=round(time.time() - T0), limit=LIMIT, test=TEST, pool_dir=pool_dir, train_parquet=train_p,
               script_sha256=BANK_SHA256, problems=problems, notes=notes), open(os.path.join(OUT, 'env.json'), 'w'), indent=1)
say('outputs', sorted(os.listdir(OUT)))
if problems:
    raise SystemExit('FAILED: ' + '; '.join(problems))
say('E2_BANK_KERNEL_DONE', 'notes:', notes or 'none')
