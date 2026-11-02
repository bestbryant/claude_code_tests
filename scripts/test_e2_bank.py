"""Tests of scripts/build_e2_bank.py (Engine 2 clone bank, docs/engine2_study.md option R step 1) on ~2,000 pool rows.

python scripts/test_e2_bank.py [pool_dir] [train_parquet] [--rows 2000] [--keep]
Defaults: /home/user/work/pool_v18c and /home/user/data/train_meta.parquet. Outputs go to a fresh directory under
$TMPDIR (removed at the end unless --keep). About 40 s on 4 CPUs.

Checks
  1. selection rule on synthetic counts: inclusive bounds 0.005 / 0.995, int64 ascending result.
  2. hand-derived fingerprints of small molecules (popcounts of each block, exact MACCS keys, failure cases).
  3. build A (procs 2, chunk 300) on N evenly spaced pool rows; shapes, dtypes, alignment with the pool, file
     naming / pickle schema of Engine 2's bank loader, MANIFEST digests and gates.
  4. independent recomputation of the 10,407-column descriptor through RDKit's older APIs (GetMorganFingerprintAsBitVect,
     RDKFingerprint, ConvertToNumpyArray) for every sample row: frequencies (exactly numpy's mean), the selected bit
     list, the packed rows (bitorder 'big', zero padding) and the popcounts; Morgan bits = unfolded ids mod 4096;
     radius-2 bits are a subset of radius-3 bits.
  5. determinism: build B (procs 1, one chunk) gives byte-identical outputs.
  6. build C on a shuffled copy of the sample with one corrupted non-training SMILES: stable mass sort with the
     permutation in bank_meta.pool_row, the failed row kept with zero bits and rdkit_ok 0, same bits, every other row
     identical to build A by key.
  7. the copy embedded in kernels/casmi26-e2-bank/notebook.py is in sync (scripts/make_e2_bank_kernel.py --check).
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_e2_bank as B  # noqa: E402

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import MACCSkeys, rdMolDescriptors  # noqa: E402
RDLogger.DisableLog('rdApp.*')

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''), flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def ref_descriptor(smi):
    """Independent route to the 10,407 columns: RDKit's older fingerprint functions with every option spelled out."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    out = []
    for r in (2, 3):
        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, r, nBits=4096)
        a = np.zeros(4096, np.uint8)
        a[list(bv.GetOnBits())] = 1
        out.append(a)
    rk = Chem.RDKFingerprint(m, minPath=1, maxPath=6, fpSize=2048, nBitsPerHash=2, useHs=True, branchedPaths=True,
                             useBondOrder=True)
    a = np.zeros(2048, np.uint8)
    a[list(rk.GetOnBits())] = 1
    out.append(a)
    mk = np.zeros(167, np.int32)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), mk)
    out.append(mk.astype(np.uint8))
    return np.concatenate(out)


def build(pool_dir, train_p, out, procs, *extra):
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, 'build_e2_bank.py'), pool_dir, train_p, out, str(procs), *extra],
                       capture_output=True, text=True)
    print(f'  build {os.path.basename(out)}: exit {p.returncode} in {time.time() - t:.1f}s; '
          + ' | '.join(l for l in p.stdout.splitlines() if 'selected' in l or 'E2_BANK' in l or 'verify' in l), flush=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
    return p.returncode


def load(d):
    z = dict(fp=np.load(f'{d}/coco_fp.npy'), mass=np.load(f'{d}/coco_mass.npy'), bits=np.load(f'{d}/fp_bits.npy'),
             freq=np.load(f'{d}/fp_freq.npy'), meta=pickle.load(open(f'{d}/coco_meta.pkl', 'rb')),
             bm=pd.read_parquet(f'{d}/bank_meta.parquet'), man=json.load(open(f'{d}/MANIFEST.json')))
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pool_dir', nargs='?', default='/home/user/work/pool_v18c')
    ap.add_argument('train_parquet', nargs='?', default='/home/user/data/train_meta.parquet')
    ap.add_argument('--rows', type=int, default=2000)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    work = tempfile.mkdtemp(prefix='e2bank_test_')
    print('work dir', work)

    # ---- 1. selection rule
    n = 1000
    counts = np.array([0, 4, 5, 6, 500, 994, 995, 996, 1000])
    bits, freq = B.select_bits(counts, n)
    check('select_bits keeps 0.005 <= f <= 0.995 inclusive', bits.tolist() == [2, 3, 4, 5, 6], bits.tolist())
    check('select_bits returns int64 ascending', bits.dtype == np.int64 and np.all(np.diff(bits) > 0))
    full = (np.random.default_rng(1).random((n, 9)) < counts / n).astype(np.uint8)
    check('freq = counts / n equals numpy mean of the 0/1 matrix exactly',
          np.array_equal(B.select_bits(full.sum(0), n)[1], full.mean(0)))

    # ---- 2. hand-derived fingerprints
    # Morgan: radius-0 environments = distinct atom invariants, each radius adds only environments not seen before.
    #   methane 1 environment; ethane CH3 (r0) + CH3-C (r1); ethanol 3 atoms x (r0, r1) = 6, r2/r3 add nothing new;
    #   benzene one environment per radius 0..2 (r2) and the whole ring at r3 (4 bits).
    # RDKit path FP counts bond paths (minPath 1): none for methane, one C-C path x 2 bits per path for ethane.
    # MACCS: benzene = aromatic (162), 6-ring (163), ring (165) only; key 0 is never set.
    hand = {'C': (1, 1, 0, None), 'CC': (2, 2, 2, None), 'CCO': (6, 6, 6, None), 'c1ccccc1': (3, 4, 12, [162, 163, 165])}
    for smi, (p2, p3, pr, maccs) in hand.items():
        d = B.descriptor(smi)[0]
        blocks = d[:4096], d[4096:8192], d[8192:10240], d[10240:]
        got = (int(blocks[0].sum()), int(blocks[1].sum()), int(blocks[2].sum()))
        ok = got == (p2, p3, pr) and blocks[3][0] == 0 and (maccs is None or np.flatnonzero(blocks[3]).tolist() == maccs)
        check(f'hand fingerprint {smi}', ok, f'popcounts {got}, MACCS {np.flatnonzero(blocks[3]).tolist()}')
    for smi in ('CCO', 'CC(=O)Oc1ccccc1C(=O)O', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'):
        m = Chem.MolFromSmiles(smi)
        d = B.descriptor(smi)[0]
        mk = set(np.flatnonzero(d[10240:]).tolist())
        bad = [k for k, (patt, cnt) in MACCSkeys.smartsPatts.items()
               if patt != '?' and cnt == 0 and (k in mk) != m.HasSubstructMatch(Chem.MolFromSmarts(patt))]
        check(f'MACCS block = SMARTS matches of the presence keys, {smi}', not bad, f'mismatched keys {bad}')
    check('descriptor() returns None for bad input',
          all(B.descriptor(s) is None for s in ('', None, 'C1CC', 'C(C)(C)(C)(C)C', 'xyz')))
    d = B.descriptor('CC(=O)Oc1ccccc1C(=O)O')
    check('descriptor shape / dtype / ExactMolWt', d[0].shape == (10407,) and d[0].dtype == np.uint8
          and set(np.unique(d[0])) <= {0, 1} and abs(d[1] - 180.042258736) < 1e-6, f'mass {d[1]:.9f}')

    # ---- 3. build A
    A_dir = f'{work}/A'
    rc = build(a.pool_dir, a.train_parquet, A_dir, 2, '--limit', str(a.rows), '--chunk', '300', '--verify', '300')
    check('build A exit 0', rc == 0)
    A = load(A_dir)
    P = pd.read_parquet(f'{a.pool_dir}/pool.parquet', columns=['key', 'smiles', 'mass', 'in_train'])
    N = len(A['mass'])
    nb = len(A['bits'])
    exp_rows = np.unique(np.linspace(0, len(P) - 1, num=a.rows).round().astype(np.int64))
    check('files named as Engine 2 reads them', all(os.path.exists(f'{A_dir}/{f}') for f in
          ('coco_fp.npy', 'coco_mass.npy', 'coco_meta.pkl', 'fp_bits.npy', 'fp_freq.npy', 'bank_meta.parquet', 'MANIFEST.json')))
    check('pool_row = evenly spaced sample in pool order', np.array_equal(A['bm'].pool_row.values, exp_rows))
    check('coco_fp uint8 (N, ceil(nbits/8))', A['fp'].dtype == np.uint8 and A['fp'].shape == (N, (nb + 7) // 8),
          f'{A["fp"].shape}, nbits {nb}')
    check('coco_mass float64 (N,), = pool mass, ascending', A['mass'].dtype == np.float64 and A['mass'].shape == (N,)
          and np.array_equal(A['mass'], P.mass.values[exp_rows]) and np.all(np.diff(A['mass']) >= 0))
    check('fp_bits int64 ascending unique in [0, 10407)', A['bits'].dtype == np.int64 and A['bits'].ndim == 1
          and np.all(np.diff(A['bits']) > 0) and A['bits'][0] >= 0 and A['bits'][-1] < 10407)
    check('fp_freq float64 (10407,)', A['freq'].dtype == np.float64 and A['freq'].shape == (10407,))
    meta = A['meta']
    check('coco_meta.pkl = {keys, smiles, nbits}, lists aligned with the pool',
          type(meta) is dict and set(meta) == {'keys', 'smiles', 'nbits'} and type(meta['keys']) is list
          and type(meta['smiles']) is list and all(type(x) is str for x in meta['keys'][:50])
          and meta['keys'] == P.key.values[exp_rows].tolist() and meta['smiles'] == P.smiles.values[exp_rows].tolist()
          and meta['nbits'] == nb)
    bm = A['bm']
    check('bank_meta columns and alignment', list(bm.columns) == ['pool_row', 'key', 'smiles', 'in_train', 'rdkit_ok', 'popcount']
          and bm.key.tolist() == meta['keys'] and np.array_equal(bm.in_train.values, P.in_train.values[exp_rows]))
    man = A['man']
    dig_ok = all(sha(f'{A_dir}/{e["file"]}') == e['sha256'] for e in man['outputs'].values())
    check('MANIFEST sha256 of every output matches', dig_ok and len(man['outputs']) == 6)
    check('MANIFEST selection bits = fp_bits, nbits, rows', man['selection']['bits'] == A['bits'].tolist()
          and man['selection']['nbits'] == nb and man['rows'] == N and man['packing']['bytes_per_row'] == (nb + 7) // 8)
    check('MANIFEST gates all ok', all(g['ok'] for g in man['gates'].values()),
          {k: g['value'] for k, g in man['gates'].items()})
    check('MANIFEST pool md5 / training keys', man['inputs']['pool_md5'] == hashlib.md5(open(f'{a.pool_dir}/pool.parquet', 'rb').read()).hexdigest()
          and man['inputs']['train_keys'] > 0, f"{man['inputs']['pool_md5']} {man['inputs']['train_keys']}")

    # ---- 4. independent recomputation
    t = time.time()
    R = [ref_descriptor(s) for s in meta['smiles']]
    ok = np.array([r is not None for r in R])
    F = np.stack([r if r is not None else np.zeros(10407, np.uint8) for r in R])
    print(f'  reference descriptors for {N} rows in {time.time() - t:.1f}s')
    check('RDKit parses every sample row; rdkit_ok all 1', ok.all() and (bm.rdkit_ok.values == 1).all())
    tr = (bm.in_train.values == 1) & ok
    ref_freq = F[tr].mean(0)
    ref_bits = np.where((ref_freq >= 0.005) & (ref_freq <= 0.995))[0]
    check('frequencies over the training rows = reference (exact)', np.array_equal(A['freq'], ref_freq),
          f'{int(tr.sum())} training rows')
    check('selected bits = reference rule over the training rows', np.array_equal(A['bits'], ref_bits), f'{nb} bits')
    unp = np.unpackbits(A['fp'], axis=1)
    check('unpack(coco_fp)[:, :nbits] = reference descriptor[:, bits]', np.array_equal(unp[:, :nb], F[:, A['bits']]))
    check('padding bits are zero', not unp[:, nb:].any(), f'{unp.shape[1] - nb} pad bits')
    check('packing round trip (packbits default big-endian, per row)',
          np.array_equal(np.packbits(unp[:, :nb], axis=1), A['fp'])
          and all(((A['fp'][i, 0] >> 7) & 1) == F[i, A['bits'][0]] for i in range(N)))
    check('popcount column = selected bits on', np.array_equal(bm.popcount.values, F[:, A['bits']].sum(1)))
    m2, m3 = F[:, :4096].astype(bool), F[:, 4096:8192].astype(bool)
    check('Morgan r2 bits are a subset of r3 bits (block order)', not (m2 & ~m3).any())
    from rdkit.Chem import rdFingerprintGenerator
    g2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    sel = np.linspace(0, N - 1, 25).astype(int)
    fold = all(set(np.flatnonzero(F[i, :4096]).tolist())
               == {k % 4096 for k in g2.GetSparseCountFingerprint(Chem.MolFromSmiles(meta['smiles'][i])).GetNonzeroElements()}
               for i in sel)
    check('Morgan r2 block = unfolded environment ids mod 4096 (25 rows)', fold)
    check('MACCS bit 0 never set', not F[:, 10240].any() and not unp[:, :nb][:, A['bits'] == 10240].any())

    # ---- 5. determinism
    B_dir = f'{work}/B'
    rc = build(a.pool_dir, a.train_parquet, B_dir, 1, '--limit', str(a.rows), '--chunk', '100000', '--verify', '0')
    same = {f: sha(f'{A_dir}/{f}') == sha(f'{B_dir}/{f}') for f in
            ('coco_fp.npy', 'coco_mass.npy', 'coco_meta.pkl', 'fp_bits.npy', 'fp_freq.npy', 'bank_meta.parquet')}
    check('build B (procs 1, one chunk) byte-identical to build A', rc == 0 and all(same.values()), same)

    # ---- 6. shuffled pool with one corrupted SMILES
    C_pool = f'{work}/poolC'
    os.makedirs(C_pool)
    S = P.iloc[exp_rows].reset_index(drop=True)
    S = S.iloc[np.random.default_rng(7).permutation(len(S))].reset_index(drop=True)
    bad_i = int(np.flatnonzero(S.in_train.values == 0)[len(S) // 4])
    bad_key = S.key[bad_i]
    S.loc[bad_i, 'smiles'] = 'C1CC'
    S.to_parquet(f'{C_pool}/pool.parquet', index=False)
    trk = pd.DataFrame({'inchikey14': S.key[S.in_train == 1].tolist() + [None]})
    trk.to_parquet(f'{work}/train_keys_C.parquet', index=False)
    C_dir = f'{work}/C'
    rc = build(C_pool, f'{work}/train_keys_C.parquet', C_dir, 2, '--limit', '1000000', '--chunk', '500', '--verify', '200')
    check('build C exit 0', rc == 0)
    C = load(C_dir)
    cbm = C['bm']
    check('C: not ascending -> stable mass sort, permutation in pool_row',
          C['man']['order']['sorted_here'] and np.array_equal(cbm.pool_row.values, np.argsort(S.mass.values, kind='stable'))
          and cbm.key.tolist() == S.key.values[cbm.pool_row.values].tolist() and np.all(np.diff(C['mass']) >= 0))
    fi = np.flatnonzero(cbm.rdkit_ok.values == 0)
    check('C: one failure, kept, zero bits, flagged', len(fi) == 1 and cbm.key[fi[0]] == bad_key
          and not C['fp'][fi[0]].any() and cbm.popcount[fi[0]] == 0 and C['man']['failures']['count'] == 1
          and len(C['mass']) == len(S), f'failed rows {fi.tolist()}')
    check('C: same selected bits as A (failed row is not a training row)', np.array_equal(C['bits'], A['bits']))
    a_of = {k: i for i, k in enumerate(meta['keys'])}
    ia = np.array([a_of[k] for k in cbm.key])
    keep = cbm.rdkit_ok.values == 1
    check('C: every other row identical to A by key', np.array_equal(C['fp'][keep], A['fp'][ia[keep]])
          and np.array_equal(C['mass'], A['mass'][ia]))

    # ---- 7. kernel copy in sync
    mk = os.path.join(HERE, 'make_e2_bank_kernel.py')
    if os.path.exists(mk):
        p = subprocess.run([sys.executable, mk, '--check'], capture_output=True, text=True)
        check('kernel notebook embeds the current build_e2_bank.py', p.returncode == 0, (p.stdout + p.stderr).strip()[-300:])

    if not a.keep:
        shutil.rmtree(work)
    nf = sum(1 for _, ok_ in RESULTS if not ok_)
    print(f'{len(RESULTS) - nf}/{len(RESULTS)} checks passed in {time.time() - t0:.0f}s' + ('' if a.keep else '; work dir removed'))
    return 1 if nf else 0


if __name__ == '__main__':
    sys.exit(main())
