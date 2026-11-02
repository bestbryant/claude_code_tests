"""Engine 2 clone, step 2: pack the Kaggle inputs of the replay (docs/engine2_study.md option R; dataset
bryantyu/casmi26-e2-replay-inputs). Our own code on our own files; nothing from Engine 2 is imported.

Usage: python scripts/e2_pack_inputs.py <split: val|train> [out_dir]        (default /home/user/work/e2_replay_inputs)

Inputs
  /home/user/work/ret18cu_<split>_pf.parquet   our stage-1 rows (qid, pid, key, label, cls, truth, q_rows, n_q, ppm,
                                               lib_n, in_coco, in_pcl, in_bio, pf_bern, pf_cos, lib, an_best, frag)
  /home/user/work/train_store/                 compact store of train.parquet (row order = train.parquet order):
                                               spec_meta + top-64 peaks <= precursor + 2 (scripts/build_spec_store.py)
  /home/user/work/pool_v18c/pool.parquet       mass, key, in_coco (md5 8afda40a...)
  /home/user/data/train_meta.parquet           inchikey14, ingest_lib (the npx set)
Outputs (all three files of a split are uploaded together; MANIFEST.json keeps every split packed so far)
  e2q_<split>.parquet     one row per query: qid int32, truth, cls int8, subset ('npx' if the truth is an
                          enveda-np-examples key, else 'np' if pool in_coco == 1, else 'other'; train_ret_ranker.py
                          L80-83), reach (truth row in_coco | in_pcl | in_bio; L28-30), q_rows (original order),
                          e_rows (the q_rows with >= 3 peaks after clean_peaks(store peaks, electron-corrected precursor,
                          rel_floor 0.005, top_k 64), i.e. our rows' query Q, dump_ret_rows.py L71-79; asserted len == n_q),
                          n_q int8, n_q_rows int8, M_ret float64 (= mass[pid] / (1 + ppm / 1e6), the ret18cu target),
                          n_ret int32, truth_pid_ret int32 (-1 if absent), lib_n_truth float32, art_ret int8
                          (cls == 1 and lib_n_truth == 0)
  e2pf_<split>.parquet    qid, pid int32 + pf_bern, pf_cos, lib, an_best, frag float32, sorted by (qid, pid)
  e2qspec_<split>.npz     the distinct q_rows: row int64 (sorted), off int64, mz float32 / it float16 (the store bytes),
                          prec float64 (raw), ik U14, npk int64 (num_peaks), adduct U16, ingest_lib U24,
                          instrument_type U48 ('' for None), ce float64 (mean collision energy, -1 if absent)
  MANIFEST.json           what, packer sha256, per split: source md5 / bytes, output md5 / bytes / rows, counts
  e2_pack_inputs.py       a copy of this packer (provenance)
"""
import datetime
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from casmi.spectra import Store, clean_peaks  # noqa: E402

W = '/home/user/work'
DEFAULTS = dict(ret=W + '/ret18cu_{split}_pf.parquet', store=W + '/train_store', pool=W + '/pool_v18c/pool.parquet',
                train_meta='/home/user/data/train_meta.parquet')
WHAT = 'Engine 2 clone replay inputs (docs/engine2_study.md option R step 2)'
ELECTRON = 0.00054858                 # scripts/dump_ret_rows.py L19
NPX_LIB = 'enveda-np-examples'
COLS = ['qid', 'pid', 'key', 'label', 'cls', 'truth', 'q_rows', 'n_q', 'ppm', 'lib_n', 'in_coco', 'in_pcl', 'in_bio',
        'pf_bern', 'pf_cos', 'lib', 'an_best', 'frag']
PF_COLS = ['pf_bern', 'pf_cos', 'lib', 'an_best', 'frag']
U = dict(ik=14, adduct=16, ingest_lib=24, instrument_type=48)


def digest(p, algo='md5'):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def corrected_prec(prec, ingest_lib, adduct):
    """dump_ret_rows.py L73-75: the hidden test carries the electron-mass fix; enveda-180 '+' rows in train do not."""
    return prec - ELECTRON if ingest_lib == 'enveda-180' and str(adduct).endswith('+') else prec


def e_row_ok(st, meta, i):
    mz, it = st.peaks(i)
    r = meta.loc[i]
    a, _ = clean_peaks(mz, it, corrected_prec(r.precursor_mz, r.ingest_lib, r.adduct), rel_floor=0.005, top_k=64)
    return len(a) >= 3


def _ustr(vals, n, name):
    s = ['' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in vals]
    m = max((len(x) for x in s), default=0)
    if m > n:
        raise ValueError(f'{name}: value of length {m} does not fit U{n}')
    return np.array(s, dtype=f'U{n}')


def pack(split, out_dir, ret=None, store=None, pool=None, train_meta=None, quiet=False):
    """Write the three files of `split` into out_dir and merge the split into out_dir/MANIFEST.json. Returns the split's
    manifest entry."""
    ret = ret or DEFAULTS['ret'].format(split=split)
    store = store or DEFAULTS['store']; pool = pool or DEFAULTS['pool']; train_meta = train_meta or DEFAULTS['train_meta']
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    D = pd.read_parquet(ret, columns=COLS).sort_values(['qid', 'pid']).reset_index(drop=True)
    st = Store(store); meta = st.meta
    P = pd.read_parquet(pool, columns=['key', 'mass', 'in_coco'])
    mass = P.mass.values
    m = pd.read_parquet(train_meta, columns=['inchikey14', 'ingest_lib'])
    npx = set(m.inchikey14[m.ingest_lib == NPX_LIB])
    coco = P.drop_duplicates('key').set_index('key').in_coco
    assert np.array_equal(P.key.astype(str).values[D.pid.values], D.key.astype(str).values), 'pid -> pool key mismatch'

    g = D.groupby('qid', sort=True)
    Q = g.agg(truth=('truth', 'first'), cls=('cls', 'first'), q_rows=('q_rows', 'first'), n_q=('n_q', 'first'),
              n_ret=('pid', 'size')).reset_index()
    for c in ('truth', 'cls', 'q_rows', 'n_q'):
        assert (g[c].nunique() == 1).all(), f'{c} varies within a query'
    mr = mass[D.pid.values] / (1 + D.ppm.values / 1e6)
    mq = pd.Series(mr).groupby(D.qid.values)
    assert float((mq.max() - mq.min()).max()) < 1e-6, 'M_ret differs within a query'
    Q['M_ret'] = mq.first().reindex(Q.qid).values
    t = D[D.label == 1].set_index('qid')
    assert t.index.is_unique, 'more than one truth row in a query'
    Q['truth_pid_ret'] = Q.qid.map(t.pid).fillna(-1).astype(np.int32)
    Q['lib_n_truth'] = Q.qid.map(t.lib_n).fillna(-1).astype(np.float32)
    reach = (t.in_coco == 1) | (t.in_pcl == 1) | (t.in_bio == 1)
    Q['reach'] = Q.qid.map(reach).fillna(False).astype(bool)
    Q['art_ret'] = ((Q.cls == 1) & (Q.lib_n_truth == 0)).astype(np.int8)
    Q['subset'] = np.where(Q.truth.isin(npx), 'npx', np.where(Q.truth.map(coco) == 1, 'np', 'other'))

    ok_row, erows, nqr, bad, allrows = {}, [], [], [], set()
    for qid, qr, nq in zip(Q.qid, Q.q_rows, Q.n_q):
        rows = [int(x) for x in str(qr).split(',')]
        allrows.update(rows)
        keep = []
        for i in rows:
            if i not in ok_row:
                ok_row[i] = e_row_ok(st, meta, i)
            if ok_row[i]:
                keep.append(i)
        if len(keep) != nq:
            bad.append(int(qid))
        erows.append(','.join(map(str, keep))); nqr.append(len(rows))
    if bad:
        raise AssertionError(f'{len(bad)} queries with len(e_rows) != n_q, e.g. {bad[:5]}')
    Q['e_rows'] = erows
    Q['n_q_rows'] = np.array(nqr, np.int8)
    truth_ik = meta.inchikey14.values
    for qid, qr, tr in zip(Q.qid, Q.q_rows, Q.truth):
        rows = [int(x) for x in str(qr).split(',')]
        assert all(truth_ik[i] == tr for i in rows), f'query {qid}: a q_row is not a spectrum of the truth'
    Q = Q.astype({'qid': 'int32', 'cls': 'int8', 'n_q': 'int8', 'n_ret': 'int32'})
    Q = Q[['qid', 'truth', 'cls', 'subset', 'reach', 'q_rows', 'e_rows', 'n_q', 'n_q_rows', 'M_ret', 'n_ret',
           'truth_pid_ret', 'lib_n_truth', 'art_ret']]
    os.makedirs(out_dir, exist_ok=True)
    f_q = os.path.join(out_dir, f'e2q_{split}.parquet')
    Q.to_parquet(f_q, index=False)

    f_pf = os.path.join(out_dir, f'e2pf_{split}.parquet')
    E = D[['qid', 'pid'] + PF_COLS].astype({'qid': 'int32', 'pid': 'int32', **{c: 'float32' for c in PF_COLS}})
    E.to_parquet(f_pf, index=False)

    rows = np.array(sorted(allrows), np.int64)
    off = np.zeros(len(rows) + 1, np.int64)
    mzs, its = [], []
    for j, i in enumerate(rows):
        a, b = st.off[i], st.off[i + 1]
        mzs.append(np.asarray(st.mz[a:b])); its.append(np.asarray(st.it[a:b])); off[j + 1] = off[j] + (b - a)
    mm = meta.loc[rows]
    ce = np.array([float(np.mean(c)) if c is not None and len(c) else -1.0 for c in mm.collision_energy_ev], np.float64)
    mzc = np.concatenate(mzs) if mzs else np.zeros(0, np.float32)
    itc = np.concatenate(its) if its else np.zeros(0, np.float16)
    assert mzc.dtype == np.float32 and itc.dtype == np.float16, (mzc.dtype, itc.dtype)
    f_qs = os.path.join(out_dir, f'e2qspec_{split}.npz')
    np.savez_compressed(f_qs, row=rows, off=off, mz=mzc, it=itc, prec=mm.precursor_mz.values.astype(np.float64),
                        ik=_ustr(mm.inchikey14.values, U['ik'], 'ik'), npk=mm.num_peaks.values.astype(np.int64),
                        adduct=_ustr(mm.adduct.values, U['adduct'], 'adduct'),
                        ingest_lib=_ustr(mm.ingest_lib.values, U['ingest_lib'], 'ingest_lib'),
                        instrument_type=_ustr(mm.instrument_type.values, U['instrument_type'], 'instrument_type'), ce=ce)

    srcs = {'ret': ret, 'spec_meta': f'{store}/spec_meta.parquet', 'spec_off': f'{store}/spec_off.npy',
            'spec_mz': f'{store}/spec_mz.npy', 'spec_int': f'{store}/spec_int.npy', 'pool': pool, 'train_meta': train_meta}
    entry = dict(
        sources={k: dict(path=p, md5=digest(p), bytes=os.path.getsize(p)) for k, p in srcs.items()},
        outputs={k: dict(file=os.path.basename(p), md5=digest(p), bytes=os.path.getsize(p), rows=n)
                 for k, p, n in (('e2q', f_q, len(Q)), ('e2pf', f_pf, len(E)), ('qspec', f_qs, len(rows)))},
        n_queries=int(len(Q)), n_rows=int(len(E)), n_q_rows_distinct=int(len(rows)),
        n_e_rows_distinct=int(sum(ok_row.values())),
        queries_with_dropped_q_rows=int((Q.n_q_rows != Q.n_q).sum()),
        counts={f'{a}_c{b}': int(v) for (a, b), v in Q.groupby(['subset', 'cls']).size().items()},
        class_counts={int(k): int(v) for k, v in Q.cls.value_counts().items()},
        truth_in_window=int((Q.truth_pid_ret >= 0).sum()), art_ret=int(Q.art_ret.sum()),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
    man_p = os.path.join(out_dir, 'MANIFEST.json')
    man = {}
    if os.path.exists(man_p):
        with open(man_p) as f:
            man = json.load(f)
    me = os.path.abspath(__file__)
    man.update(what=WHAT, packer=dict(file='e2_pack_inputs.py', sha256=digest(me, 'sha256')))
    man.setdefault('splits', {})[split] = entry
    with open(man_p + '.tmp', 'w') as f:
        json.dump(man, f, indent=1)
    os.replace(man_p + '.tmp', man_p)
    if os.path.abspath(os.path.join(out_dir, 'e2_pack_inputs.py')) != me:
        shutil.copy(me, os.path.join(out_dir, 'e2_pack_inputs.py'))
    say(split, 'queries', len(Q), 'rows', len(E), 'distinct q_rows', len(rows), 'counts', entry['counts'])
    for k, o in entry['outputs'].items():
        say(f'  {o["file"]}: {o["bytes"]:,} B md5 {o["md5"]} rows {o["rows"]:,}')
    return entry


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('val', 'train'):
        sys.exit(__doc__)
    pack(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else W + '/e2_replay_inputs')
