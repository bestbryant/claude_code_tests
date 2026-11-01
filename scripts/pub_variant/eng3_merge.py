"""ENG3 (ours, MIT): Engine 2's lists merged with our stage 1 by reciprocal rank, for the rank line (casmi26-pubm-eng3*).

Written next to the pipeline modules by notebook cell 8 (SHA-256 pinned) and imported by the patched frozen cell 39,
the single point where Engine 2's lists change. ENG is created in frozen cell 32 (the original Engine 2's
eng_lists.json, loaded verbatim) and first read by the ICE / GLACIER input union of cell 41; no cell in between reads
it, and the pipeline's `chem` (chem.score_key) is bound from cell 37 on (`from casmi import fpnet, chem`). At the end
of cell 39 ENG becomes, per molecule,

    the top n identities by   1 / (k + r_engine2) + w / (k + r_ours)

r = the 1-based position in each list (an identity absent from a list gets no term from it). The identity is the key
both lists carry: the pipeline's chem.score_key, the first block (14 characters) of the InChIKey of RDKit's canonical
tautomer -- the connectivity block: stereo is not stripped before canonicalisation, but this block carries none.
Engine 2 wrote its keys with its canon_key (the same RDKit calls; for an empty InChIKey it writes '', score_key None);
ours are written by our subprocess with the same function and re-checked with chem.score_key itself before the merge
(eng2_merge.verify_keys, as in elig1: a different key is replaced, a falsy or repeated one dropped). An Engine 2 entry
without a key ('' / None: RDKit gave no InChIKey) is not dropped: it stays its own identity at its own position, scored
1 / (k + r), with its SMILES and its key as written (the SMILES readers of pubm see it there). A repeated key is the
same identity again and scores nothing (neither engine writes one). k = 3 is the lineage's own RRF offset: KRR of
frozen cell 45's fuse2, also the offset of the metric-identity fusion (cell 49) and the window-40 overlay (cell 52),
so the merged list follows the same rank decay as every fusion that reads it. Ties: the lower Engine 2 position first,
then the lower position in ours (an identity missing from a list ranks after every listed one). The SMILES of an
identity is Engine 2's when Engine 2 lists it, else ours. A molecule without a list of ours keeps Engine 2's entry
unchanged (same SMILES, keys and length); a molecule without an Engine 2 list (should not happen) gets ours (top n).
Scores are exact fractions (fractions.Fraction): ties are exact, the order does not depend on float rounding.
Pure functions; only write_lists touches the disk.
"""
import json
import math
import os
from fractions import Fraction

K_RRF = 3          # frozen cell 45: ALPHA, KRR = 0.6, 3.0
N_TOP = 40         # Engine 2 wrote its top 40 (eng_runner.py of frozen cell 32); the ICE union reads [:40]


def _usable(v):
    """A list entry {'smiles': [...], 'keys': [...]} with at least one SMILES."""
    return isinstance(v, dict) and isinstance(v.get('smiles'), list) and bool(v.get('smiles'))


def _positions(v, keyless=False):
    """{identity: 1-based position}, {identity: smiles}, {identity: key as written} of a list; positions are those of
    the list as written. An entry with an empty or non-string SMILES, or whose key repeats an earlier one, keeps its
    slot but scores nothing. An entry without a (truthy) key scores nothing in our list (keyless=False; our key
    re-check drops such entries anyway); in Engine 2's list (keyless=True) it is its own identity ('', position),
    so it keeps its slot, its score 1 / (k + r), its SMILES and its key as written ('' or None)."""
    pos, smi, key = {}, {}, {}
    keys = v.get('keys') or []
    for r, s in enumerate(v.get('smiles') or [], 1):
        k = keys[r - 1] if r <= len(keys) else None
        if not isinstance(s, str) or not s:
            continue
        if not k:
            if not keyless:
                continue
            x = ('', r)                                    # never equal to a key (keys are str)
        elif k in pos:
            continue
        else:
            x = k
        pos[x] = r
        smi[x] = s
        key[x] = k
    return pos, smi, key


def _copy(v):
    """Engine 2's entry as it is (lists copied, other fields kept), so no caller aliases the original."""
    return {f: (list(x) if isinstance(x, list) else x) for f, x in v.items()} if isinstance(v, dict) else v


def rrf_merge(eng2, ours, w, k=K_RRF, n=N_TOP):
    """eng2, ours: {molecule_id: {'smiles': [...], 'keys': [...]}} -> (merged, stats).

    merged holds every molecule of eng2 (first, in its order) and then every molecule that only ours lists, as str ids.
    stats: molecules (entries of merged), engine2 / ours (molecules with a non-empty list), merged (both: RRF applied),
    engine2_only (kept unchanged), ours_only (ours taken), shared (identities in both input lists, summed over merged
    molecules), new_from_ours (identities of ours not in Engine 2's list that made the merged top n), displaced (Engine 2
    identities pushed out of the top n), top1_changed / top1_from_ours (merged leader differs from Engine 2's / comes from
    ours only), engine2_keyless / engine2_keyless_kept (Engine 2 entries without a key in merged molecules / of them in
    the merged top n), per_molecule rows for the report."""
    w = float(w)
    assert math.isfinite(w) and w > 0, f'w {w!r}: a finite weight > 0'
    assert isinstance(n, int) and n > 0 and float(k) >= 0, (k, n)
    wf, kf = Fraction(w), Fraction(k)
    eng2 = eng2 if isinstance(eng2, dict) else {}
    ours = ours if isinstance(ours, dict) else {}
    e_of, o_of = {}, {}
    for m, v in eng2.items():
        e_of.setdefault(str(m), v)
    for m, v in ours.items():
        o_of.setdefault(str(m), v)
    st = dict(w=w, k=k, n=n, molecules=0, engine2=sum(1 for v in e_of.values() if _usable(v)),
              ours=sum(1 for v in o_of.values() if _usable(v)), merged=0, engine2_only=0, ours_only=0, shared=0,
              new_from_ours=0, displaced=0, top1_changed=0, top1_from_ours=0, engine2_keyless=0,
              engine2_keyless_kept=0, ours_only_ids=[], per_molecule=[])
    out = {}
    for m in list(e_of) + [m for m in o_of if m not in e_of]:
        e, o = e_of.get(m), o_of.get(m)
        if not _usable(o):                                 # no list of ours: Engine 2's entry exactly as it was
            if m in e_of:
                out[m] = _copy(e)
                st['engine2_only'] += int(_usable(e))
            continue
        o_pos, o_smi, _ = _positions(o)
        if not _usable(e):                                 # no Engine 2 list (should not happen): ours, top n
            keys = sorted(o_pos, key=o_pos.get)[:n]
            out[m] = dict(smiles=[o_smi[x] for x in keys], keys=keys)
            st['ours_only'] += 1
            if len(st['ours_only_ids']) < 20:
                st['ours_only_ids'].append(m)
            continue
        e_pos, e_smi, e_key = _positions(e, keyless=True)
        score = {}
        for x, r in e_pos.items():
            score[x] = Fraction(1) / (kf + r)
        for x, r in o_pos.items():
            score[x] = score.get(x, Fraction(0)) + wf / (kf + r)
        inf = len(e_pos) + len(o_pos) + 1
        order = sorted(score, key=lambda x: (-score[x], e_pos.get(x, inf), o_pos.get(x, inf)))[:n]
        out[m] = dict(smiles=[e_smi[x] if x in e_smi else o_smi[x] for x in order],
                      keys=[e_key[x] if x in e_key else x for x in order])
        kept = set(order)
        shared = sum(1 for x in o_pos if x in e_pos)
        new = sum(1 for x in order if x not in e_pos)
        gone = sum(1 for x in e_pos if x not in kept)
        keyless = [x for x in e_pos if isinstance(x, tuple)]
        e_top = min(e_pos, key=e_pos.get) if e_pos else None
        st['merged'] += 1
        st['shared'] += shared
        st['new_from_ours'] += new
        st['displaced'] += gone
        st['top1_changed'] += int(bool(order) and order[0] != e_top)
        st['top1_from_ours'] += int(bool(order) and order[0] not in e_pos)
        st['engine2_keyless'] += len(keyless)
        st['engine2_keyless_kept'] += sum(1 for x in keyless if x in kept)
        st['per_molecule'].append(dict(molecule_id=m, engine2=len(e_pos), ours=len(o_pos), shared=shared,
                                       new_from_ours=new, displaced=gone, top1_changed=bool(order) and order[0] != e_top,
                                       engine2_keyless=len(keyless)))
    st['molecules'] = len(out)
    return out, st


def check_keys(lists, key_of, head=None):
    """Log only: compare the keys a list carries with key_of(smiles) (the pipeline's chem.score_key); nothing changes.
    head = n: the first n entries of each list only."""
    rep = dict(mode='all' if not head else f'first {int(head)} per molecule', molecules=0, checked=0, mismatches=0,
               examples=[])
    for m, v in (lists or {}).items():
        if not _usable(v):
            continue
        rep['molecules'] += 1
        pairs = list(zip(v['smiles'], v.get('keys') or []))
        for s, kk in pairs[:int(head)] if head else pairs:
            rep['checked'] += 1
            k2 = key_of(s)
            if k2 != kk:
                rep['mismatches'] += 1
                if len(rep['examples']) < 5:
                    rep['examples'].append(dict(molecule_id=str(m), smiles=s, key=kk, score_key=k2))
    return rep


def log_line(stats, seconds):
    """'ENG3 | w W | molecules N | ours N | merged N | shared identities S | new from ours M | S s' (S s: our
    subprocess's wall time, the cost of the extra engine)."""
    return (f"ENG3 | w {stats['w']!r} | molecules {stats['molecules']} | ours {stats['ours']} | merged {stats['merged']} | "
            f"shared identities {stats['shared']} | new from ours {stats['new_from_ours']} | {seconds} s")


def detail_line(stats):
    return (f"ENG3 merge | k {stats['k']} | top {stats['n']} | top1 changed {stats['top1_changed']} (from ours "
            f"{stats['top1_from_ours']}) | engine 2 entries displaced {stats['displaced']} | engine 2 only "
            f"{stats['engine2_only']} | ours only {stats['ours_only']} | engine 2 keyless kept "
            f"{stats.get('engine2_keyless_kept', 0)} / {stats.get('engine2_keyless', 0)}")


def keys_line(ours_check, eng2_check, seconds):
    c, e = ours_check or {}, eng2_check or {}
    return (f"ENG3 keys | ours ({c.get('mode')}): entries {c.get('entries')}, checked {c.get('checked')}, mismatches "
            f"{c.get('mismatches')}, dropped {c.get('dropped')}, emptied {c.get('emptied')} | engine 2 (log only, "
            f"{e.get('mode')}): checked {e.get('checked')}, mismatches {e.get('mismatches')} | {seconds:.0f} s")


def write_lists(eng, path):
    """eng_lists.json (Engine 2's schema), written atomically."""
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(eng, handle, default=str)
    os.replace(tmp, path)
    return path
