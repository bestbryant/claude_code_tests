"""Null Engine 2 of the eligible fork (elig0; fix F2a of docs/fork_eligibility.md). Ours, MIT.

Engine 2 of the frozen 0.465 pipeline (prvsiyan FP models / ranker rows / COCONUT bank, megayak rows, sergiomalv
ChEBI/LMSD, franciscoangulo PubChemLite) is not run. Every reader of its output (ENG, eng_lists.json) gets Engine 1's
own lists instead, in Engine 2's schema (eng_runner.py of frozen cell 32):

    {molecule_id: {'smiles': [smiles, ...], 'keys': [key, ...]}}      (top 40 per molecule)

`null_engine_lists` builds that dict: each molecule's ranked list is copied verbatim (same SMILES, same order, the
placeholder 'CCO' and empty entries dropped), then the tail is appended (only SMILES whose key is new for the
molecule) up to n entries. Keys come with the list (`keys`) or from `key_of`; they are the tautomer-canonical
InChIKey14 (Ahmed's chem.score_key), the identity Engine 2 wrote (its canon_key) and that cell 45 fuses on.
Copying the list verbatim (repeats included) makes a weighted reciprocal-rank fusion of the list with itself
order-preserving, whatever identity the fusion uses: every entry scores (1 + alpha) times its own score.
Pure functions; only write_lists touches the disk.
"""
import json
import os

PLACEHOLDER = 'CCO'      # the frozen pipeline's empty-list placeholder; Engine 2 never emitted it


def _get(mapping, mid):
    """mapping[mid] for a str or raw molecule id; None when absent (or mapping is None)."""
    if mapping is None:
        return None
    if mid in mapping:
        return mapping[mid]
    return mapping.get(str(mid))


def _usable(smiles):
    """The frozen readers' own filter (cells 45, 49, 52: `s and s != 'CCO'`), so list positions line up exactly."""
    return isinstance(smiles, str) and smiles != '' and smiles != PLACEHOLDER


def null_engine_lists(mids, lists, keys=None, tails=None, key_of=None, n=40):
    """Engine 2's schema from Engine 1's lists.

    mids:    molecule ids to cover; each one gets an entry (empty lists when it has no candidate), so the frozen
             asserts that Engine 2 covers every active molecule hold (cells 48 and 50).
    lists:   {mid: ranked SMILES}, copied verbatim (placeholder / empty entries dropped), at most n.
    keys:    {mid: keys parallel to lists[mid]} (Engine 1's own keys), or None to take key_of(smiles).
    tails:   {mid: ranked SMILES} appended after the list; an entry is kept only when its key (key_of) is truthy and
             new for the molecule and its SMILES is not listed yet.
    key_of:  smiles -> key (required when keys is None or tails is given).
    n:       maximum entries per molecule (Engine 2 wrote its top 40).
    Returns {str(mid): {'smiles': [...], 'keys': [...]}} in the order of mids (first occurrence of each id)."""
    assert isinstance(n, int) and n > 0, n
    out = {}
    for mid in mids:
        m = str(mid)
        if m in out:
            continue
        head = list(_get(lists, mid) or [])
        if keys is not None:
            head_keys = list(_get(keys, mid) or [])
            assert len(head_keys) == len(head), (m, len(head_keys), len(head))
        else:
            assert key_of is not None or not head, 'key_of is required without keys'
            head_keys = [key_of(s) if _usable(s) else None for s in head]
        smiles, ids = [], []
        for s, k in zip(head, head_keys):
            if len(smiles) >= n:
                break
            if _usable(s):
                smiles.append(s)
                ids.append(k)
        tail = list(_get(tails, mid) or [])
        if tail and len(smiles) < n:
            assert key_of is not None, 'key_of is required for tails'
            seen_keys, seen_smiles = {k for k in ids if k}, set(smiles)
            for s in tail:
                if len(smiles) >= n:
                    break
                if not _usable(s) or s in seen_smiles:
                    continue
                k = key_of(s)
                if not k or k in seen_keys:
                    continue
                smiles.append(s)
                ids.append(k)
                seen_keys.add(k)
                seen_smiles.add(s)
        out[m] = {'smiles': smiles, 'keys': ids}
    return out


def summary(eng, mids=None):
    """One log line; 'engine lists <N> ' matches the notebook's compact-trace filter."""
    rows = list(eng.values()) if mids is None else [eng.get(str(m), {'smiles': []}) for m in mids]
    sizes = [len(r['smiles']) for r in rows]
    return (f'engine lists {len(eng)} | nonempty {sum(1 for s in sizes if s)} | candidates {sum(sizes)} | '
            f'min {min(sizes) if sizes else 0} max {max(sizes) if sizes else 0}')


def write_lists(eng, path):
    """Write eng_lists.json (Engine 2's file) atomically; keys that are numpy strings are written as str."""
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(eng, handle, default=str)
    os.replace(tmp, path)
    return path
