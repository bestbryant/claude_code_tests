"""Merge novel-structure candidates into a finished submission.

For each molecule whose main pipeline evidence is weak (library similarity below `lib_gate`), novel candidates
with score >= `min_score` are inserted at fixed 1-based `slots`; the list is deduplicated by metric key and cut to 25.
"""
import pandas as pd

from .metric import key14


def merge_lists(main, novel, slots):
    """main: list of smiles; novel: list of smiles (best first); slots: 1-based target positions for novel items."""
    out = list(main)
    keys = [key14(s) for s in out]
    ins = 0
    for slot, s in zip(slots, novel):
        k = key14(s)
        if k is None or k in keys:
            continue
        pos = min(slot - 1, len(out))
        out.insert(pos, s); keys.insert(pos, k); ins += 1
    seen, res = set(), []
    for s, k in zip(out, keys):
        if k is None or k in seen:
            continue
        seen.add(k); res.append(s)
    return res[:25], ins


def apply(sub: pd.DataFrame, novel: dict, lib_sim: dict, lib_gate=0.7, min_score=-1e9, slots=(2, 4, 6, 8, 10),
          max_insert=5):
    rows, n_mol, n_ins = [], 0, 0
    for mid, ans in zip(sub.molecule_id.astype(str), sub.smiles):
        main = [t for t in str(ans).split(';') if t]
        cand = [c[0] for c in (novel.get(mid) or []) if c[1] >= min_score][:max_insert]
        if cand and lib_sim.get(mid, 0.0) < lib_gate:
            new, k = merge_lists(main, cand, slots)
            if k:
                n_mol += 1; n_ins += k
            rows.append(';'.join(new))
        else:
            rows.append(ans)
    out = sub.copy()
    out['smiles'] = rows
    return out, dict(molecules=n_mol, inserted=n_ins)
