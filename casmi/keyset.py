"""Compact InChIKey14 membership: first 13 letters packed base-26 into uint64, sorted array + searchsorted."""
import numpy as np


def pack(keys):
    """keys: iterable of 14-letter InChIKey first blocks -> uint64 array (first 13 letters, base 26)."""
    a = np.frombuffer(''.join(k[:13] for k in keys).encode('ascii'), dtype=np.uint8).reshape(-1, 13).astype(np.uint64) - 65
    v = np.zeros(len(a), np.uint64)
    for c in range(13):
        v = v * np.uint64(26) + a[:, c]
    return v


class KeySet:
    def __init__(self, arr):
        self.a = np.sort(arr) if not np.all(arr[:-1] <= arr[1:]) else arr

    @classmethod
    def load(cls, path):
        return cls(np.load(path, mmap_mode='r'))

    def contains(self, keys):
        keys = [k for k in keys]
        ok = np.array([isinstance(k, str) and len(k) >= 13 and k[:13].isalpha() and k[:13].isupper() for k in keys])
        out = np.zeros(len(keys), bool)
        if ok.any():
            v = pack([k for k, o in zip(keys, ok) if o])
            i = np.searchsorted(self.a, v)
            i = np.minimum(i, len(self.a) - 1)
            out[np.where(ok)[0]] = np.asarray(self.a)[i] == v
        return out
