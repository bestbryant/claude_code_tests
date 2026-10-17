"""Modification-site localisation from a parent (analog) spectrum and the query spectrum.

Parent peaks that reappear unshifted in the query come from fragments that do not contain the modification;
peaks that reappear shifted by delta come from fragments that do. A candidate site is scored by how many
(intensity-weighted) parent peaks have an explaining fragment consistent with that pattern.
"""
import numpy as np

from .frag import fragments, annotate
from .spectra import modcos_shifted_flags


class ParentEvidence:
    def __init__(self, parent_mol, pmz, pit, qmz, qit, delta, positive=True, tol=0.01):
        self.flags, self.w = modcos_shifted_flags(pmz.astype(np.float64), pit.astype(np.float64),
                                                  qmz.astype(np.float64), qit.astype(np.float64), delta, tol)
        idx = np.where(self.flags > 0)[0]
        self.frags = fragments(parent_mol) if len(idx) else []
        ann = annotate(self.frags, pmz[idx], positive, tol) if len(idx) else []
        self.peaks = [(int(self.flags[i]), float(self.w[i]), [a[0] for a in ann[t]]) for t, i in enumerate(idx) if ann[t]]
        self.n_shift = int((self.flags == 2).sum()); self.n_unshift = int((self.flags == 1).sum())

    def score(self, sites, removed):
        mask = 0
        for i in list(sites) + list(removed):
            mask |= (1 << i)
        if mask == 0 or not self.peaks:
            return 0.0, 0.0
        tot = good = 0.0
        for flag, w, masks in self.peaks:
            want = flag == 2
            ok = any(((m & mask) != 0) == want for m in masks)
            # a fragment set that is identical regardless of site carries no information
            tot += w
            good += w if ok else -w
        return good / tot if tot > 0 else 0.0, tot
