"""Shared spectrum-metadata encodings (instrument family) used in training and inference."""


def inst_family(s):
    s = s.lower() if isinstance(s, str) else ''
    if 'tims' in s:
        return 1
    if 'orbitrap' in s or 'q exactive' in s or 'exploris' in s or 'ft' in s or 'hcd' in s:
        return 2
    if 'tof' in s:
        return 3
    if 'trap' in s or 'it' == s:
        return 4
    return 0
