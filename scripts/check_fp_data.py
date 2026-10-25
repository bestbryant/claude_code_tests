"""Check a prep_fp_data.py output dir against the published casmi26-fp-data (Oct 17; docs/reproduce.md G7).

Pool v1 itself is not published: structs.key / pool_idx, negs.npy and negfp.npy carry its key set and row order, so a
match here also checks the pool v1 rebuild (scripts/build_pool_v1.py). Compares dtype, shape and the md5 of the raw
array bytes of all 16 arrays, one array at a time (peak RSS ~0.4 GB, a few seconds).
Usage: python scripts/check_fp_data.py <fp_data_dir>
"""
import hashlib, sys
import numpy as np

REF = {  # file: {array: (dtype, shape, md5 of the C-order bytes)}
    'spectra.npz': {
        'mz': ('<f4', (97906647,), '1c269b20a7b475899a44c6ce6723ba26'),
        'it': ('<f2', (97906647,), '6e8ce4104dac58f83cc971874d875559'),
        'off': ('<i8', (2531378,), '42fea51f6867fcb613298bdcce120c50'),
        'prec': ('<f4', (2531377,), 'f901eb39f859365f3b04feb4812fc900'),
        'ce': ('<f4', (2531377,), '4ede788540d710d8c19b386a398c7e8b'),
        'add_id': ('|i1', (2531377,), 'b668814719178b4818c789be67de9801'),
        'inst_id': ('|i1', (2531377,), 'fc8b2a306ee064522bd08e4ca1f381e3'),
        'pol': ('|i1', (2531377,), '3812e01ed0fd3237f19de386e6ea285a'),
        'sidx': ('<i4', (2531377,), '9d774c477a9253ea5d12bf1503be84e0'),
        'split': ('|i1', (2531377,), 'a7a81afa909d04ad965c13f0c5710385')},
    'structs.npz': {
        'fp': ('|u1', (275158, 277), 'c422e15b33836b1a459625dbf439f306'),
        'mass': ('<f8', (275158,), '222111313324008316be6dc00fa72c96'),
        'key': ('<U14', (275158,), '03300e1d48b8ef2ba01fb549ff6aba09'),
        'pool_idx': ('<i8', (275158,), '938210f541d612023cee624fc52e5b82')},
    'negs.npy': {'negs': ('<i4', (275158, 64), 'a8e44a322853902fdf8826589327043d')},
    'negfp.npy': {'negfp': ('|u1', (623595, 277), '133340cbf655b8ef6d1b87528abd8a82')},
}

d = sys.argv[1]
bad = []
for f, arrs in REF.items():
    z = {f[:-4]: np.load(f'{d}/{f}', mmap_mode='r')} if f.endswith('.npy') else np.load(f'{d}/{f}')
    for k, ref in arrs.items():
        a = np.ascontiguousarray(z[k])
        got = (a.dtype.str, a.shape, hashlib.md5(a.view(np.uint8)).hexdigest())
        print(f, k, *got, 'ok' if got == ref else f'MISMATCH (expected {ref})', flush=True)
        if got != ref:
            bad.append(f'{f}:{k}')
        del a
assert not bad, f'differs from published casmi26-fp-data: {bad}'
print('fp-data ok')
