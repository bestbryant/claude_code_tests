"""CASMI26 class-3 probe: submit ONLY novel-structure candidates (analog derivatives + building-block conjugates,
all absent from PubChem and the training set). The public score of this list measures how much of the hidden
test our class-3 engine can reach on its own.
"""
import glob, os, subprocess, sys, time
os.environ['NUMBA_NUM_THREADS'] = '1'
T0 = time.time()
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
LOCAL = os.environ.get('CASMI_LOCAL') == '1'
if not LOCAL:
    whl = [w for w in glob.glob('/kaggle/input/**/rdkit-2026.3.3-*.whl', recursive=True) if tag in w]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
CODE = os.environ.get('CASMI_CODE') or os.path.dirname(os.path.dirname(
    sorted(glob.glob('/kaggle/input/**/casmi/novel.py', recursive=True), key=len)[0]))
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from multiprocessing import Pool
from casmi.build import build_analog_lib, structures, hydrolyse_all
from casmi.analog import AnalogLib
from casmi.conjugate import BlockLib
from casmi.keyset import KeySet, pack
from casmi.novel import NovelEngine, interleave

COMP = os.environ.get('CASMI_COMP') or os.path.dirname(sorted(glob.glob('/kaggle/input/**/test.parquet', recursive=True), key=len)[0])
PCK = os.environ.get('CASMI_PCK') or sorted(glob.glob('/kaggle/input/**/pc_keys_u64.npy', recursive=True), key=len)[0]
TRAIN = f'{COMP}/train.parquet'
W = os.environ.get('CASMI_WORK', '/kaggle/working')

S = structures(TRAIN); print('structures', len(S), f'{time.time()-T0:.0f}s', flush=True)
n = build_analog_lib(TRAIN, f'{W}/analog_lib.npz'); print('analog lib', n, f'{time.time()-T0:.0f}s', flush=True)
P = hydrolyse_all(S, procs=4); print('pieces', len(P), f'{time.time()-T0:.0f}s', flush=True)

L = AnalogLib(f'{W}/analog_lib.npz')
B = BlockLib(P, min_count=1)
E = NovelEngine(L, B, KeySet.load(PCK), KeySet(np.unique(pack(list(S.inchikey14)))), dict(zip(S.inchikey14, S.mass)))

test = pd.read_parquet(f'{COMP}/test.parquet')
mids = list(dict.fromkeys(test.molecule_id))


def run(mid):
    g = test[test.molecule_id == mid]
    spectra = [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float),
                    prec=float(r.precursor_mz), adduct=r.adduct, positive=(r.ionization_mode == 'positive'))
               for r in g.itertuples()]
    try:
        d, c, M = E.predict(spectra)
        return mid, [x[0] for x in interleave(d, c, first='d')]
    except Exception as e:
        print('fail', mid, repr(e), flush=True)
        return mid, []


with Pool(4) as pool:
    res = dict(pool.imap_unordered(run, mids, chunksize=2))
print('predicted', f'{time.time()-T0:.0f}s', flush=True)
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
sub['smiles'] = [';'.join(res.get(m) or ['C']) for m in sub.molecule_id]
sub.to_csv(f'{W}/submission.csv', index=False)
print('rows with candidates', sum(bool(res.get(m)) for m in sub.molecule_id), 'of', len(sub), f'{time.time()-T0:.0f}s')
