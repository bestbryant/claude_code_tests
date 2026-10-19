# Kaggle CPU job: DreaMS embeddings of analog-library shard 1/5 (self-contained copy of casmi/dreams_emb.py).
"""DreaMS spectrum embeddings (Bushuiev et al. 2025, MIT; TorchScript weights from huggingface.co/roman-bushuiev/DreaMS).

Preprocessing as dreams.utils.data.SpectrumPreprocessor (DataFormatA defaults): the 100 most intense peaks,
intensities relative to the base peak, zero padding, precursor prepended with intensity 1.1 -> (101, 2) float32.
python -m casmi.dreams_emb <model.pt> <train.parquet> <rows.npy> <shard> <n_shards> <out.npz> [threads]
embeds train.parquet rows rows[shard::n_shards] (float16, L2-normalised) for the analog library.
"""
import sys, time
import numpy as np

N_PEAKS, PREC_INTENS = 100, 1.1


def prep(mz, it, prec, n=N_PEAKS):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    x = np.zeros((n + 1, 2), np.float32)
    x[0] = (prec, PREC_INTENS)
    if len(mz) and it.max() > 0:
        o = np.argsort(it, kind='stable')[-n:]
        x[1:len(o) + 1, 0] = mz[o]; x[1:len(o) + 1, 1] = it[o] / it[o].max()
    return x


class DreamsEmbedder:
    def __init__(self, path, threads=4):
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.m = torch.jit.load(path, map_location='cpu').eval()

    def embed(self, spectra, bs=64):
        """spectra: iterable of (mz, it, prec) -> (n, 1024) float32, L2-normalised."""
        X = np.stack([prep(*s) for s in spectra]) if len(spectra) else np.zeros((0, N_PEAKS + 1, 2), np.float32)
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(X), bs):
                out.append(self.m(self.torch.from_numpy(X[i:i + bs])).float().numpy())
        E = np.concatenate(out) if out else np.zeros((0, 1024), np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


import glob, os
import pyarrow.parquet as pq
first = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
model, train, rows_p = first('DreaMS_embedding_model_torchscript.pt'), first('train.parquet'), first('lib_rows.npy')
SHARD, N_SH = 1, 5
rows = np.load(rows_p)[SHARD::N_SH]
want = np.zeros(int(rows.max()) + 1, bool); want[rows] = True
pf = pq.ParquetFile(train)
em = DreamsEmbedder(model, os.cpu_count() or 4)
got_rows, embs, base, t0 = [], [], 0, time.time()
for g in range(pf.metadata.num_row_groups):
    n = pf.metadata.row_group(g).num_rows
    idx = np.arange(base, base + n)
    loc = (idx[idx < len(want)][want[idx[idx < len(want)]]] - base).tolist()
    if loc:
        t = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz']).take(loc).to_pandas()
        for i in range(0, len(t), 2048):
            c = t.iloc[i:i + 2048]
            embs.append(em.embed(list(zip(c.ms2_mzs, c.ms2_normalized_intensities, c.precursor_mz))).astype(np.float16))
            got_rows.extend(base + loc[j] for j in range(i, i + len(c)))
            print(f'{len(got_rows)}/{len(rows)} {time.time() - t0:.0f}s {len(got_rows) / (time.time() - t0):.1f}/s', flush=True)
    base += n
np.savez(f'/kaggle/working/dreams_lib_{SHARD}.npz', row=np.array(got_rows, np.int64), emb=np.concatenate(embs))
print('done', len(got_rows), f'{time.time() - t0:.0f}s')
