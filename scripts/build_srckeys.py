"""Source key sets of the LB probes and the source-tiering layer (olast): casmi_probe_srckeys.parquet (bryantyu/casmi26-srckeys).

python scripts/build_srckeys.py <pool_dir> <out.parquet>       (pool v4: /home/user/work/pool4)
Columns key (InChIKey14) and src: 'train' = a training structure (in_train), 'coco' = a COCONUT structure that is not a
training structure; every other key is 'other' (absent from the file). Reproduces the file used since Oct 20.
"""
import sys
import pandas as pd

pool_dir, out = sys.argv[1:3]
P = pd.read_parquet(f'{pool_dir}/pool.parquet', columns=['key', 'in_train', 'in_coco']).groupby('key').max()
P = P[(P.in_train == 1) | (P.in_coco == 1)].reset_index()
P['src'] = P.in_train.map({1: 'train', 0: 'coco'})
P[['key', 'src']].to_parquet(out, index=False)
print(P.src.value_counts())
