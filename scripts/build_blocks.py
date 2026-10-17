"""Building-block census: hydrolyse every training structure at amide / ester / glycosidic bonds.

Writes work/blocks_pieces.parquet (key, bond, acid, partner).
"""
import sys
import pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.build import hydrolyse_all

if __name__ == '__main__':
    ts = pd.read_parquet('/home/user/pub/pool/train_structs.parquet')
    df = hydrolyse_all(ts, procs=4)
    df.to_parquet('/home/user/work/blocks_pieces.parquet')
    print(len(df), df.bond.value_counts().to_dict())
