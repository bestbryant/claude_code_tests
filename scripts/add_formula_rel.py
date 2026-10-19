"""Rows with FP features -> + formula and within-formula relative features (casmi.own.formula_rel).

python scripts/add_formula_rel.py <rows_fp.parquet> <pool_dir> <out.parquet>
"""
import sys
import pandas as pd
sys.path.insert(0, '/home/user/claude_code_tests')
from casmi.own import formula_rel

rows, pool, out = sys.argv[1:4]
D = pd.read_parquet(rows)
D['formula'] = pd.read_parquet(f'{pool}/pool.parquet', columns=['formula']).formula.values[D.pid.values]
formula_rel(D).to_parquet(out)
print(out, D.shape)
