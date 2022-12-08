import warnings
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np


def expand_columns(ref_columns, columns):
  new_columns = []
  for c in columns:
    n = len(new_columns)
    for cc in ref_columns:
      if re.match(c, cc):
        new_columns.append(cc)
    if len(new_columns) == n:
      warnings.warn(f'Warning! No columns found with pattern "{c}"') 
  if len(new_columns) == 0 and len(columns) != 0:
    warnings.warn(f'Warning! No columns found with patterns: {columns}')
  new_columns = list(set(new_columns))  # remove duplicates
  return new_columns


def loop(executor, future2callable):
  for f in as_completed(future2callable):
    try:
      r = f.result()
    except Exception as e:
      c, ks = future2callable[f]
      print(f'Exception: {f}, {c}, {ks}')
    else:
      c, ks = future2callable[f]
      print(f'Done: {f}, {c}, {ks}')
    finally:  # Resubmit
      c, ks = future2callable[f]
      f = executor.submit(c, **ks)
      f2c = {f: [c, ks]}
      loop(executor, f2c)

      
def delta2value(deltas, kind, initial_value=1):
  if kind == 'lograt':
    deltas = np.exp(np.cumsum(deltas))
  elif kind == 'rat':
    deltas = np.cumprod(deltas)
  elif kind == 'rel':
    deltas = np.cumprod(1 + deltas)
  else:
    raise NotImplementedError(kind)
  return deltas * initial_value
