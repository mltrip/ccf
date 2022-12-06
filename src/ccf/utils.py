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
  for future in as_completed(future2callable):
    try:
      r = future.result()
    except Exception as e:
      print(f'Exception: {future} - {e}')
    else:
      print(f'Done: {future} - {r}')
    finally:  # Resubmit
      c, kwargs = future2callable[future]
      new_future = executor.submit(c, **kwargs)
      new_future2callable = {new_future: [c, kwargs]}
      loop(executor, new_future2callable)

      
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
