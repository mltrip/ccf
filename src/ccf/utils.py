import warnings
import re

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


def rat2val(ratios, initial_value=1):
  if 'lograt_' in ratios.name:
    ratios = np.exp(ratios.fillna(0).cumsum())
  elif 'rat_' in ratios.name:
    ratios = ratios.fillna(1).cumprod()
  elif 'pct_' in ratios.name:
    ratios = 1 + ratios.fillna(0).cumsum()
  else:
    raise NotImplementedError(ratios.name)
  ratios = ratios * initial_value
  return ratios
