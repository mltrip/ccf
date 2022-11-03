import warnings
import re

import pandas as pd
import numpy as np


def expand_columns(df, columns):
  new_columns = []
  for c in columns:
    n = len(new_columns)
    for cc in df.columns:
      if re.match(c, cc):
        new_columns.append(cc)
    if len(new_columns) == n:
      warnings.warn(f'Warning! No columns found with pattern "{c}"') 
  if len(new_columns) == 0 and len(columns) != 0:
    warnings.warn(f'Warning! No columns found with patterns: {columns}')
  new_columns = list(set(new_columns))  # remove duplicates
  return new_columns
