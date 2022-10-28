import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yaml


def make_features(dfs, resample_kwargs, aggregate_kwargs, interpolate_kwargs,
                  pre_features, post_features):
  for g in dfs:
    for f in pre_features:
      if isinstance(f, str):
        if f == 'm_p':
          df = dfs[g]['orderbook']
          dfs[g]['orderbook']['m_p'] = 0.5*(df['a_p_0'] + df['b_p_0'])
    # Quantize
    for n, df in dfs[g].items():
      aks = aggregate_kwargs[n]
      iks = interpolate_kwargs[n]
      df = df.resample(**resample_kwargs).aggregate(**aks)
      if iks is not None:
        df = df.interpolate(**iks)
      dfs[g][n] = df
    df = pd.concat([x for x in dfs[g].values()], axis=1).sort_index()
    for f in post_features:
      if isinstance(f, str):
        if f == 'group':
          df['group'] = g
        elif f == 'time_idx':
          df['time_idx'] = np.arange(len(df))
    dfs[g] = df
  df = pd.concat([x for x in dfs.values()], axis=0)
  return df
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'make_features.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  df = make_features(**kwargs)
  print(df)