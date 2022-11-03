import sys
from datetime import datetime, timedelta, timezone
import time
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from ccf.read_data import read_data
from ccf.utils import expand_columns


def extract_features(raw_data_kwargs, feature_data_kwargs, pre_features, post_features, 
                     resample_kwargs, aggregate_kwargs, interpolate_kwargs, 
                     remove_pre_features=False,
                     delay=0, verbose=False):
  feature_io = deepcopy(feature_data_kwargs['query']['feature'])
  for n in feature_io:
    ek = feature_io[n]['engine_kwargs']
    feature_io[n]['write_kwargs']['con'] = create_engine(**ek)
  # Init pre features
  new_pre_features = []
  for f in pre_features:
    if isinstance(f, str):
      c = getattr(sys.modules[__name__], f)
      new_pre_features.append([c, {}])
    elif isinstance(f, dict):
      c = getattr(sys.modules[__name__], f.pop('feature'))
      new_pre_features.append([c, f])
    else:
      raise ValueError(f)
  pre_features = new_pre_features
  # Init post features
  new_post_features = []
  for f in post_features:
    if isinstance(f, str):
      c = getattr(sys.modules[__name__], f)
      new_post_features.append([c, {}])
    elif isinstance(f, dict):
      c = getattr(sys.modules[__name__], f.pop('feature'))
      new_post_features.append([c, f])
    else:
      raise ValueError(f)
  post_features = new_post_features
  while True:
    t0 = time.time()
    # Read
    raw = read_data(**raw_data_kwargs)
    old_feature = read_data(**feature_data_kwargs)['feature']
    pre_feature = {}
    # Pre resample features
    for n in raw:
      dfs = []
      for f, kwargs in pre_features:
        dfs.extend(f(n, raw, old_feature, None, **kwargs))
      pre_feature[n] = pd.concat(dfs, axis=1)
    # Resample features
    for n, df in pre_feature.items():
      pre_feature[n] = df.resample(**resample_kwargs).aggregate(**aggregate_kwargs).interpolate(**interpolate_kwargs)
    # Post resample features
    new_feature = {}
    for n in raw:
      dfs = [pre_feature[n]] if not remove_pre_features else []
      for f, kwargs in post_features:
        dfs.extend(f(n, raw, old_feature, pre_feature, **kwargs))
      new_feature[n] = pd.concat(dfs, axis=1)
    # Write
    for n, old_df in old_feature.items():
      new_df = new_feature[n]
      d = new_df.index.difference(old_df.index)
      new_df = new_df[new_df.index.isin(d)]
      if verbose:
        print(n)
        print(d)
        print(old_df.tail(1))
        print(new_df.tail(1))
      wk = feature_io[n]['write_kwargs']
      new_df.to_sql(**wk)
    dt = time.time() - t0
    if verbose:
      print(f'{datetime.utcnow()}, dt: {dt:.3f}')
    time.sleep(max(0, delay - dt))
  
  
def m_p(name, raw, old_feature, pre_feature, depth=1):
  o = raw[name]['o']
  depth = len(o.columns) if depth is None else depth  # +-
  dfs = []
  for d in range(depth):
    a, b, m = f'a_p_{d}', f'b_p_{d}', f'o_m_p_{d}'
    if a in o and b in o:
      df = 0.5*(o[a] + o[b])
      df.name = m
      dfs.append(df)
  return dfs
 

def get(name, raw, old_feature, pre_feature, columns=None):
  columns = list(df.columns) if columns is None else columns
  dfs = []
  for n, df in raw[name].items():
    columns = expand_columns(df, columns)
    if len(columns) > 0:
      df = df[columns].add_prefix(f'{n}_')
      dfs.append(df)
  return dfs


def relative(name, raw, old_feature, pre_feature, 
             kind='pct', column=None, columns=None, shift=0):
  df = pre_feature[name]
  columns = list(df.columns) if columns is None else columns
  columns = expand_columns(df, columns)
  if len(columns) == 0:
    return []
  a = df[columns]
  prefix = f'{kind}_{str(shift)}'
  if column is not None:
    b = df[column].shift(shift) if shift != 0 else df[column]
    new_columns = {x: '-'.join([prefix, x, column]) for x in df.columns}
  else:
    b = df[columns].shift(shift) if shift != 0 else df[columns]
    new_columns = {x: '-'.join([prefix, x, x]) for x in df.columns}
  if kind == 'pct':
    new_df = a.div(b, axis=0) - 1
  elif kind == 'rat':
    new_df = a.div(b, axis=0)
  elif kind == 'lograt':
    new_df = np.log(a.div(b, axis=0))
  new_df = new_df.rename(columns=new_columns)
  return [new_df]

  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  extract_features(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
