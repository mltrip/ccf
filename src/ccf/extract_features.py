import sys
from datetime import datetime, timedelta, timezone
import time
from copy import deepcopy
import gc

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
  # Initialize pre features
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
  # Initialize post features
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
    print(f'{datetime.utcnow()}')
    t0 = time.time()
    # Create engine
    feature_io = deepcopy(feature_data_kwargs['query']['feature'])
    for n in feature_io:
      ek = feature_io[n]['engine_kwargs']
      feature_io[n]['write_kwargs']['con'] = create_engine(**ek)
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
      if len(old_df) > 0:
        new_df = new_df[new_df.index > old_df.index.max()]
      if verbose:
        print(f'{n}, old: {old_df.shape}, new: {new_df.shape}')
      if len(new_df) > 0:
        wk = feature_io[n]['write_kwargs']
        new_df.to_sql(**wk)
    dt = time.time() - t0
    wt = max(0, delay - dt)
    print(f'dt: {dt:.3f}, wt: {wt:.3f}')
    time.sleep(wt)
  
  
def pqv(name, raw, old_feature, pre_feature, depth=None):
  o = raw[name]['o']
  # Columns
  a_q_cs = [x for x in o if x.startswith('a_q_')]
  b_q_cs = [x for x in o if x.startswith('b_q_')]
  a_p_cs = [x for x in o if x.startswith('a_p_')]
  b_p_cs = [x for x in o if x.startswith('b_p_')]
  a_v_cs = ['_'.join(['o', 'a', 'v', x.split('_')[2]]) for x in a_q_cs]
  b_v_cs = ['_'.join(['o', 'b', 'v', x.split('_')[2]]) for x in b_q_cs]
  dfs = []
  # Volumes
  a_vs = []
  for a_q_c, a_p_c, a_v_c in zip(a_q_cs, a_p_cs, a_v_cs):
    df = (o[a_q_c]*o[a_p_c]).rename(a_v_c)
    a_vs.append(df)
  a_vs = pd.concat(a_vs, axis=1)
  dfs.append(a_vs)
  b_vs = []
  for b_q_c, b_p_c, b_v_c in zip(b_q_cs, b_p_cs, b_v_cs):
    df = (o[b_q_c]*o[b_p_c]).rename(b_v_c)
    b_vs.append(df)
  b_vs = pd.concat(b_vs, axis=1)
  dfs.append(b_vs)
  # Sums
  a_q = o[a_q_cs].sum(axis=1).rename('o_a_q')
  b_q = o[b_q_cs].sum(axis=1).rename('o_b_q')
  a_v = a_vs.sum(axis=1).rename('o_a_v')
  b_v = b_vs.sum(axis=1).rename('o_b_v')
  dfs.append(a_q)
  dfs.append(b_q)
  dfs.append(a_v)
  dfs.append(b_v)
  # Mids
  m_q = (0.5*(a_q + b_q)).rename('o_m_q')
  m_v = (0.5*(a_v + b_v)).rename('o_m_v')
  m_p = (0.5*(o['a_p_0'] + o['b_p_0'])).rename('o_m_p')
  dfs.append(m_q)
  dfs.append(m_v)
  dfs.append(m_p)
  return dfs


def get(name, raw, old_feature, pre_feature, columns=None):
  dfs = []
  for n, df in raw[name].items():
    columns = list(df.columns) if columns is None else columns
    columns = expand_columns(df.columns, columns)
    if len(columns) > 0:
      df = df[columns].add_prefix(f'{n}_')
      dfs.append(df)
  return dfs


def relative(name, raw, old_feature, pre_feature, 
             kind='pct', column=None, columns=None, shift=0):
  df = pre_feature[name]
  columns = list(df.columns) if columns is None else columns
  columns = expand_columns(df.columns, columns)
  if len(columns) == 0:
    return []
  a = df[columns]
  prefix = f'{kind}_{str(shift)}'
  if column is not None:
    b = df[column].shift(shift) if shift != 0 else df[column]
    new_columns = {x: '-'.join([prefix, x, column]) for x in columns}
  else:
    b = df[columns].shift(shift) if shift != 0 else df[columns]
    new_columns = {x: '-'.join([prefix, x, x]) for x in columns}
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
