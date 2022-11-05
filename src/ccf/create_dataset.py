import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gc
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import pytorch_forecasting as pf
from sqlalchemy import create_engine
import yaml

from ccf.read_data import read_data
from ccf.utils import expand_columns


def create_dataset(feature_data_kwargs, dataset_kwargs,
                   split=None, target_prefix='tgt', df_only=False):
  features = read_data(**feature_data_kwargs)['feature']
  for n, df in features.items():
    df['group'] = n
    df['time_idx'] = np.arange(len(df))
  df = pd.concat(features.values(), axis=0)
  if len(df) == 0:
    return None, None, None, None
  # Dataset
  columns = set()
  time_idx = dataset_kwargs['time_idx']
  columns.add(time_idx)
  for key in ['group_ids', 
              'time_varying_unknown_reals',
              'time_varying_known_reals',
              'static_reals',
              'time_varying_unknown_categoricals',
              'time_varying_known_categoricals',
              'static_categoricals',
              'target']:
    cs = dataset_kwargs.get(key, [])
    cs = expand_columns(df.columns, cs)
    dataset_kwargs[key] = cs
    columns.update(cs)
  target = dataset_kwargs.get('target', [])
  target = [target] if isinstance(target, str) else target
  for i, t in enumerate(target):
    tt = f'{target_prefix}-{t}'
    df[tt] = df[t]
    target[i] = tt
  dataset_kwargs['target'] = target if len(target) != 1 else target[0]
  columns.update(target)
  # Scalers
  all_scalers = {}
  scalers = dataset_kwargs.get('scalers', {})
  if isinstance(scalers, dict):
    if 'class' in scalers:
      black_list = target + ['group', 'time_idx']
      scalers = {x: deepcopy(scalers) for x in columns 
                 if x not in black_list}
    for column, scaler_kwargs in scalers.items():
      if isinstance(scaler_kwargs, dict):
        c = scaler_kwargs.pop('class')
        s = getattr(pf.data.encoders, c)(**scaler_kwargs)
      else:
        s = scaler_kwargs
      scalers[column] = s
    dataset_kwargs['scalers'] = scalers
  target_scaler = dataset_kwargs.get('target_normalizer', 'auto')
  if isinstance(target_scaler, dict):
    if len(target) > 1:
      c = target_scaler.pop('class')
      target_scaler = getattr(pf.data.encoders, c)(**target_scaler)
      target_scalers = [target_scaler for _ in target]
      target_scaler = pf.MultiNormalizer(target_scalers)
    else:
      c = target_scaler.pop('class')
      target_scaler = getattr(pf.data.encoders, c)(**target_scaler)
  dataset_kwargs['target_normalizer'] = target_scaler
  # Filter
  df = df[list(columns)]
  # df = df.replace([np.inf, -np.inf, np.nan], 0)
  df = df.dropna()  # Requires allow_missing_timesteps = True
  if split is not None:
    df1s, df2s = [], []
    for g, gdf in df.groupby('group'):
      split_idx = int(split*len(gdf))
      df1s.append(gdf[:split_idx])
      df2s.append(gdf[split_idx:])
    df, df2 = pd.concat(df1s), pd.concat(df2s)
  else:
    df, df2 = df, None
  if not df_only:
    dataset_kwargs['data'] = df.reset_index()
    ds = pf.TimeSeriesDataSet(**dataset_kwargs)  # FIXME Memory leak!
    if df2 is not None:
      ds2 = pf.TimeSeriesDataSet.from_dataset(ds, df2.reset_index(), stop_randomization=True)
    else:
      ds2 = None
    return ds, ds2, df, df2
  else:
    return None, None, df, df2
    
    
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  create_dataset(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()

