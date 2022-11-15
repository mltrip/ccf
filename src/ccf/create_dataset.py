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
                   split=None, target_prefix='tgt', df_only=False, verbose=False):
  features = read_data(**feature_data_kwargs)['feature']
  time_idx = dataset_kwargs['time_idx']
  max_enc_len = dataset_kwargs['max_encoder_length']
  max_pred_len = dataset_kwargs['max_prediction_length']
  dfs = []
  cnt = 0
  for n, df in features.items():
    min_len = max_enc_len + max_pred_len
    if dataset_kwargs.get('predict_mode', False) and len(df) > 1:  # 2 timesteps min
      new_rows = [df]
      last_dt = df.index.to_series().diff().dt.total_seconds().iloc[-1]
      last_row = df.iloc[[-1]]
      for _ in range(max_pred_len):
        last_row.index = last_row.index + timedelta(seconds=last_dt)
        new_rows.append(last_row.copy())
      df = pd.concat(new_rows)
    df_len = len(df)
    nan_len = df.isna().any(axis=1).sum()
    cnt_len = df_len - nan_len
    if cnt_len < min_len:
      print(f'Warning! {n} data length {cnt_len} < {min_len} = {df_len} - {nan_len} < {max_enc_len} + {max_pred_len}')
      continue
    df[time_idx] = np.arange(df_len)
    df = df.reset_index()
    df = df.set_index(np.arange(cnt, cnt + df_len))
    cnt += df_len
    df['group'] = n
    dfs.append(df)
  if len(dfs) == 0:
    return None, None, None, None
  df = pd.concat(dfs)
  df = df.set_index('time')
  if len(df) == 0:
    return None, None, None, None
  # Dataset
  columns = set()
  columns.add(time_idx)
  for key in ['target',
              'group_ids', 
              'time_varying_unknown_reals',
              'time_varying_unknown_categoricals',
              'time_varying_known_reals',
              'time_varying_known_categoricals',
              'static_reals',
              'static_categoricals']:
    cs = dataset_kwargs.get(key, [])
    cs = [cs] if isinstance(cs, str) else cs
    cs = expand_columns(df.columns, cs)
    if key == 'target':  # rename target with prefix
      for i, c in enumerate(cs):
        cc = f'{target_prefix}-{c}'
        df[cc] = df[c]
        cs[i] = cc
      dataset_kwargs[key] = cs if len(cs) != 1 else cs[0]
    else:
      dataset_kwargs[key] = cs
    columns.update(cs)
  # Scalers
  all_scalers = {}
  scalers = dataset_kwargs.get('scalers', {})
  target = dataset_kwargs.get('target', [])
  target = [target] if isinstance(target, str) else target
  if isinstance(scalers, dict):
    if 'class' in scalers:
      black_list = target + ['group', time_idx]
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
  # df = df.replace([np.inf, -np.inf, np.nan], 0)  # Fill missing values with 0
  df = df.dropna()  # Remove rows with missing values (Requires allow_missing_timesteps = True)
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
    if verbose:
      print(ds)
    return ds, ds2, df, df2
  else:
    return None, None, df, df2
    
    
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  create_dataset(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()

