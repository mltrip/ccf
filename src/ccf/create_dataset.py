import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gc
from copy import deepcopy
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import pytorch_forecasting as pf
from sqlalchemy import create_engine
import yaml

from ccf.utils import expand_columns
from ccf import agents as ccf_agents


class Dataset:
  def __init__(self, quant, size, agents, dataset_kwargs, executor=None, replace_nan=None,
               split=None, target_prefix='tgt', df_only=False, verbose=False):
    self.quant = int(quant) if quant is not None else quant
    self.size = size
    for name, kwargs in agents.items():
      if quant is not None:
        kwargs['quant'] = quant
      if size is not None:
        kwargs['size'] = size
    self.agents = agents
    self.dataset_kwargs = dataset_kwargs
    self.executor = {} if executor is None else executor
    self.replace_nan = replace_nan
    self.split = split
    self.target_prefix = target_prefix
    self.df_only = df_only
    self.verbose = verbose
  
  def __call__(self):
    features = self.get_features()
    return self.create_dataset(features)
  
  def __iter__(self):  # TODO async
    agents = deepcopy(self.agents)
    for name, kwargs in agents.items():
      class_name = kwargs.pop('class')
      kwargs['size'] = 1
      agents[name] = getattr(ccf_agents, class_name)(**kwargs)
    data = {}
    e = ThreadPoolExecutor(**self.executor)
    while True:
      if len(agents) == 1:
        for name, agent in agents.items():
          result = agent()
      else:
        f2c = {}
        for name, agent in agents.items():
          future_kwargs = {}
          future = e.submit(agent, **future_kwargs)
          f2c[future] = [agent, future_kwargs]
        for future in as_completed(f2c):
          result = future.result()
      for feature, df in result.items():
        data.setdefault(feature, deque()).append(df)
      len_data = [len(v) for v in data.values()]
      print(len_data)
      if all([v >= self.size for v in len_data]):
        features = {k: pd.concat(v) for k, v in data.items()}
        for f, d in data.items():
          d.popleft()
        yield self.create_dataset(features)

  def get_features(self):
    # Initialize agents
    agents = deepcopy(self.agents)
    for name, kwargs in agents.items():
      class_name = kwargs.pop('class')
      agents[name] = getattr(ccf_agents, class_name)(**kwargs)
    # Collect features
    features = {}
    if len(agents) == 1:
      for name, agent in agents.items():
        agent_features = agent()
        features.update(agent_features)
    else:
      e = ThreadPoolExecutor(**executor)
      f2c = {}
      for name, agent in agents.items():
        print(name)
        future_kwargs = {}
        future = e.submit(agent, **future_kwargs)
        f2c[future] = [agent, future_kwargs]
      for future in as_completed(f2c):
        result = future.result()
        features.update(result)
    return features

  def create_dataset(self, features):
    dataset_kwargs = deepcopy(self.dataset_kwargs)
    time_idx = dataset_kwargs['time_idx']
    group_ids = dataset_kwargs['group_ids']
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
      if self.replace_nan is not None:
        num_cols = df.select_dtypes(np.number).columns
        obj_cols = df.select_dtypes('object').columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf, np.nan], self.replace_nan)
        # df = df.replace([np.inf, -np.inf, np.nan], self.replace_nan)
      nan_len = df.isna().any(axis=1).sum()
      cnt_len = df_len - nan_len
      if cnt_len < min_len:
        print(f'Warning! {n} data length {cnt_len} < {min_len} = {df_len} - {nan_len} < {max_enc_len} + {max_pred_len}')
        continue
      df[time_idx] = np.arange(df_len)
      df = df.reset_index()
      df = df.set_index(np.arange(cnt, cnt + df_len))
      cnt += df_len
      dfs.append(df)
    if len(dfs) == 0:
      return None, None, None, None
    df = pd.concat(dfs)
    df = df.set_index('timestamp')
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
          cc = f'{self.target_prefix}-{c}'
          df[cc] = df[c]
          cs[i] = cc
        dataset_kwargs[key] = cs if len(cs) != 1 else cs[0]
      else:
        dataset_kwargs[key] = cs
      columns.update(cs)
    # Scalers/Encoders
    all_scalers = {}
    scalers = dataset_kwargs.get('scalers', {})
    encoders = dataset_kwargs.get('categorical_encoders', {})
    target = dataset_kwargs.get('target', [])
    target = [target] if isinstance(target, str) else target
    if isinstance(encoders, dict):
      if 'class' in encoders:
        white_list = []
        for key in ['time_varying_unknown_categoricals',
                    'time_varying_known_categoricals',
                    'static_categoricals']:
          white_list += dataset_kwargs.get(key,[])
        white_list = list(set(white_list))  
        encoders = {x: deepcopy(encoders) for x in white_list}
      for column, encoder_kwargs in encoders.items():
        if isinstance(encoder_kwargs, dict):
          c = encoder_kwargs.pop('class')
          s = getattr(pf.data.encoders, c)(**encoder_kwargs)
        else:
          s = encoder_kwargs
        encoders[column] = s
      dataset_kwargs['categorical_encoders'] = encoders
    if isinstance(scalers, dict):
      if 'class' in scalers:
        white_list = []
        for key in ['time_varying_unknown_reals',
                    'time_varying_known_reals',
                    'static_reals']:
          white_list += dataset_kwargs.get(key,[])
        white_list = list(set(white_list))  
        scalers = {x: deepcopy(scalers) for x in white_list}
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
    all_df = df
    df = df[list(columns)]
    if dataset_kwargs.get('allow_missing_timesteps', False):   
      df = df.dropna()  # Remove rows with nan values
    if self.split is not None:
      df1s, df2s = [], []
      cnt1, cnt2 = 0, 0
      for g, gdf in df.groupby(group_ids):
        split_idx = int(self.split*len(gdf))
        # part 1
        df1 = gdf[:split_idx]
        df1_len = len(df1)
        df1 = df1.reset_index()
        df1 = df1.set_index(np.arange(cnt, cnt + df1_len))
        df1[time_idx] = np.arange(df1_len)
        cnt1 += df1_len
        df1s.append(df1)
        # part 2
        df2 = gdf[split_idx:]
        df2_len = len(df2)
        df2 = df2.reset_index()
        df2 = df2.set_index(np.arange(cnt, cnt + df2_len))
        df2[time_idx] = np.arange(df2_len)
        cnt2 += df2_len
        df2s.append(df2)
      df, df2 = pd.concat(df1s), pd.concat(df2s)
      df = df.set_index('timestamp')
      df2 = df2.set_index('timestamp')
    else:
      df, df2 = all_df, None
    if not self.df_only:
      dataset_kwargs['data'] = df.reset_index()
      try:
        ds = pf.TimeSeriesDataSet(**dataset_kwargs)
      except Exception as e:
        print(e)
        ds = None
      if df2 is not None and ds is not None:
        ds2 = pf.TimeSeriesDataSet.from_dataset(ds, df2.reset_index(), stop_randomization=True)
      else:
        ds2 = None
      if self.verbose:
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

