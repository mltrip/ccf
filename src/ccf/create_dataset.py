import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gc
from copy import deepcopy
from pprint import pprint
import os
import concurrent.futures
from collections import deque

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import pytorch_forecasting as pf
from sqlalchemy import create_engine
import yaml

from ccf.utils import expand_columns, initialize_time
from ccf import agents as ccf_agents
from ccf import transformations as ccf_transformations


class Dataset:
  def __init__(self, agents, dataset_kwargs, quant=None, size=None, 
               start=None, stop=None, executor=None, replace_nan=None,
               split=None, target_prefix='tgt', replace_dot=' ',
               default_group_column='group_id',
               df_only=False, watermark=None, verbose=False, 
               merge_features=False, sep='-'):
    # Preinit
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    if 'max_workers' not in executor:
      executor['max_workers'] = len(agents)
    # Init
    start, stop, size, quant = initialize_time(start, stop, size, quant)
    self.start = start
    self.stop = stop
    self.size = size
    self.quant = quant
    self.watermark = int(watermark) if watermark is not None else watermark
    self.verbose = verbose
    for name, kwargs in agents.items():
      if 'quant' not in kwargs and self.quant is not None:
        kwargs['quant'] = self.quant
      if 'size' not in kwargs and self.size is not None:
        kwargs['size'] = self.size
      if 'start' not in kwargs and self.start is not None:
        kwargs['start'] = self.start
      if 'stop' not in kwargs and self.stop is not None:
        kwargs['stop'] = self.stop
      if 'watermark' not in kwargs and self.watermark is not None:
        kwargs['watermark'] = self.watermark
      if 'verbose' not in kwargs and self.verbose is not None:
        kwargs['verbose'] = self.verbose
      class_name = kwargs.pop('class')
      agents[name] = getattr(ccf_agents, class_name)(**kwargs)
    self.agents = agents
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    self.executor = executor
    self.dataset_kwargs = dataset_kwargs
    self.replace_nan = replace_nan
    self.replace_dot = replace_dot
    self.default_group_column = default_group_column
    if split is None:
      split = os.getenv('CCF_DATASET_SPLIT', None)
      split = float(split) if isinstance(split, str) else split
    self.split = split
    self.target_prefix = target_prefix
    self.df_only = df_only
    self.buffer = {}
    self.merge_features = merge_features
    self.sep = sep
    
  def get_features(self):
    features = {}
    if len(self.agents) == 1:  # Run without executor
      for name, agent in self.agents.items():
        print(name)
        r = agent()
        if isinstance(r, dict):
          features.update(r)
        else:
          features[name] = r
    else:
      future_to_name = {}
      for name, agent in self.agents.items():
        print(name)
        future_to_name[self.executor.submit(agent)] = name
      for f in concurrent.futures.as_completed(future_to_name):
        n = future_to_name[f]
        try:
          r = f.result()
        except Exception as e:
          print(f'Exception of {n}: {e}')
          raise e
        else:
          print(f'Result of {n}: {r}')
          if isinstance(r, dict):
            features.update(r)
          else:
            features[n] = r
    if self.merge_features:
      all_features = pd.concat([v.add_prefix(f'{k}{self.sep}')
                                for k, v in features.items()], axis=1)
      features = {'all': all_features}
    return features
  
  def __call__(self):
    features = self.get_features()
    return self.create_dataset(features)
  
  def __iter__(self):  # TODO async?
    self.buffer = {}
    return self
  
  def __next__(self):  # TODO async?
    t0 = time.time()
    features = self.get_features()
    for feature, df in features.items():
      old_df = self.buffer.get(feature, None)
      if old_df is not None:
        self.buffer[feature] = pd.concat([df, old_df]).drop_duplicates().sort_index()
      else:
        self.buffer[feature] = df
    if self.watermark is not None:
      for feature, df in self.buffer.items():
        if len(df) > 0:
          min_time = df.index.min()
          max_time = df.index.max()
          watermark_time = max_time.timestamp()*1e9 - self.watermark
          watermark_datetime = pd.to_datetime(watermark_time, unit='ns')
          print(f'\ncreate_dataset watermark')
          print(f'feature:   {feature}')
          print(f'now:       {datetime.utcnow()}')
          print(f'min:       {min_time}')
          print(f'max:       {max_time}')
          print(f'watermark: {watermark_datetime}')
          print(f'rows:      {len(df)}')
          new_df = df[df.index > watermark_datetime]          
          print(f'new_min:   {new_df.index.min()}')
          print(f'new_max:   {new_df.index.max()}')
          print(f'new_rows:  {len(new_df)}')
          self.buffer[feature] = new_df
    print({k: v.shape for k, v in self.buffer.items()})
    return self.create_dataset(self.buffer)

  def create_dataset(self, features):
    t0 = time.time()
    dataset_kwargs = deepcopy(self.dataset_kwargs)
    time_idx = dataset_kwargs['time_idx']
    max_enc_len = dataset_kwargs['max_encoder_length']
    max_pred_len = dataset_kwargs['max_prediction_length']
    dfs = []
    cnt = 0
    for n, df in features.items():
      min_len = max_enc_len + max_pred_len
      if dataset_kwargs.get('predict_mode', False) and len(df) > 1:  # 2 timesteps min
        new_rows = [df]
        if self.quant is not None:
          last_dt = self.quant/1e9  # ns -> s
        else:
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
    # Groups
    group_ids = dataset_kwargs.get('group_ids', [])
    if len(group_ids) == 0:  # If you have only one timeseries, set this to the name of column that is constant.
      group_ids.append(self.default_group_column)
      dataset_kwargs['group_ids'] = group_ids
      df[self.default_group_column] = 0
    # Features
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
          cc = f'{self.target_prefix}{self.sep}{c}'
          df[cc] = df[c]
          cs[i] = cc
        dataset_kwargs[key] = cs if len(cs) != 1 else cs[0]
      else:
        dataset_kwargs[key] = [x for x in cs if not x.startswith(self.target_prefix)]
      columns.update(cs)
      if self.verbose:
        print(key)
        pprint(cs)
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
          if 'transformation' in scaler_kwargs:
            transformation = scaler_kwargs['transformation']
            if isinstance(transformation, dict):
              for t_type, t_kwargs in transformation.items():
                t_class = t_kwargs.pop('class')
                transformation[t_type] = getattr(ccf_transformations, t_class)(**t_kwargs)
          s = getattr(pf.data.encoders, c)(**scaler_kwargs)
        else:
          s = scaler_kwargs
        scalers[column] = s
      dataset_kwargs['scalers'] = scalers
    target_scaler = dataset_kwargs.get('target_normalizer', 'auto')
    if isinstance(target_scaler, dict):
      c = target_scaler.pop('class')
      if 'transformation' in target_scaler:
        transformation = target_scaler['transformation']
        if isinstance(transformation, dict):
          for t_type, t_kwargs in transformation.items():
            t_class = t_kwargs.pop('class')
            transformation[t_type] = getattr(ccf_transformations, t_class)(**t_kwargs)
      target_scaler = getattr(pf.data.encoders, c)(**target_scaler)
      if len(target) > 1:
        target_scaler = pf.MultiNormalizer([target_scaler for _ in target])
    dataset_kwargs['target_normalizer'] = target_scaler
    # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ',' 
    old2new = {x: x.replace('.', self.replace_dot) for x in df.columns}
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
      new_cs = [old2new[x] for x in cs]
      if key == 'target':  # rename target with prefix
        dataset_kwargs[key] = new_cs if len(new_cs) != 1 else new_cs[0]
      else:
        dataset_kwargs[key] = new_cs
      if self.verbose:
        print(key)
        pprint(new_cs)
    encoders = dataset_kwargs.get('categorical_encoders')
    dataset_kwargs['categorical_encoders'] = {old2new[k]: v for k, v in encoders.items()}
    scalers = dataset_kwargs.get('scalers', {})
    dataset_kwargs['scalers'] = {old2new[k]: v for k, v in scalers.items()}
    columns = [old2new[x] for x in columns]
    df = df.rename(columns=old2new)
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
        df1 = df1.set_index(np.arange(cnt1, cnt1 + df1_len))
        df1[time_idx] = np.arange(df1_len)
        cnt1 += df1_len
        df1s.append(df1)
        # part 2
        df2 = gdf[split_idx:]
        df2_len = len(df2)
        df2 = df2.reset_index()
        df2 = df2.set_index(np.arange(cnt2, cnt2 + df2_len))
        df2[time_idx] = np.arange(df2_len)
        cnt2 += df2_len
        df2s.append(df2)
      df, df2 = pd.concat(df1s), pd.concat(df2s)
      df = df.set_index('timestamp')
      df2 = df2.set_index('timestamp')
    else:
      df, df2 = all_df, None
    if self.verbose:
      print(df)
      pprint(df.columns)
      print(df.describe(include='all'))
      if df2 is not None:
        print(df2)
        pprint(df2.columns)
        print(df2.describe(include='all'))
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
        print(ds2)
      print(f'dt dataset: {time.time() - t0:.3f}')
      return ds, ds2, df, df2
    else:
      print(f'dt dataset: {time.time() - t0:.3f}')
      return None, None, df, df2
    
    
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  create_dataset(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
