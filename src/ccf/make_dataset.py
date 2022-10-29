import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gc
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import pytorch_forecasting as pf
from sqlalchemy import create_engine
import yaml

from ccf.make_features import make_features


def make_dataset(engine_kwargs, read_kwargs, 
                 features_kwargs, dataset_kwargs,
                 start=None, end=None, split=None, 
                 target_prefix='tgt', df_only=False):
  now = datetime.utcnow()
  if isinstance(start, (int, float)):
    start = now + timedelta(seconds=start)
  elif isinstance(start, str):
    start = datetime.fromisoformat(start)
  else:
    start = None
  if isinstance(end, (int, float)):
    end = now + timedelta(seconds=end)
  elif isinstance(end, str):
    end = datetime.fromisoformat(end)
  else:
    end = None
  # Data
  dfs = {}
  for g, eks in engine_kwargs.items():
    dfs[g] = {}
    for n, ek in eks.items():
      rk = read_kwargs[g][n]
      rk['con'] = create_engine(**ek)
      rk['index_col'] = 'time'
      rk['parse_dates'] = ['time']
      t = rk.pop('name')
      if start is not None and end is not None:
        rk['sql'] = f"SELECT * FROM '{t}' WHERE time > '{start}' AND time < '{end}'"
      elif start is not None:
        rk['sql'] = f"SELECT * FROM '{t}' WHERE time > '{start}'"
      elif end is not None:
        rk['sql'] = f"SELECT * FROM '{t}' WHERE time < '{end}'"
      else:  # start is None and end is None
        rk['sql'] = f"SELECT * FROM '{t}'"
      df = pd.read_sql(**rk)
      if end is not None and len(df) > 0 and df.index[-1] < end:
        row = pd.DataFrame(data=[[None for _ in df.columns]], 
                           columns=df.columns,
                           index=[end]).rename_axis(df.index.name)
        df = pd.concat([df, row])
      dfs[g][n] = df  
  # Features
  features_kwargs['dfs'] = dfs
  df = make_features(**features_kwargs)
  if len(df) == 0:
    return None, None, None, None
  # Dataset
  columns = set()
  time_idx = dataset_kwargs['time_idx']
  columns.add(time_idx)
  gs = dataset_kwargs.get('group_ids', [])
  columns.update(gs)
  urs = dataset_kwargs.get('time_varying_unknown_reals', [])
  columns.update(urs)
  krs = dataset_kwargs.get('time_varying_known_reals', [])
  columns.update(krs)
  srs = dataset_kwargs.get('static_reals', [])
  columns.update(srs)
  ucs = dataset_kwargs.get('time_varying_unknown_categoricals', [])
  columns.update(ucs)
  kcs = dataset_kwargs.get('time_varying_known_categoricals', [])
  columns.update(kcs)
  scs = dataset_kwargs.get('static_categoricals ', []) 
  columns.update(scs)
  target = dataset_kwargs.get('target', [])
  target = [target] if isinstance(target, str) else target
  for i, t in enumerate(target):
    if t in columns:
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
    
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'make_dataset.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  ds = make_dataset(**kwargs)
  # p = Path(cfg)
  # if ds[1] is None:
  #   ds.save(p.with_suffix('.pt'))
  # else:
  #   for i, dss in enumerate(ds):
  #     dss.save(p.with_name(f'{p.name}_{i + 1}.pt'))
