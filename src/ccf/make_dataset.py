import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from sqlalchemy import create_engine
import yaml

from ccf.make_features import make_features


def make_dataset(engine_kwargs, read_kwargs, 
                 features_kwargs, dataset_kwargs,
                 start=None, end=None, split=None, target_prefix='tgt'):
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
  for n, ek in engine_kwargs.items():
    rk = read_kwargs[n]
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
    dfs[n] = df
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
  columns.update(target)
  df = df[list(columns)]
  # df = df.replace([np.inf, -np.inf, np.nan], 0)
  df = df.dropna()  # Requires allow_missing_timesteps = True
  if split is not None:
    split_idx = int(split*len(df))
    df, df2 = df[:split_idx], df[split_idx:]
  else:
    df, df2 = df, None
  dataset_kwargs['data'] = df
  ds = TimeSeriesDataSet(**dataset_kwargs)
  if df2 is not None:
    ds2 = TimeSeriesDataSet.from_dataset(ds, df2, stop_randomization=True)
  else:
    ds2 = None
  return ds, ds2, df, df2
    
  
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
