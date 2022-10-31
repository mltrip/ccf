import sys
from datetime import datetime, timedelta, timezone
from copy import deepcopy

import yaml
from sqlalchemy import create_engine
import pandas as pd


def read_data(query, start=None, end=None):
  now = datetime.utcnow()
  if isinstance(start, (int, float)):
    start = now + timedelta(seconds=start)
  elif isinstance(start, str):
    start = datetime.fromisoformat(start)
  elif isinstance(start, datetime):
    pass
  else:
    start = None
  if isinstance(end, (int, float)):
    end = now + timedelta(seconds=end)
  elif isinstance(end, str):
    end = datetime.fromisoformat(end)
  elif isinstance(end, datetime):
    pass
  else:
    end = None
  result = {}
  for n, d in query.items():
    dfs = []
    for nn, dd in d.items():
      ek = deepcopy(dd['engine_kwargs'])
      rk = deepcopy(dd['read_kwargs'])
      rk['index_col'] = 'time'
      rk['parse_dates'] = ['time']
      name = rk.pop('name')
      resample_kwargs = rk.pop('resample_kwargs', None)
      aggregate_kwargs = rk.pop('aggregate_kwargs', None)
      group = rk.pop('group', None)
      columns = rk.pop('columns', None)
      sql_columns = ','.join([f'`{x}`' for x in columns]) if columns is not None else '*'
      sql_group = f" AND `group` = '{group}'" if group is not None else ''
      if start is not None and end is not None:
        rk['sql'] = f"SELECT {sql_columns} FROM '{name}' WHERE time > '{start}' AND time < '{end}'{sql_group}"
      elif start is not None:
        rk['sql'] = f"SELECT {sql_columns} FROM '{name}' WHERE time > '{start}'{sql_group}"
      elif end is not None:
        rk['sql'] = f"SELECT {sql_columns} FROM '{name}' WHERE time < '{end}'{sql_group}"
      else:  # start is None and end is None
        rk['sql'] = f"SELECT {sql_columns} FROM '{name}'{sql_group}"
      rk['con'] = create_engine(**ek)
      df = pd.read_sql(**rk)
      if resample_kwargs is not None and aggregate_kwargs is not None:
        df = df.resample(**resample_kwargs).aggregate(**aggregate_kwargs)
      dfs.append(df.add_prefix(f'{nn}_'))
    result[n] = pd.concat(dfs, axis=1)
  return result
    
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'read_data.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  df = read_data(**kwargs)
  print(df)