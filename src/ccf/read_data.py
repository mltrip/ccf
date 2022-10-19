import sys
from datetime import datetime, timedelta, timezone

import yaml
from sqlalchemy import create_engine
import pandas as pd


def read_data(engine_kwargs=None, read_kwargs=None):
  engine_kwargs = {'url': 'sqlite:///data.db'} if engine_kwargs is None else engine_kwargs
  read_kwargs = {} if read_kwargs is None else read_kwargs
  read_kwargs['con'] = create_engine(**engine_kwargs)
  df = pd.read_sql(**read_kwargs)
  return df  
    
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'read_data.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  df = read_data(**kwargs)
  print(df) 