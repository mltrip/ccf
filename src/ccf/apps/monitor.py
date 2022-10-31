import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import yaml
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite

from ccf.read_data import read_data


def app(read_data_kwargs, delay=0, start=None, end=None, log_dir='monitor',
        test_kwargs=None, report_kwargs=None, column_mapping_kwargs=None):
  # Initialize Test
  if test_kwargs is not None:
    for i, t in enumerate(test_kwargs['tests']):
      c = t.pop('class')
      t = getattr(evidently.test_preset, c)(**t)
      test_kwargs['tests'][i] = t
  # Initialize Report
  if report_kwargs is not None:
    for i, m in enumerate(report_kwargs['metrics']):
      c = m.pop('class')
      m = getattr(evidently.metric_preset, c)(**m)
      report_kwargs['metrics'][i] = m
  # Initialize ColumnMapping
  if column_mapping_kwargs is not None:
    cm = ColumnMapping(**column_mapping_kwargs)
  else:
    cm = None
  d0, d1 = None, None
  while True:
    print(datetime.utcnow())
    d0 = d1
    d1 = read_data(**read_data_kwargs)
    if d0 is not None and d1 is not None:
      for name, df1 in d1.items():
        df0 = d0[name]
        print(name, len(df0), len(df1))
        if len(df0) == 0 or len(df1) == 0:
          continue
        names = [f'-'.join(['test', name]), f'-'.join(['report', name])]
        monitors = [TestSuite(**test_kwargs) if test_kwargs is not None else None, 
                    Report(**report_kwargs) if report_kwargs is not None else None]
        for n, m in zip(names, monitors):
          if m is None:
            continue
          filename = Path(f'{log_dir}/{n}.html')
          print(filename)
          m.run(reference_data=df0, current_data=df1, column_mapping=cm)
          filename.parent.mkdir(parents=True, exist_ok=True)
          filename.unlink(missing_ok=True)
          m.save_html(filename)
    time.sleep(delay)
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'monitor.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  app(**kwargs)