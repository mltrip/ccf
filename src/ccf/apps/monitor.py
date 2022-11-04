import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import yaml
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite

from ccf.read_data import read_data


def monitor(read_data_kwargs, delay=0, start=None, end=None, log_dir='monitor',
            test_kwargs=None, report_kwargs=None, column_mapping_kwargs=None, 
            verbose=False):
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
  concat = read_data_kwargs.get('concat', True)
  while True:
    print(datetime.utcnow())
    t0 = time.time()
    d0 = d1
    d1 = read_data(**read_data_kwargs)
    if concat:
      d1 = {'default': d1}
    if d0 is not None and d1 is not None:
      for n, dd0 in d0.items():
        dd1 = d1[n]
        for nn, df1 in dd1.items():
          df0 = dd0[nn]
          if verbose:
            print(f'{n}, {nn}, old: {len(df0)}, new: {len(df1)}')
          if len(df0) == 0 or len(df1) == 0:
            continue
          prefixes = [f'-'.join(['test', n, nn]), f'-'.join(['report', n, nn])]
          monitors = [TestSuite(**test_kwargs) if test_kwargs is not None else None, 
                      Report(**report_kwargs) if report_kwargs is not None else None]
          for p, m in zip(prefixes, monitors):
            if m is None:
              continue
            filename = Path(f'{log_dir}/{p}.html')
            if verbose:
              print(filename)
            m.run(reference_data=df0, current_data=df1, column_mapping=cm)
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.unlink(missing_ok=True)
            m.save_html(filename)
    dt = time.time() - t0
    wt = max(0, delay - dt)
    print(f'dt: {dt:.3f}, wt: {wt:.3f}')
    time.sleep(wt)
  

@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  monitor(**OmegaConf.to_object(cfg))


if __name__ == "__main__":
  app()  
