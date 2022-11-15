import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gc
from copy import deepcopy
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import pytorch_forecasting as pf
from sqlalchemy import create_engine
import yaml
import sklearn.metrics
from pandas.tseries.frequencies import to_offset

from ccf.read_data import read_data
from ccf.utils import expand_columns, rat2val


def main(data_kwargs, raw_data_kwargs, feature_data_kwargs, predict_data_kwargs,
         metrics, kinds, delay=0, verbose=False, horizons=None, relative=False):
  # Write data kwargs
  write_kwargs = data_kwargs['query']['metric']['write_kwargs']
  engine_kwargs = data_kwargs['query']['metric']['engine_kwargs'] 
  write_kwargs['con'] = create_engine(**engine_kwargs)
  while True:
    print(datetime.utcnow())
    t0 = time.time()
    # Read
    raw = read_data(**raw_data_kwargs)
    features = read_data(**feature_data_kwargs)
    preds = read_data(**predict_data_kwargs)
    # Preprocess
    dfs, cnt = [], 0
    for n, df in raw.items():
      df['group'] = n
      df = df.reset_index()
      df = df.set_index(np.arange(cnt, cnt + len(df)))
      cnt += len(df)
      dfs.append(df)
    raw = pd.concat(dfs)
    raw = raw.set_index('time').sort_index()
    raw['o_m_p'] = 0.5*(raw['o_a_p_0'] + raw['o_b_p_0'])
    dfs, cnt = [], 0
    for n, df in features['feature'].items():
      df['group'] = n
      df = df.reset_index()
      df = df.set_index(np.arange(cnt, cnt + len(df)))
      cnt += len(df)
      dfs.append(df)
    features = pd.concat(dfs)
    features = features.set_index('time')
    dfs, cnt = [], 0
    for n, df in preds.items():
      df = df['prediction']
      df['model'] = n
      df = df.reset_index()
      df = df.set_index(np.arange(cnt, cnt + len(df)))
      cnt += len(df)
      dfs.append(df)
    preds = pd.concat(dfs)
    preds = preds.set_index('time')
    results = []
    for (model, group), model_df in preds.groupby(['model', 'group']):
      prediction_cols = [x for x in model_df if 'pred' in x]
      target_cols = ['-'.join(x.split('-')[1:]) for x in prediction_cols]
      raw_cols = [x.split('-')[-2] for x in prediction_cols]
      for kind in kinds:
        for prediction_col, target_col, raw_col in zip(prediction_cols, target_cols, raw_cols):
          target_df = features[features['group'] == group][target_col]
          prediction_df = model_df[[prediction_col, 'horizon']]
          if relative and kind in ['rat', 'val']:
            # Target
            freq = target_df.index.inferred_freq
            if freq is None:
              continue
            if kind == 'val':
              init_time = target_df.index[0]
              raw_index = raw.index.get_loc(init_time, method='nearest')
              raw_time = raw.index[raw_index]
              time_delta = abs(init_time - raw_time).total_seconds()
              freq_delta = pd.to_timedelta(to_offset(freq)).total_seconds()
              if time_delta > freq_delta:
                continue
              initial_value = raw.iloc[raw_index][raw_col]
            else:
              initial_value = 1
            target_df = target_df.resample(freq).first()
            mask = target_df.isnull()
            target_df = rat2val(target_df, initial_value)
            target_df[mask] = np.nan
            # Prediction
            dfs, cnt = [], 0
            for g, df in prediction_df.groupby(level=0):
              df = df.reset_index()
              df = df.set_index('horizon').sort_index()
              df[prediction_col] = rat2val(df[prediction_col], initial_value)
              df = df.reset_index()
              df = df.set_index(np.arange(cnt, cnt + len(df)))
              cnt += len(df)
              dfs.append(df)
            prediction_df = pd.concat(dfs)
            prediction_df = prediction_df.set_index('time')
          for horizon, horizon_df in prediction_df.groupby('horizon'):
            if horizons is not None and horizon not in horizons:
              continue
            all_df = pd.merge(horizon_df[prediction_col], target_df, 
                              left_index=True, right_index=True)
            all_df = all_df.dropna()
            if len(all_df) == 0:
              continue
            for m in metrics:
              c = m.pop('class')
              label = m.pop('label')
              if 'sklearn' in c:
                f = getattr(sklearn.metrics, 
                            c.replace('sklearn.metrics.', ''))
                m['y_true'] = all_df[target_col]
                m['y_pred'] = all_df[prediction_col]
                value = f(**m)
              else:
                raise NotImplementedError(c)
              m['class'] = c
              m['label'] = label
              prediction = prediction_col.split('-')[0]
              result = {
                'time': all_df.index.max(),
                'horizon': horizon,
                'group': group,
                'metric': c,
                'label': label,
                'value': value,
                'model': model,
                'prediction': prediction,
                'kind': kind,
                'target': target_col}
              results.append(result)
    if len(results) > 0:
      df = pd.DataFrame(results).set_index('time')
      df.to_sql(**write_kwargs)
    dt = time.time() - t0
    wt = max(0, delay - dt)
    if verbose:
      print(f'n: {len(results)}, dt: {dt:.3f}, wt: {wt:.3f}')
    time.sleep(wt)

    
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  main(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()

