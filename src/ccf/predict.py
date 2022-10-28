import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from copy import deepcopy

import pandas as pd
import pytorch_forecasting as pf
import pytorch_lightning as pl
from sqlalchemy import create_engine
import yaml

from ccf.make_dataset import make_dataset


def predict(model_path, train_kwargs, engine_kwargs, write_kwargs, 
            predict_kwargs, past, verbose=False, prediction_suffix='pred',
            dataloader_kwargs=None):
  with open(train_kwargs) as f:
    train_kwargs = yaml.safe_load(f)
  c = getattr(pf.models, train_kwargs['model_kwargs']['class'])
  model = c.load_from_checkpoint(model_path)
  dks = train_kwargs['dataset_kwargs']
  max_prediction_length = dks['dataset_kwargs']['max_prediction_length']
  max_encoder_length = dks['dataset_kwargs']['max_encoder_length']
  min_length = max_encoder_length + max_prediction_length
  resample_rule = dks['features_kwargs']['resample_kwargs']['rule']
  resample_seconds = pd.to_timedelta(resample_rule).total_seconds()
  dks['split'] = None
  dks['start'] = -past
  dks['end'] = max_prediction_length*resample_seconds
  dks['dataset_kwargs']['predict_mode'] = True
  while True:
    t0 = time.time()
    ds, _, df, _ = make_dataset(**deepcopy(dks))
    if verbose:
      dt_data = time.time() - t0
    if ds is None:
      status = None
    else:
      status = True
    # else:
    #   status = True if len(df) >= min_length else False
    # dl = ds.to_dataloader(**dataloader_kwargs)
    if status is not None and status:
      # df = df.tail(min_length)
      # df_past = df.head(max_encoder_length)
      # df_future = df.tail(max_prediction_length)
      # pred_time_idx = df_future.iloc[0].time_idx
      # predict_kwargs['data'] = ds.filter(
      #   lambda x: x.time_idx_first_prediction == pred_time_idx)
      predict_kwargs['data'] = ds
      pred, idxs = model.predict(**predict_kwargs)
      pred_dfs = []
      for g, gdf in df.groupby('group'):
        g_idx = idxs[idxs['group'] == g]
        p_idx, t_idx = g_idx.iloc[0].name, g_idx.iloc[0].time_idx
        df_future = gdf[gdf['time_idx'] >= t_idx]
        if predict_kwargs['mode'] == 'quantiles':
          ps = pred[p_idx].tolist()
          data = [x + [g] for x in ps]
          qs = model.loss.quantiles
          columns = [f'{prediction_suffix}-{x}' for x in qs]
          columns += ['group']
          pred_df = pd.DataFrame(
            data=data, 
            columns=columns,
            index=df_future.index)
        elif predict_kwargs['mode'] == 'prediction':
          ps = pred[p_idx].tolist()
          data = [[x, g] for x in ps]
          pred_df = pd.DataFrame(
            data=data, 
            columns=[prediction_suffix, 'group'],
            index=df_future.index)
        else:
          raise NotImplementedError(predict_kwargs['mode'])
        pred_dfs.append(pred_df)
      pred_df = pd.concat(pred_dfs)
      write_kwargs['con'] = create_engine(**engine_kwargs)
      pred_df.to_sql(**write_kwargs)
    dt_total = time.time() - t0
    if verbose:
      dt_pred = time.time() - (t0 + dt_data)
      print(f'{datetime.utcnow()}, status: {status}, dt_data: {dt_data:.3f}, dt_pred: {dt_pred:.3f}, dt_total: {dt_total:.3f}')
    time.sleep(max(0, resample_seconds - dt_total))
    
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'predict.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  predict(**kwargs)
