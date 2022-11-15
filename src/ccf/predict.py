import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytorch_forecasting as pf
import pytorch_lightning as pl
from sqlalchemy import create_engine

from ccf.create_dataset import create_dataset
import ccf


def predict(model_path, train_kwargs, data_kwargs, 
            predict_kwargs, past, verbose=False, prediction_prefix='pred',
            rule='1S', dataloader_kwargs=None):
  engine_kwargs = data_kwargs['query']['prediction']['engine_kwargs'] 
  write_kwargs = data_kwargs['query']['prediction']['write_kwargs']
  if model_path is None:
    model = pf.models.Baseline()
  else:
    model_name = train_kwargs['model_kwargs']['class']
    c = getattr(pf.models, model_name, None)
    if c is None:
      c = getattr(ccf.models, model_name, None)
    if c is None:
      raise NotImplementedError(model_name) 
    model = c.load_from_checkpoint(model_path)
  dks = train_kwargs['create_dataset_kwargs']
  max_prediction_length = dks['dataset_kwargs']['max_prediction_length']
  max_encoder_length = dks['dataset_kwargs']['max_encoder_length']
  min_length = max_encoder_length + max_prediction_length
  resample_seconds = pd.to_timedelta(rule).total_seconds()
  dks['split'] = None
  dks['feature_data_kwargs']['start'] = -past
  # dks['feature_data_kwargs']['start'] = -max_encoder_length*resample_seconds
  # dks['feature_data_kwargs']['end'] = max_prediction_length*resample_seconds
  # dks['feature_data_kwargs']['end'] = None
  dks['feature_data_kwargs']['end'] = None
  dks['dataset_kwargs']['predict_mode'] = True
  while True:
    print(datetime.utcnow())
    t0 = time.time()
    ds, _, df, _ = create_dataset(**deepcopy(dks))
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
      pred = [pred] if len(ds.target_names) == 1 else pred
      pred_dfs = []
      for g, gdf in df.groupby('group'):
        g_idx = idxs[idxs['group'] == g]
        p_idx, t_idx = g_idx.iloc[0].name, g_idx.iloc[0].time_idx
        t_last = gdf.index[gdf['time_idx'] == t_idx - 1].tolist()[0]
        df_future = gdf[gdf['time_idx'] >= t_idx]
        horizons = (df_future.index - t_last).total_seconds().tolist() 
        tgt_dfs = []
        for tgt_idx, tgt in enumerate(ds.target_names):
          pred_suffix = '-'.join(tgt.split('-')[1:])  # remove target prefix
          if predict_kwargs['mode'] == 'quantiles':
            ps = pred[tgt_idx][p_idx].tolist()
            data = [x + [g] + [y] for x, y in zip(ps, horizons)]
            qs = model.loss.quantiles
            if len(ds.target_names) > 1:
              qs = qs[tgt_idx]
            columns = ['-'.join([f'{prediction_prefix}_{x}', pred_suffix]) for x in qs]
            columns.append('group')
            columns.append('horizon')
            pred_df = pd.DataFrame(
              data=data, 
              columns=columns,
              index=df_future.index)
            tgt_dfs.append(pred_df)
          elif predict_kwargs['mode'] == 'prediction':
            ps = pred[tgt_idx][p_idx].tolist()
            data = [[x, g, y] for x, y in zip(ps, horizons)] 
            pred_df = pd.DataFrame( 
              data=data, 
              columns=['-'.join([prediction_prefix, pred_suffix]), 'group', 'horizon'],
              index=df_future.index)
            tgt_dfs.append(pred_df)
          else:
            raise NotImplementedError(predict_kwargs['mode'])
        tgt_df = pd.concat(tgt_dfs, axis=1)
        tgt_df = tgt_df.loc[:, ~tgt_df.columns.duplicated()]
        pred_dfs.append(tgt_df)
      pred_df = pd.concat(pred_dfs)
      write_kwargs['con'] = create_engine(**engine_kwargs)
      pred_df.to_sql(**write_kwargs)
    dt_total = time.time() - t0
    wt = max(0, resample_seconds - dt_total)
    if verbose:
      dt_pred = time.time() - (t0 + dt_data)
      print(f'status: {status}, n: {len(pred_df) if status else 0}, dt_data: {dt_data:.3f}, dt_pred: {dt_pred:.3f}, dt_total: {dt_total:.3f}, wt: {wt:.3f}')
    time.sleep(wt)
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  predict(**OmegaConf.to_object(cfg))


if __name__ == "__main__":
  app()
