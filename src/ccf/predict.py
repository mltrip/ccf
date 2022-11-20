import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from copy import deepcopy
import gc
import os
import functools

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytorch_forecasting as pf
import pytorch_lightning as pl
from sqlalchemy import create_engine

from ccf.create_dataset import create_dataset
import ccf


def predict(model_path, train_kwargs, data_kwargs, 
            predict_kwargs, past, verbose=False, 
            prediction_prefix='pred', dataloader_kwargs=None, delay=0):
  engine_kwargs = data_kwargs['query']['prediction']['engine_kwargs'] 
  write_kwargs = data_kwargs['query']['prediction']['write_kwargs']
  model_path = Path(model_path)
  if model_path.is_file():
    model_name = train_kwargs['model_kwargs']['class']
  else:
    model_name = str(model_path)
  c = getattr(pf.models, model_name, None)
  if c is None:
    c = getattr(ccf.models, model_name, None)
  if c is None:
    raise NotImplementedError(model_name)
  if model_path.is_file():
    model = c.load_from_checkpoint(model_path)
  else:
    model = c()
  dks = train_kwargs['create_dataset_kwargs']
  dks['split'] = None
  dks['feature_data_kwargs']['start'] = -past
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
    gc.collect()
    wrappers = [x for x in gc.get_objects() 
                if isinstance(x, functools._lru_cache_wrapper)]
    for wrapper in wrappers:
      wrapper.cache_clear()
    if status is not None and status:
      # dl = ds.to_dataloader(**dataloader_kwargs)
      predict_kwargs['data'] = ds
      pred, idxs = model.predict(**predict_kwargs)
      pred = [pred] if len(ds.target_names) == 1 else pred
      pred_dfs = []
      for g, gdf in df.groupby('group'):
        g_idx = idxs[idxs['group'] == g]
        if len(g_idx) == 0:
          print(f'Warning! No group {g} in predictions')
          continue
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
      if len(pred_dfs) > 0:
        pred_df = pd.concat(pred_dfs)
        write_kwargs['con'] = create_engine(**engine_kwargs)
        pred_df.to_sql(**write_kwargs)
    dt = time.time() - t0
    wt = max(0, delay - dt)
    if verbose:
      dt_pred = time.time() - (t0 + dt_data)
      print(f'status: {status}, n: {len(pred_df) if status else 0}, horizon: {pred_df.index.max() if status else ""}, dt_data: {dt_data:.3f}, dt_pred: {dt_pred:.3f}, dt: {dt:.3f}, wt: {wt:.3f}')
    time.sleep(wt)
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  predict(**OmegaConf.to_object(cfg))


if __name__ == "__main__":
  app()
