import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from copy import deepcopy
import gc
import os
import functools
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import pytorch_forecasting as pf
import pytorch_lightning as pl
from sqlalchemy import create_engine

from ccf.create_dataset import Dataset
from ccf import models as ccf_models
from ccf import agents as ccf_agents
from ccf.utils import delta2value


def predict(model_path, train_kwargs, data_kwargs, 
            predict_kwargs, size, agents, verbose=False, horizons=None, executor=None,
            model_version=None, prediction_prefix='pred', dataloader_kwargs=None, delay=0):
  executor = {} if executor is None else executor
  # Initialize agents
  for name, kwargs in agents.items():
    class_name = kwargs.pop('class')
    agents[name] = getattr(ccf_agents, class_name)(**kwargs)
  # Initialize model
  model_path = Path(model_path)
  if model_path.is_file():
    model_class = train_kwargs['model_kwargs']['class']
    model_name = model_path.stem
  else:
    model_name = str(model_path)
  print(model_name)
  c = getattr(pf.models, model_class, None)
  if c is None:
    c = getattr(ccf_models, model_class, None)
  if c is None:
    raise NotImplementedError(model_class)
  if model_path.is_file():
    model = c.load_from_checkpoint(model_path)
  else:
    model = c()
  # Initialize dataset
  dks = train_kwargs['create_dataset_kwargs']
  dks['split'] = None
  dks['size'] = size
  quant = dks.get('quant', None)
  dks['dataset_kwargs']['predict_mode'] = True
  dataset = Dataset(**deepcopy(dks))
  # Initialize exectutor
  executor = ThreadPoolExecutor(**executor)  # TODO async
  # Predict in infinite loop
  t0 = time.time()
  for ds, _, df, _ in dataset:
    dt_data = time.time() - t0
    t0 = time.time()
    print(datetime.utcnow(), str(df.tail(1).index))
    if verbose:
      print(df)
    # Clean LRU cache (due to memory leak)
    gc.collect()
    wrappers = [x for x in gc.get_objects() 
                if isinstance(x, functools._lru_cache_wrapper)]
    for wrapper in wrappers:
      wrapper.cache_clear()
    if ds is None:
      continue
    # Predict
    # dl = ds.to_dataloader(**dataloader_kwargs)
    predict_kwargs['data'] = ds
    pred, idxs = model.predict(**predict_kwargs)
    # Send
    messages = {}
    pred = [pred] if len(ds.target_names) == 1 else pred
    pred_dfs = []
    if verbose:
      print(pred)
      print(idxs)
    cols = idxs.columns
    for index, row in idxs.iterrows():
      last_known_idx = row[ds.time_idx] - 1
      pred_df = df[np.logical_and.reduce([
        df[x] == y if x != ds.time_idx else df[x] >= last_known_idx for x, y in zip(cols, row)])]
      for j, tgt in enumerate(pred):  # Foreach target
        tgt_name = ds.target_names[j]
        tgt_tokens = tgt_name.split('-')
        if len(tgt_tokens) == 4 and any(['lograt' in tgt_tokens[1], 
                                         'rat' in tgt_tokens[1], 
                                         'rel' in tgt_tokens[1]]):
          tgt_rel = tgt_tokens[1].split('_')[0]
          base_tgt = tgt_tokens[-1]
        else:
          tgt_rel = None
          base_tgt = '-'.join(tgt_tokens[1:])
        if tgt_rel is not None and base_tgt in pred_df:
          actual = pred_df[base_tgt]
        else:
          actual = pred_df[tgt_name]
        prediction = tgt[index]
        if predict_kwargs['mode'] == 'quantiles':
          quantiles = model.loss.quantiles
          if len(ds.target_names) > 1:
            quantiles = quantiles[index]
          hs = np.arange(1, 1 + len(prediction)) if horizons is None else horizons
          for horizon in hs:
            horizon_idx = last_known_idx + horizon
            horizon_row = pred_df[pred_df[ds.time_idx] == horizon_idx].iloc[-1]
            quantile_prediction = prediction[horizon - 1]
            if tgt_rel is not None:
              last_known_actual = actual[0]
              quantile_prediction = delta2value(quantile_prediction, tgt_rel, last_known_actual)
            for quantile_idx, q in enumerate(quantiles):
              message = {
                f'quantile_{q}': float(quantile_prediction[quantile_idx]),
                'exchange': horizon_row['exchange'],
                'base': horizon_row['base'],
                'quote': horizon_row['quote'],
                'quant': int(quant),
                'feature': horizon_row['feature'],
                'model': model_name,
                'version': model_version,
                'target': base_tgt,
                'horizon': horizon,
                'timestamp': int(horizon_row.name.timestamp()*1e9)}
              message_key = '/'.join([horizon_row['exchange'], horizon_row['base'], horizon_row['quote']])
              messages.setdefault(message_key, []).append(message)
        elif predict_kwargs['mode'] == 'prediction':
          if tgt_rel is not None:
            last_known_actual = actual[0]
            prediction = delta2value(prediction, tgt_rel, last_known_actual)
          hs = np.arange(1, 1 + len(prediction)) if horizons is None else horizons
          for horizon in hs:
            horizon_idx = last_known_idx + horizon
            # print(horizon, horizon_idx)
            horizon_row = pred_df[pred_df[ds.time_idx] == horizon_idx].iloc[-1]
            # print(horizon_row)
            message = {
              'value': float(prediction[horizon - 1]),
              'exchange': horizon_row['exchange'],
              'base': horizon_row['base'],
              'quote': horizon_row['quote'],
              'quant': int(quant),
              'feature': horizon_row['feature'],
              'model': model_name,
              'version': model_version,
              'target': base_tgt,
              'horizon': horizon,
              'timestamp': int(horizon_row.name.timestamp()*1e9)}
            message_key = '/'.join([horizon_row['exchange'], horizon_row['base'], horizon_row['quote']])
            messages.setdefault(message_key, []).append(message)
    dt_pred = time.time() - t0
    t0 = time.time()
    if verbose:
      pprint(messages)
    futures = []
    for agent_name, agent in agents.items():
      for producer_name, topic_keys in agent.producers_topic_keys.items():
        producer = agent.producers[producer_name]
        for topic, keys in topic_keys.items():
          for key in keys:
            for message_key, key_messages in messages.items():
              if message_key == key:
                for message in key_messages:
                  future = executor.submit(producer.send, topic, key=key, value=message)
                  futures.append(future)
    for future in as_completed(futures):
      result = future.result()
    dt_send = time.time() - t0
    dt = dt_data + dt_pred + dt_send
    wt = max(0, delay - dt)
    print(f'dt_data: {dt_data:.3f}, dt_pred: {dt_pred:.3f}, dt_send: {dt_send:.3f}, dt: {dt:.3f}, wt: {wt:.3f}, msg {sum(len(v) for v in messages.values())}')
    time.sleep(wt)
    t0 = time.time()
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  predict(**OmegaConf.to_object(cfg))


if __name__ == "__main__":
  app()
