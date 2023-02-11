import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from copy import deepcopy
import gc
import os
import functools
from pprint import pprint
import concurrent.futures

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
from ccf.model_mlflow import load_model


def predict(model_name, predict_kwargs, create_dataset_kwargs,
            agents, verbose=False, horizons=None, executor=None,
            watermark=None, kind=None, interpret_kwargs=None,
            interpret_attention=True, interpret_importance=True,
            model_version=None, model_stage=None,
            prediction_prefix='pred', dataloader_kwargs=None, delay=0):
  """Prediction with interpretation. Tested with TFT model only
  
  Pipeline:
  1. Initialization
  2. Data retrieving
  3. LRU cleaning (due to memory leak)
  4. Model updating
  5. Model prediction
  6. Model interpretation
  7. Message preparation (TODO Refactoring)
  8. Message sending
  9. Statistics plotting
  """
  # 1. Initialization
  executor = {'class': 'ThreadPoolExecutor'} if executor is None else executor
  interpret_kwargs = {} if interpret_kwargs is None else interpret_kwargs
  # Initialize agents
  for name, kwargs in agents.items():
    class_name = kwargs.pop('class')
    agents[name] = getattr(ccf_agents, class_name)(**kwargs)
  # Initialize model
  model, last_version, last_stage = load_model(model_name, model_version, model_stage)
  # Initialize dataset
  dataset = Dataset(**deepcopy(create_dataset_kwargs))
  # Initialize exectutor
  executor_class = executor.pop('class')
  executor = getattr(concurrent.futures, executor_class)(**executor)
  # Predict in infinite loop
  t_data = time.time()
  for ds, _, df, _ in dataset:  # 2. Data retrieving
    dt_data = time.time() - t_data
    print(datetime.utcnow(), df.index.min(), df.index.max())
    if verbose:
      print(df)
    # 3. LRU cleaning (due to memory leak)
    gc.collect()
    wrappers = [x for x in gc.get_objects() 
                if isinstance(x, functools._lru_cache_wrapper)]
    for wrapper in wrappers:
      wrapper.cache_clear()
    if ds is None:
      continue
    # 4. Model updating
    t_model = time.time()
    if kind == 'auto_update':
      _, cur_version, cur_stage = load_model(model_name, model_version, model_stage,
                                             metadata_only=True)
      if int(cur_version) > int(last_version):
        print(f'Updating model from {last_version} to {cur_version} version')
        model, last_version, last_stage = load_model(model_name, model_version, model_stage)
    dt_model = time.time() - t_model
    # 5. Model prediction
    t_pred = time.time()
    # dl = ds.to_dataloader(**dataloader_kwargs)
    predict_kwargs_ = deepcopy(predict_kwargs)
    predict_kwargs_['data'] = ds
    predict_kwargs_['mode'] = 'raw'
    preds, idxs = model.unwrap_python_model().predict_model(predict_kwargs_)
    dt_pred = time.time() - t_pred
    if verbose:
      pprint(preds)
    # 6. Model interpretation
    t_interpret = time.time()
    if interpret_attention or interpret_importance:
      interpret_kwargs_ = deepcopy(interpret_kwargs)
      interpret_kwargs_['out'] = preds
      default_horizon = interpret_kwargs_.pop('attention_prediction_horizon', None)
      hs = [default_horizon] if default_horizon is not None else horizons
      hs = [1] if hs is None else hs  # e.g. if horizons is None
      attentions, importances = {}, {}
      for horizon in hs:
        interpret_kwargs_['attention_prediction_horizon'] = horizon - 1  # horizon starts from 0 here
        interpretation = model.unwrap_python_model().model.interpret_output(**interpret_kwargs_)
        if len(ds.target_names) == 1:  # add dummy dim
          attentions[horizon] = [interpretation['attention']]
        else:
          attentions[horizon] == interpretation['attention']
        if not importances:  # Update once because importance doesn't depend on horizon
          if len(ds.target_names) == 1:  # add dummy dim
            importances['decoder_variables'] = [interpretation['decoder_variables']]
            importances['encoder_variables'] = [interpretation['encoder_variables']]
            importances['static_variables'] = [interpretation['static_variables']]
          else:
            importances['decoder_variables'] = interpretation['decoder_variables']
            importances['encoder_variables'] = interpretation['encoder_variables']
            importances['static_variables'] = interpretation['static_variables']
    dt_interpret = time.time() - t_interpret
    # 7. Message preparation
    t_prep = time.time()
    if predict_kwargs['mode'] == 'quantiles':
      pred = model.unwrap_python_model().model.to_quantiles(preds)
    elif predict_kwargs['mode'] == 'prediction':
      pred = model.unwrap_python_model().model.to_prediction(preds)
    if len(ds.target_names) == 1:  # add dummy dim
      pred = [pred]
    messages = {'prediction': {}, 'metric': {}}
    pred_dfs = []
    cols = idxs.columns
    for index, row in idxs.iterrows():  # Foreach time
      last_known_idx = row[ds.time_idx] - 1
      pred_df = df[np.logical_and.reduce([
        df[x] == y if x != ds.time_idx else df[x] >= last_known_idx for x, y in zip(cols, row)])]
      for tgt_i, tgt in enumerate(pred):  # Foreach target
        tgt_name = ds.target_names[tgt_i]
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
        last_known_actual = actual[0]
        attention_flag, importance_flag = True, True
        if predict_kwargs['mode'] == 'quantiles':
          quantiles = model.unwrap_python_model().model.loss.quantiles
          if len(ds.target_names) > 1:
            quantiles = quantiles[index]
          for quantile_idx, q in enumerate(quantiles):
            quantile_prediction = prediction[:, quantile_idx]
            if tgt_rel is not None:
              quantile_prediction = delta2value(quantile_prediction, tgt_rel, last_known_actual)
            hs = np.arange(1, 1 + len(prediction)) if horizons is None else horizons
            for horizon in hs:
              horizon_idx = last_known_idx + horizon
              horizon_row = pred_df[pred_df[ds.time_idx] == horizon_idx].iloc[-1]
              message = {
                'last': float(last_known_actual),
                f'quantile_{q}': float(quantile_prediction[horizon - 1]),
                'exchange': horizon_row['exchange'],
                'base': horizon_row['base'],
                'quote': horizon_row['quote'],
                'quant': dataset.quant,
                'feature': horizon_row['feature'],
                'model': model_name,
                'version': last_version,
                'target': base_tgt.replace(',', '.'),  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ','
                'horizon': horizon,
                'timestamp': int(horizon_row.name.timestamp()*1e9)}
              message_key = '-'.join([horizon_row['exchange'], horizon_row['base'], horizon_row['quote']])
              messages['prediction'].setdefault(message_key, []).append(message)
              if attention_flag and interpret_attention:
                if horizon in attentions:
                  attention = attentions[horizon][tgt_i][index]
                  attention_message = {
                    'exchange': horizon_row['exchange'],
                    'base': horizon_row['base'],
                    'quote': horizon_row['quote'],
                    'quant': dataset.quant,
                    'feature': horizon_row['feature'],
                    'model': model_name,
                    'version': last_version,
                    'target': base_tgt.replace(',', '.'),  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ','
                    'horizon': horizon,
                    'prediction': 'quantile',
                    'metric': 'attention',
                    'timestamp': int(horizon_row.name.timestamp()*1e9)}
                  for a_i, a in enumerate(attention):
                    attention_message[f'{a_i+1}'] = a.item()
                  messages['metric'].setdefault(message_key, []).append(attention_message)
              if importance_flag and interpret_importance:
                importance_messages = []
                last_row = pred_df[pred_df[ds.time_idx] == last_known_idx].iloc[-1]
                for k, v in importances.items():
                  importance2metric = {
                    'encoder_variables': 'encoder_importance',
                    'decoder_variables': 'decoder_importance',
                    'static_variables': 'static_importance'}
                  importance_message = {
                    'exchange': horizon_row['exchange'],
                    'base': horizon_row['base'],
                    'quote': horizon_row['quote'],
                    'quant': dataset.quant,
                    'feature': horizon_row['feature'],
                    'model': model_name,
                    'version': last_version,
                    'target': base_tgt.replace(',', '.'),  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ','
                    'horizon': '-1',
                    'prediction': 'quantile',
                    'metric': importance2metric[k],
                    'timestamp': int(last_row.name.timestamp()*1e9)}
                  features = getattr(model.unwrap_python_model().model, k)
                  for f_i, f in enumerate(features):
                    f = f.replace(',', '.')  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ','
                    if not f in importance_message:
                      importance_message[f] = v[tgt_i][index][f_i].item()
                    else:
                      importance_message[f'feature_{f}'] = v[tgt_i][index][f_i].item()
                  messages['metric'].setdefault(message_key, []).append(importance_message)
                importance_flag = False  # Once
            attention_flag = False  # Once per horizon
        elif predict_kwargs['mode'] == 'prediction':
          if tgt_rel is not None:
            prediction = delta2value(prediction, tgt_rel, last_known_actual)
          hs = np.arange(1, 1 + len(prediction)) if horizons is None else horizons
          for horizon in hs:
            horizon_idx = last_known_idx + horizon
            horizon_row = pred_df[pred_df[ds.time_idx] == horizon_idx].iloc[-1]
            message = {
              'last': float(last_known_actual),
              'value': float(prediction[horizon - 1]),
              'exchange': horizon_row['exchange'],
              'base': horizon_row['base'],
              'quote': horizon_row['quote'],
              'quant': dataset.quant,
              'feature': horizon_row['feature'],
              'model': model_name,
              'version': last_version,
              'target': base_tgt.replace(',', '.'),  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ',',
              'horizon': horizon,
              'timestamp': int(horizon_row.name.timestamp()*1e9)}
            message_key = '-'.join([horizon_row['exchange'], horizon_row['base'], horizon_row['quote']])
            messages['prediction'].setdefault(message_key, []).append(message)
            if attention_flag and interpret_attention:
              if horizon in attentions:
                attention = attentions[horizon][tgt_i][index]
                attention_message = {
                  'exchange': horizon_row['exchange'],
                  'base': horizon_row['base'],
                  'quote': horizon_row['quote'],
                  'quant': dataset.quant,
                  'feature': horizon_row['feature'],
                  'model': model_name,
                  'version': last_version,
                  'target': base_tgt.replace(',', '.'),  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ','
                  'horizon': horizon,
                  'prediction': 'value',
                  'metric': 'attention',
                  'timestamp': int(horizon_row.name.timestamp()*1e9)}
                for a_i, a in enumerate(attention):
                  attention_message[f'{a_i+1}'] = a.item()
                messages['metric'].setdefault(message_key, []).append(attention_message)
            if importance_flag and interpret_importance:
              importance_messages = []
              last_row = pred_df[pred_df[ds.time_idx] == last_known_idx].iloc[-1]
              for k, v in importances.items():
                importance2metric = {
                  'encoder_variables': 'encoder_importance',
                  'decoder_variables': 'decoder_importance',
                  'static_variables': 'static_importance'}
                importance_message = {
                  'exchange': horizon_row['exchange'],
                  'base': horizon_row['base'],
                  'quote': horizon_row['quote'],
                  'quant': dataset.quant,
                  'feature': horizon_row['feature'],
                  'model': model_name,
                  'version': last_version,
                  'target': base_tgt.replace(',', '.'),  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ','
                  'horizon': '-1',
                  'prediction': 'value',
                  'metric': importance2metric[k],
                  'timestamp': int(last_row.name.timestamp()*1e9)}
                features = getattr(model.unwrap_python_model().model, k)
                for f_i, f in enumerate(features):
                  f = f.replace(',', '.')  # Workaround of pytorch forecasting "column names must not contain '.' characters" -> Replace '.' to ','
                  if not f in importance_message:
                    importance_message[f] = v[tgt_i][index][f_i].item()
                  else:
                    importance_message[f'feature_{f}'] = v[tgt_i][index][f_i].item()
                messages['metric'].setdefault(message_key, []).append(importance_message)
              importance_flag = False  # Once
          attention_flag = False  # Once per horizon
    dt_prep = time.time() - t_prep
    # 8. Message sending
    t_send = time.time()
    if verbose:
      pprint(messages)
    futures = []
    for agent_name, agent in agents.items():
      agent_messages = messages.get(agent_name, {})
      if not agent_messages:
        continue
      for producer_name, topic_keys in agent.producers_topic_keys.items():
        producer = agent.producers[producer_name]
        for topic, keys in topic_keys.items():
          for key in keys:
            for message_key, key_messages in agent_messages.items():
              if message_key == key:
                for message in key_messages:
                  future = executor.submit(producer.send, topic, key=key, value=message)
                  futures.append(future)
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
    dt_send = time.time() - t_send
    # 9. Statistics plotting
    dt = dt_data + dt_model + dt_pred + dt_prep + dt_send + dt_interpret
    wt = max(0, delay - dt)
    print(f'dt_data: {dt_data:.3f}, dt_model: {dt_model:.3f}, dt_pred: {dt_pred:.3f}, dt_prep: {dt_prep:.3f}, dt_send: {dt_send:.3f}, dt_interpret: {dt_interpret:.3f}, dt: {dt:.3f}, wt: {wt:.3f}, msg_sent: {len(futures)}')
    time.sleep(wt)
    t_data = time.time()
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  predict(**OmegaConf.to_object(cfg))


if __name__ == "__main__":
  app()
