import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from copy import deepcopy
import gc
import os
import functools
from pprint import pprint

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from kafka import KafkaConsumer, KafkaProducer, TopicPartition

from ccf.create_dataset import Dataset
from ccf.model_mlflow import update_model
from ccf import partitioners as ccf_partitioners


def predict(
  model_name, key,
  create_dataset_kwargs, 
  producer, partitioner, 
  topic='prediction',
  model_kind=None, model_version=None, model_stage=None,
  max_model_delay=None,
  verbose=False
):
  # Init producer
  producer_class = partitioner.pop('class')
  partitioner = getattr(ccf_partitioners, producer_class)(**partitioner)
  partitioner.update()
  producer['partitioner'] = partitioner
  producer['key_serializer'] = partitioner.serialize_key
  producer['value_serializer'] = partitioner.serialize_value
  producer = KafkaProducer(**producer)
  # Init dataset
  dataset = Dataset(**create_dataset_kwargs)
  # Init model
  model, last_version, last_stage = update_model(
    model=None, kind=model_kind, name=model_name, 
    version=model_version, stage=model_stage)
  last_model_t =  time.time_ns()
  # Run
  prev_t = time.time_ns()
  while True:
    cur_t = time.time_ns()
    delta_t = cur_t - prev_t
    print(f'now:     {datetime.fromtimestamp(cur_t/1e9, tz=timezone.utc)}')
    print(f'delta_t: {delta_t/1e9}')
    ds_t, ds_v, df_t, df_v = dataset()
    if ds_t is None:
      prev_t = cur_t
      continue
    print('Obtaining observation')
    dl_t = ds_t.to_dataloader(train=False, batch_size=1, batch_sampler='synchronized')
    x, y = next(iter(dl_t))
    if verbose:
      pprint(x)
      pprint(y)
    obs = x['encoder_cont'][0]
    print('Observation:')
    print(obs)
    decoder_first_index = len(df_t) - x['decoder_lengths'][0].item()
    encoder_first_index = len(df_t) - x['decoder_lengths'][0].item() - x['encoder_lengths'][0].item()
    encoder_last_index = decoder_first_index - 1
    last_known_index = encoder_first_index - 1
    print(last_known_index, encoder_first_index, encoder_last_index, decoder_first_index, len(df_t))
    last_known_row = df_t.iloc[last_known_index]
    encoder_first_row = df_t.iloc[encoder_first_index]  # first prediction
    encoder_last_row = df_t.iloc[encoder_last_index]  # last prediction
    decoder_first_row = df_t.iloc[decoder_first_index]
    if verbose:
      print(last_known_row)
      print(encoder_first_row)
      print(encoder_last_row)
      print(decoder_first_row)
    last_t = int(last_known_row.name.timestamp()*1e9)
    delta_last = cur_t - last_t
    print(f'delta_last: {delta_last/1e9}')
    print('Cleaning LRU cache')
    gc.collect()
    wrappers = [x for x in gc.get_objects() 
                if isinstance(x, functools._lru_cache_wrapper)]
    for wrapper in wrappers:
      wrapper.cache_clear()
    print('LRU cache Cleaned')
    # Update model
    prev_version = last_version
    model, last_version, last_stage = update_model(
      model=model, kind=model_kind, name=model_name, 
      version=last_version, stage=last_stage)
    print(f'prev_version: {prev_version}, last_version: {last_version}, model_version: {model_version}')
    if last_version != prev_version:
      last_model_t = time.time_ns()
    delta_model = cur_t - last_model_t
    print(f'delta_model: {delta_model/1e9}/{max_model_delay/1e9}')
    if max_model_delay is not None:
      if delta_model > max_model_delay:
        print('Skipping: max model delay')
        continue
    print('Prediction action')
    action, _ = model.predict(obs)  # 0: HOLD, 1: BUY, 2: SELL
    print(f'action: {action}')
    print('Creating message')
    message = {
      'last': None,
      'action': int(action.item()),
      'exchange': last_known_row['exchange'],
      'base': last_known_row['base'],
      'quote': last_known_row['quote'],
      'quant': int(last_known_row['quant']),
      'feature': None,
      'model': model_name,
      'version': last_version,
      'target': None,
      'horizon': None,
      'timestamp': int(last_known_row.name.timestamp()*1e9)}
    pprint(message)
    print('Sending message')
    producer.send(topic, key=key, value=message)
    print('Done')
    prev_t = cur_t
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  predict(**OmegaConf.to_object(cfg))


if __name__ == "__main__":
  app()
