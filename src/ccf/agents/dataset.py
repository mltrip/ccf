# import asyncio
import concurrent.futures
from collections import deque
from copy import deepcopy
import time
import random
from pprint import pprint
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from kafka import KafkaConsumer, TopicPartition

from ccf.agents.base import Agent
from ccf.utils import expand_columns
from ccf import partitioners as ccf_partitioners


class KafkaDataset(Agent):      
  def __init__(self, quant, size, consumer, verbose=False, 
               topics=None, feature_keys=None, delay=0, executor_kwargs=None):
    self.quant = quant
    self.size = size
    self.delay = delay
    self.topics = ['feature'] if topics is None else topics
    self.feature_keys = {} if feature_keys is None else feature_keys
    if executor_kwargs is None:
      executor_kwargs = {'class': 'ThreadPoolExecutor'}
    self.executor_kwargs = executor_kwargs
    self.verbose = verbose
    self.aggregate_kwargs = {'func': {'.*': 'last', '.*_': 'mean'}}
    self.interpolate_kwargs = {'method': 'pad'}
    self.consumer_kwargs = consumer
    self.futures = None
  
  class OnFeature:
    def __init__(self, topic, key, feature, size, quant, 
                 partitioner_kwargs, consumer_kwargs,
                 aggregate_kwargs, interpolate_kwargs):
      self.topic = topic
      self.key = key
      self.partitioner_kwargs = partitioner_kwargs
      self.consumer_kwargs = consumer_kwargs
      self.feature = feature
      self.size = size
      self.quant = quant
      self.aggregate_kwargs = aggregate_kwargs
      self.interpolate_kwargs = interpolate_kwargs
      self.consumer = None
  
    def __call__(self):
      if self.consumer is None:
        logger = logging.getLogger('kafka')
        logger.setLevel(logging.CRITICAL)
        consumer_kwargs = deepcopy(self.consumer_kwargs)
        partitioner_kwargs = deepcopy(self.partitioner_kwargs)
        partitioner_class = partitioner_kwargs.pop('class')
        partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner_kwargs)
        partitioner.update()
        consumer_kwargs['key_deserializer'] = partitioner.deserialize_key
        consumer_kwargs['value_deserializer'] = partitioner.deserialize_value
        self.consumer = KafkaConsumer(**consumer_kwargs)
        partitions = partitioner[self.key]
        self.consumer.assign([TopicPartition(self.topic, x) for x in partitions])
      data = {}
      for message in self.consumer:
        t = time.time()
        message_feature = message.value.get('feature')
        if message_feature != self.feature:
          continue
        # print(message.key, message.topic)
        key_feature = '-'.join([x if x is not None else '' 
                                for x in [message.key, '', self.feature]])
        data.setdefault(key_feature, deque()).append(message.value)
        data_lengths = {k: len(v) for k, v in data.items()}
        dfs = {}
        if all(v >= self.size for v in data_lengths.values()):
          for n, d in data.items():
            df = pd.DataFrame(d)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df = df.set_index('timestamp')
            ak = deepcopy(self.aggregate_kwargs)
            ak['func'] = {kk: v for k, v in ak['func'].items() 
                          for kk in expand_columns(df.columns, [k])}
            df = df.resample(pd.Timedelta(self.quant, unit='ns')).aggregate(**ak)
            df = df.interpolate(**self.interpolate_kwargs)
            if len(df) < self.size:
              break
            dfs[n] = df
            data[n].popleft()
        dfs_lengths = {k: len(v) for k, v in dfs.items()}
        dt = time.time() - t
        print(f'{key_feature}, dt: {dt:.3f}, dl: {list(data_lengths.values())}, dfl: {list(dfs_lengths.values())}')
        if len(dfs) == len(data) and all(v >= self.size for v in dfs_lengths.values()):
          return dfs

  def __call__(self):
    executor_kwargs = deepcopy(self.executor_kwargs)
    executor_class = executor_kwargs.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor_kwargs)
    futures = []
    for topic in self.topics:
      for feature, keys in self.feature_keys.items():
        for key in keys:
          consumer_kwargs = deepcopy(self.consumer_kwargs)
          partitioner_kwargs = consumer_kwargs.pop('partitioners')['feature']
          on_feature_kwargs = {
            'topic': topic,
            'key': key,
            'feature': feature,
            'partitioner_kwargs': partitioner_kwargs,
            'consumer_kwargs': consumer_kwargs,
            'size': self.size,
            'quant': self.quant,
            'aggregate_kwargs': self.aggregate_kwargs,
            'interpolate_kwargs': self.interpolate_kwargs}
          on_feature = self.OnFeature(**on_feature_kwargs)
          future = executor.submit(on_feature)
          futures.append(future)
    features = {}
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      features.update(result)
    # print(features)
    return features
