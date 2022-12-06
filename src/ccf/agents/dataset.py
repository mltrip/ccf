# import asyncio
# from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from collections import deque
from copy import deepcopy
import time
import random
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ccf.agents.base import Kafka
from ccf.utils import expand_columns


class KafkaDataset(Kafka):      
  def __init__(self, quant, size, consumer, consumers=None, producers=None, verbose=False, 
               topics=None, feature_keys=None, delay=0, executor=None):
    self.quant = quant
    self.size = size
    self.delay = delay
    self.topics = ['feature'] if topics is None else topics
    self.feature_keys = {} if feature_keys is None else feature_keys
    self.executor = {} if executor is None else executor
    self.verbose = verbose
    self.producers = {}
    self.aggregate_kwargs = {'func': {'.*': 'last', '.*_': 'mean'}}
    self.interpolate_kwargs = {'method': 'pad'}
    feature_consumers = {}
    for topic in self.topics:
      for feature, keys in self.feature_keys.items():
        for key in keys:
          kwargs = deepcopy(consumer)
          kwargs['topic_keys'] = {topic: [ key ]}
          c = self.init_consumer(kwargs)
          feature_consumers.setdefault(feature, []).append(c)
    self.feature_consumers = feature_consumers
    # super().__init__(consumers, producers, verbose)
  
  def consume(self, consumer, feature):
    data = {}
    for message in consumer:
      t = time.time()
      message_feature = message.value.get('feature')
      if message_feature != feature:
        continue
      # print(message.key, message.topic)
      key_feature = '-'.join([x if x is not None else '' 
                              for x in [message.key, '', feature]])
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
      wt = max(0, self.delay - dt)
      print(f'{key_feature}, dt: {dt:.3f}, wt: {wt:.3f}, dl: {list(data_lengths.values())}, dfl: {list(dfs_lengths.values())}')
      if len(dfs) == len(data) and all(v >= self.size for v in dfs_lengths.values()):
        print('Features for dataset collected!')
        return dfs
      time.sleep(wt)
  
  def __call__(self):
    e = ThreadPoolExecutor(**self.executor)
    features = {}
    f2c = {}  # future - [callable, kwargs]
    for feature, consumers in self.feature_consumers.items():
      for consumer in consumers:
        c = self.consume  # callable
        kwargs = {'consumer': consumer, 'feature': feature}
        future = e.submit(c, **kwargs)
        f2c[future] = [c, kwargs]
    for future in as_completed(f2c):
      result = future.result()
      features.update(result)
    return features
