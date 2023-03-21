# import asyncio
from datetime import datetime, timezone
import concurrent.futures
from collections import deque
from copy import deepcopy
import time
import random
from pprint import pprint
import logging
import os

from influxdb_client import InfluxDBClient
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from kafka import KafkaConsumer, TopicPartition

from ccf.agents.base import Agent
from ccf.agents import InfluxDB
from ccf.utils import expand_columns, initialize_time
from ccf import partitioners as ccf_partitioners


class KafkaDataset(Agent):      
  def __init__(self, quant, size, consumer, start=None, stop=None, verbose=False, 
               watermark=None, topics=None, feature_keys=None, delay=0, executor=None):
    self.quant = quant
    self.size = size
    self.start = start
    self.stop = stop
    self.watermark = watermark
    self.delay = delay
    self.topics = ['feature'] if topics is None else topics
    self.feature_keys = {} if feature_keys is None else feature_keys
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    self.verbose = verbose
    self.aggregate = {'func': {'.*': 'last', '.*_': 'mean'}}
    self.interpolate = {'method': 'pad'}
    self.consumer_kwargs = consumer
    self.futures = None
  
  class OnFeature:
    def __init__(self, topic, key, feature, size, quant, 
                 partitioner_kwargs, consumer_kwargs,
                 aggregate, interpolate):
      self.topic = topic
      self.key = key
      self.partitioner_kwargs = partitioner_kwargs
      self.consumer_kwargs = consumer_kwargs
      self.feature = feature
      self.size = size
      self.quant = quant
      self.aggregate = aggregate
      self.interpolate = interpolate
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
            ak = deepcopy(self.aggregate)
            ak['func'] = {kk: v for k, v in ak['func'].items() 
                          for kk in expand_columns(df.columns, [k])}
            df = df.resample(pd.Timedelta(self.quant, unit='ns')).aggregate(**ak)
            df = df.interpolate(**self.interpolate)
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
    executor = deepcopy(self.executor)
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
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
            'aggregate': self.aggregate,
            'interpolate': self.interpolate}
          on_feature = self.OnFeature(**on_feature_kwargs)
          future = executor.submit(on_feature)
          futures.append(future)
    features = {}
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      features.update(result)
    # print(features)
    return features


class InfluxdbDataset(Agent):      
  def __init__(self, quant, size=None, start=None, stop=None, client=None, 
               bucket=None, topics=None, feature_keys=None, delay=0,
               executor=None, aggregate=None, interpolate=None, watermark=None,
               from_env_properties=False, batch=None, verbose=False):
    self.quant = quant
    self.size = size
    self.start = start
    self.stop = stop
    self.delay = delay
    self.watermark = watermark
    self.topics = ['feature'] if topics is None else topics
    self.feature_keys = {} if feature_keys is None else feature_keys
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    self.verbose = verbose
    if aggregate is None:
      aggregate = {'func': {'.*': 'last', '.*_': 'mean'}}
    self.aggregate = aggregate
    if interpolate is None:
      interpolate = {'method': 'pad'}
    self.interpolate = interpolate
    self.client = {} if client is None else client
    self.client.setdefault('token', os.getenv('INFLUXDB_V2_TOKEN', None))
    self.client.setdefault('url', os.getenv('INFLUXDB_V2_URL', 'https://influxdb:8086'))
    self.client.setdefault('org', os.getenv('INFLUXDB_V2_ORG', 'mltrip'))
    self.client.setdefault('timeout', os.getenv('INFLUXDB_V2_TIMEOUT', None))
    self.client.setdefault('verify_ssl', os.getenv(
      'INFLUXDB_V2_VERIFY_SSL', 'true').lower() in ['yes', 'true', '1'])
    self.client.setdefault('proxy', os.getenv('INFLUXDB_V2_PROXY', None))
    self.bucket = os.getenv('INFLUXDB_V2_BUCKET', 'ccf') if bucket is None else bucket
    self.futures = None
    self.from_env_properties = from_env_properties
    self.batch = batch
  
  class OnFeature:
    def __init__(self, topic, key, feature, size, start, stop, quant, 
                 client, bucket, aggregate, interpolate, from_env_properties, batch, 
                 verbose):
      self.topic = topic
      self.key = key
      self.feature = feature
      self.size = size
      self.start = start
      self.stop = stop
      self.quant = quant
      self.client = client
      self.bucket = bucket
      self.aggregate = aggregate
      self.interpolate = interpolate
      self.from_env_properties = from_env_properties
      self.batch = batch
      self.verbose = verbose
      self._client = None
      
  
    def __call__(self):
      t = time.time()
      # Initialize client
      if self._client is None:
        if self.from_env_properties:
          self._client = InfluxDBClient.from_env_properties()
        else:
          self._client = InfluxDBClient(**self.client)
      # Make query
      if self.start is not None and self.stop is not None and self.batch is not None:
        dfs = []
        cur_start = self.start
        while cur_start < self.stop:
          if cur_start + self.batch < self.stop:
            cur_stop = cur_start + self.batch
          else:
            cur_stop = self.stop
          exchange, base, quote = self.key.split('-')
          stop_str = f", stop: {int(cur_stop/10**9)}"
          rename_str = '|> rename(columns: {_time: "timestamp"})'
          query = f'''
          from(bucket: "{self.bucket}")
          |> range(start: {int(cur_start/10**9)}{stop_str})
          |> filter(fn:(r) => r._measurement == "{self.topic}")
          |> filter(fn:(r) => r.exchange == "{exchange}")
          |> filter(fn:(r) => r.base == "{base}")
          |> filter(fn:(r) => r.quote == "{quote}")
          |> filter(fn:(r) => r.feature == "{self.feature}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> drop(columns: ["_start", "_stop", "_measurement", "host"])
          {rename_str}'''
          if self.verbose:
            print(query)
          df = self._client.query_api().query_data_frame(query=query)
          dfs.append(df)
          cur_start = cur_stop
        df = pd.concat(dfs, ignore_index=True, sort=False)
      else:
        exchange, base, quote = self.key.split('-')
        stop_str = f", stop: {int(self.stop/10**9)}" if self.stop is not None else ''
        rename_str = '|> rename(columns: {_time: "timestamp"})'
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: {int(self.start/10**9)}{stop_str})
        |> filter(fn:(r) => r._measurement == "{self.topic}")
        |> filter(fn:(r) => r.exchange == "{exchange}")
        |> filter(fn:(r) => r.base == "{base}")
        |> filter(fn:(r) => r.quote == "{quote}")
        |> filter(fn:(r) => r.feature == "{self.feature}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> drop(columns: ["_start", "_stop", "_measurement", "host"])
        {rename_str}'''
        if self.verbose:
          print(query)
        df = self._client.query_api().query_data_frame(query=query)
      # Raw data
      if self.verbose:
        print(df)
        print(df.columns)
        print(df.dtypes)
      # Process data
      df = df.drop(columns=['result', 'table'])  # https://community.influxdata.com/t/get-rid-of-result-table-columns/14887/3
      df = df.set_index('timestamp')
      td = pd.Timedelta(self.quant, unit='ns')
      a = deepcopy(self.aggregate)
      a['func'] = {kk: v for k, v in a['func'].items() 
                   for kk in expand_columns(df.columns, [k])}
      df = df.resample(td).aggregate(**a).interpolate(**self.interpolate)
      if self.verbose:
        print(df)
      key_feature = '-'.join([self.key, '', self.feature])
      dfs = {key_feature: df}
      dfs_shape = {k: v.shape for k, v in dfs.items()}
      dfs_interval = {k: (v.index.min(), v.index.max()) for k, v in dfs.items()}
      dt = time.time() - t
      pprint(dfs_shape)
      pprint(dfs_interval)
      print(f'{datetime.utcnow()}, dt: {dt:.3f}')
      return dfs

  def __call__(self):
    executor = deepcopy(self.executor)
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    futures = []
    for topic in self.topics:
      for feature, keys in self.feature_keys.items():
        for key in keys:
          on_feature_kwargs = {
            'topic': topic,
            'key': key,
            'feature': feature,
            'client': deepcopy(self.client),
            'size': self.size,
            'start': self.start,
            'stop': self.stop,
            'quant': self.quant,
            'bucket': self.bucket,
            'aggregate': self.aggregate,
            'interpolate': self.interpolate,
            'from_env_properties': self.from_env_properties,
            'batch': self.batch,
            'verbose': self.verbose}
          on_feature = self.OnFeature(**on_feature_kwargs)
          future = executor.submit(on_feature)
          futures.append(future)
    features = {}
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      features.update(result)
    # print(features)
    return features
  
  
class StreamDatasetInfluxDB(InfluxDB):
  """Dataset streaming from InfluxDB
  
  Note:
    * Stream is aggregated from set of OnFeature objects
    * OnFeature object get features from topic-key-feature combination
    * Use ThreadPoolExecutor because ProcessPoolExecutor resets OnFeature
    
  Todo:
    * Refactor lazy initialization
  """
  
  def __init__(self, client=None, bucket=None, query_api=None, write_api=None,
               executor=None, topic='feature', feature_keys=None, 
               quant=None, start=None, stop=None, size=None, 
               watermark=None, delay=None,
               batch_size=86400e9, replace_nan=1.0,
               resample=None, aggregate=None, interpolate=None,
               verbose=False):
    super().__init__(client, bucket, query_api, write_api, verbose)
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    self.topic = topic
    self.feature_keys = {} if feature_keys is None else feature_keys
    self.start = start
    self.stop = stop
    self.size = size
    self.quant = quant
    self.watermark = int(watermark) if watermark is not None else int(quant)
    self.delay = self.watermark if delay is None else int(delay)
    self.batch_size = batch_size
    self.replace_nan = replace_nan
    self.resample = {} if resample is None else resample
    self.aggregate = {'func': 'last'} if aggregate is None else aggregate
    self.interpolate = {'method': 'pad'} if interpolate is None else interpolate
    self._executor = None
    
  class OnFeature:
    def __init__(self, client, bucket, query_api,
                 topic, key, feature, start, stop, size, quant, watermark, delay,
                 batch_size, replace_nan, resample, aggregate, interpolate,
                 verbose):
      self.client = client
      self.bucket = bucket
      self.query_api = query_api
      self.topic = topic
      self.key = key
      self.feature = feature
      self.size = size
      self.start = start
      self.stop = stop
      self.quant = quant
      self.watermark = watermark
      self.delay = delay
      self.batch_size = batch_size
      self.replace_nan = replace_nan
      self.resample = resample
      self.aggregate = aggregate
      self.interpolate = interpolate
      self.verbose = verbose
      self.stream = None
      self.cur_t = None
      self.start_t = None
      self.delay_t = None
      self.stop_t = None
      self.buffer = None
    
    def init(self):  # Lazy init TODO using properties?
      if self.stream is None:
        self.start, self.stop, self.size, self.quant = initialize_time(self.start, self.stop, 
                                                                       self.size, self.quant)
        client = InfluxDB.init_client(self.client)
        query_api = InfluxDB.get_query_api(client, self.query_api)
        exchange, base, quote = self.key.split('-')
        if self.topic == 'feature':
          self.stream = InfluxDB.get_feature_batch_stream(query_api, bucket=self.bucket, 
                                                          start=self.start, stop=self.stop, 
                                                          batch_size=self.batch_size,
                                                          exchange=exchange, base=base, quote=quote, 
                                                          feature=self.feature, quant=self.quant,
                                                          verbose=self.verbose)
        else:
          raise NotImplementedError(self.topic)
        self.cur_t = (self.start // self.quant)*self.quant
        self.start_t = self.cur_t
        self.delay_t = self.cur_t + self.delay
        self.stop_t = (self.stop // self.quant)*self.quant
        self.buffer = []
        
    def step(self):
      # Update timestamps
      time_t = time.time()
      next_t = self.cur_t + self.quant
      watermark_t = next_t - self.watermark
      print('\ndataset step')
      print(f'now:       {datetime.utcnow()}')
      print(f'start:     {datetime.fromtimestamp(self.start_t/10**9, tz=timezone.utc)}')
      print(f'delay:     {datetime.fromtimestamp(self.delay_t/10**9, tz=timezone.utc)}')
      print(f'watermark: {datetime.fromtimestamp(watermark_t/10**9, tz=timezone.utc)}')
      print(f'current:   {datetime.fromtimestamp(self.cur_t/10**9, tz=timezone.utc)}')
      print(f'next:      {datetime.fromtimestamp(next_t/10**9, tz=timezone.utc)}')
      print(f'stop:      {datetime.fromtimestamp(self.stop_t/10**9, tz=timezone.utc)}')
      # Check stop
      if next_t > self.stop_t:
        raise StopIteration
      # Update buffer
      cur_buffer_t = watermark_t
      while cur_buffer_t < next_t:
        try:
          message = next(self.stream)
          cur_buffer_t = message['timestamp']
          self.buffer.append(message)
        except StopIteration:
          break
      # Sink buffer
      self.buffer = [x for x in self.buffer if x['timestamp'] > watermark_t]
      # Update current timestamp
      self.cur_t = next_t
      # Update dataframe
      df = pd.DataFrame(self.buffer)
      if len(df) != 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df = df.set_index('timestamp')
        # Preprocess dataframe
        resample = deepcopy(self.resample)
        resample['rule'] = pd.Timedelta(self.quant, unit='ns')
        aggregate = deepcopy(self.aggregate)
        if isinstance(aggregate.get('func', {}), dict):
          aggregate['func'] = {kk: v for k, v in aggregate.get('func', {}).items() 
                               for kk in expand_columns(df.columns, [k])}
        interpolate = deepcopy(self.interpolate) 
        df = df.resample(**resample).aggregate(**aggregate).interpolate(**interpolate)
        if self.replace_nan is not None:
          df = df.replace([np.inf, -np.inf], np.nan).replace({np.nan: self.replace_nan})
      # Print stats
      time_dt = time.time() - time_t
      print(f'end:       {datetime.utcnow()}')
      print(f'dt: {time_dt:.3f}, rows: {len(df)}, cols: {len(df.columns)}')
      if self.verbose:
        print(df)
      return df
    
    def __iter__(self):
      return self
    
    def __next__(self):
      # Lazy init
      self.init()
      # Delay
      while self.cur_t + self.quant < self.delay_t:
        self.step()
      # Step
      df = self.step()
      key_feature = '-'.join([self.key, '', self.feature])
      dfs = {key_feature: df}
      return dfs
    
    def __call__(self):
      return next(self)
  
  def init(self):  # Lazy init TODO using properties?
    if self._executor is None:
      executor = deepcopy(self.executor)
      executor_class = executor.pop('class')
      executor = getattr(concurrent.futures, executor_class)(**executor)
      self._executor = executor
      self.on_features = []
      for feature, keys in self.feature_keys.items():
        for key in keys:
          on_feature_kwargs = {
            'topic': self.topic,
            'key': key,
            'feature': feature,
            'client': deepcopy(self.client),
            'size': self.size,
            'start': self.start,
            'stop': self.stop,
            'quant': self.quant,
            'query_api': self.query_api,
            'bucket': self.bucket,
            'watermark': self.watermark,
            'delay': self.delay,
            'resample': self.resample,
            'aggregate': self.aggregate,
            'interpolate': self.interpolate,
            'replace_nan': self.replace_nan,
            'batch_size': self.batch_size,
            'verbose': self.verbose}
          on_feature = self.OnFeature(**on_feature_kwargs)
          self.on_features.append(on_feature)
  
  def __iter__(self):
      return self
    
  def __next__(self):
    # Lazy init
    self.init()
    # Step
    features = {}
    futures = [self._executor.submit(x) for x in self.on_features]
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      features.update(result)
    # print(features)
    return features
  
  def __call__(self):
    return next(self)
