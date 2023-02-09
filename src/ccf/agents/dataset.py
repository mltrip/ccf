# import asyncio
from datetime import datetime
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
from ccf.utils import expand_columns
from ccf import partitioners as ccf_partitioners


class KafkaDataset(Agent):      
  def __init__(self, quant, size, consumer, start=None, stop=None, verbose=False, 
               topics=None, feature_keys=None, delay=0, executor=None):
    self.quant = quant
    self.size = size
    self.start = start
    self.stop = stop
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
               executor=None, aggregate=None, interpolate=None, 
               from_env_properties=False, verbose=False):
    self.quant = quant
    self.size = size
    self.start = start
    self.stop = stop
    self.delay = delay
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
  
  class OnFeature:
    def __init__(self, topic, key, feature, size, start, stop, quant, 
                 client, bucket, aggregate, interpolate, from_env_properties, verbose):
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
