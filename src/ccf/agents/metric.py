# import asyncio
import concurrent.futures
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import time
import random
from pprint import pprint
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import sklearn.metrics

from ccf.agents.base import Agent
from ccf.agents import InfluxDB
from ccf import partitioners as ccf_partitioners
from ccf.utils import wait_first_future, initialize_time


class Metric(Agent):      
  def __init__(self, keys, metrics, watermark=None,
               feature_topic='feature',
               prediction_topic='prediction', metric_topic='metric',
               consumer_partitioner=None, consumer=None, 
               producer_partitioner=None, producer=None, 
               executor=None, verbose=False):
    self.keys = keys
    self.metrics = metrics
    self.watermark = watermark
    self.feature_topic = feature_topic
    self.prediction_topic = prediction_topic
    self.metric_topic = metric_topic
    if consumer_partitioner is None:
      consumer_partitioner = {}
    self.consumer_partitioner = consumer_partitioner
    self.consumer = {} if consumer is None else consumer
    if producer_partitioner is None:
      producer_partitioner = {}
    self.producer_partitioner = producer_partitioner
    self.producer = {} if producer is None else producer
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    self.verbose = verbose
  
  class OnMessage:
    def __init__(self, key, metric, watermark,
                 feature_topic, prediction_topic, metric_topic,
                 consumer_partitioner, consumer,
                 producer_partitioner, producer, verbose):
      self.key = key
      self.metric = metric
      self.watermark = int(watermark) if watermark is not None else watermark
      self.feature_topic = feature_topic
      self.prediction_topic = prediction_topic
      self.metric_topic = metric_topic
      self.consumer_partitioner = consumer_partitioner
      self.consumer = consumer
      self.producer_partitioner = producer_partitioner
      self.producer = producer
      self._consumer = None
      self._producer = None
      self._metric = None
      self._metric_name = None
      self._metric_kwargs = None
      self.verbose = verbose
    
    def init_consumer(self):
      consumer = deepcopy(self.consumer)
      partitioner = deepcopy(self.consumer_partitioner)
      partitioner_class = partitioner.pop('class')
      partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
      partitioner.update()
      consumer['key_deserializer'] = partitioner.deserialize_key
      consumer['value_deserializer'] = partitioner.deserialize_value
      self._consumer = KafkaConsumer(**consumer)
      partitions = partitioner[self.key]
      self._consumer.assign([TopicPartition(y, x) 
                             for x in partitions 
                             for y in [self.prediction_topic, self.feature_topic]])
    
    def init_producer(self):
      producer = deepcopy(self.producer)
      partitioner = deepcopy(self.producer_partitioner)
      partitioner_class = partitioner.pop('class')
      partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
      partitioner.update()
      producer['key_serializer'] = partitioner.serialize_key
      producer['value_serializer'] = partitioner.serialize_value
      self._producer = KafkaProducer(**producer)
      
    def init_metric(self):
      metric = deepcopy(self.metric)
      self._metric_class = metric.pop('class')
      self._metric_name = metric.pop('name')
      self._metric_kwargs = metric
      if 'sklearn' in self._metric_class:
        self._metric = getattr(sklearn.metrics, 
                               self._metric_class.replace('sklearn.metrics.', ''))
      elif self._metric_class in globals():
        self._metric = globals()[self._metric_class]
      else:
        raise NotImplementedError(self._metric_class)
    
    def evaluate_metric(self, prediction, target):
      target_value = target[prediction['target']]
      last_value = prediction['last']
      message_keys = ['exchange', 'base', 'quote', 'quant', 'feature', 
                      'model', 'version', 'target', 'horizon', 'timestamp']
      base_message = {k: v for k, v in prediction.items() 
                      if k in message_keys}
      prediction_values = {k: v for k, v in prediction.items() 
                           if k not in message_keys + ['last']}
      messages = []
      for name, value in prediction_values.items():
        message = deepcopy(base_message)
        message['prediction'] = name
        if 'sklearn' in self._metric_class:
          metric = self._metric(y_true=[target_value], y_pred=[value], **self._metric_kwargs)
          message['metric'] = 'sklearn'
        elif self._metric_class in globals():
          metric = self._metric(y_true=target_value, y_pred=value, y_last=last_value,
                                name=name, **self._metric_kwargs)
          message['metric'] = 'ccf'
        else:
          raise NotImplementedError(self._metric_class)
        message[self._metric_name] = metric
        messages.append(message)
      return messages
      
    def __call__(self):
      # logger = logging.getLogger('kafka')
      # logger.setLevel(logging.CRITICAL)
      # Lazy init
      if self._consumer is None:
        self.init_consumer()
      if self._producer is None:
        self.init_producer()
      if self._metric is None:
        self.init_metric()
      # Metric
      predictions = {}
      for message in self._consumer:
        t0 = time.time()
        cnt = 0
        timestamp = message.value['timestamp']
        if message.topic == self.prediction_topic:
          target = message.value['target']
          predictions.setdefault(target, {})
          predictions[target].setdefault(timestamp, []).append(message.value)
        elif message.topic == self.feature_topic:
          for prediction, timestamps in predictions.items():
            if prediction in message.value: 
              if timestamp in timestamps:
                for prediction_value in timestamps[timestamp]:
                  target_value = message.value
                  messages = self.evaluate_metric(prediction_value, target_value)
                  for m in messages:
                    self._producer.send(topic=self.metric_topic, key=message.key, value=m)
                    cnt += 1
                predictions[prediction].pop(timestamp)
            timestamps_to_delete = []
            for t in timestamps:
              if timestamp - t > self.watermark:
                timestamps_to_delete.append(t)
            for t in timestamps_to_delete:
              predictions[prediction].pop(t)
        dt = time.time() - t0
        if self.verbose:
          ts = datetime.fromtimestamp(timestamp / 1e9)
          cache = {k: len(v) for k, v in predictions.items()}
          print(f'{datetime.utcnow()}, {ts}, {message.key}, {message.topic}, {self._metric_name}, cnt: {cnt}, cache: {cache}, dt: {dt:.3f}')
  
  def __call__(self):
    executor = deepcopy(self.executor)
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    futures = []
    for metric in self.metrics:
      for key in self.keys:
        on_message = {
          'key': key,
          'metric': metric,
          'watermark': self.watermark,
          'feature_topic': self.feature_topic,
          'prediction_topic': self.prediction_topic,
          'metric_topic': self.metric_topic,
          'consumer_partitioner': self.consumer_partitioner, 
          'consumer': self.consumer,
          'producer_partitioner': self.producer_partitioner, 
          'producer': self.producer,
          'verbose': self.verbose}
        on_message = self.OnMessage(**on_message)
        future = executor.submit(on_message)
        futures.append(future)
    wait_first_future(_executor, futures)

    
class MetricInfluxDB(InfluxDB):
  def __init__(
    self, keys, metrics,
    client=None, bucket=None, query_api=None, write_api=None,
    start=None, stop=None, size=None, quant=None, watermark=None, delay=None,
    batch_size=86400e9, executor=None,
    feature_topic='feature', prediction_topic='prediction', metric_topic='metric',
    verbose=False
  ):
    super().__init__(client, bucket, query_api, write_api, verbose)
    self.keys = keys
    self.metrics = metrics
    self.start = start
    self.stop = stop
    self.size = size
    self.quant = quant
    self.watermark = int(watermark) if watermark is not None else int(quant)
    self.delay = 0 if delay is None else int(delay)
    self.batch_size = batch_size
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    self.feature_topic = feature_topic
    self.prediction_topic = prediction_topic
    self.metric_topic = metric_topic
    self.verbose = verbose
  
  class OnMessage:
    def __init__(
      self, client, bucket, query_api, write_api, 
      key, metric, start, stop, size, quant, watermark, delay, batch_size,
      feature_topic, prediction_topic, metric_topic, verbose
    ):
      self.client = client
      self.bucket = bucket
      self.query_api = query_api
      self.write_api = write_api
      self.key = key
      self.metric = metric
      self.start = start
      self.stop = stop
      self.size = size
      self.quant = quant
      self.watermark = watermark
      self.delay = delay
      self.batch_size = batch_size
      self.feature_topic = feature_topic
      self.prediction_topic = prediction_topic
      self.metric_topic = metric_topic
      self.verbose = verbose
      self._metric = None
      self._metric_name = None
      self._metric_class = None
      self._metric_kwargs = None
    
    def init_metric(self):
      metric = deepcopy(self.metric)
      self._metric_class = metric.pop('class')
      self._metric_name = metric.pop('name')
      self._metric_kwargs = metric
      if 'sklearn' in self._metric_class:
        self._metric = getattr(sklearn.metrics, 
                               self._metric_class.replace('sklearn.metrics.', ''))
      elif self._metric_class in globals():
        self._metric = globals()[self._metric_class]
      else:
        raise NotImplementedError(self._metric_class)
    
    def evaluate_metric(self, prediction, target):
      target_value = target[prediction['target']]
      last_value = prediction['last']
      message_keys = ['exchange', 'base', 'quote', 'quant', 'feature', 
                      'model', 'version', 'target', 'horizon', 'timestamp']
      base_message = {k: v for k, v in prediction.items() 
                      if k in message_keys}
      prediction_values = {k: v for k, v in prediction.items() 
                           if k not in message_keys + ['last']}
      messages = []
      for name, value in prediction_values.items():
        message = deepcopy(base_message)
        message['prediction'] = name
        if 'sklearn' in self._metric_class:
          metric = self._metric(y_true=[target_value], y_pred=[value], **self._metric_kwargs)
          message['metric'] = 'sklearn'
        elif self._metric_class in globals():
          metric = self._metric(y_true=target_value, y_pred=value, y_last=last_value,
                                name=name, **self._metric_kwargs)
          message['metric'] = 'ccf'
        else:
          raise NotImplementedError(self._metric_class)
        message[self._metric_name] = metric
        messages.append(message)
      return messages
      
    def __call__(self):
      # Lazy init
      if self._metric is None:
        self.init_metric()
      start, stop, size, quant = initialize_time(self.start, self.stop, 
                                                 self.size, self.quant)
      client = InfluxDB.init_client(self.client)
      query_api = InfluxDB.get_query_api(client, self.query_api)
      write_api = InfluxDB.get_write_api(client, self.write_api)
      # Update streams
      streams = {}
      exchange, base, quote = self.key.split('-')
      streams[self.feature_topic] = InfluxDB.get_feature_batch_stream(
        query_api, bucket=self.bucket, 
        start=start, stop=stop, batch_size=self.batch_size,
        exchange=exchange, base=base, quote=quote, feature=None, quant=None, 
        verbose=self.verbose)
      streams[self.prediction_topic] = InfluxDB.get_prediction_batch_stream(
        query_api, bucket=self.bucket,
        start=start, stop=stop, batch_size=self.batch_size,
        exchange=exchange, base=base, quote=quote, feature=None, quant=None, 
        model=None, version=None, horizon=None, target=None, 
        verbose=self.verbose)
      # Update buffers
      cur_t = (start // quant)*quant
      start_t = cur_t
      delay_t = cur_t + self.delay
      stop_t = (stop // quant)*quant
      buffers = {}
      while cur_t < stop_t:
        time_t = time.time()
        next_t = cur_t + quant
        watermark_t = next_t - self.watermark
        print('\nmetric')
        print(f'now:       {datetime.utcnow()}')
        print(f'start:     {datetime.fromtimestamp(start_t/1e9, tz=timezone.utc)}')
        print(f'delay:     {datetime.fromtimestamp(delay_t/1e9, tz=timezone.utc)}')
        print(f'watermark: {datetime.fromtimestamp(watermark_t/1e9, tz=timezone.utc)}')
        print(f'current:   {datetime.fromtimestamp(cur_t/1e9, tz=timezone.utc)}')
        print(f'next:      {datetime.fromtimestamp(next_t/1e9, tz=timezone.utc)}')
        print(f'stop:      {datetime.fromtimestamp(stop_t/1e9, tz=timezone.utc)}')
        for topic, stream in streams.items():
          buffer = buffers.get(topic, [])
          # Append to buffer
          cur_buffer_t = watermark_t
          while cur_buffer_t < next_t:
            try:
              message = next(stream)
              cur_buffer_t = message['timestamp']
              buffer.append(message)
            except StopIteration:
              break
          # Sink buffer
          buffer = [x for x in buffer if x['timestamp'] > watermark_t]
          buffers[topic] = buffer
          print(f'buffer "{topic}": {len(buffer)}')
        print(f'end:       {datetime.utcnow()}')
        time_dt = time.time() - time_t
        cur_t = next_t
        # Evaluate metric
        messages = []
        for prediction in buffers[self.prediction_topic]:
          for target in buffers[self.feature_topic]:
            if prediction['timestamp'] == target['timestamp']:
              ms = self.evaluate_metric(prediction, target)
              messages.extend(ms)
        # Send
        for message in messages:
          record = InfluxDB.message_to_record(message, self.metric_topic)
          results = write_api.write(bucket=self.bucket, record=record,
                                    write_precision='ns')
        print(f'messages:  {len(messages)}')
        print(f'dt:        {time_dt}')
  
  def __call__(self):
    executor = deepcopy(self.executor)
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    futures = []
    for metric in self.metrics:
      for key in self.keys:
        on_message = {
          'client': self.client,
          'bucket': self.bucket,
          'query_api': self.query_api,
          'write_api': self.write_api,
          'key': key,
          'metric': metric,
          'start': self.start,
          'stop': self.stop,
          'size': self.size,
          'quant': self.quant,
          'watermark': self.watermark, 
          'delay': self.delay,
          'batch_size': self.batch_size,
          'feature_topic': self.feature_topic,
          'prediction_topic': self.prediction_topic,
          'metric_topic': self.metric_topic,
          'verbose': self.verbose}
        on_message = self.OnMessage(**on_message)
        future = executor.submit(on_message)
        futures.append(future)
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
    # wait_first_future(executor, futures)
    
    
def MASE(y_true, y_pred, y_last, name):
  mae_naive = abs(y_true - y_last)
  mae = abs(y_true - y_pred)
  if mae_naive != 0:
    return mae/mae_naive
  else:
    return None


def ROR(y_true, y_pred, y_last, name, 
        kind='all', threshold=0.0, fees=0.0, random_guess=None):
  if random_guess is None:
    dy_pred = y_pred / y_last - 1.0
  else:
    dy_pred = random.uniform(-random_guess, random_guess)
  dy_true = y_true / y_last - 1.0
  if kind == 'all':
    if name == 'value':
      if dy_pred > threshold:  # Long
        ror = dy_true - fees
      elif dy_pred < -threshold:  # short
        ror = -dy_true - fees
      else:  # hold
        ror = 0.0
    elif name.startswith('quantile'):
      quantile = float(name.split('_')[1])
      if quantile < 0.5:  # long
        if dy_pred > threshold:
          ror = dy_true - fees
        else:
          ror = 0.0
      elif quantile > 0.5:  # short
        if dy_pred < -threshold:
          ror = -dy_true - fees
        else:
          ror = 0.0
      else:  # q == 0.5 -> all
        if dy_pred > threshold:  # Long
          ror = dy_true - fees
        elif dy_pred < -threshold:  # short
          ror = -dy_true - fees
        else:  # hold
          ror = 0.0
    else:
      raise NotImplementedError(name)
  elif kind == 'long':
    if dy_pred > threshold:
      ror = dy_true - fees
    else:
      ror = 0.0
  elif kind == 'short':
    if dy_pred < -threshold:
      ror = -dy_true - fees
    else:
      ror = 0.0
  else:
    raise NotImplementedError(kind)
  return ror
