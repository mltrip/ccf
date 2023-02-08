# import asyncio
import concurrent.futures
from collections import deque
from copy import deepcopy
from datetime import datetime
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
from ccf import partitioners as ccf_partitioners
from ccf.utils import wait_first_future


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
  
  class _OnMessage:
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
    
    def _init_consumer(self):
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
    
    def _init_producer(self):
      producer = deepcopy(self.producer)
      partitioner = deepcopy(self.producer_partitioner)
      partitioner_class = partitioner.pop('class')
      partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
      partitioner.update()
      producer['key_serializer'] = partitioner.serialize_key
      producer['value_serializer'] = partitioner.serialize_value
      self._producer = KafkaProducer(**producer)
      
    def _init_metric(self):
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
    
    def _evaluate(self, prediction, target):
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
                                **self._metric_kwargs)
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
        self._init_consumer()
      if self._producer is None:
        self._init_producer()
      if self._metric is None:
        self._init_metric()
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
                  messages = self._evaluate(prediction_value, target_value)
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
    _executor = getattr(concurrent.futures, executor_class)(**executor)
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
        on_message = self._OnMessage(**on_message)
        future = _executor.submit(on_message)
        futures.append(future)
    wait_first_future(_executor, futures)

        
def MASE(y_true, y_pred, y_last):
  mae_naive = abs(y_true - y_last)
  mae = abs(y_true - y_pred)
  if mae_naive != 0:
    return mae/mae_naive
  else:
    return None


def ROR(y_true, y_pred, y_last, kind, threshold, fees, random_guess=None):
  if random_guess is None:
    dy_pred = y_pred / y_last - 1.0
  else:
    dy_pred = random.uniform(-random_guess, random_guess)
  dy_true = y_true / y_last - 1.0
  if kind == 'all':
    if abs(dy_pred) > threshold:
      if dy_pred > 0:
        return dy_true - fees
      elif dy_pred < 0:
        return -dy_true - fees
      else:
        return 0
    else:
      return 0
  elif kind == 'long':
    if dy_pred > threshold:
      return dy_true - fees
    else:
      return 0
  elif kind == 'short':
    if -dy_pred > threshold:
      return -dy_true - fees
    else:
      return 0
  else:
    raise NotImplementedError(kind)
