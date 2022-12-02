"""
See Also:
  https://websocket-client.readthedocs.io/en/latest
  https://binance-docs.github.io/apidocs/spot/en/#how-to-manage-a-local-order-book-correctly
  https://kafka-python.readthedocs.io/en/master/usage.html
  https://pythonhosted.org/feedparser
"""
import sys
import json
from datetime import datetime, timedelta, timezone
import time
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
import websocket
import numpy as np
from sqlalchemy import create_engine
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import feedparser
from kafka import KafkaProducer
from kafka import KafkaAdminClient
from kafka.admin.new_partitions import NewPartitions


KEY2PART = {
  'binance/btc/usdt': 0, 
  'binance/eth/usdt': 1,
  'binance/eth/btc': 2
}


def partitioner(key, all_partitions, available):
  if key is None:
    if len(available) > 0:
      return random.choice(available)
    return random.choice(all_partitions)
  else:
    idx = KEY2PART[key.decode('ascii')]
    return all_partitions[idx]
  

def loop(executor, future2callable):
  for future in as_completed(future2callable):
    try:
      r = future.result()
    except Exception as e:
      print(f'Exception: {future} - {e}')
    else:
      print(f'Done: {future} - {r}')
    finally:  # Resubmit
      c, kwargs = future2callable[future]
      new_future = executor.submit(c, **kwargs)
      new_future2callable = {new_future: [c, kwargs]}
      loop(executor, new_future2callable)


def get_data(producers, producer_kwargs, client_kwargs,
             executor_kwargs=None, app_kwargs=None, run_kwargs=None, verbose=False):
  executor_kwargs = {} if executor_kwargs is None else executor_kwargs
  app_kwargs = {} if app_kwargs is None else app_kwargs
  run_kwargs = {} if run_kwargs is None else run_kwargs
  websocket.enableTrace(verbose)
  executor = ThreadPoolExecutor(**executor_kwargs)
  f2c = {}  # future: [callable, kwargs]
  for name, p in producers.items():
    print(name)
    c = p.pop('class')
    if c in ['OnLob', 'OnTrade']:
      c = globals()[c]
      on_message = c(producer_kwargs=producer_kwargs, client_kwargs=client_kwargs, **p)
      app_kwargs['on_message'] = on_message
      app_kwargs['url'] = on_message.url
      app = websocket.WebSocketApp(**app_kwargs)
      # app.run_forever(**run_kwargs)
      future = executor.submit(app.run_forever, **run_kwargs)
      f2c[future] = [app.run_forever, run_kwargs]
    elif c in ['OnFeed']:
      c = globals()[c]
      on_feed = c(producer_kwargs=producer_kwargs, client_kwargs=client_kwargs, **p)
      future = executor.submit(on_feed)
      f2c[future] = [on_feed, {}]
    else:
      raise NotImplementedError(c)
  loop(executor, f2c)

  
class OnLob:
  def __init__(self, topic, key, depth, producer_kwargs, client_kwargs, delay=0, verbose=False):
    super().__init__()
    self.topic = topic
    self.key = key
    self.depth = depth
    self.delay = delay
    self.verbose = verbose
    exchange, base, quote = key.split('/')
    self.exchange = exchange
    self.base = base
    self.quote = quote
    if exchange == 'binance':
      self.url = f'wss://stream.binance.com:9443/ws/{base}{quote}@depth{depth}@{int(delay*1000)}ms'
    else:
      raise NotImplementedError(exchange)
    producer_kwargs['partitioner'] = partitioner
    producer_kwargs['key_serializer'] = lambda x: x.encode('ascii') if isinstance(x, str) else x
    producer_kwargs['value_serializer'] = lambda x: json.dumps(x).encode('ascii')
    self.producer = KafkaProducer(**producer_kwargs)
    if len(self.producer.partitions_for(self.topic)) != len(KEY2PART):
      client = KafkaAdminClient(**client_kwargs)
      result = client.create_partitions({self.topic: NewPartitions(len(KEY2PART))})
      print(result)
    
  def __call__(self, wsapp, message):
    d = {'timestamp': time.time_ns(),
         'exchange': self.exchange,
         'base': self.base,
         'quote': self.quote}
    if self.exchange == 'binance':
      data = json.loads(message)
      for i, a in enumerate(data['asks']):
        if a[1] != '0':
          d[f'a_p_{i}'] = float(a[0])
          d[f'a_q_{i}'] = float(a[1])
        else:
          d[f'a_p_{i}'] = None
          d[f'a_q_{i}'] = None
      for i, b in enumerate(data['bids']):
        if b[1] != '0':
          d[f'b_p_{i}'] = float(b[0])
          d[f'b_q_{i}'] = float(b[1])
        else:
          d[f'b_p_{i}'] = None
          d[f'b_q_{i}'] = None
    else:
      raise NotImplementedError(self.exchange)
    # mid price
    if d.get('a_p_0', None) is not None and d.get('b_p_0', None) is not None:
      d['m_p'] = 0.5*(d['a_p_0'] + d['b_p_0']) 
    else:
      d['m_p'] = None
    if self.verbose:
      print(self.topic, self.key, d)
    self.producer.send(self.topic, key=self.key, value=d)

    
class OnTrade:
  def __init__(self, topic, key, producer_kwargs, client_kwargs, delay=0, verbose=False):
    super().__init__()
    self.topic = topic
    self.key = key
    self.delay = delay
    self.verbose = verbose
    exchange, base, quote = key.split('/')
    self.exchange = exchange
    self.base = base
    self.quote = quote
    if exchange == 'binance':
      self.url = f'wss://stream.binance.com:9443/ws/{base}{quote}@trade'
    else:
      raise NotImplementedError(exchange)
    producer_kwargs['partitioner'] = partitioner
    producer_kwargs['key_serializer'] = lambda x: x.encode('ascii') if isinstance(x, str) else x
    producer_kwargs['value_serializer'] = lambda x: json.dumps(x).encode('ascii')
    self.producer = KafkaProducer(**producer_kwargs)
    if len(self.producer.partitions_for(self.topic)) != len(KEY2PART):
      client = KafkaAdminClient(**client_kwargs)
      result = client.create_partitions({self.topic: NewPartitions(len(KEY2PART))})
      print(result)
    self.t = time.time()
    
  def __call__(self, wsapp, message):
    d = {'exchange': self.exchange,
         'base': self.base,
         'quote': self.quote}
    if self.exchange == 'binance':
      data = json.loads(message)
      d['timestamp'] = int(data['T']*1e6)  # ms -> ns
      d['t_p'] = float(data['p'])
      d['t_q'] = float(data['q'])
      # trade side: 0 - maker_sell/taker_buy or 1 - maker_buy/taker_sell
      d['t_s'] = int(data['m']) 
    else:
      raise NotImplementedError(self.exchange)
    dt = time.time() - self.t
    if dt > self.delay:
      if self.verbose:
        print(self.topic, self.key, d)
      self.producer.send(self.topic, key=self.key, value=d)
      self.t = time.time()

      
class OnFeed:
  def __init__(self, topic, feeds, producer_kwargs, client_kwargs, delay=0, start=0, verbose=False):
    super().__init__()
    self.topic = topic
    self.feeds = feeds
    self.delay = delay
    self.start = start
    self.verbose = verbose
    producer_kwargs['value_serializer'] = lambda x: json.dumps(x).encode('ascii')
    self.producer = KafkaProducer(**producer_kwargs)
    self.cache = set()
    self.t = time.time()
    
  def __call__(self):
    while True:
      dt = time.time() - self.t
      if dt <= self.delay:
        continue
      else:
        self.t = time.time()
      for f, flag in self.feeds.items():
        if not flag:
          continue
        try:
          r = feedparser.parse(f)
        except Exception:
          d, s = [], None
        else:
          d, s = r.get('entries', []), r.get('status', None)
        for e in d:
          i = e.get('id', None)
          if i not in self.cache:
            self.cache.add(i)
            t = e.get('published_parsed', None)
            if t is not None:
              t = datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
              start_t = datetime.now(timezone.utc) + timedelta(seconds=self.start)
              if t < start_t:
                continue
            else:
              continue
            authors = e.get('authors', None)
            if authors is not None:
              authors = '|'.join(x.get('name', '') for x in authors)
            tags = e.get('tags', None)
            if tags is not None:
              tags = '|'.join(x.get('term', '') for x in tags)
            links = e.get('links', None)
            if links is not None:
              links = '|'.join(x.get('href', '') for x in links)
            ee = {'id': i,
                  'title': e.get('title', None),
                  'links': links,
                  'timestamp': int(t.timestamp()*1e9),  # s to ns
                  'authors': authors,
                  'tags': tags,
                  'summary': e.get('summary', None)}
            if self.verbose:
              print(self.topic, ee)
            self.producer.send(self.topic, value=ee)

            
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  get_data(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
