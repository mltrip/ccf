"""
See Also:
  https://websocket-client.readthedocs.io/en/latest
  https://binance-docs.github.io/apidocs/spot/en/#how-to-manage-a-local-order-book-correctly
  https://kafka-python.readthedocs.io/en/master/usage.html
  https://pythonhosted.org/feedparser
"""
import concurrent.futures
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import time
from pprint import pprint

import feedparser
from kafka import KafkaProducer
import websocket

from ccf.agents.base import Agent
from ccf import partitioners as ccf_partitioners

  
class Lob(Agent):
  def __init__(self, topic, keys, 
               partitioner=None, producer=None, 
               app=None, run=None, 
               executor=None,
               depth=None, delay=None, verbose=False):
    super().__init__()
    self.topic = topic
    self.keys = keys
    self.partitioner = {} if partitioner is None else partitioner
    self.producer = {} if producer is None else producer
    self.app = {} if app is None else app
    self.run = {} if run is None else run
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    self.depth = depth
    self.delay = delay
    self.verbose = verbose
    
  class OnError:
    def __init__(self):
      pass
    
    def __call__(self, wc, error, *args, **kwargs):
      raise error
    
  class OnMessage:
    def __init__(self, topic, key, partitioner, producer,
                 exchange, base, quote, verbose=0):
      self.topic = topic
      self.key = key
      self.partitioner = partitioner
      self.producer = producer
      self.exchange = exchange
      self.base = base
      self.quote = quote
      self.verbose = verbose
      self._producer = None  # Lazy init see __call__
    
    def __call__(self, wc, message):
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
        print(datetime.utcnow(), self.topic, self.key, len(d))
      if self.verbose > 1:
        pprint(d)
      if self._producer is None:  # Lazy init
        partitioner = deepcopy(self.partitioner)
        producer = deepcopy(self.producer)
        partitioner_class = partitioner.pop('class')
        partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
        partitioner.update()
        producer['partitioner'] = partitioner
        producer['key_serializer'] = partitioner.serialize_key
        producer['value_serializer'] = partitioner.serialize_value
        self._producer = KafkaProducer(**producer)
      self._producer.send(self.topic, key=self.key, value=d)

  def __call__(self):
    websocket.enableTrace(True if self.verbose > 1 else False)
    executor = deepcopy(self.executor)
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    futures = []
    for key in self.keys:
      partitioner = deepcopy(self.partitioner)
      producer = deepcopy(self.producer)
      app = deepcopy(self.app)
      run = deepcopy(self.run)
      exchange, base, quote = key.split('-')
      if exchange == 'binance':
        suffix = '@'.join(x for x in [
          f'{base}{quote}', 
          f'depth{self.depth}' if self.depth is not None else 'depth',
          f'{int(self.delay*1000)}ms' if self.delay is not None else None
        ] if x is not None)
        url = f'wss://stream.binance.com:9443/ws/{suffix}'
      else:
        raise NotImplementedError(exchange)
      print(url)
      on_message = {
        'topic': self.topic,
        'key': key,
        'partitioner': partitioner,
        'producer': producer,
        'exchange': exchange,
        'base': base,
        'quote': quote,
        'verbose': self.verbose}
      app['on_message'] = self.OnMessage(**on_message)
      app['on_error'] = self.OnError()
      app['url'] = url
      app = websocket.WebSocketApp(**app)
      future = executor.submit(app.run_forever, **run)
      futures.append(future)
    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    
class Trade(Agent):
  def __init__(self, topic, keys, 
               partitioner=None, producer=None, 
               app=None, run=None, 
               executor=None,
               delay=None, verbose=False):
    super().__init__()
    self.topic = topic
    self.keys = keys
    self.partitioner = {} if partitioner is None else partitioner
    self.producer = {} if producer is None else producer
    self.app = {} if app is None else app
    self.run = {} if run is None else run
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    self.delay = delay
    self.verbose = verbose
    
  class OnError:
    def __init__(self):
      pass
    
    def __call__(self, wc, error, *args, **kwargs):
      raise error
    
  class OnMessage:
    def __init__(self, topic, key, partitioner, producer,
                 exchange, base, quote, delay=None, verbose=0):
      self.topic = topic
      self.key = key
      self.partitioner = partitioner
      self.producer = producer
      self.exchange = exchange
      self.base = base
      self.quote = quote
      self.delay = delay
      self.verbose = verbose
      self._producer = None  # Lazy init see __call__
      self.t = time.time()
    
    def __call__(self, wc, message):
      if self.delay is not None:
        if time.time() - self.t < self.delay:
          return
        else:
          self.t = time.time()
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
      if self.verbose:
        print(datetime.utcnow(), self.topic, self.key, len(d))
      if self.verbose > 1:
        pprint(d)
      if self._producer is None:  # Lazy init
        partitioner = deepcopy(self.partitioner)
        producer = deepcopy(self.producer)
        partitioner_class = partitioner.pop('class')
        partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
        partitioner.update()
        producer['partitioner'] = partitioner
        producer['key_serializer'] = partitioner.serialize_key
        producer['value_serializer'] = partitioner.serialize_value
        self._producer = KafkaProducer(**producer)
      self._producer.send(self.topic, key=self.key, value=d)

  def __call__(self):
    websocket.enableTrace(True if self.verbose > 1 else False)
    executor = deepcopy(self.executor)
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    futures = []
    for key in self.keys:
      partitioner = deepcopy(self.partitioner)
      producer = deepcopy(self.producer)
      app = deepcopy(self.app)
      run = deepcopy(self.run)
      exchange, base, quote = key.split('-')
      if exchange == 'binance':
        suffix = '@'.join(x for x in [f'{base}{quote}', 'trade'])
        url = f'wss://stream.binance.com:9443/ws/{base}{quote}@trade'
      else:
        raise NotImplementedError(exchange)
      print(url)
      on_message = {
        'topic': self.topic,
        'key': key,
        'partitioner': partitioner,
        'producer': producer,
        'exchange': exchange,
        'base': base,
        'quote': quote,
        'delay': self.delay,
        'verbose': self.verbose}
      app['on_message'] = self.OnMessage(**on_message)
      app['on_error'] = self.OnError()
      app['url'] = url
      app = websocket.WebSocketApp(**app)
      future = executor.submit(app.run_forever, **run)
      futures.append(future)
    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)    
    
     
class Feed(Agent):
  def __init__(self, topic, feeds, 
               partitioner=None, producer=None, executor=None, 
               start=0, delay=None, feeds_per_group=1, max_cache=1e5, verbose=False):
    super().__init__()
    self.topic = topic
    self.feeds = feeds
    self.delay = delay
    self.start = start
    self.delay = delay
    self.feeds_per_group = feeds_per_group 
    self.max_cache = int(max_cache)
    self.verbose = verbose
    self.partitioner = {} if partitioner is None else partitioner
    self.producer = {} if producer is None else producer
    if executor is None:
      executor = {'class': 'ThreadPoolExecutor'}
    self.executor = executor
    
  class OnFeed:
    def __init__(self, topic, feeds, partitioner=None, producer=None, 
                 start=None, delay=None, max_cache=1e5, verbose=False):
      self.topic = topic
      self.feeds = feeds
      self.partitioner = partitioner
      self.producer = producer
      self.start = start
      self.delay = delay
      self.verbose = verbose
      self._producer = None  # Lazy init
      self.cache = set()
      self.max_cache = int(max_cache)
  
    def __call__(self):
      if self._producer is None:  # Lazy init
        partitioner = deepcopy(self.partitioner)
        producer = deepcopy(self.producer)
        partitioner_class = partitioner.pop('class')
        partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
        partitioner.update()
        producer['partitioner'] = partitioner
        producer['key_serializer'] = partitioner.serialize_key
        producer['value_serializer'] = partitioner.serialize_value
        self._producer = KafkaProducer(**producer)
      while True:
        if len(self.cache) > self.max_cache:
          self.cache = set()
        t0 = time.time()
        for feed in self.feeds:
          try:
            r = feedparser.parse(feed)
          except Exception:
            d, s = [], None
          else:
            d, s = r.get('entries', []), r.get('status', None)
          if self.verbose > 1:
            print(f'{datetime.utcnow()} {self.topic} {feed}: {s}')
          for e in d:
            i = e.get('id', None)
            if i not in self.cache:
              self.cache.add(i)
              t = e.get('published_parsed', None)
              if t is not None:
                t = datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
                if self.start is not None:
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
              message = {'id': i,
                         'title': e.get('title', None),
                         'links': links,
                         'timestamp': int(t.timestamp()*1e9),  # s to ns
                         'authors': authors,
                         'tags': tags,
                         'summary': e.get('summary', None)}
              if self.verbose:
                print(datetime.utcnow(), self.topic, message['title'], t)
              if self.verbose > 2:
                pprint(message)
              self._producer.send(self.topic, value=message)
          dt = time.time() - t0
        if self.delay is not None:
          wt = max(0, self.delay - dt)
          if verbose:
            print(f'dt: {dt:.3f}, wt: {wt:.3f}')
          time.sleep(wt)
      
  def __call__(self):
    feeds = [x for x, y in self.feeds.items() if y]
    print(f'Number of feeds: {len(feeds)}')
    groups = [feeds[i:i+self.feeds_per_group] 
              for i in range(0, len(feeds), self.feeds_per_group)]
    print(f'Number of groups of feeds: {len(groups)} {[len(x) for x in groups]}')
    executor = deepcopy(self.executor)
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    futures = []
    for g in groups:
      on_feed = {
        'topic': self.topic,
        'feeds': g,
        'partitioner': deepcopy(self.partitioner),
        'producer': deepcopy(self.producer),
        'start': self.start,
        'delay': self.delay,
        'max_cache': self.max_cache,
        'verbose': self.verbose}
      on_feed = self.OnFeed(**on_feed)
      future = executor.submit(on_feed)
      futures.append(future)
    # print(futures)
    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
      