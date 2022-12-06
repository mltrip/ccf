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

import numpy as np
import pandas as pd
import feedparser
from kafka import KafkaProducer, KafkaConsumer

from ccf.agents.base import Agent


# class OnLob(Agent):
#   def __init__(self, in_topic_keys, depth, out_topic_keys, partitioners, consumers, producers, 
#                depth, delay=0, verbose=False):
#     super().__init__(in_topic_keys, depth, out_topic_keys, partitioners, consumers, producers)
#     self.depth = depth
#     self.delay = delay
#     self.verbose = verbose
#     if isinstance(self.consumers, dict):
#       for topic, keys in self.in_topic_keys.items():
#         producer_kwargs['partitioner'] = partitioner
#         producer_kwargs['key_serializer'] = lambda x: x.encode('ascii') if isinstance(x, str) else x
#         producer_kwargs['value_serializer'] = lambda x: json.dumps(x).encode('ascii')
#     # exchange, base, quote = key.split('-')
#     # self.exchange = exchange
#     # self.base = base
#     # self.quote = quote
#     # if exchange == 'binance':
#     #   self.url = f'wss://stream.binance.com:9443/ws/{base}{quote}@depth{depth}@{int(delay*1000)}ms'
#     # else:
#     #   raise NotImplementedError(exchange)
#     # producer_kwargs['partitioner'] = partitioner
#     # producer_kwargs['key_serializer'] = lambda x: x.encode('ascii') if isinstance(x, str) else x
#     # producer_kwargs['value_serializer'] = lambda x: json.dumps(x).encode('ascii')
#     # self.producer = KafkaProducer(**producer_kwargs)
#     # if len(self.producer.partitions_for(self.topic)) != len(KEY2PART):
#     #   client = KafkaAdminClient(**client_kwargs)
#     #   result = client.create_partitions({self.topic: NewPartitions(len(KEY2PART))})
#     #   print(result)
    
#   def __call__(self, ws, message):
#     d = {'timestamp': time.time_ns(),
#          'exchange': self.exchange,
#          'base': self.base,
#          'quote': self.quote}
#     if self.exchange == 'binance':
#       data = json.loads(message)
#       for i, a in enumerate(data['asks']):
#         if a[1] != '0':
#           d[f'a_p_{i}'] = float(a[0])
#           d[f'a_q_{i}'] = float(a[1])
#         else:
#           d[f'a_p_{i}'] = None
#           d[f'a_q_{i}'] = None
#       for i, b in enumerate(data['bids']):
#         if b[1] != '0':
#           d[f'b_p_{i}'] = float(b[0])
#           d[f'b_q_{i}'] = float(b[1])
#         else:
#           d[f'b_p_{i}'] = None
#           d[f'b_q_{i}'] = None
#     else:
#       raise NotImplementedError(self.exchange)
#     # mid price
#     if d.get('a_p_0', None) is not None and d.get('b_p_0', None) is not None:
#       d['m_p'] = 0.5*(d['a_p_0'] + d['b_p_0']) 
#     else:
#       d['m_p'] = None
#     if self.verbose:
#       print(self.topic, self.key, d)
#     self.producer.send(self.topic, key=self.key, value=d)

    
# class OnTrade:
#   def __init__(self, topic, key, producer_kwargs, client_kwargs, delay=0, verbose=False):
#     super().__init__()
#     self.topic = topic
#     self.key = key
#     self.delay = delay
#     self.verbose = verbose
#     exchange, base, quote = key.split('-')
#     self.exchange = exchange
#     self.base = base
#     self.quote = quote
#     if exchange == 'binance':
#       self.url = f'wss://stream.binance.com:9443/ws/{base}{quote}@trade'
#     else:
#       raise NotImplementedError(exchange)
#     producer_kwargs['partitioner'] = partitioner
#     producer_kwargs['key_serializer'] = lambda x: x.encode('ascii') if isinstance(x, str) else x
#     producer_kwargs['value_serializer'] = lambda x: json.dumps(x).encode('ascii')
#     self.producer = KafkaProducer(**producer_kwargs)
#     if len(self.producer.partitions_for(self.topic)) != len(KEY2PART):
#       client = KafkaAdminClient(**client_kwargs)
#       result = client.create_partitions({self.topic: NewPartitions(len(KEY2PART))})
#       print(result)
#     self.t = time.time()
    
#   def __call__(self, wsapp, message):
#     d = {'exchange': self.exchange,
#          'base': self.base,
#          'quote': self.quote}
#     if self.exchange == 'binance':
#       data = json.loads(message)
#       d['timestamp'] = int(data['T']*1e6)  # ms -> ns
#       d['t_p'] = float(data['p'])
#       d['t_q'] = float(data['q'])
#       # trade side: 0 - maker_sell/taker_buy or 1 - maker_buy/taker_sell
#       d['t_s'] = int(data['m']) 
#     else:
#       raise NotImplementedError(self.exchange)
#     dt = time.time() - self.t
#     if dt > self.delay:
#       if self.verbose:
#         print(self.topic, self.key, d)
#       self.producer.send(self.topic, key=self.key, value=d)
#       self.t = time.time()

      
# class OnFeed:
#   def __init__(self, topic, feeds, producer_kwargs, client_kwargs, delay=0, start=0, verbose=False):
#     super().__init__()
#     self.topic = topic
#     self.feeds = feeds
#     self.delay = delay
#     self.start = start
#     self.verbose = verbose
#     producer_kwargs['value_serializer'] = lambda x: json.dumps(x).encode('ascii')
#     self.producer = KafkaProducer(**producer_kwargs)
#     self.cache = set()
#     self.t = time.time()
    
#   def __call__(self):
#     while True:
#       dt = time.time() - self.t
#       if dt <= self.delay:
#         continue
#       else:
#         self.t = time.time()
#       for f, flag in self.feeds.items():
#         if not flag:
#           continue
#         try:
#           r = feedparser.parse(f)
#         except Exception:
#           d, s = [], None
#         else:
#           d, s = r.get('entries', []), r.get('status', None)
#         for e in d:
#           i = e.get('id', None)
#           if i not in self.cache:
#             self.cache.add(i)
#             t = e.get('published_parsed', None)
#             if t is not None:
#               t = datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
#               start_t = datetime.now(timezone.utc) + timedelta(seconds=self.start)
#               if t < start_t:
#                 continue
#             else:
#               continue
#             authors = e.get('authors', None)
#             if authors is not None:
#               authors = '|'.join(x.get('name', '') for x in authors)
#             tags = e.get('tags', None)
#             if tags is not None:
#               tags = '|'.join(x.get('term', '') for x in tags)
#             links = e.get('links', None)
#             if links is not None:
#               links = '|'.join(x.get('href', '') for x in links)
#             ee = {'id': i,
#                   'title': e.get('title', None),
#                   'links': links,
#                   'timestamp': int(t.timestamp()*1e9),  # s to ns
#                   'authors': authors,
#                   'tags': tags,
#                   'summary': e.get('summary', None)}
#             if self.verbose:
#               print(self.topic, ee)
#             self.producer.send(self.topic, value=ee)