"""
See Also:
  https://websocket-client.readthedocs.io/en/latest
  https://binance-docs.github.io/apidocs/spot/en/#how-to-manage-a-local-order-book-correctly
"""
import sys
import json
from datetime import datetime, timedelta, timezone
import time

import yaml
import websocket
from concurrent.futures import ThreadPoolExecutor, wait
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import feedparser


def get_data(markets=None, streams=None, feeds=None, 
             engine_kwargs=None, 
             market_kwargs=None, feeds_kwargs=None,
             executor_kwargs=None, app_kwargs=None, 
             run_kwargs=None):
  markets = [] if markets is None else markets
  streams = [] if streams is None else streams
  feeds = [] if feeds is None else feeds
  engine_kwargs = {'url': 'sqlite:///data.db'} if engine_kwargs is None else engine_kwargs
  executor_kwargs = {} if executor_kwargs is None else executor_kwargs
  feeds_kwargs = {} if feeds_kwargs is None else feeds_kwargs
  market_kwargs = {} if market_kwargs is None else market_kwargs
  app_kwargs = {} if app_kwargs is None else app_kwargs
  run_kwargs = {} if run_kwargs is None else run_kwargs
  if market_kwargs.pop('verbose', False):
    websocket.enableTrace(True)
  executor = ThreadPoolExecutor(**executor_kwargs)
  futures = []
  for m in markets:
    for s in streams:
      if 'depth' in s:
        app_kwargs['url'] = f'wss://stream.binance.com:9443/ws/{m}@{s}'
      elif 'trade' in s:
        app_kwargs['url'] = f'wss://stream.binance.com:9443/ws/{m}@{s}'
      else:
        raise NotImplementedError(s)
      app_kwargs['on_message'] = OnMessage(engine_kwargs=engine_kwargs, 
                                           app_kwargs=app_kwargs)
      app = websocket.WebSocketApp(**app_kwargs)
      futures.append(executor.submit(app.run_forever, **run_kwargs))
  feeds = [k for k, v in feeds.items() if v]
  feeds = np.array_split(feeds, feeds_kwargs.pop('split', 1))
  feeds_kwargs['engine_kwargs'] = engine_kwargs
  for f in feeds:
    feeds_kwargs['feeds'] = f
    on_feed = OnFeed(**feeds_kwargs)
    futures.append(executor.submit(on_feed))
  wait(futures)


class OnMessage:
  def __init__(self, engine_kwargs, app_kwargs):
    super().__init__()
    url = engine_kwargs['url']
    if not 'sqlite' in url:
      self.con = create_engine(**engine_kwargs)
      self.is_sqlite = False
    else:
      tag = app_kwargs['url'].split('/')[-1]
      tokens = url.split('/')
      tokens[-1] = f'{tag}.db'
      engine_kwargs['url'] = '/'.join(tokens)
      self.con = create_engine(**engine_kwargs)
      self.is_sqlite = True

  def __call__(self, ws, message):
    d = {}
    if 'depth' in ws.url:
      d['time'] = datetime.now(timezone.utc)
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
    elif 'trade' in ws.url:
      data = json.loads(message)
      d['time'] = datetime.fromtimestamp(float(data['T']) / 1000.0, tz=timezone.utc)
      d['t_p'] = float(data['p'])
      d['t_q'] = float(data['q'])
      d['t_t'] = data['m']  # True - Sell, False - Buy
    else:
      raise NotImplementedError(s)
    df = pd.DataFrame.from_records([d], index='time')
    if not self.is_sqlite:
      name = ws.url.split('/')[-1]
    else:
      name = 'data'
    df.to_sql(name, self.con, if_exists='append')

    
class OnFeed:
  def __init__(self, engine_kwargs, feeds, delay=3, before=None, verbose=False):
    self.feeds = feeds
    self.delay = delay
    self.before = before
    self.verbose = verbose
    url = engine_kwargs['url']
    if not 'sqlite' in url:
      self.con = create_engine(**engine_kwargs)
      self.is_sqlite = False
    else:
      tokens = url.split('/')
      tokens[-1] = f'news.db'
      engine_kwargs['url'] = '/'.join(tokens)
      self.con = create_engine(**engine_kwargs)
      self.is_sqlite = True
    self.cache = set()
    
  def __call__(self):
    while True:
      t0 = time.time()
      for f in self.feeds:
        try:
          r = feedparser.parse(f)
        except Exception:
          d, s = [], None
        else:
          d, s = r.get('entries', []), r.get('status', None)
        dd = []
        for e in d:
          i = e.get('id', None)
          if i not in self.cache:
            self.cache.add(i)
            t = e.get('published_parsed', None)
            if t is not None:
              t = datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
              if self.before is not None:
                min_t = datetime.now(timezone.utc) - timedelta(seconds=self.before)
                if t < min_t:
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
                  'links': links,
                  'title': e.get('title', None),
                  'time': t,
                  'authors': authors,
                  'tags': tags,
                  'summary': e.get('summary', None)}
            dd.append(ee)
        if self.verbose:
          print(f'feed: {f}, status: {s}, news: {len(dd)}')
        if len(dd) > 0:
          df = pd.DataFrame.from_records(dd, index='id')
          if not self.is_sqlite:
            name = 'news'
          else:
            name = 'data'
          df.to_sql(name, self.con, if_exists='append')
      print(self.delay - ((time.time() - t0) % self.delay))
      time.sleep(self.delay - ((time.time() - t0) % self.delay))

  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'get_data.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  get_data(**kwargs)