"""
See Also:
  https://websocket-client.readthedocs.io/en/latest
  https://binance-docs.github.io/apidocs/spot/en/#how-to-manage-a-local-order-book-correctly
"""
import sys
import json
from datetime import datetime, timezone

import yaml
import websocket
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from sqlalchemy import create_engine
import pandas as pd


def get_data(markets, streams, verbose=False,
             executor_kwargs=None, app_kwargs=None, 
             run_kwargs=None, engine_kwargs=None):
  engine_kwargs = {'url': 'sqlite:///data.db'} if engine_kwargs is None else engine_kwargs
  executor_kwargs = {} if executor_kwargs is None else executor_kwargs
  app_kwargs = {} if app_kwargs is None else app_kwargs
  run_kwargs = {} if run_kwargs is None else run_kwargs
  if verbose:
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
  for future in as_completed(futures):
    try:
        data = future.result()
    except Exception as exc:
        print('%r generated an exception: %s' % (url, exc))
    else:
        print('%r page is %d bytes' % (url, len(data)))


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
      d['time'] = datetime.utcnow()
      data = json.loads(message)
      d['lastUpdateId'] = data['lastUpdateId']
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
      d['t_i'] = data['t']
      d['t_t'] = 'sell' if data['m'] else 'buy'
    else:
      raise NotImplementedError(s)
    df = pd.DataFrame.from_records([d], index='time')
    if not self.is_sqlite:
      name = ws.url.split('/')[-1]
    else:
      name = 'data'
    df.to_sql(name, self.con, if_exists='append')
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'get_data.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  get_data(**kwargs)