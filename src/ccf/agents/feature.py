# import asyncio
# from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from collections import deque
from copy import deepcopy
import time
import random
from pprint import pprint

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ccf.agents.base import Agent
from ccf.utils import expand_columns


class Delta(Agent):      
  def __init__(self, quant, consumers, producers, verbose=False,
               feature=None, minsize=5, maxsize=5, delay=0, kind='lograt'):
    super().__init__(consumers, producers, verbose)
    self.quant = quant
    if feature is None:
      feature = f'-'.join([self.__class__.__name__, 'default', kind])
    self.feature = feature
    self.minsize = minsize
    self.maxsize = maxsize
    self.delay = 0
    self.kind = kind
    self.data = deque()
    
  def __call__(self):
    if len(self.consumers) > 1:  # TODO implement with aiostream, asyncio or concurrent.futures
      raise NotImplementedError('Many consumers')
    else:
      consumer = self.consumers[list(self.consumers.keys())[0]]
    if len(self.producers) > 1:  # TODO implement with aiostream, asyncio or concurrent.futures
      raise NotImplementedError('Many producers')
    else:
      name = list(self.producers.keys())[0]
      producer = self.producers[name]
      producer_topic_keys = self.producers_topic_keys[name]
    aggregate_kwargs = {'func': {'.*': 'last', '.*_q.*': 'sum', '.*_p.*': 'mean'}}
    last_datetime = None
    for message in consumer:
      t = time.time()
      print(message.key, message.topic)
      topic = message.topic
      value = message.value
      if topic == 'trade':
        if value['t_s']:  # taker sell
          value['t_p_s'] = value['t_p']
          value['t_q_s'] = value['t_q']
          value['t_v_s'] = value['t_p']*value['t_q']
          value['t_p_b'] = None
          value['t_q_b'] = None
          value['t_v_b'] = None
        else:  # taker buy
          value['t_p_s'] = None
          value['t_q_s'] = None
          value['t_v_s'] = None
          value['t_p_b'] = value['t_p']
          value['t_q_b'] = value['t_q']
          value['t_v_b'] = value['t_p']*value['t_q']
        value.pop('t_s')
        value.pop('t_p')
        value.pop('t_q')
      self.data.append(value)
      df = pd.DataFrame(self.data)
      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
      df = df.set_index('timestamp')
      ak = deepcopy(aggregate_kwargs)
      ak['func'] = {kk: v for k, v in ak['func'].items() for kk in expand_columns(df.columns, [k])}
      df = df.resample(pd.Timedelta(self.quant, unit='ns')).aggregate(**ak)
      df = df.interpolate('pad')
      if len(df) < self.minsize:
        continue
      # print(df[expand_columns(df.columns, ['t_.*'])])
      dfs = qv(df)
      df2 = pd.concat([df] + dfs, axis=1)
      dfs1 = delta(df2, kind=self.kind, shift=1)
      dfs2 = delta(df2, kind=self.kind, column='m_p', columns=['a_p_.*', 'b_p_.*', 't_p_.*'], shift=0)
      dfs3 = delta(df2, kind=self.kind, column='a_q', columns=['^b_q$'], shift=0)
      dfs4 = delta(df2, kind=self.kind, column='a_q', columns=['a_q_.*', 't_q_.*'], shift=0)
      dfs5 = delta(df2, kind=self.kind, column='b_q', columns=['b_q_.*', 't_q_.*'], shift=0)
      dfs6 = delta(df2, kind=self.kind, column='a_v', columns=['^b_v$'], shift=0)
      dfs7 = delta(df2, kind=self.kind, column='a_v', columns=['a_v_.*', 't_v_.*'], shift=0)
      dfs8 = delta(df2, kind=self.kind, column='b_v', columns=['b_v_.*', 't_v_.*'], shift=0)
      df3 = pd.concat(dfs1 + dfs2 + dfs3 + dfs4 + dfs5 + dfs6 + dfs7 + dfs8, axis=1)
      df3['m_p'] = df['m_p']
      df = df3.replace([np.inf, -np.inf], np.nan).replace({np.nan: None})
      last_row = df.iloc[-1]
      new_last_datetime = last_row.name
      # pprint(sorted(list(df3.columns)))
      # print(df[expand_columns(df.columns, ['lograt_0-t_.*'])])
      if last_datetime is None or new_last_datetime > last_datetime:
        print(f'New: {last_datetime} -> {new_last_datetime}')
        last_dict = last_row.to_dict()
        last_dict['exchange'] = value['exchange']
        last_dict['base'] = value['base']
        last_dict['quote'] = value['quote']
        last_dict['timestamp'] = int(new_last_datetime.timestamp()*1e9)
        last_dict['feature'] = self.feature
        last_dict['quant'] = self.quant
        last_datetime = new_last_datetime
        for producer_topic, producer_keys in producer_topic_keys.items():
          print(producer_topic, producer_keys)
          if producer_keys is None:
            producer.send(self.topic, value=last_dict)
          else:
            for producer_key in producer_keys:
              # last_dict = {k: v for k, v in last_dict.items() if v is not None}
              # pprint(last_dict)
              producer.send(producer_topic, key=producer_key, value=last_dict)
      if self.maxsize is not None and len(df) >= self.maxsize:
        self.data.popleft()
      dt = time.time() - t
      wt = max(0, self.delay - dt)
      print(f'dt: {dt:.3f}, wt: {wt:.3f}, rows: {len(df)}, cols: {len(df.columns)}, nans: {df3.iloc[-1].isnull().sum()}')
      time.sleep(wt)
      
  
def delta(df, kind='lograt', column=None, columns=None, shift=0):
  columns = [x for x in df.columns if is_numeric_dtype(df[x])] if columns is None else columns
  columns = expand_columns(df.columns, columns)
  if len(columns) == 0:
    return []
  a = df[columns]
  prefix = f'{kind}_{str(shift)}'
  if column is not None:
    b = df[column].shift(shift) if shift != 0 else df[column]
    new_columns = {x: '-'.join([prefix, x, column]) for x in columns}
  else:
    b = df[columns].shift(shift) if shift != 0 else df[columns]
    new_columns = {x: '-'.join([prefix, x, x]) for x in columns}
  if kind == 'rel':
    new_df = a.div(b, axis=0) - 1
  elif kind == 'rat':
    new_df = a.div(b, axis=0)
  elif kind == 'lograt':
    new_df = np.log(a.div(b, axis=0))
  new_df = new_df.rename(columns=new_columns)
  return [new_df]


def qv(df):
  # Columns
  a_q_cs = [x for x in df if x.startswith('a_q_')]
  b_q_cs = [x for x in df if x.startswith('b_q_')]
  a_p_cs = [x for x in df if x.startswith('a_p_')]
  b_p_cs = [x for x in df if x.startswith('b_p_')]
  a_v_cs = ['_'.join(['a', 'v', x.split('_')[2]]) for x in a_q_cs]
  b_v_cs = ['_'.join(['b', 'v', x.split('_')[2]]) for x in b_q_cs]
  dfs = []
  # Volumes
  a_vs = []
  for a_q_c, a_p_c, a_v_c in zip(a_q_cs, a_p_cs, a_v_cs):
    df_c = (df[a_q_c]*df[a_p_c]).rename(a_v_c)
    a_vs.append(df_c)
  a_vs = pd.concat(a_vs, axis=1)
  dfs.append(a_vs)
  b_vs = []
  for b_q_c, b_p_c, b_v_c in zip(b_q_cs, b_p_cs, b_v_cs):
    df_c = (df[b_q_c]*df[b_p_c]).rename(b_v_c)
    b_vs.append(df_c)
  b_vs = pd.concat(b_vs, axis=1)
  dfs.append(b_vs)
  # Sums
  a_q = df[a_q_cs].sum(axis=1).rename('a_q')
  b_q = df[b_q_cs].sum(axis=1).rename('b_q')
  a_v = a_vs.sum(axis=1).rename('a_v')
  b_v = b_vs.sum(axis=1).rename('b_v')
  dfs.append(a_q)
  dfs.append(b_q)
  dfs.append(a_v)
  dfs.append(b_v)
  return dfs


def talib(df, function='SMA', function_kwargs=None):
  function_kwargs = {} if function_kwargs is None else function_kwargs
  f = getattr(ta, function)
  ohlc = ['open', 'high', 'low', 'close']
  tokens = [function] + [str(v) for k, v in function_kwargs.items() if k not in ohlc]
  prefix = '_'.join(tokens)
  dfs = []
  if 'open' in function_kwargs or 'high' in function_kwargs or 'low' in function_kwargs:
    cs = []
    for c in ohlc:
      if c in function_kwargs:
        function_kwargs[c] = df[c]
        cs.append(c)
    try:
      dfc = f(**function_kwargs)
    except Exception as e:
      print(e)
      dfc = df[cc].replace('?', np.NaN)
    dfc.name = '-'.join([prefix] + cs)
    dfs.append(dfc)
  elif 'close' in function_kwargs:
    c = function_kwargs.pop('close')
    for cc in expand_columns(df.columns, [c]):
      if not is_numeric_dtype(df[cc]):
        continue
      try:
        dfc = f(df[cc], **function_kwargs)
        
      except Exception as e:
        print(e)
        dfc = df[cc].replace('?', np.NaN)
      dfc.name = '-'.join([prefix, cc])
      dfs.append(dfc)
  return dfs