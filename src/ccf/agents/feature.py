# import asyncio
# from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from collections import deque
from copy import deepcopy
import time
import random
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ccf.agents.base import Kafka
from ccf.utils import expand_columns


class Delta(Kafka):
  def __init__(self, quant, consumers, producers, verbose=False, replace_nan=None,
               feature=None, minsize=5, maxsize=5, delay=0, kind='rat',
               ema_keys=None, ema_alphas=None, vwaps=None, deltas=None, qv=None,
               resample=None, aggregate=None, interpolate=None):
    super().__init__(consumers, producers, verbose)
    self.quant = quant
    if feature is None:
      feature = f'-'.join([self.__class__.__name__, 'default', kind])
    self.feature = feature
    self.minsize = minsize
    self.maxsize = maxsize
    self.delay = delay
    self.kind = kind
    self.data = deque()
    self.replace_nan = replace_nan
    self.ema_keys = [] if ema_keys is None else ema_keys
    self.ema_alphas = [] if ema_alphas is None else ema_alphas
    self.vwaps = [] if vwaps is None else vwaps
    self.deltas = [] if deltas is None else deltas
    self.qv = {} if qv is None else qv
    self.resample = {} if resample is None else resample
    self.aggregate = {'func': {'.*': 'last'}} if aggregate is None else aggregate
    self.interpolate = {'method': 'pad'} if interpolate is None else interpolate
    
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
    last_datetime = None
    topic_old_values = {}
    for message in consumer:
      t = time.time()
      if self.verbose:
        print(message.key, message.topic)
      topic = message.topic
      value = message.value
      if topic in topic_old_values:
        old_values = topic_old_values[topic][-1]
      else:
        old_values = None
      if topic == 'trade':
        value = unwrap_trade(value)
      elif topic == 'lob':
        value.update(evaluate_qv(values=value, **self.qv))
        for v in self.vwaps:
          vwap_ask = vwap(value, side='ask', **v)
          vwap_bid = vwap(value, side='bid', **v)
          vwap_mid = {}
          for (ask_key, ask_value), (bid_key, bid_value) in zip(vwap_ask.items(), vwap_bid.items()):
            mid_key = ask_key.replace('ask', 'mid')
            mid_value = 0.5*(ask_value + bid_value)
            vwap_mid[mid_key] = mid_value
          value.update(vwap_ask)
          value.update(vwap_bid)
          value.update(vwap_mid)
      expanded_ema_keys = expand_columns(value.keys(), self.ema_keys)
      for ema_key in expanded_ema_keys:
        for ema_alpha in self.ema_alphas:
          if ema_key in value:
            value.update(ema(value, ema_key, ema_alpha, old_values))
      topic_old_values.setdefault(topic, deque()).append(value)
      if self.maxsize is not None and len(topic_old_values[topic]) > self.maxsize:
        topic_old_values[topic].popleft()
      # pprint(value)
      self.data.append(value)
      # Create DataFrame
      df = pd.DataFrame(self.data)
      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
      df = df.set_index('timestamp')
      if self.verbose:
        print(df)
      # Resample, aggregate, interpolate DataFrame
      resample = deepcopy(self.resample)
      resample['rule'] = pd.Timedelta(self.quant, unit='ns')
      aggregate = deepcopy(self.aggregate)
      aggregate['func'] = {kk: v for k, v in aggregate['func'].items() 
                           for kk in expand_columns(df.columns, [k])}
      interpolate = deepcopy(self.interpolate)
      df = df.resample(**resample).aggregate(**aggregate).interpolate(**interpolate)
      if self.verbose:
        print(df)
      if len(df) < self.minsize:
        continue
      # Evaluate deltas of DataFrame
      dfs = []
      for d in self.deltas:
        dfs.append(delta(df, kind=self.kind, **d))
      df2 = pd.concat(dfs, axis=1)
      df2['m_p'] = df['m_p']
      df = df2.replace([np.inf, -np.inf], np.nan).replace({np.nan: self.replace_nan})
      last_row = df.iloc[-1]
      new_last_datetime = last_row.name
      if self.verbose:
        print(df)
      if last_datetime is None or new_last_datetime > last_datetime:
        print(f'New feature: {last_datetime} -> {new_last_datetime}')
        last_dict = last_row.to_dict()
        last_dict['exchange'] = value['exchange']
        last_dict['base'] = value['base']
        last_dict['quote'] = value['quote']
        last_dict['timestamp'] = int(new_last_datetime.timestamp()*1e9)
        last_dict['feature'] = self.feature
        last_dict['quant'] = self.quant
        last_datetime = new_last_datetime
        if self.verbose:
          pprint(last_dict)
        for producer_topic, producer_keys in producer_topic_keys.items():
          print(producer_topic, producer_keys)
          if producer_keys is None:
            producer.send(self.topic, value=last_dict)
          else:
            for producer_key in producer_keys:
              producer.send(producer_topic, key=producer_key, value=last_dict)
      if self.maxsize is not None and len(df) >= self.maxsize:
        self.data.popleft()
      dt = time.time() - t
      wt = max(0, self.delay - dt)
      print(f'{datetime.utcnow()}, dt: {dt:.3f}, wt: {wt:.3f}, rows: {len(df)}, cols: {len(df.columns)}')
      time.sleep(wt)
      
      
def delta(df, kind='lograt', column=None, columns=None, shift=0):
  columns_ = [x for x in df.columns if is_numeric_dtype(df[x])] if columns is None else columns
  columns = expand_columns(df.columns, columns_)
  if len(columns) == 0:
    raise ValueError(f'No columns found with patterns: {columns_}')
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
  return new_df


def unwrap_trade(values):
  if values['t_s']:  # taker sell
    values['t_p_s'] = values['t_p']
    values['t_q_s'] = values['t_q']
    values['t_v_s'] = values['t_p']*values['t_q']
    values['t_p_b'] = None
    values['t_q_b'] = None
    values['t_v_b'] = None
  else:  # taker buy
    values['t_p_s'] = None
    values['t_q_s'] = None
    values['t_v_s'] = None
    values['t_p_b'] = values['t_p']
    values['t_q_b'] = values['t_q']
    values['t_v_b'] = values['t_p']*values['t_q']
  values.pop('t_s')
  values.pop('t_p')
  values.pop('t_q')
  return values


def evaluate_qv(values, maxdepth=None):
  new_values = {}
  if maxdepth is None:
    a_q_keys = [x for x in values if x.startswith('a_q_')]
    b_q_keys = [x for x in values if x.startswith('b_q_')]
    a_p_keys = [x for x in values if x.startswith('a_p_')]
    b_p_keys = [x for x in values if x.startswith('b_p_')]
  else:
    a_q_keys = [x for x in values if x.startswith('a_q_') and int(x.split('_')[2]) < maxdepth]
    b_q_keys = [x for x in values if x.startswith('b_q_') and int(x.split('_')[2]) < maxdepth]
    a_p_keys = [x for x in values if x.startswith('a_p_') and int(x.split('_')[2]) < maxdepth]
    b_p_keys = [x for x in values if x.startswith('b_p_') and int(x.split('_')[2]) < maxdepth]
  a_v_keys = ['_'.join(['a', 'v', x.split('_')[2]]) for x in a_q_keys]
  b_v_keys = ['_'.join(['b', 'v', x.split('_')[2]]) for x in b_q_keys]
  new_values['a_q'] = 0
  new_values['b_q'] = 0
  new_values['a_v'] = 0
  new_values['b_v'] = 0
  for a_q_key, a_p_key, a_v_key in zip(a_q_keys, a_p_keys, a_v_keys):
    p = values[a_p_key] 
    q = values[a_q_key]
    v = p*q
    new_values[a_v_key] = v
    new_values['a_q'] += q
    new_values['a_v'] += v
  for b_q_key, b_p_key, b_v_key in zip(b_q_keys, b_p_keys, b_v_keys):
    p = values[b_p_key]
    q = values[b_q_key]
    v = p*q
    new_values[b_v_key] = v
    new_values['b_q'] += q
    new_values['b_v'] += v
  return new_values


def vwap(values, side, currency='base', quantity=None, maxdepth=None):
  assert side in ['ask', 'bid']
  assert currency in ['base', 'quote']
  p_key = 'a_p' if side == 'ask' else 'b_p'
  q_key = 'a_q' if side == 'ask' else 'b_q'
  quantity_str = 'max' if quantity is None else np.format_float_positional(quantity)
  maxdepth_str = str(maxdepth) if maxdepth is not None else 'max'
  vwap_key = '_'.join(['vwap', side, currency, quantity_str, maxdepth_str])
  ps, qs, i, s = [], [], 0, 0  # prices, quantities, index, sum
  while True:
    if maxdepth is not None and i == maxdepth:
      break
    p = values.get(f'{p_key}_{i}', None)
    q = values.get(f'{q_key}_{i}', None)
    if p is None or q is None:
      break
    if currency == 'quote': 
      q *= p
    ps.append(p)
    s += q
    if quantity is None or s <= quantity:
      qs.append(q)
    else:
      dq = s - quantity
      qs.append(q - dq)
      break
    i += 1
  vwap_value = np.average(ps, weights=qs)
  return {vwap_key: vwap_value}


def ema(new_values, key, alpha, old_values=None):
  alpha_str = np.format_float_positional(alpha)
  ema_key = '_'.join(['ema', alpha_str]) + '-' + key
  new_value = new_values[key]
  old_ema = old_values[ema_key] if old_values is not None else new_value
  new_ema = old_ema + alpha*(new_value - old_ema)
  return {ema_key: new_ema}


# def talib(df, function='SMA', function_kwargs=None):
#   function_kwargs = {} if function_kwargs is None else function_kwargs
#   f = getattr(ta, function)
#   ohlc = ['open', 'high', 'low', 'close']
#   tokens = [function] + [str(v) for k, v in function_kwargs.items() if k not in ohlc]
#   prefix = '_'.join(tokens)
#   dfs = []
#   if 'open' in function_kwargs or 'high' in function_kwargs or 'low' in function_kwargs:
#     cs = []
#     for c in ohlc:
#       if c in function_kwargs:
#         function_kwargs[c] = df[c]
#         cs.append(c)
#     try:
#       dfc = f(**function_kwargs)
#     except Exception as e:
#       print(e)
#       dfc = df[cc].replace('?', np.NaN)
#     dfc.name = '-'.join([prefix] + cs)
#     dfs.append(dfc)
#   elif 'close' in function_kwargs:
#     c = function_kwargs.pop('close')
#     for cc in expand_columns(df.columns, [c]):
#       if not is_numeric_dtype(df[cc]):
#         continue
#       try:
#         dfc = f(df[cc], **function_kwargs)
        
#       except Exception as e:
#         print(e)
#         dfc = df[cc].replace('?', np.NaN)
#       dfc.name = '-'.join([prefix, cc])
#       dfs.append(dfc)
#   return dfs