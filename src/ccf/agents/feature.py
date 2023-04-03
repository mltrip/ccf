# import asyncio
# from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from collections import deque
from copy import deepcopy
import time
import random
from pprint import pprint
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ccf.agents.base import Kafka
from ccf.agents.influxdb import InfluxDB
from ccf.utils import expand_columns, initialize_time


class DeltaKafka(Kafka):
  def __init__(self, consumers, producers, 
               feature=None, kind='rat', replace_nan=1.0,
               quant=None, delay=0.0, watermark=None,
               depth=None, deltas=None, emas=None, 
               vwaps=None, verbose=False):
    super().__init__(consumers, producers, verbose)
    if feature is None:
      feature = f'-'.join([self.__class__.__name__, 'default', kind])
    self.feature = feature
    self.kind = kind
    self.quant = quant
    self.delay = delay
    self.watermark = watermark if watermark is not None else quant
    self.depth = depth
    self.replace_nan = replace_nan
    self.emas = [] if emas is None else emas
    self.vwaps = [] if vwaps is None else vwaps
    self.deltas = [] if deltas is None else deltas
    
  def __call__(self):
    # Initialize consumers and producers
    if len(self.consumers) > 1:  # TODO implement with aiostream, asyncio or concurrent.futures
      raise NotImplementedError('Many consumers')
    else:
      consumer_name = list(self.consumers.keys())[0]
      consumer = self.consumers[consumer_name]
      consumer_topic_keys = self.consumers_topic_keys[consumer_name]
    if len(self.producers) > 1:  # TODO implement with aiostream, asyncio or concurrent.futures
      raise NotImplementedError('Many producers')
    else:
      producer_name = list(self.producers.keys())[0]
      producer = self.producers[producer_name]
      producer_topic_keys = self.producers_topic_keys[producer_name]
    # Update streams
    # Update buffers
    cur_t = (time.time_ns() // self.quant)*self.quant
    start_t = cur_t
    delay_t = cur_t + self.delay
    # stop_t = (stop // quant)*quant
    topic_key_buffers = {}
    topic_key_dataframes = {}
    topic_key_features = {}
    while True:
      t0 = time.time()
      next_t = cur_t + self.quant
      watermark_t = next_t - self.watermark
      print(f'\nnow:       {datetime.utcnow()}')
      print(f'start:     {datetime.fromtimestamp(start_t/10**9, tz=timezone.utc)}')
      print(f'delay:     {datetime.fromtimestamp(delay_t/10**9, tz=timezone.utc)}')
      print(f'watermark: {datetime.fromtimestamp(watermark_t/10**9, tz=timezone.utc)}')
      print(f'current:   {datetime.fromtimestamp(cur_t/10**9, tz=timezone.utc)}')
      print(f'next:      {datetime.fromtimestamp(next_t/10**9, tz=timezone.utc)}')
      # print(f'stop {datetime.fromtimestamp(stop_t/10**9, tz=timezone.utc)}')
      # Append to buffer
      cur_buffer_t = watermark_t
      while cur_buffer_t < next_t:
        message = next(consumer)
        topic, key = message.topic, message.key
        buffer = topic_key_buffers.setdefault(topic, {}).setdefault(key, deque())
        value = message.value
        cur_buffer_t = value['timestamp']
        buffer.append(value)
      # Popleft from buffers (timestamp >= watermark_t)
      for topic, key_buffer in topic_key_buffers.items():
        for key, buffer in key_buffer.items():
          if len(buffer) != 0:
            min_buffer_t = buffer[0]['timestamp']
            while min_buffer_t < watermark_t:
              if len(buffer) != 0:
                min_buffer_t = buffer.popleft()['timestamp']
              else:
                break
      cur_t = next_t
      # Extract features
      evaluate_features_delta_ema_qv_vwap(topic_key_buffers, topic_key_dataframes, topic_key_features,
                                          self.quant, self.depth, next_t, deltas=self.deltas,
                                          emas=self.emas, vwaps=self.vwaps, kind=self.kind,
                                          replace_nan=self.replace_nan, verbose=self.verbose)
      # Write features
      if next_t >= delay_t:
        for key, features in topic_key_features.get('feature', {}).items():
          df = features['delta']
          exchange, base, quote = key.split('-')
          df['exchange'] = exchange
          df['base'] = base
          df['quote'] = quote
          df['quant'] = self.quant
          df['feature'] = self.feature
          data_frame_tag_columns = ['exchange', 'base', 'quote', 'quant', 'feature']
          if self.verbose:
            print(df)
          if len(df) > 0:        
            last_row = df.iloc[-1]
            last_dict = last_row.to_dict()
            last_dict['timestamp'] = int(last_row.name.timestamp()*1e9)
            for producer_topic, producer_keys in producer_topic_keys.items():
              if producer_keys is None:
                pass
                producer.send(producer_topic, value=last_dict)
                print(f'{key} len: {len(last_dict)}')
              else:
                for producer_key in producer_keys:
                  if key == producer_key:
                    producer.send(producer_topic, key=producer_key, value=last_dict)
                    print(f'{key}: {len(last_dict)}')
      dt = time.time() - t0
      print(f'dt: {dt}')
      

class DeltaInfluxDB(InfluxDB):
  def __init__(self, client=None, bucket=None, query_api=None, write_api=None,
               feature=None, kind='rat', replace_nan=1.0, topic_keys=None, 
               quant=None, start=None, stop=None, size=None, watermark=None,
               delay=0.0,
               batch_size=86400e9, depth=None, deltas=None, emas=None, 
               vwaps=None, verbose=False):
    super().__init__(client, bucket, query_api, write_api, verbose)
    if feature is None:
      feature = f'-'.join([self.__class__.__name__, 'default', kind])
    self.feature = feature
    self.kind = kind
    self.topic_keys = {} if topic_keys is None else topic_keys
    self.quant = quant
    self.start = start
    self.stop = stop
    self.size = size
    self.watermark = watermark if watermark is not None else quant
    self.delay = delay
    self.batch_size = batch_size
    self.depth = depth
    self.replace_nan = 1.0
    self.emas = [] if emas is None else emas
    self.vwaps = [] if vwaps is None else vwaps
    self.deltas = [] if deltas is None else deltas
    
  def __call__(self):
    start, stop, size, quant = initialize_time(self.start, self.stop, 
                                               self.size, self.quant)
    client = self.init_client(self.client)
    query_api = self.get_query_api(client, self.query_api)
    write_api = self.get_write_api(client, self.write_api)
    # Update streams
    topic_key_streams = {}
    for topic, keys in self.topic_keys.items():
      for key in keys:
        exchange, base, quote = key.split('-')
        if topic == 'lob':
          stream = self.get_lob_batch_stream(query_api, bucket=self.bucket, 
                                             start=start, stop=stop, 
                                             batch_size=self.batch_size,
                                             exchange=exchange, base=base, quote=quote, 
                                             verbose=self.verbose)
        elif topic == 'trade':
          stream = self.get_trade_batch_stream(query_api, bucket=self.bucket, 
                                               start=start, stop=stop, 
                                               batch_size=self.batch_size,
                                               exchange=exchange, base=base, quote=quote, 
                                               verbose=self.verbose)
        else:
          raise NotImplementedError(topic)
        topic_key_streams.setdefault(topic, {})[key] = stream
    # Update buffers
    cur_t = (start // quant)*quant
    start_t = cur_t
    delay_t = cur_t + self.delay
    stop_t = (stop // quant)*quant
    topic_key_buffers = {}
    topic_key_dataframes = {}
    topic_key_features = {}
    while cur_t < stop_t:
      t0 = time.time()
      next_t = cur_t + quant
      watermark_t = next_t - self.watermark
      print(f'\nnow:       {datetime.utcnow()}')
      print(f'start:     {datetime.fromtimestamp(start_t/10**9, tz=timezone.utc)}')
      print(f'delay:     {datetime.fromtimestamp(delay_t/10**9, tz=timezone.utc)}')
      print(f'watermark: {datetime.fromtimestamp(watermark_t/10**9, tz=timezone.utc)}')
      print(f'current:   {datetime.fromtimestamp(cur_t/10**9, tz=timezone.utc)}')
      print(f'next:      {datetime.fromtimestamp(next_t/10**9, tz=timezone.utc)}')
      print(f'stop:      {datetime.fromtimestamp(stop_t/10**9, tz=timezone.utc)}')
      for topic, key_stream in topic_key_streams.items():
        for key, stream in key_stream.items():
          buffer = topic_key_buffers.setdefault(topic, {}).setdefault(key, [])
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
          topic_key_buffers[topic][key] = buffer
      cur_t = next_t
      # Extract features
      evaluate_features_delta_ema_qv_vwap(topic_key_buffers, topic_key_dataframes, topic_key_features,
                                          quant, self.depth, next_t, deltas=self.deltas,
                                          emas=self.emas, vwaps=self.vwaps, kind=self.kind,
                                          replace_nan=self.replace_nan, verbose=self.verbose)
      # Write features
      if next_t >= delay_t:
        for key, features in topic_key_features.get('feature', {}).items():
          df = features['delta']
          exchange, base, quote = key.split('-')
          df['exchange'] = exchange
          df['base'] = base
          df['quote'] = quote
          df['quant'] = quant
          df['feature'] = self.feature
          data_frame_tag_columns = ['exchange', 'base', 'quote', 'quant', 'feature']
          if self.verbose:
            print(df.tail(-1))
          write_api.write(bucket=self.bucket, 
                          record=df.tail(-1),  # skip first row because delta shift 1 is 0
                          data_frame_measurement_name='feature',
                          data_frame_tag_columns=data_frame_tag_columns)
      dt = time.time() - t0
      print(f'dt: {dt}')     
    # Close
    t00 = time.time()
    write_api.close()
    client.close()

          
def evaluate_features_delta_ema_qv_vwap(
  topic_key_buffers, topic_key_dataframes, topic_key_features, quant, depth, next_t, 
  deltas, emas, vwaps, kind, replace_nan, verbose=False):
  # Create DataFrames (timestamp < next_t)
  for topic, key_buffer in topic_key_buffers.items():
    for key, buffer in key_buffer.items():
      buffer_ = [x for x in buffer if x['timestamp'] < next_t]
      if len(buffer_) == 0:
        continue
      elif topic == 'lob':
        df = pd.DataFrame(buffer_)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df = df.set_index('timestamp')
        df = preprocess_lob(df, depth=depth, quant=quant, verbose=verbose)
      elif topic == 'trade':
        df = pd.DataFrame([unwrap_trade(x) for x in buffer_])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df = df.set_index('timestamp')
        df = preprocess_trade(df, quant=quant, verbose=verbose)
      else:
        raise NotImplementedError(topic)
      topic_key_dataframes.setdefault(topic, {})[key] = df
  # Extract features
  for topic, key_df in topic_key_dataframes.items():
    for key, df in key_df.items():
      features = topic_key_features.setdefault(topic, {}).setdefault(key, {})
      if topic == 'lob':
        qv = evaluate_qv_df(df, depth)
        features['qv'] = qv
        for i, v in enumerate(vwaps):
          vwap = evaluate_vwap_df(df, depth=depth, **v)
          features[f'vwap_{i}'] = vwap
        features['lob'] = df
      elif topic == 'trade':
        features['trade'] = df
        features['trade_2'] = evaluate_trade(df)
      else:
        raise NotImplementedError(topic)
  key_features = {}
  for topic, key_fs in topic_key_features.items():
    if topic != 'feature':
      for key, fs in key_fs.items():
        key_features.setdefault(key, {}).update(fs)
  for key, features in key_features.items():
    fs = topic_key_features.setdefault('feature', {}).setdefault(key, {})
    df = pd.concat([x for x in features.values()], axis=1)
    dfs = [df]
    for i, e in enumerate(emas):
      old_ema = fs.get(f'ema_{i}', None)
      ema_df = evaluate_ema_df(df, old_ema=old_ema, **e)
      fs[f'ema_{i}'] = ema_df
      dfs.append(ema_df)
    df = pd.concat(dfs, axis=1)
    dfs = []
    for d in deltas:
      try:
        dfs.append(evaluate_delta_df(df, kind=kind, **d))
      except ValueError as e:
        print(e)
    delta_df = pd.concat(dfs, axis=1)
    delta_df = delta_df.replace([np.inf, -np.inf], np.nan).replace({np.nan: replace_nan})
    delta_df['m_p'] = df['m_p']
    fs['delta'] = delta_df
    if verbose > 1:
      print([delta_df[x] for x in delta_df])
      print(len(delta_df.columns))
      
      
def preprocess_lob(df, depth=None, quant=None, verbose=False):
  # Check depth
  if depth is not None:
    ask_price_depth = f'a_p_{depth - 1}'
    if not ask_price_depth in df:
      raise ValueError(f"Data doesn't have a sufficient depth {depth}!")
    df = df[df[ask_price_depth].notna()]  # Drop rows with insufficient depth
    if verbose:
      print('Check depth')
      print(df)
  # Quantize data
  if quant is not None:
    resample = {'rule': pd.Timedelta(quant, unit='ns'), 
                'origin': 'epoch',
                'closed': 'left',
                'label': 'right'}
    df = df.resample(**resample).aggregate(func='last').interpolate(method='pad')
    if verbose:
      print('Resample')
      print(df)
  return df
  

def preprocess_trade(df, quant=None, verbose=False):
  if quant is not None:
    resample = {'rule': pd.Timedelta(quant, unit='ns'), 
                'origin': 'epoch',
                'closed': 'left',
                'label': 'right'}
    aggreagate = {
      'func': { 
        't_p_s': 'mean', 
        't_q_s': 'sum', 
        't_v_s': 'sum',
        't_p_b': 'mean', 
        't_q_b': 'sum', 
        't_v_b': 'sum'}}
    df = df.resample(**resample).aggregate(**aggreagate)
    if verbose:
      print('Resample')
      print(df)
  return df
  

def unwrap_trade(values):
  new_values = deepcopy(values)
  if values['t_s']:  # taker sell
    new_values['t_p_s'] = values['t_p']
    new_values['t_q_s'] = values['t_q']
    new_values['t_v_s'] = values['t_p']*values['t_q']
    new_values['t_p_b'] = None
    new_values['t_q_b'] = None
    new_values['t_v_b'] = None
  else:  # taker buy
    new_values['t_p_s'] = None
    new_values['t_q_s'] = None
    new_values['t_v_s'] = None
    new_values['t_p_b'] = values['t_p']
    new_values['t_q_b'] = values['t_q']
    new_values['t_v_b'] = values['t_p']*values['t_q']
  new_values.pop('t_s')
  new_values.pop('t_p')
  new_values.pop('t_q')
  return new_values  
  
  
def evaluate_delta_df(df, kind='lograt', columns_bottom=None, columns_up=None, 
                      shift=0, self_only=False):
  if columns_up is None:
    columns_up = [x for x in df.columns if is_numeric_dtype(df[x])]
  columns_up_2 = expand_columns(df.columns, columns_up)
  if columns_bottom is None:
    columns_bottom = [x for x in df.columns if is_numeric_dtype(df[x])]
  columns_bottom_2 = expand_columns(df.columns, columns_bottom)
  if len(columns_up_2) == 0:
    raise ValueError(f'No columns found with patterns: {columns_up}')
  if len(columns_bottom_2) == 0:
    raise ValueError(f'No columns found with patterns: {columns_bottom}')
  prefix = f'{kind}_{str(shift)}'
  dfs = []
  for column_bottom in columns_bottom_2:
    if shift == 0:  # Remove self
      columns_up_3 = [x for x in columns_up_2 if x != column_bottom]
    else:
      if self_only:
        columns_up_3 = [x for x in columns_up_2 if x == column_bottom]
      else:
        columns_up_3 = columns_up_2
    new_names = {x: '-'.join([prefix, x, column_bottom]) for x in columns_up_3}
    b = df[column_bottom].shift(shift) if shift != 0 else df[column_bottom]
    us = df[columns_up_3]
    if kind == 'rel':
      new_df = us.div(b.replace({0: np.nan}), axis=0) - 1
    elif kind == 'rat':
      new_df = us.div(b.replace({0: np.nan}), axis=0)
    elif kind == 'lograt':
      new_df = us.log(u.div(b.replace({0: np.nan}), axis=0))
    new_df = new_df.rename(columns=new_names)
    dfs.append(new_df)
  new_df = pd.concat(dfs, axis=1)
  return new_df


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


def evaluate_ema_df(df, columns, alphas, old_ema=None, skipna=True):
  n2c = {}
  cols = expand_columns(df.columns, columns)
  for alpha in alphas:
    alpha_str = np.format_float_positional(alpha)
    col2ema = {x: '_'.join(['ema', alpha_str]) + '-' + x for x in cols}
    for index, row in df.iterrows():
      for col in cols:
        ema_col = col2ema[col]
        n2c.setdefault(ema_col, [])
        value = row[col]
        isna = value is None or np.isnan(value)
        if len(n2c[ema_col]) == 0:
          if old_ema is not None:
            prev_ema = old_ema.head(1)[ema_col].item()
            if skipna and isna:
              ema = prev_ema
            else:
              ema = prev_ema + alpha*(value - prev_ema)
          else:
            ema = value
        else:
          prev_ema = n2c[ema_col][-1]
          if skipna and isna:
            ema = prev_ema
          else:
            ema = prev_ema + alpha*(value - prev_ema)
        n2c[ema_col].append(ema)
  n2c['timestamp'] = df.index
  new_ema = pd.DataFrame(n2c)
  new_ema = new_ema.set_index('timestamp')
  return new_ema


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
  new_values['o_v'] = new_values['a_v'] + new_values['b_v']
  return new_values


def evaluate_trade(df):
  n2c = {}
  n2c['t_p'] = df[['t_p_b', 't_p_s']].mean(axis=1)
  n2c['t_v'] = df[['t_v_b', 't_v_s']].sum(axis=1)
  n2c['t_q'] = df[['t_q_b', 't_q_s']].sum(axis=1)
  new_df = pd.DataFrame(n2c)
  return new_df


def evaluate_qv_df(df, depth=None):
  n2c = {}  # name to column map
  if depth is None:
    a_q_cols = [x for x in df.columns if x.startswith('a_q_')]
    b_q_cols = [x for x in df.columns if x.startswith('b_q_')]
    a_p_cols = [x for x in df.columns if x.startswith('a_p_')]
    b_p_cols = [x for x in df.columns if x.startswith('b_p_')]
  else:
    a_q_cols = [x for x in df.columns 
                if x.startswith('a_q_') and int(x.split('_')[2]) < depth]
    b_q_cols = [x for x in df.columns 
                if x.startswith('b_q_') and int(x.split('_')[2]) < depth]
    a_p_cols = [x for x in df.columns 
                if x.startswith('a_p_') and int(x.split('_')[2]) < depth]
    b_p_cols = [x for x in df.columns 
                if x.startswith('b_p_') and int(x.split('_')[2]) < depth]
  a_v_cols = ['_'.join(['a', 'v', x.split('_')[2]]) for x in a_q_cols]
  b_v_cols = ['_'.join(['b', 'v', x.split('_')[2]]) for x in b_q_cols]
  for a_q_key, a_p_key, a_v_key in zip(a_q_cols, a_p_cols, a_v_cols):
    p = df[a_p_key] 
    q = df[a_q_key]
    n2c[a_v_key] = p*q
  for b_q_key, b_p_key, b_v_key in zip(b_q_cols, b_p_cols, b_v_cols):
    p = df[b_p_key]
    q = df[b_q_key]
    n2c[b_v_key] = p*q  
  new_df = pd.DataFrame(n2c)
  new_df['a_v'] = new_df[a_v_cols].sum(axis=1)
  new_df['b_v'] = new_df[b_v_cols].sum(axis=1)
  new_df['a_q'] = df[a_q_cols].sum(axis=1)
  new_df['b_q'] = df[b_q_cols].sum(axis=1)
  new_df['o_v'] = new_df[['a_v', 'b_v']].sum(axis=1)
  return new_df


def vwap_main(row, col, quantity=None, currency='base', depth=None):
  ps, qs, i, s = [], [], 0, 0  # prices, quantities, index, sum
  while True:
    if depth is not None and i == depth:
      break
    cp = f'{col}_p_{i}'
    cq = f'{col}_q_{i}'
    if not cp in row or not cq in row:
      break
    p = row[cp]
    q = row[cq]
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
  vwap = np.average(ps, weights=qs)
  return vwap


def evaluate_vwap_df(df, currency='base', quantities=None, depth=None):
  assert currency in ['base', 'quote']
  quantities = [] if quantities is None else quantities
  n2c = {}
  for quantity in quantities:
    quantity_str = 'max' if quantity is None else np.format_float_positional(quantity)
    depth_str = str(depth) if depth is not None else 'max'
    c_ask = '_'.join(['vwap', 'ask', currency, quantity_str, depth_str])
    c_bid = '_'.join(['vwap', 'bid', currency, quantity_str, depth_str])
    c_mid = '_'.join(['vwap', 'mid', currency, quantity_str, depth_str])
    n2c[c_ask] = df.apply(vwap_main, col='a', quantity=quantity, currency=currency, axis=1)
    n2c[c_bid] = df.apply(vwap_main, col='b', quantity=quantity, currency=currency, axis=1)
    n2c[c_mid] = 0.5*(n2c[c_ask] + n2c[c_bid])
  new_df = pd.DataFrame(n2c)
  return new_df


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