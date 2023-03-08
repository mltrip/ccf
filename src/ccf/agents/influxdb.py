from datetime import datetime
import os

from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd

from ccf.agents.base import Agent
from ccf.utils import initialize_time


class InfluxDB(Agent):
  def __init__(self, client=None, bucket=None, query_api=None, write_api=None,
               verbose=False):
    super().__init__()
    self.client = {} if client is None else client
    self.client.setdefault('token', os.getenv('INFLUXDB_V2_TOKEN', None))
    self.client.setdefault('url', os.getenv('INFLUXDB_V2_URL', 'https://influxdb:8086'))
    self.client.setdefault('org', os.getenv('INFLUXDB_V2_ORG', 'mltrip'))
    self.client.setdefault('timeout', os.getenv('INFLUXDB_V2_TIMEOUT', None))
    self.client.setdefault('verify_ssl', os.getenv(
      'INFLUXDB_V2_VERIFY_SSL', 'true').lower() in ['yes', 'true', '1'])
    self.client.setdefault('proxy', os.getenv('INFLUXDB_V2_PROXY', None))
    self.bucket = os.getenv('INFLUXDB_V2_BUCKET', 'ccf') if bucket is None else bucket
    self.query_api = query_api
    self.write_api = write_api
    self.verbose = verbose
  
  @staticmethod
  def init_client(client):
    return InfluxDBClient(**client)

  @staticmethod
  def get_query_api(client, query_api_kwargs=None):
    query_api_kwargs = {} if query_api_kwargs is None else query_api_kwargs
    return client.query_api(**query_api_kwargs)
  
  @staticmethod
  def get_write_api(client, write_api_kwargs=None):
    write_api_kwargs = {} if write_api_kwargs is None else write_api_kwargs
    if 'write_options' in write_api_kwargs:
      wo = WriteOptions(**write_api_kwargs['write_options'])
    else:
      wo = SYNCHRONOUS
    write_api_kwargs['write_options'] = wo  
    return client.write_api(**write_api_kwargs)
  
  @staticmethod
  def record_to_message(record):
    message = record.values
    message['timestamp'] = int(message.pop('_time').timestamp())*int(10**9)
    message.pop('_start')
    message.pop('_stop')
    message.pop('_measurement')
    message.pop('host', None)
    message.pop('result')
    message.pop('table')
    return message
  
  @staticmethod
  def iterrow_to_message(index, row):
    message = dict(row)
    message['timestamp'] = int(index.timestamp())*int(10**9)
    return message
  
  @staticmethod
  def message_to_record_lob(message, topic='lob'):
    record = {
      'measurement': topic,
      'time': message['timestamp'],
      'tags': {'exchange': message['exchange'], 
               'base': message['base'],
               'quote': message['quote']},
      'fields': {k: v for k, v in message.items() 
                 if k not in ['exchange', 'base', 'quote', 'timestamp', 'topic']}}
    return record
  
  @staticmethod
  def message_to_record_trade(message, topic='trade'):
    record = {
      'measurement': topic,
      'time': message['timestamp'],
      'tags': {'exchange': message['exchange'], 
               'base': message['base'],
               'quote': message['quote']},
      'fields': {k: v for k, v in message.items() 
                 if k not in ['exchange', 'base', 'quote', 'timestamp', 'topic']}}
    return record
  
  @staticmethod
  def get_lob_subquery(topic='lob', exchange=None, base=None, quote=None):
    query = f'''
      |> filter(fn:(r) => r._measurement == "{topic}")'''
    if exchange is not None:
      query += f'''
        |> filter(fn:(r) => r.exchange == "{exchange}")'''
    if base is not None:
      query += f'''
        |> filter(fn:(r) => r.base == "{base}")'''
    if quote is not None:
      query += f'''
        |> filter(fn:(r) => r.quote == "{quote}")'''
    return query

  @staticmethod
  def get_trade_subquery(topic='trade', exchange=None, base=None, quote=None):
    query = f'''
      |> filter(fn:(r) => r._measurement == "{topic}")'''
    if exchange is not None:
      query += f'''
        |> filter(fn:(r) => r.exchange == "{exchange}")'''
    if base is not None:
      query += f'''
        |> filter(fn:(r) => r.base == "{base}")'''
    if quote is not None:
      query += f'''
        |> filter(fn:(r) => r.quote == "{quote}")'''
    return query
  
  @staticmethod
  def read_dataframe(query_api, subquery, batch_size, bucket='ccf',
                     start=None, stop=None, size=None, quant=None,  
                     verbose=False):
    start, stop, size, quant = initialize_time(start, stop, size, quant)
    dfs = []
    num_batches, last_batch_size = divmod(stop - start, batch_size)
    num_batches = num_batches + 1 if last_batch_size > 0 else num_batches
    batch_idx = 0
    cur_start = start
    while batch_idx < num_batches:
      if last_batch_size > 0 and batch_idx + 1 == num_batches:
        cur_batch_size = last_batch_size
      else:
        cur_batch_size = batch_size
      cur_stop = cur_start + cur_batch_size
      print(f'batch {batch_idx + 1}/{int(num_batches)}')
      rename_str = '|> rename(columns: {_time: "timestamp"})'
      query = f'''
        from(bucket: "{bucket}")
        |> range(start: {int(cur_start/10**9)}, stop: {int(cur_stop/10**9)})'''
      query += subquery
      query += '''
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> drop(columns: ["_start", "_stop", "_measurement", "host"])
        |> rename(columns: {_time: "timestamp"})'''
      if verbose:
        print(query)
      df = query_api.query_data_frame(query=query)
      dfs.append(df)
      batch_idx += 1
      cur_start = cur_stop
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = df.set_index('timestamp')
    df = df.drop(columns=['result', 'table'])  # https://community.influxdata.com/t/get-rid-of-result-table-columns/14887/3
    if verbose:
      print(df)
      print(df.columns)
      print(df.dtypes)
    return df
  
  @staticmethod
  def get_batch_stream(query_api, subquery, batch_size, bucket='ccf',
                       start=None, stop=None, size=None, quant=None,  
                       verbose=False):
    start, stop, size, quant = initialize_time(start, stop, size, quant)
    dfs = []
    num_batches, last_batch_size = divmod(stop - start, batch_size)
    num_batches = num_batches + 1 if last_batch_size > 0 else num_batches
    batch_idx = 0
    cur_start = start
    while batch_idx < num_batches:
      if last_batch_size > 0 and batch_idx + 1 == num_batches:
        cur_batch_size = last_batch_size
      else:
        cur_batch_size = batch_size
      cur_stop = cur_start + cur_batch_size
      print(f'batch {batch_idx + 1}/{int(num_batches)}')
      rename_str = '|> rename(columns: {_time: "timestamp"})'
      query = f'''
        from(bucket: "{bucket}")
        |> range(start: {int(cur_start/10**9)}, stop: {int(cur_stop/10**9)})'''
      query += subquery
      query += '''
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> drop(columns: ["_start", "_stop", "_measurement", "host"])
        |> rename(columns: {_time: "timestamp"})'''
      if verbose:
        print(query)
      df = query_api.query_data_frame(query=query)
      dfs.append(df)
      batch_idx += 1
      cur_start = cur_stop
    df = pd.concat(dfs, ignore_index=True, sort=False)
    if len(df) > 0:
      df = df.set_index('timestamp')
      df = df.drop(columns=['result', 'table'])  # https://community.influxdata.com/t/get-rid-of-result-table-columns/14887/3
    if verbose:
      print(df)
      print(df.columns)
      print(df.dtypes)
    return df.iterrows()
  
  @staticmethod
  def get_stream(query_api, subquery, bucket='ccf',
                 start=None, stop=None, size=None, quant=None, 
                 verbose=False):
    start, stop, size, quant = initialize_time(start, stop, size, quant)
    start_str = f'start: {int(start/10**9)}'
    stop_str = '' if stop is None else f', stop: {int(stop/10**9)}'
    query = f'''
      from(bucket: "{bucket}")
      |> range({start_str}{stop_str})'''
    query += subquery
    query += '''
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> group(columns: ["_time"])'''
    if verbose:
      print(query)
    return query_api.query_stream(query)
 
  @staticmethod
  def read_lob_dataframe(query_api, batch_size, bucket='ccf',
                         start=None, stop=None, size=None, quant=None,
                         topic='lob', exchange=None, base=None, quote=None, 
                         verbose=False):
    subquery = InfluxDB.get_lob_subquery(topic, exchange, base, quote)
    return InfluxDB.get_dataframe(query_api, subquery, batch_size, bucket,
                                  start, stop, size, quant, verbose)
  @staticmethod
  def read_trade_dataframe(query_api, batch_size, bucket='ccf',
                           start=None, stop=None, size=None, quant=None,
                           topic='trade', exchange=None, base=None, quote=None, 
                           verbose=False):
    subquery = InfluxDB.get_trade_subquery(topic, exchange, base, quote)
    return InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket,
                                  start, stop, size, quant, verbose)

  @staticmethod
  def get_lob_batch_stream(query_api, batch_size, bucket='ccf',
                           start=None, stop=None, size=None, quant=None,
                           topic='lob', exchange=None, base=None, quote=None, 
                           verbose=False):
    subquery = InfluxDB.get_lob_subquery(topic, exchange, base, quote)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket,
                                     start, stop, size, quant, verbose)
  
  @staticmethod
  def get_trade_batch_stream(query_api, batch_size, bucket='ccf',
                             start=None, stop=None, size=None, quant=None,
                             topic='trade', exchange=None, base=None, quote=None, 
                             verbose=False):
    subquery = InfluxDB.get_trade_subquery(topic, exchange, base, quote)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket,
                                     start, stop, size, quant, verbose)
  
  @staticmethod
  def get_lob_stream(query_api, bucket='ccf', 
                     start=None, stop=None, size=None, quant=None,
                     topic='lob', exchange=None, base=None, quote=None, 
                     verbose=False):
    subquery = InfluxDB.get_lob_subquery(topic, exchange, base, quote)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                                start, stop, size, quant, verbose)
  
  @staticmethod
  def get_trade_stream(query_api, bucket='ccf', 
                       start=None, stop=None, size=None, quant=None,
                       topic='trade', exchange=None, base=None, quote=None, 
                       verbose=False):
    subquery = InfluxDB.get_trade_subquery(topic, exchange, base, quote)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                                start, stop, size, quant, verbose)
