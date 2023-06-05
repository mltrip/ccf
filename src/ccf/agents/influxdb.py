from copy import deepcopy
from datetime import datetime, timezone
import os

from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd

from ccf.agents.base import Agent


class InfluxDB(Agent):
  def __init__(self, client=None, bucket=None, query_api=None, write_api=None,
               verbose=False, consumers=None, producers=None):
    super().__init__()
    self.client = {} if client is None else client
    self.client.setdefault('token', os.getenv('INFLUXDB_V2_TOKEN', None))
    self.client.setdefault('url', os.getenv('INFLUXDB_V2_URL', 'https://influxdb:8086'))
    self.client.setdefault('org', os.getenv('INFLUXDB_V2_ORG', 'mltrip'))
    self.client.setdefault('timeout', os.getenv('INFLUXDB_V2_TIMEOUT', None))
    self.client.setdefault('verify_ssl', os.getenv(
      'INFLUXDB_V2_VERIFY_SSL', 'no').lower() in ['yes', 'true', '1'])
    self.client.setdefault('proxy', os.getenv('INFLUXDB_V2_PROXY', None))
    self.bucket = os.getenv('INFLUXDB_V2_BUCKET', 'ccf') if bucket is None else bucket
    self.query_api = query_api
    self.write_api = write_api
    self.verbose = verbose
    self.consumers = {} if consumers is None else consumers
    self.producers = {} if producers is None else producers
    # Init consumers
    self.consumers_topic_keys = {}
    for name, consumer in self.consumers.items():
      self.consumers_topic_keys[name] = consumer.pop('topic_keys', {})
    for name, consumer in self.consumers.items():
      self.consumers[name] = self.Consumer(**consumer)
    # Init producers
    self.producers_topic_keys = {}
    for name, producer in self.producers.items():
      self.producers_topic_keys[name] = producer.pop('topic_keys', {})
      producer.setdefault('client', self.client)
      producer.setdefault('write_api', self.write_api)
      producer.setdefault('bucket', self.bucket)
    for name, producer in self.producers.items():
      self.producers[name] = self.Producer(**producer)
  
  class Producer:
    def __init__(self, client, write_api, bucket):
      super().__init__()
      self.client = client
      self.write_api = write_api
      self.bucket = bucket
      self._client = None
      self._write_api = None
    
    def init(self):  # Lazy init TODO with properties?
      if self._client is None:
        self._client = InfluxDB.init_client(self.client)
      if self._write_api is None:
        self._write_api = InfluxDB.get_write_api(self._client, self.write_api)
    
    def send(self, topic, key, value):
      self.init()
      record = InfluxDB.message_to_record(value, topic)
      results = self._write_api.write(bucket=self.bucket, record=record, 
                                      write_precision='ns')
      return results
      
  class Consumer:
    def __init__(self):
      super().__init__()
      
    def receive(self, topic, key):
      raise NotImplementedError()
  
  @staticmethod
  def init_client(client_kwargs):
    return InfluxDBClient(**client_kwargs)

  @staticmethod
  def get_query_api(client, query_api_kwargs=None):
    query_api_kwargs = {} if query_api_kwargs is None else deepcopy(query_api_kwargs)
    return client.query_api(**query_api_kwargs)
  
  @staticmethod
  def get_write_api(client, write_api_kwargs=None):
    write_api_kwargs = {} if write_api_kwargs is None else deepcopy(write_api_kwargs)
    if 'write_options' in write_api_kwargs:
      wo = WriteOptions(**write_api_kwargs['write_options'])
    else:
      wo = SYNCHRONOUS
    write_api_kwargs['write_options'] = wo  
    return client.write_api(**write_api_kwargs)
  
  @staticmethod
  def record_to_message(record):
    """Convert InfluxDB record to Kafka message"""
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
  def message_to_record(message, topic):
    """Convert Kafka message to InfluxDB record by topic"""
    time_key = 'timestamp'
    if topic in ['trade', 'lob']:
      tags_keys = ['exchange', 'base', 'quote']
    elif topic == 'news':
      tags_keys = ['id', 'title', 'summary', 'links', 'authors', 'tags']
    elif topic == 'feature':
      tags_keys = ['exchange', 'base', 'quote', 'quant', 'feature']
    elif topic == 'prediction':
      tags_keys = ['exchange', 'base', 'quote', 'quant', 'feature', 
                   'model', 'version', 'target', 'horizon']
    elif topic == 'metric':
      tags_keys = ['exchange', 'base', 'quote', 'quant', 'feature', 
                   'model', 'version', 'target', 'horizon', 'prediction', 'metric']
    else:
      raise ValueError(topic)
    if 'quant' in message:  # FIXME 3000000000 to 3e9
      if isinstance(message['quant'], int):
        message['quant'] = f"{message['quant']:.0e}"
    tags_time_keys = tags_keys + [time_key]
    record = {
      'measurement': topic,
      'time': message[time_key],
      'tags': {k: message[k] for k in tags_keys},
      'fields': {k: v for k, v in message.items() if k not in tags_time_keys}}
    return record
  
  @staticmethod
  def iterrow_to_message(iterrow):
    message = dict(iterrow[1])
    message['timestamp'] = int(iterrow[0].timestamp())*int(10**9)
    return message
  
  @staticmethod
  def get_subquery(topic, exchange=None, base=None, quote=None,
                   id=None, title=None, summary=None, links=None,
                   authors=None, tags=None, feature=None, quant=None, 
                   model=None, version=None, target=None, horizon=None,
                   prediction=None, metric=None):
    key2values = {k: v for k, v in locals().items() if v is not None and k != 'topic'}
    query = f'''
          |> filter(fn:(r) => r._measurement == "{topic}")'''
    for k, v in key2values.items():
      if k != 'quant':
        if isinstance(v, str) and v.startswith('/') and v.endswith('/'):
          query += f'''
              |> filter(fn:(r) => r.{k} =~ {v})'''
        else:
          query += f'''
              |> filter(fn:(r) => r.{k} == "{v}")'''
      else:
        query += f'''
            |> filter(fn:(r) => r.{k} == "{v:.0e}" or r.{k} == "{int(v)}")'''
    return query
  
  @staticmethod
  def read_dataframe(query_api, subquery, batch_size, bucket='ccf',
                     start=None, stop=None, verbose=False):
    batch_size = int(batch_size)
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
      start_timestamp = int(cur_start/10**9)
      stop_timestamp = int(cur_stop/10**9)
      start_time = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
      stop_time = datetime.fromtimestamp(stop_timestamp, tz=timezone.utc) 
      print(f'batch {batch_idx + 1}/{int(num_batches)} from {start_time} to {stop_time}')
      if start_timestamp != stop_timestamp:
        rename_str = '|> rename(columns: {_time: "timestamp"})'
        query = f'''
          from(bucket: "{bucket}")
          |> range(start: {start_timestamp}, stop: {stop_timestamp})'''
        query += subquery
        query += '''
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> drop(columns: ["_start", "_stop", "_measurement", "host"])
          |> rename(columns: {_time: "timestamp"})'''
        if verbose:
          print(query)
        df = query_api.query_data_frame(query=query)
        if isinstance(df, pd.DataFrame):
          dfs.append(df)
        elif isinstance(df, list):
          dfs.extend(df)
        else:
          raise NotImplementedError(df)
      else:
        print(f'batch skipped because start {start_timestamp} equals to stop {stop_timestamp}')
      batch_idx += 1
      cur_start = cur_stop
    df = pd.concat(dfs, ignore_index=True, sort=False)
    if len(df) > 0:
      df = df.set_index('timestamp')
      df = df.drop(columns=['result', 'table'])  # https://community.influxdata.com/t/get-rid-of-result-table-columns/14887/3
      df = df.sort_index()
    if verbose:
      print(df)
      print(df.columns)
      print(df.dtypes)
    return df
  
  @staticmethod
  def read_dataframe_by_topic(
    topic, query_api, batch_size, bucket='ccf',
    start=None, stop=None,
    exchange=None, base=None, quote=None, 
    verbose=False, **kwargs
  ):
    topic2func = {
      'lob': InfluxDB.read_lob_dataframe,
      'trade': InfluxDB.read_trade_dataframe,
      'news': InfluxDB.read_news_dataframe,
      'feature': InfluxDB.read_feature_dataframe,
      'prediction': InfluxDB.read_prediction_dataframe,
      'metric': InfluxDB.read_metric_dataframe
    }
    func = topic2func[topic]
    return func(query_api=query_api, 
                batch_size=batch_size, bucket=bucket,
                start=start, stop=stop,
                exchange=exchange, base=base, quote=quote, 
                verbose=verbose, **kwargs)
  
  @staticmethod
  def get_batch_stream(query_api, subquery, batch_size, bucket='ccf',
                       start=None, stop=None, verbose=False):
    df = InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket=bucket,
                                 start=start, stop=stop, verbose=verbose)
    message_stream = map(InfluxDB.iterrow_to_message, df.iterrows())
    return message_stream
  
  @staticmethod
  def get_stream(query_api, subquery, bucket='ccf',
                 start=None, stop=None, verbose=False):
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
  def get_lob_batch_stream(query_api, batch_size, bucket='ccf',
                           start=None, stop=None,
                           exchange=None, base=None, quote=None, 
                           verbose=False):
    subquery = InfluxDB.get_subquery('lob', 
                                     exchange=exchange, base=base, quote=quote)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket,
                                     start, stop, verbose)
  
  @staticmethod
  def read_lob_dataframe(query_api, batch_size, bucket='ccf',
                         start=None, stop=None,
                         exchange=None, base=None, quote=None, 
                         verbose=False):
    subquery = InfluxDB.get_subquery('lob', 
                                     exchange=exchange, base=base, quote=quote)
    return InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket,
                                   start, stop, verbose)
  
  @staticmethod
  def get_lob_stream(query_api, bucket='ccf', 
                     start=None, stop=None, 
                     exchange=None, base=None, quote=None, 
                     verbose=False):
    subquery = InfluxDB.get_subquery('lob', 
                                     exchange=exchange, base=base, quote=quote)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                               start, stop, verbose)  
  
  @staticmethod
  def get_trade_batch_stream(query_api, batch_size, bucket='ccf', 
                             start=None, stop=None,
                             exchange=None, base=None, quote=None,
                             verbose=False):
    subquery = InfluxDB.get_subquery('trade', 
                                     exchange=exchange, base=base, quote=quote)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket,
                                     start, stop, verbose)
  
  @staticmethod
  def read_trade_dataframe(query_api, batch_size, bucket='ccf',
                           start=None, stop=None,
                           exchange=None, base=None, quote=None, 
                           verbose=False):
    subquery = InfluxDB.get_subquery('trade', 
                                     exchange=exchange, base=base, quote=quote)
    return InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket,
                                   start, stop, verbose)
  
  @staticmethod
  def get_trade_stream(query_api, bucket='ccf', 
                       start=None, stop=None,
                       exchange=None, base=None, quote=None, 
                       verbose=False):
    subquery = InfluxDB.get_subquery('trade', 
                                     exchange=exchange, base=base, quote=quote)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                               start, stop, verbose)
  
  @staticmethod
  def get_news_batch_stream(query_api, batch_size, bucket='ccf',
                            start=None, stop=None,
                            id=None, title=None, summary=None, 
                            links=None, authors=None, tags=None, 
                            verbose=False):
    subquery = InfluxDB.get_subquery('news', id=id, title=title, summary=summary,
                                     links=links, authors=authors, tags=tags)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket,
                                     start, stop, verbose)
  
  @staticmethod
  def read_news_dataframe(query_api, batch_size, bucket='ccf',
                          start=None, stop=None, verbose=False,
                          id=None, title=None, summary=None, 
                          links=None, authors=None, tags=None):
    subquery = InfluxDB.get_subquery('news', id=id, title=title, summary=summary,
                                     links=links, authors=authors, tags=tags)
    return InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket,
                                   start, stop, verbose)
  
  @staticmethod
  def get_news_stream(query_api, bucket='ccf', 
                      start=None, stop=None,
                      id=None, title=None, summary=None, 
                      links=None, authors=None, tags=None, 
                      verbose=False):
    subquery = InfluxDB.get_subquery('news', id=id, title=title, summary=summary,
                                     links=links, authors=authors, tags=tags)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                               start, stop, verbose)
  
  
  @staticmethod
  def read_feature_dataframe(query_api, batch_size, bucket='ccf',
                             start=None, stop=None,
                             exchange=None, base=None, quote=None, 
                             feature=None, quant=None, 
                             verbose=False):
    subquery = InfluxDB.get_subquery('feature', 
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant)
    return InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket, 
                                   start, stop, verbose)

  
  @staticmethod
  def get_feature_batch_stream(query_api, batch_size, bucket='ccf',
                               start=None, stop=None,
                               exchange=None, base=None, quote=None, 
                               feature=None, quant=None, 
                               verbose=False):
    subquery = InfluxDB.get_subquery('feature', exchange=exchange,
                                     base=base, quote=quote, 
                                     feature=feature, quant=quant)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket,
                                     start, stop, verbose)
 
  @staticmethod
  def get_feature_stream(query_api, bucket='ccf',
                         start=None, stop=None,
                         exchange=None, base=None, quote=None, 
                         feature=None, quant=None, 
                         verbose=False):
    subquery = InfluxDB.get_subquery('feature',
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                                start, stop, verbose)
  
  
  @staticmethod
  def read_prediction_dataframe(query_api, batch_size, bucket='ccf',
                                start=None, stop=None,
                                exchange=None, base=None, quote=None, 
                                feature=None, quant=None, 
                                model=None, version=None, horizon=None, target=None,
                                verbose=False):
    subquery = InfluxDB.get_subquery('prediction', 
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant, 
                                     model=model, version=version,
                                     horizon=horizon, target=target)
    return InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket,
                                   start, stop, verbose)

  @staticmethod
  def get_prediction_batch_stream(query_api, batch_size, bucket='ccf',
                                  start=None, stop=None,
                                  exchange=None, base=None, quote=None, 
                                  feature=None, quant=None, 
                                  model=None, version=None, horizon=None, target=None, 
                                  verbose=False):
    subquery = InfluxDB.get_subquery('prediction', 
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant, 
                                     model=model, version=version,
                                     horizon=horizon, target=target)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket,
                                     start, stop, verbose)
 
  @staticmethod
  def get_prediction_stream(query_api, bucket='ccf',
                            start=None, stop=None,
                            exchange=None, base=None, quote=None, 
                            feature=None, quant=None, 
                            model=None, version=None, horizon=None, target=None, 
                            verbose=False):
    subquery = InfluxDB.get_subquery('prediction', 
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant, 
                                     model=model, version=version,
                                     horizon=horizon, target=target)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                               start, stop, verbose)
  
  @staticmethod
  def read_metric_dataframe(query_api, batch_size, bucket='ccf',
                            start=None, stop=None, 
                            exchange=None, base=None, quote=None, 
                            feature=None, quant=None,  
                            model=None, version=None, horizon=None, target=None,
                            prediction=None, metric=None,
                            verbose=False):
    subquery = InfluxDB.get_subquery('metric', 
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant, 
                                     model=model, version=version,
                                     horizon=horizon, target=target,
                                     prediction=prediction, metric=metric)
    return InfluxDB.read_dataframe(query_api, subquery, batch_size, bucket,
                                   start, stop, verbose)

  @staticmethod
  def get_metric_batch_stream(query_api, batch_size, bucket='ccf',
                              start=None, stop=None,
                              exchange=None, base=None, quote=None, 
                              feature=None, quant=None,  
                              model=None, version=None, horizon=None, target=None,
                              prediction=None, metric=None,
                              verbose=False):
    subquery = InfluxDB.get_subquery('metric', 
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant, 
                                     model=model, version=version,
                                     horizon=horizon, target=target,
                                     prediction=prediction, metric=metric)
    return InfluxDB.get_batch_stream(query_api, subquery, batch_size, bucket, 
                                     start, stop, verbose)
 
  @staticmethod
  def get_metric_stream(query_api, bucket='ccf',
                        start=None, stop=None,
                        exchange=None, base=None, quote=None, 
                        feature=None, quant=None,  
                        model=None, version=None, horizon=None, target=None,
                        prediction=None, metric=None,
                        verbose=False):
    subquery = InfluxDB.get_subquery('metric', 
                                     exchange=exchange, base=base, quote=quote, 
                                     feature=feature, quant=quant, 
                                     model=model, version=version,
                                     horizon=horizon, target=target,
                                     prediction=prediction, metric=metric)
    return InfluxDB.get_stream(query_api, subquery, bucket, 
                                start, stop, verbose)