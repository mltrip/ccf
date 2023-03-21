import os
import pytest

from ccf.agents import InfluxDB, StreamDatasetInfluxDB
from ccf.utils import initialize_time

  
@pytest.fixture
def client():
  return InfluxDB()


@pytest.mark.influxdb
def test_lob_batch_stream(client):
  start = -60e9
  stop = None
  size = None
  quant = None
  batch_size = 6e9
  bucket = client.bucket
  exchange = 'binance'
  base = 'btc'
  quote = 'usdt'
  verbose = True
  start, stop, size, quant = initialize_time(start, stop, size, quant)
  client_ = client.init_client(client.client)
  query_api = client.get_query_api(client_, client.query_api)
  stream = client.get_lob_batch_stream(query_api, bucket=bucket, 
                                       start=start, stop=stop, 
                                       batch_size=batch_size,
                                       exchange=exchange, base=base, quote=quote, 
                                       verbose=verbose)
  for message in stream:
    print(message)
    
    
@pytest.mark.influxdb
def test_trade_batch_stream(client):
  start = -60e9
  stop = None
  size = None
  quant = None
  batch_size = 6e9
  bucket = client.bucket
  exchange = 'binance'
  base = 'btc'
  quote = 'usdt'
  verbose = True
  start, stop, size, quant = initialize_time(start, stop, size, quant)
  client_ = client.init_client(client.client)
  query_api = client.get_query_api(client_, client.query_api)
  stream = client.get_trade_batch_stream(query_api, bucket=bucket, 
                                         start=start, stop=stop, 
                                         batch_size=batch_size,
                                         exchange=exchange, base=base, quote=quote, 
                                         verbose=verbose)
  for message in stream:
    print(message)
  
  
@pytest.mark.influxdb
def test_news_batch_stream(client):
  start = -60e9
  stop = None
  size = None
  quant = None
  batch_size = 6e9
  bucket = client.bucket
  id = None
  title = None
  summary = None 
  links = None
  authors = None
  tags = None
  verbose = True
  start, stop, size, quant = initialize_time(start, stop, size, quant)
  client_ = client.init_client(client.client)
  query_api = client.get_query_api(client_, client.query_api)
  stream = client.get_news_batch_stream(query_api, bucket=bucket, 
                                        start=start, stop=stop, 
                                        batch_size=batch_size,
                                        id=id, title=title, summary=summary, 
                                        links=links, authors=authors, tags=tags,
                                        verbose=verbose)
  for message in stream:
    print(message)
  
    
@pytest.mark.influxdb
def test_feature_batch_stream(client):
  start = -60e9
  stop = None
  size = None
  quant = None
  batch_size = 6e9
  bucket = client.bucket
  exchange = 'binance'
  base = 'btc'
  quote = 'usdt'
  feature = None
  verbose = True
  start, stop, size, quant = initialize_time(start, stop, size, quant)
  client_ = client.init_client(client.client)
  query_api = client.get_query_api(client_, client.query_api)
  stream = client.get_feature_batch_stream(query_api, bucket=bucket, 
                                           start=start, stop=stop, 
                                           batch_size=batch_size,
                                           exchange=exchange, base=base, quote=quote, 
                                           feature=feature, quant=quant,
                                           verbose=verbose)
  for message in stream:
    assert 'quant' in message
    assert 'feature' in message
    print(message)
    
    
@pytest.mark.influxdb
def test_prediction_batch_stream(client):
  start = -60e9
  stop = None
  size = None
  quant = None
  batch_size = 6e9
  bucket = client.bucket
  exchange = 'binance'
  base = 'btc'
  quote = 'usdt'
  feature = None
  model = None
  version = None
  horizon = None
  target = None
  verbose = True
  start, stop, size, quant = initialize_time(start, stop, size, quant)
  client_ = client.init_client(client.client)
  query_api = client.get_query_api(client_, client.query_api)
  stream = client.get_prediction_batch_stream(query_api, bucket=bucket, 
                                              start=start, stop=stop, 
                                              batch_size=batch_size,
                                              exchange=exchange, base=base, quote=quote, 
                                              feature=feature, quant=quant,
                                              model=model, version=version, 
                                              horizon=horizon, target=target,
                                              verbose=verbose)
  for message in stream:
    assert 'model' in message
    assert 'version' in message
    assert 'horizon' in message
    assert 'target' in message
    print(message)
   
    
@pytest.mark.influxdb
def test_metric_batch_stream(client):
  start = -60e9
  stop = None
  size = None
  quant = None
  batch_size = 6e9
  bucket = client.bucket
  exchange = 'binance'
  base = 'btc'
  quote = 'usdt'
  feature = None
  model = None
  version = None
  horizon = None
  target = None
  prediction = None
  metric = None
  verbose = True
  start, stop, size, quant = initialize_time(start, stop, size, quant)
  client_ = client.init_client(client.client)
  query_api = client.get_query_api(client_, client.query_api)
  stream = client.get_metric_batch_stream(query_api, bucket=bucket, 
                                          start=start, stop=stop, 
                                          batch_size=batch_size,
                                          exchange=exchange, base=base, quote=quote, 
                                          feature=feature, quant=quant,
                                          model=model, version=version, 
                                          horizon=horizon, target=target,
                                          prediction=prediction, metric=metric,
                                          verbose=verbose)
  for message in stream:
    assert 'prediction' in message
    assert 'metric' in message
    print(message)
    
    
@pytest.mark.influxdb
def test_dataset_stream():
  dataset = StreamDatasetInfluxDB(
    client=None,
    bucket=None,
    query_api=None,
    write_api=None,
    executor={'class': 'ThreadPoolExecutor'},
    topic='feature',
    feature_keys={'rat-ema-qv-vwap_20': ['binance-btc-usdt']}, 
    start=-300e9,
    stop=-270e9,
    size=None,
    quant=3e9,
    watermark=18e9, 
    delay=18e9,
    batch_size=6e9, 
    replace_nan=1.0,
    resample=None, 
    aggregate=None, 
    interpolate=None,
    verbose=True)
  n_samples = dataset.watermark // dataset.quant
  for sample in dataset:
    print({k: v.shape for k, v in sample.items()})
    assert all(len(v) >= n_samples for v in sample.values())
  with pytest.raises(StopIteration):   
    sample = dataset()
