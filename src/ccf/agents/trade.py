import concurrent.futures
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import uuid
import os
import socket
import time
from pprint import pprint
import hmac
import hashlib
import logging

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import pandas as pd
import websocket

from ccf.agents.base import Agent
from ccf import partitioners as ccf_partitioners
from ccf.utils import loop_futures


class Trader(Agent):
  def __init__(self, strategy, api_url=None, api_key=None, secret_key=None):
    super().__init__()
    self.strategy = strategy
    self.api_url = os.getenv('TRADE_API_URL', api_url)
    self.api_key = os.getenv('TRADE_API_KEY', api_key)
    self.secret_key = os.getenv('TRADE_SECRET_KEY', secret_key)
    
  def compute_signature(self, params, exchange=None):
    query = '&'.join(f'{k}={v}' for k, v in sorted(params.items(), key=lambda x: x[0]) if k != 'signature')
    signature = hmac.new(self.secret_key.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature
  
  def send_request(self, ws, method, params=None,
                   add_api_key=True, add_signature=True,
                   timestamp=None, recv_window=None, timeout=None, exchange=None):
    if exchange == 'binance':
      params = {} if params is None else params
      assert recv_window is None or recv_window <= 60000  # recommended <= 5000
      request = {"id": str(uuid.uuid4()), "method": method}
      if timestamp is not None:
        params['timestamp'] = timestamp
      if self.api_key is not None and add_api_key:
        params['apiKey'] = self.api_key
      if recv_window is not None:
        params['recvWindow'] = recv_window
      if self.secret_key is not None and add_signature:
        signature = self.compute_signature(params, exchange=exchange)
        params['signature'] = signature
      if len(params) > 0:
        request["params"] = params
      ws.send(json.dumps(request))
      try:
        # if timestamp < server_time + 1000 and server_time - timestamp < recv_window:  # process request
        # else: # reject request
        response = json.loads(ws.recv())
      except Exception as e:
        print(e)
        return None
      else:
        status = response.get('status', {})
        if status == 200:
          result = response.get('result', {})
          return result
        else:
          print(response)
          return None
    else:
      raise NotImplementedError(exchange)
      
  def get_server_time(self, ws, timeout=None, exchange=None):
    response = self.send_request(ws, 'time', timeout=timeout)
    if response is not None:
      server_time = response.get('serverTime', None)
    else:
      server_time = None
    return server_time
  
  def ping_server(self, ws, timeout=None, exchange=None):
    response = self.send_request(ws, 'ping', timeout=timeout)
    if response is not None:
      return True
    else:
      return False
  
  def exchange_info(self, ws, symbols=None, timeout=None, exchange=None):
    if symbols is not None:
      if isinstance(symbols, str):
        params = {"symbol": symbols}
      elif isinstance(symbols, list):
        params = {"symbols": symbols}
      else:
        raise ValueError(symbols)
    else:
      params = None
    return self.send_request(ws, 'exchangeInfo', params, timeout=timeout)
  
  def get_account_status(self, ws, timeout=None, exchange=None):
    timestamp = int(time.time_ns()/1e6)
    return self.send_request(ws, 'account.status', None,
                               timestamp=timestamp, timeout=timeout, exchange=exchange)

  def get_ticker_orderbook(self, ws, symbols=None, timeout=None, exchange=None):
    if symbols is not None:
      if isinstance(symbols, str):
        params = {"symbol": symbols}
      elif isinstance(symbols, list):
        params = {"symbols": symbols}
      else:
        raise ValueError(symbols)
    else:
      params = None
    result = self.send_request(ws, 'ticker.book', params, 
                               add_api_key=False, add_signature=False, 
                               timeout=timeout, exchange=exchange)
    if result is not None:
      for key in ['bidPrice', 'bidQty', 'askPrice', 'askQty']:
        result[key] = float(result[key])
    return result
  
  def get_ticker_price(self, ws, symbols=None, timeout=None, exchange=None):
    if symbols is not None:
      if isinstance(symbols, str):
        params = {"symbol": symbols}
      elif isinstance(symbols, list):
        params = {"symbols": symbols}
      else:
        raise ValueError(symbols)
    else:
      params = None
    return self.send_request(ws, 'ticker.price', params,
                             add_api_key=False, add_signature=False, 
                             timeout=timeout, exchange=exchange)
  
  def buy_market(self, ws, symbol, quantity, is_base_quantity=True, is_test=False,
                 recv_window=5000, timeout=None, exchange=None):
    params = {
        "symbol": symbol,
        "side": "BUY",
        "type": "MARKET"
    }
    if is_base_quantity:
      params['quantity'] = quantity
    else:  # quote
      params['quoteOrderQty'] = quantity
    timestamp = int(time.time_ns()/1e6)
    method = 'order.place' if not is_test else 'order.test'
    return self.send_request(ws, method, params,
                             timestamp=timestamp, recv_window=recv_window, 
                             timeout=timeout, exchange=exchange)
    
  def sell_market(self, ws, symbol, quantity, is_base_quantity=True, is_test=False,
                  recv_window=5000, timeout=None, exchange=None):
    params = {
        "symbol": symbol,
        "side": "SELL",
        "type": "MARKET"
    }
    if is_base_quantity:
      params['quantity'] = quantity
    else:  # quote
      params['quoteOrderQty'] = quantity
    timestamp = int(time.time_ns()/1e6)
    method = 'order.place' if not is_test else 'order.test'
    return self.send_request(ws, method, params,
                             timestamp=timestamp, recv_window=recv_window, 
                             timeout=timeout, exchange=exchange)
  
  def buy_oco(self, ws, symbol, quantity, price, stop_price, stop_limit_price,
              time_in_force='GTC', new_order_resp_type='RESULT',
              recv_window=5000, timeout=None, exchange=None):
    # time_in_force
    # GTC Good 'til Canceled – the order will remain on the book until you cancel it, or the order is completely filled.
    # IOC Immediate or Cancel – the order will be filled for as much as possible, the unfilled quantity immediately expires.
    # FOK Fill or Kill – the order will expire unless it cannot be immediately filled for the entire quantity.
    # newOrderRespType Select response format: ACK, RESULT, FULL.
    # MARKET and LIMIT orders use FULL by default, other order types default to ACK.
    # BUY price < market price < stopPrice < stopLimitPrice
    # SELL price > market price > stopPrice > stopLimitPrice
    # Any LIMIT or LIMIT_MAKER order can be made into an iceberg order by specifying the icebergQty.
    # An order with an icebergQty must have timeInForce set to GTC.
    # Trigger order price rules for STOP_LOSS/TAKE_PROFIT orders:
    # stopPrice must be above market price: STOP_LOSS BUY, TAKE_PROFIT SELL
    # stopPrice must be below market price: STOP_LOSS SELL, TAKE_PROFIT BUY
    params = {
        "symbol": symbol,
        "quantity": quantity,
        "side": "BUY",
        "price": price,  # desired price
        "stopPrice": stop_price,  # trigger
        "stopLimitPrice": stop_limit_price,  # new price after trigger
        "stopLimitTimeInForce": time_in_force,
        "newOrderRespType": new_order_resp_type
    }
    timestamp = int(time.time_ns()/1e6)
    return self.send_request(ws, 'orderList.place', params,
                               timestamp, recv_window, timeout, exchange=exchange)
  
  def sell_oco(self, ws,
               symbol, quantity, price, stop_price, stop_limit_price,
               time_in_force='GTC', new_order_resp_type='RESULT',
               recv_window=5000, timeout=None, exchange=None):
    params = {
        "symbol": symbol,
        "quantity": quantity,
        "side": "SELL",
        "price": price,  # desired price
        "stopPrice": stop_price,  # trigger
        "stopLimitPrice": stop_limit_price,  # new price after trigger
        "stopLimitTimeInForce": time_in_force,
        "newOrderRespType": new_order_resp_type
    }
    timestamp = int(time.time_ns()/1e6)
    return self.send_request(ws, 'orderList.place', params,
                               timestamp, recv_window, timeout, exchange=exchange)
  
  def __call__(self):
    raise NotImplementedError()


class MomentumTrader(Trader):
  def __init__(
    self, strategy, 
    key, 
    api_url=None, api_key=None, secret_key=None,
    prediction_topic='prediction',
    metric_topic='metric',
    consumer_partitioner=None, consumer=None, 
    producer_partitioner=None, producer=None, 
    timeout=None, 
    start=None, stop=None, quant=None, size=None, watermark=None, delay=None,
    feature=None, target=None, model=None, version=None, horizon=None, prediction=None,
    threshold=0.0, num_forecasts=3, num_d_forecasts=2, num_d2_forecasts=0,
    quantity=None, is_base_quantity=True, position='none',
    is_test=False, verbose=False
  ):
    super().__init__(strategy=strategy, api_url=api_url, api_key=api_key, secret_key=secret_key)
    self.key = key
    self.prediction_topic = prediction_topic
    self.metric_topic = metric_topic
    if consumer_partitioner is None:
      consumer_partitioner = {}
    self.consumer_partitioner = consumer_partitioner
    self.consumer = {} if consumer is None else consumer
    if producer_partitioner is None:
      producer_partitioner = {}
    self.producer_partitioner = producer_partitioner
    self.producer = {} if producer is None else producer
    self.timeout = timeout
    self.start = start
    self.stop = stop
    self.quant = quant
    self.size = size
    self.watermark = int(watermark) if watermark is not None else watermark
    self.delay = delay
    self.model = model
    self.version = version
    self.feature = feature
    self.target = target
    self.horizon = horizon
    self.prediction = prediction
    self.threshold = threshold
    self.num_forecasts = num_forecasts
    self.num_d_forecasts = num_d_forecasts
    self.num_d2_forecasts = num_d2_forecasts
    self.quantity = quantity
    self.is_base_quantity = is_base_quantity
    self.position = position
    self.is_test = is_test
    self.verbose = verbose
    self._consumer = None
    self._producer = None
    self._ws = None
    
  def init_consumer(self):
    consumer = deepcopy(self.consumer)
    partitioner = deepcopy(self.consumer_partitioner)
    partitioner_class = partitioner.pop('class')
    partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
    partitioner.update()
    consumer['key_deserializer'] = partitioner.deserialize_key
    consumer['value_deserializer'] = partitioner.deserialize_value
    self._consumer = KafkaConsumer(**consumer)
    partitions = partitioner[self.key]
    self._consumer.assign([TopicPartition(y, x) 
                           for x in partitions 
                           for y in [self.prediction_topic]])

  def init_producer(self):
    producer = deepcopy(self.producer)
    partitioner = deepcopy(self.producer_partitioner)
    partitioner_class = partitioner.pop('class')
    partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
    partitioner.update()
    producer['key_serializer'] = partitioner.serialize_key
    producer['value_serializer'] = partitioner.serialize_value
    self._producer = KafkaProducer(**producer)
      
  def __call__(self):
    # logger = logging.getLogger('kafka')
    # logger.setLevel(logging.CRITICAL)
    # websocket.enableTrace(True if self.verbose > 1 else False)
    # if self.timeout is not None:
    #   websocket.setdefaulttimeout(self.timeout)
    # Lazy init
    if self._consumer is None:
      self.init_consumer()
    if self._producer is None:
      self.init_producer()
    if self._ws is None:
      self._ws = websocket.WebSocket()
      self._ws.connect(self.api_url, timeout=self.timeout)
    # Strategy
    buffer = {}
    exchange, base, quote = self.key.split('-')
    symbol = f'{base}{quote}'.upper()
    base_symbol = base.upper()
    quote_symbol = quote.upper()
    last_timestamp = None
    last_price = None
    last_quantity = None
    threshold_up = 1 + self.threshold
    threshold_down = 1 - self.threshold
    for message in self._consumer:
      try:
        value = message.value
        if value['horizon'] != self.horizon:
          continue
        if self.prediction not in value:
          continue
        if self.model is not None:
          if value['model'] != self.model:
            continue
        else:
          self.model = value['model']
        if self.version is not None:
          if value['version'] != self.version:
            continue
        else:
          self.version = value['version']
        if self.quant is None:
          self.quant = value['quant']
        if self.feature is None:
          self.feature = value['feature']
        if self.target is None:
          self.target = value['target']
        timestamp = value['timestamp']
        buffer.setdefault(timestamp, {}).update(value)
        if last_timestamp is not None and timestamp != last_timestamp:
          ticker_orderbook = self.get_ticker_orderbook(
            self._ws, symbol, self.timeout, exchange=exchange)
          buffer.setdefault(timestamp, {}).update(ticker_orderbook)
          current_timestamp = time.time_ns()
          watermark_timestamp = current_timestamp - self.watermark
          print(f'\ntrade')
          print(f'watermark:  {datetime.fromtimestamp(watermark_timestamp/1e9, tz=timezone.utc)}')
          print(f'now:        {datetime.fromtimestamp(current_timestamp/1e9, tz=timezone.utc)}')
          print(f'last:       {datetime.fromtimestamp(last_timestamp/1e9, tz=timezone.utc)}')
          print(f'current:    {datetime.fromtimestamp(timestamp/1e9, tz=timezone.utc)}')
          print(f'buffer:     {len(buffer)}')
          buffer = {k: v for k, v in buffer.items() if k > watermark_timestamp}
          print(f'new_buffer: {len(buffer)}')
          df = pd.DataFrame(buffer.values())
          if len(df) == 0:
            continue
          df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
          df = df.set_index('timestamp').sort_index()
          print(f'min:        {df.index.min()}')
          print(f'max:        {df.index.max()}')
          print(f'rows:       {len(df)}')
          print(f'columns:    {len(df.columns)}')
          if self.verbose:
            print(df)
          if self.prediction in df:
            df['midPrice'] = df[['askPrice', 'bidPrice']].mean(axis=1)
            df['forecast_last'] = df[self.prediction] / df['last']
            df['forecast_mid'] = df[self.prediction] / df['midPrice']
            df['d_forecast_last'] = df['forecast_last'] - df['forecast_last'].shift(1)
            df['d2_forecast_last'] = df['d_forecast_last'] - df['d_forecast_last'].shift(1)
            df['d_forecast_mid'] = df['forecast_mid'] - df['forecast_mid'].shift(1)
            df['d2_forecast_mid'] = df['d_forecast_mid'] - df['d_forecast_mid'].shift(1)
            flag_forecasts_last_up = False if self.num_forecasts != 0 else True
            flag_forecasts_mid_up = False if self.num_forecasts != 0 else True 
            flag_forecasts_last_down = False if self.num_forecasts != 0 else True
            flag_forecasts_mid_down = False if self.num_forecasts != 0 else True
            flag_d_forecasts_last_up = False if self.num_d_forecasts != 0 else True
            flag_d_forecasts_mid_up = False if self.num_d_forecasts != 0 else True
            flag_d_forecasts_last_down = False if self.num_d_forecasts != 0 else True
            flag_d_forecasts_mid_down = False if self.num_d_forecasts != 0 else True
            flag_d2_forecasts_last_up = False if self.num_d2_forecasts != 0 else True
            flag_d2_forecasts_mid_up = False if self.num_d2_forecasts != 0 else True
            flag_d2_forecasts_last_down = False if self.num_d2_forecasts != 0 else True
            flag_d2_forecasts_mid_down = False if self.num_d2_forecasts != 0 else True
            if self.num_forecasts > 0 and len(df) >= self.num_forecasts:
              flag_forecasts_last_up = all(df['forecast_last'][-self.num_forecasts:] > threshold_up)
              flag_forecasts_mid_up = all(df['forecast_mid'][-self.num_forecasts:] > threshold_up)
              flag_forecasts_last_down = all(df['forecast_last'][-self.num_forecasts:] < threshold_down)
              flag_forecasts_mid_down = all(df['forecast_mid'][-self.num_forecasts:] < threshold_down)
            if self.num_d_forecasts > 0 and len(df) >= self.num_d_forecasts:
              flag_d_forecasts_last_up = all(df['d_forecast_last'][-self.num_d_forecasts:] > 0)
              flag_d_forecasts_mid_up = all(df['d_forecast_mid'][-self.num_d_forecasts:] > 0)
              flag_d_forecasts_last_down = all(df['d_forecast_last'][-self.num_d_forecasts:] < 0)
              flag_d_forecasts_mid_down = all(df['d_forecast_mid'][-self.num_d_forecasts:] < 0)
            if self.num_d2_forecasts > 0 and len(df) >= self.num_d2_forecasts:
              flag_d2_forecasts_last_up = all(df['d2_forecast_last'][-self.num_d2_forecasts:] > 0)
              flag_d2_forecasts_mid_up = all(df['d2_forecast_mid'][-self.num_d2_forecasts:] > 0)
              flag_d2_forecasts_last_down = all(df['d2_forecast_last'][-self.num_d2_forecasts:] < 0)
              flag_d2_forecasts_mid_down = all(df['d2_forecast_mid'][-self.num_d2_forecasts:] < 0)
            if all([self.position != 'long',
                    flag_forecasts_last_down, 
                    flag_forecasts_mid_down,
                    flag_d_forecasts_last_up,
                    flag_d_forecasts_mid_up,
                    flag_d2_forecasts_last_up,
                    flag_d2_forecasts_mid_up]):
              action = 'buy'
              self.position = 'long' if self.position == 'none' else 'none'
              result = self.buy_market(self._ws, 
                                       symbol, 
                                       quantity=self.quantity, 
                                       is_base_quantity=self.is_base_quantity, 
                                       is_test=self.is_test,
                                       timeout=self.timeout, 
                                       exchange=exchange)
              pprint(result)
            elif all([self.position != 'short',
                    flag_forecasts_last_up, 
                    flag_forecasts_mid_up,
                    flag_d_forecasts_last_down,
                    flag_d_forecasts_mid_down,
                    flag_d2_forecasts_last_down,
                    flag_d2_forecasts_mid_down]):
              action = 'sell'
              self.position = 'short' if self.position == 'none' else 'none'
              result = self.sell_market(self._ws,
                                        symbol, 
                                        quantity=self.quantity, 
                                        is_base_quantity=self.is_base_quantity, 
                                        is_test=self.is_test,
                                        timeout=self.timeout,
                                        exchange=exchange)
              pprint(result)
            else:  # hold
              action = 'hold'
              trade_timestamp = None
            print(action, self.position)
            if action in ['sell', 'buy']:
              if self.verbose:
                print(df['forecast_last'][-self.num_forecasts:])
                print(df['d_forecast_last'][-self.num_d_forecasts:])
              value = {
                'metric': 'trade',
                'timestamp': current_timestamp,
                'exchange': exchange,
                'base': base,
                'quote': quote,
                'quant': self.quant,
                'feature': self.feature,
                'prediction': self.prediction,
                'horizon': self.horizon,
                'model': self.model,
                'version': self.version,
                'target': self.target,
                'strategy': self.strategy,
                'threshold': self.threshold,
                'num_forecasts': self.num_forecasts,
                'num_d_forecasts': self.num_d_forecasts,
                'num_d2_forecasts': self.num_d2_forecasts,
                'quantity': self.quantity,
                'is_base_quantity': self.is_base_quantity,
                'position': self.position,
                'action': action,
                'api_url': self.api_url,
                'api_key': self.api_key
              }
              account_status = self.get_account_status(
                self._ws, timeout=self.timeout, exchange=exchange)
              if self.verbose:
                print(account_status)
              for b in account_status.get('balances'):
                asset = b['asset']
                if asset in [base_symbol, quote_symbol]:
                  value[f'free_{asset}'] = float(b['free'])
                  value[f'locked_{asset}'] = float(b['locked'])
              pprint(value)
              self._producer.send(topic=self.metric_topic, key=self.key, value=value)
        last_timestamp = timestamp
      except Exception as e:
        print(e)
    if self._consumer is not None:
      self._consumer.close()
    if self._ws is not None:
      self._ws.close()
