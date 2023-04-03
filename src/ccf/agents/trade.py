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
import stable_baselines3

from ccf.agents.base import Agent
from ccf import partitioners as ccf_partitioners
from ccf.utils import loop_futures
from ccf.model_mlflow import CCFRLModel, load_model
from ccf.train_rl_mlflow import preprocess_data


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
    if exchange == 'binance':
      key_map = {'bidPrice': 'b_p_0', 
                 'bidQty': 'b_q_0', 
                 'askPrice': 'a_p_0',
                 'askQty': 'a_q_0'}
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
        result = {key_map.get(k, k): v for k, v in result.items()}
    else:
      raise NotImplementedError(exchange)
    if result is not None:
      for key in result:
        if key in ['b_p_0', 'b_q_0', 'a_p_0', 'a_q_0']:
          result[key] = float(result[key])
      result['m_p'] = 0.5*(result['a_p_0'] + result['b_p_0'])
      result['s_p'] = result['a_p_0'] - result['b_p_0']
      result['s_p-m_p'] = result['s_p'] / result['m_p']
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
  
  def buy_taker(self, ws, symbol, quantity, is_base_quantity=True, is_test=False,
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
  
  def buy_maker(self, ws, symbol, price, quantity, time_in_force='GTC', 
                is_base_quantity=True, is_test=False,
                recv_window=5000, timeout=None, exchange=None):
    params = {
        "symbol": symbol,
        "side": "BUY",
        "type": "LIMIT",
        "price": price,
        "timeInForce": time_in_force
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
    
  def sell_taker(self, ws, symbol, quantity, is_base_quantity=True, is_test=False,
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
  
  def sell_maker(self, ws, symbol, price, quantity, time_in_force='GTC', 
                 is_base_quantity=True, is_test=False,
                 recv_window=5000, timeout=None, exchange=None):
    params = {
        "symbol": symbol,
        "side": "SELL",
        "type": "LIMIT",
        "price": price,
        "timeInForce": time_in_force
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
  
  def current_open_orders(self, ws, symbol, recv_window=5000, timeout=None, exchange=None):
    params = {"symbol": symbol}
    method = 'openOrders.status'
    timestamp = int(time.time_ns()/1e6)
    return self.send_request(ws, method, params,
                             timestamp=timestamp, recv_window=recv_window, 
                             timeout=timeout, exchange=exchange)
  
  def query_order(self, ws, symbol, order_id, recv_window=5000, timeout=None, exchange=None):
    params = {
      "symbol": symbol,
      "orderId": order_id
    }
    method = 'order.status'
    timestamp = int(time.time_ns()/1e6)
    return self.send_request(ws, method, params,
                             timestamp=timestamp, recv_window=recv_window, 
                             timeout=timeout, exchange=exchange)
  
  def cancel_order(self, ws, symbol, client_order_id, recv_window=5000, timeout=None, exchange=None):
    params = {
      "symbol": symbol,
      "origClientOrderId": client_order_id
    }
    method = 'order.cancel'
    timestamp = int(time.time_ns()/1e6)
    return self.send_request(ws, method, params,
                             timestamp=timestamp, recv_window=recv_window, 
                             timeout=timeout, exchange=exchange)
  
  def cancel_open_orders(self, ws, symbol, recv_window=5000, timeout=None, exchange=None):
    params = {"symbol": symbol}
    method = 'openOrders.cancelAll'
    timestamp = int(time.time_ns()/1e6)
    return self.send_request(ws, method, params,
                             timestamp=timestamp, recv_window=recv_window, 
                             timeout=timeout, exchange=exchange)
  
  
  def buy_oco(self, ws, symbol, quantity, price, stop_price, stop_limit_price,
              time_in_force='GTC', new_order_resp_type='RESULT',
              recv_window=5000, timeout=None, exchange=None):
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
    start=None, stop=None, quant=None, size=None, watermark=None, delay=None, max_delay=None,
    feature=None, target=None, model=None, version=None, horizon=None, prediction=None,
    max_spread=None, is_max_spread_none_only=False, time_in_force='GTC', max_open_orders=None,
    do_cancel_open_orders=True,
    t_x=0.0, t_v=0.0, t_a=0.0, n_x=0, n_v=0, n_a=0, 
    b_t_fs=None, s_t_fs=None, b_m_fs=None, s_m_fs=None,
    quantity=None, is_base_quantity=True, position='none',
    is_test=False, verbose=False
  ):
    """Momentum Trader
    time_in_force
    * GTC Good 'til Canceled – the order will remain on the book until you cancel it, or the order is completely filled.
    * IOC Immediate or Cancel – the order will be filled for as much as possible, the unfilled quantity immediately expires.
    * FOK Fill or Kill – the order will expire unless it cannot be immediately filled for the entire quantity.
    status
    * NEW – The order has been accepted by the engine.
    * PARTIALLY_FILLED – A part of the order has been filled.
    * FILLED – The order has been completed.
    * CANCELED – The order has been canceled by the user.
    * PENDING_CANCEL – Currently unused
    * REJECTED – The order was not accepted by the engine and not processed.
    * EXPIRED – The order was canceled according to the order type's rules (e.g. LIMIT FOK orders with no fill, LIMIT IOC or MARKET orders that partially fill) or by the exchange, (e.g. orders canceled during liquidation, orders canceled during maintenance)
    * EXPIRED_IN_MATCH – The order was canceled by the exchange due to STP trigger. (e.g. an order with EXPIRE_TAKER will match with existing orders on the book with the same account or same tradeGroupId)
    """
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
    self.max_delay = max_delay
    self.model = model
    self.version = version
    self.feature = feature
    self.target = target
    self.horizon = horizon
    self.prediction = prediction
    self.max_spread = max_spread
    self.is_max_spread_none_only = is_max_spread_none_only
    self.do_cancel_open_orders = do_cancel_open_orders
    self.time_in_force = time_in_force
    self.max_open_orders = max_open_orders
    self.t_x = t_x
    self.t_v = t_v
    self.t_a = t_a
    self.n_x = n_x
    self.n_v = n_v
    self.n_a = n_a
    self.b_t_fs = [] if b_t_fs is None else b_t_fs
    self.s_t_fs = [] if s_t_fs is None else s_t_fs
    self.b_m_fs = [] if b_m_fs is None else b_m_fs
    self.s_m_fs = [] if s_m_fs is None else s_m_fs
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
    order_id = None
    base_balance = 0.0
    quote_balance = 0.0
    last_base_balance = 0.0
    last_quote_balance = 0.0
    if self.do_cancel_open_orders:
      cancel_result = self.cancel_open_orders(
        self._ws, symbol, timeout=self.timeout, exchange=exchange)
      pprint(cancel_result)
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
          prediction_timestamp = timestamp - self.horizon*self.quant
          current_timestamp = time.time_ns()
          watermark_timestamp = current_timestamp - self.watermark
          print(f'\ntrade {self.strategy}')
          print(f'watermark:  {datetime.fromtimestamp(watermark_timestamp/1e9, tz=timezone.utc)}')
          print(f'prediction: {datetime.fromtimestamp(prediction_timestamp/1e9, tz=timezone.utc)}')
          print(f'now:        {datetime.fromtimestamp(current_timestamp/1e9, tz=timezone.utc)}')
          print(f'last:       {datetime.fromtimestamp(last_timestamp/1e9, tz=timezone.utc)}')
          print(f'current:    {datetime.fromtimestamp(timestamp/1e9, tz=timezone.utc)}')
          print(f'buffer:     {len(buffer)}')
          buffer = {k: v for k, v in buffer.items() if k > watermark_timestamp}
          print(f'new_buffer: {len(buffer)}')
          if self.max_delay is not None:
            delay = current_timestamp - prediction_timestamp
            if delay > self.max_delay:
              print(f'Skipping: delay {delay} > {self.max_delay}')
              continue
          if self.max_spread is not None:
            spread = ticker_orderbook['s_p-m_p']
            do_skip = False
            if spread > self.max_spread:
              if self.is_max_spread_none_only:
                if self.position == 'none':
                  do_skip = True
              else:  # any position 
                do_skip = True
            if do_skip:
              print(f'Skipping: spread {spread} > {self.max_spread} and position {self.position}')
              continue
          # if self.max_open_orders is not None:
          #   if n_open_orders >= self.max_open_orders:
          #     print(f'Skipping: number of open orders {n_open_orders} >= {self.max_open_orders}')
          #     continue
          #  self.cancel_open_orders(
          #  self._ws, symbol, timeout=self.timeout, exchange=exchange)
          df = pd.DataFrame(buffer.values())
          if len(df) == 0:
            continue
          df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
          df = df.set_index('timestamp').sort_index()
          print(f'min:        {df.index.min()}')
          print(f'max:        {df.index.max()}')
          print(f'rows:       {len(df)}')
          print(f'columns:    {len(df.columns)}')
          if self.verbose > 1:
            print(df)
          if self.prediction in df:
            t_cs = ['last', 'm_p', 'a_p_0', 'b_p_0']
            x_cs = [f'x_{x}' for x in t_cs]
            v_cs = [f'v_{x}' for x in t_cs]
            a_cs = [f'a_{x}' for x in t_cs]
            for x_c, t_c in zip(x_cs, t_cs):
              df[x_c] = df[self.prediction] / df[t_c] - 1.0
            for v_c, x_c in zip(v_cs, x_cs):
              df[v_c] = df[x_c] - df[x_c].shift(1)
            for a_c, v_c in zip(a_cs, v_cs):
              df[a_c] = df[v_c] - df[v_c].shift(1)
            # Flags
            flags = {x: [None for _ in range(3)] for x in t_cs}  
            for i, (n, t, cs) in enumerate([[self.n_x, self.t_x, x_cs], 
                                            [self.n_v, self.t_v, v_cs],
                                            [self.n_a, self.t_a, a_cs]]):
              if n != 0 and len(df) >= n:
                for c, t_c in zip(cs, t_cs):
                  if all(df[c][-n:] > t):
                    flag = 1
                  elif all(df[c][-n:] < -t):
                    flag = -1
                  else:
                    flag = 0
                  flags[t_c][i] = flag
            # Check
            is_b_t = False
            for f in self.b_t_fs:
              if f == flags['a_p_0']:
                is_b_t = True
                print(f'is_b_t {is_b_t}: {f}')
                break
            is_b_m = False
            for f in self.b_m_fs:
              if f == flags['b_p_0']:
                is_b_m = True
                print(f'is_b_m {is_b_m}: {f}')
                break
            is_s_t = False
            for f in self.s_t_fs:
              if f == flags['b_p_0']:
                is_s_t = True
                print(f'is_s_t {is_s_t}: {f}')
                break
            is_s_m = False
            for f in self.s_m_fs:
              if f == flags['a_p_0']:
                is_s_m = True
                print(f'is_s_m {is_s_m}: {f}')
                break
            if self.verbose:
              print(df[['s_p-m_p', 
                        'x_b_p_0', 'x_m_p', 'x_a_p_0', 
                        'v_b_p_0', 'v_m_p', 'v_a_p_0',
                        'a_b_p_0', 'a_m_p', 'a_a_p_0']])
            pprint(ticker_orderbook)
            # Act
            # Evaluate order
            print(f'Position: {self.position}')
            print(f'Active order: {order_id}')
            if order_id is not None:
              order = self.query_order(self._ws, symbol, order_id, 
                                       timeout=self.timeout, 
                                       exchange=exchange)
              pprint(order)
              order_side = order['side']
              order_status = order['status']
              if order_status in ['FILLED']:
                if order_side == 'BUY':
                  if self.position == 'none':
                    self.position = 'long'
                  elif self.position == 'short':
                    self.position = 'none'
                  else:
                    raise ValueError(f"Position can't be {self.position}!")
                else:  # SELL
                  if self.position == 'none':
                    self.position = 'short'
                  elif self.position == 'long':
                    self.position = 'none'
                  else:
                    raise ValueError(f"Position can't be {self.position}!")
                quote_quantity = float(order.get('cummulativeQuoteQty', 0.0))  # origQty
                base_quantity = float(order.get('executedQty', 0.0))
                if order_side == 'BUY':
                  quote_quantity = -quote_quantity
                  base_quantity = -base_quantity
                base_balance += base_quantity
                quote_balance += quote_quantity
                if self.position == 'none':
                  base_delta = base_balance - last_base_balance
                  quote_delta = quote_balance - last_quote_balance
                  last_base_balance = base_balance
                  last_quote_balance = quote_balance
                else:
                  base_delta = None
                  quote_delta = None
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
                  'n_x': self.n_x,
                  'n_v': self.n_v,
                  'n_a': self.n_a,
                  't_x': self.t_x,
                  't_v': self.t_v,
                  't_a': self.t_a,
                  'base_balance': base_balance,
                  'quote_balance': quote_balance,
                  'base_delta': base_delta,
                  'quote_delta': quote_delta,
                  'base_quantity': base_quantity,
                  'quote_quantity': quote_quantity,
                  'quantity': self.quantity,
                  'is_base_quantity': self.is_base_quantity,
                  'position': self.position,
                  'action': order_side.lower(),
                  'api_url': self.api_url,
                  'api_key': self.api_key
                }
                # Get status
                account_status = self.get_account_status(
                  self._ws, timeout=self.timeout, exchange=exchange)
                if self.verbose:
                  print(account_status)
                for b in account_status.get('balances'):
                  asset = b['asset']
                  if asset in [base_symbol, quote_symbol]:
                    value[f'free_{asset}'] = float(b['free'])
                    value[f'locked_{asset}'] = float(b['locked'])
                if self.verbose:
                  pprint(value)
                if not self.is_test:
                  self._producer.send(topic=self.metric_topic, key=self.key, value=value)
                order, order_id = None, None
              elif order_status in ['NEW', 'PARTIALLY_FILLED']:  # Process
                do_b_t = self.position != 'long' and is_b_t
                do_b_m = self.position != 'long' and is_b_m
                do_s_t = self.position != 'short' and is_s_t
                do_s_m = self.position != 'short' and is_s_m
                do_a = any([do_b_t, do_b_m, do_s_t, do_s_m])
                print(f'Order {order_id} in process with status: {order_status}')
                if do_a:  # act
                  print(f'Cancel order {order_id}')
                  client_order_id = order['clientOrderId']
                  self.cancel_order(self._ws, symbol, client_order_id, 
                                    timeout=self.timeout, exchange=exchange)
                  order, order_id = None, None
                else:  # hold
                  print(f'Check order {order_id}')
                  order_type = order['type']
                  if order_type == 'LIMIT':
                    order_side = order['side']
                    client_order_id = order['clientOrderId']
                    order_price = float(order['price'])
                    current_bid = ticker_orderbook['b_p_0']
                    current_ask = ticker_orderbook['a_p_0']
                    if order_side == 'BUY' and order_price != current_bid:
                      self.cancel_order(self._ws, symbol, client_order_id, 
                                        timeout=self.timeout, exchange=exchange)
                      order, order_id = None, None
                      print(f'Cancel {order_side} order with price {order_price} != {current_bid} top bid')
                    elif order_side == 'SELL' and order_price != current_ask:
                      self.cancel_order(self._ws, symbol, client_order_id, 
                                        timeout=self.timeout, exchange=exchange)
                      order, order_id = None, None
                      print(f'Cancel {order_side} order with price {order_price} != {current_ask} top ask')  
              else:  # End
                print(f'Order {order_id} ends with status: {order_status}')
                order, order_id = None, None
            print(f'Position: {self.position}')
            print(f'Active order: {order_id}')
            do_b_t = self.position != 'long' and is_b_t
            do_b_m = self.position != 'long' and is_b_m
            do_s_t = self.position != 'short' and is_s_t
            do_s_m = self.position != 'short' and is_s_m
            do_a = any([do_b_t, do_b_m, do_s_t, do_s_m])
            print(f'Action: {do_a}')
            if do_a and order_id is None:
              if do_b_t:
                print('buy_taker')
                result = self.buy_taker(self._ws, 
                                         symbol, 
                                         quantity=self.quantity, 
                                         is_base_quantity=self.is_base_quantity, 
                                         is_test=self.is_test,
                                         timeout=self.timeout, 
                                         exchange=exchange)
              elif do_b_m:
                print('buy_maker')
                result = self.buy_maker(self._ws, 
                                        symbol,
                                        price=ticker_orderbook['b_p_0'],
                                        time_in_force=self.time_in_force,
                                        quantity=self.quantity, 
                                        is_base_quantity=self.is_base_quantity, 
                                        is_test=self.is_test,
                                        timeout=self.timeout, 
                                        exchange=exchange)
              elif do_s_t:
                print('sell_taker')
                result = self.sell_taker(self._ws,
                                          symbol, 
                                          quantity=self.quantity, 
                                          is_base_quantity=self.is_base_quantity, 
                                          is_test=self.is_test,
                                          timeout=self.timeout,
                                          exchange=exchange)
              elif do_s_m:
                print('sell_maker')
                result = self.sell_maker(self._ws,
                                         symbol, 
                                         price=ticker_orderbook['a_p_0'],
                                         time_in_force=self.time_in_force,
                                         quantity=self.quantity, 
                                         is_base_quantity=self.is_base_quantity, 
                                         is_test=self.is_test,
                                         timeout=self.timeout,
                                         exchange=exchange)
              else:
                raise NotImplementedError()
              order_id = result.get('orderId', None)
              print(f'New order {order_id}')
              pprint(result)
        last_timestamp = timestamp
      except Exception as e:
        print(e)
        if self.position != 'none':
          print('Closing open positions after exception')
          ws = websocket.WebSocket()
          ws.connect(self.api_url, timeout=self.timeout)
          if self.position == 'short':
            result = self.buy_taker(ws, 
                                    symbol, 
                                    quantity=self.quantity, 
                                    is_base_quantity=self.is_base_quantity, 
                                    is_test=self.is_test,
                                    timeout=self.timeout, 
                                    exchange=exchange)
            pprint(result)
          elif self.position == 'long':
            result = self.sell_taker(ws,
                                     symbol, 
                                     quantity=self.quantity, 
                                     is_base_quantity=self.is_base_quantity, 
                                     is_test=self.is_test,
                                     timeout=self.timeout,
                                     exchange=exchange)
            pprint(result)
          self.position = 'none'
          ws.close()
        if self._consumer is not None:
          self._consumer.close()
        if self._ws is not None:
          self._ws.close()
        raise e
    if self._consumer is not None:
      self._consumer.close()
    if self._ws is not None:
      self._ws.close()

      
class RLTrader(Trader):
  """Reinforcement Learning Trader"""
  
  def __init__(
    self, 
    model_name_rl, 
    strategy, 
    key, 
    api_url=None, api_key=None, secret_key=None,
    prediction_topic='prediction',
    metric_topic='metric',
    consumer_partitioner=None, consumer=None, 
    producer_partitioner=None, producer=None, 
    timeout=None, 
    start=None, stop=None, quant=None, size=None, watermark=None, delay=None, max_delay=None,
    feature=None, target=None, model=None, version=None, horizon=None, prediction=None,
    max_spread=None, is_max_spread_none_only=False, time_in_force='GTC', max_open_orders=None,
    do_cancel_open_orders=True, window_size=20,
    quantity=None, is_base_quantity=True, position='none',
    is_test=False, verbose=False,
    kind_rl=None, model_version_rl=None, model_stage_rl=None
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
    self.max_delay = max_delay
    self.model = model
    self.version = version
    self.feature = feature
    self.target = target
    self.horizon = horizon
    self.prediction = prediction
    self.max_spread = max_spread
    self.is_max_spread_none_only = is_max_spread_none_only
    self.do_cancel_open_orders = do_cancel_open_orders
    self.time_in_force = time_in_force
    self.max_open_orders = max_open_orders
    self.window_size = window_size
    self.quantity = quantity
    self.is_base_quantity = is_base_quantity
    self.position = position
    self.is_test = is_test
    self.verbose = verbose
    self.kind_rl = kind_rl
    self.model_name_rl = model_name_rl
    self.model_version_rl = model_version_rl
    self.cur_model_version_rl = model_version_rl
    self.model_stage_rl = model_stage_rl
    self.cur_model_stage_rl = model_stage_rl
    self._consumer = None
    self._producer = None
    self._model_rl = None
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
      
  def update_model_rl(self):
    if self._model_rl is None:
      print('Initalizing model')
      last_model, last_version, last_stage = load_model(
          self.model_name_rl, self.model_version_rl, self.model_stage_rl)
      self._model_rl = last_model.unwrap_python_model().model
      self.cur_model_version_rl = last_version
      self.cur_model_stage_rl = last_stage
    elif self.kind_rl == 'auto_update':
      print('Auto-updating model')
      _, last_version, last_stage = load_model(self.model_name_rl, 
                                               self.model_version_rl, 
                                               self.model_stage_rl,
                                               metadata_only=True)
      if int(last_version) > int(self.cur_model_version_rl):
        print(f'Updating model from {self.cur_model_version_rl} to {last_version} version')
        last_model, last_version, last_stage = load_model(
          self.model_name_rl, self.model_version_rl, self.model_stage_rl)
        self._model_rl = last_model.unwrap_python_model().model
        self.cur_model_version_rl = last_version
        self.cur_model_stage_rl = last_stage
    else:
      pass
    
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
    if self._model_rl is None:
      self.update_model_rl()
    # Strategy
    buffer = {}
    exchange, base, quote = self.key.split('-')
    symbol = f'{base}{quote}'.upper()
    base_symbol = base.upper()
    quote_symbol = quote.upper()
    last_timestamp = None
    order_id = None
    base_balance = 0.0
    quote_balance = 0.0
    last_base_balance = 0.0
    last_quote_balance = 0.0    
    if self.do_cancel_open_orders:
      cancel_result = self.cancel_open_orders(
        self._ws, symbol, timeout=self.timeout, exchange=exchange)
      pprint(cancel_result)
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
          ticker_orderbook = {} if ticker_orderbook is None else ticker_orderbook
          buffer.setdefault(timestamp, {}).update(ticker_orderbook)
          prediction_timestamp = timestamp - self.horizon*self.quant
          current_timestamp = time.time_ns()
          watermark_timestamp = current_timestamp - self.watermark
          print(f'\ntrade {self.strategy}')
          print(f'watermark:  {datetime.fromtimestamp(watermark_timestamp/1e9, tz=timezone.utc)}')
          print(f'prediction: {datetime.fromtimestamp(prediction_timestamp/1e9, tz=timezone.utc)}')
          print(f'now:        {datetime.fromtimestamp(current_timestamp/1e9, tz=timezone.utc)}')
          print(f'last:       {datetime.fromtimestamp(last_timestamp/1e9, tz=timezone.utc)}')
          print(f'current:    {datetime.fromtimestamp(timestamp/1e9, tz=timezone.utc)}')
          print(f'buffer:     {len(buffer)}')
          buffer = {k: v for k, v in buffer.items() if k > watermark_timestamp}
          print(f'new_buffer: {len(buffer)}')
          if self.max_delay is not None:
            delay = current_timestamp - prediction_timestamp
            if delay > self.max_delay:
              print(f'Skipping: delay {delay} > {self.max_delay}')
              continue
          if self.max_spread is not None:
            spread = ticker_orderbook['s_p-m_p']
            do_skip = False
            if spread > self.max_spread:
              if self.is_max_spread_none_only:
                if self.position == 'none':
                  do_skip = True
              else:  # any position 
                do_skip = True
            if do_skip:
              print(f'Skipping: spread {spread} > {self.max_spread} and position {self.position}')
              continue
          # if self.max_open_orders is not None:
          #   if n_open_orders >= self.max_open_orders:
          #     print(f'Skipping: number of open orders {n_open_orders} >= {self.max_open_orders}')
          #     continue
          #  self.cancel_open_orders(
          #  self._ws, symbol, timeout=self.timeout, exchange=exchange)
          df = pd.DataFrame(buffer.values())
          if len(df) == 0:
            print(f'Skipping: empty data')
            continue
          if len(df) < self.window_size:
            print(f'Skipping: data length {len(df)} < window size {self.window_size}')
            continue
          df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
          df = df.set_index('timestamp').sort_index()
          print(f'min:        {df.index.min()}')
          print(f'max:        {df.index.max()}')
          print(f'rows:       {len(df)}')
          print(f'columns:    {len(df.columns)}')
          if self.verbose > 1:
            print(df)
          if self.prediction in df:
            self.update_model_rl()
            obs = preprocess_data(df=df,  
                                  window_size=self.window_size,
                                  forecast_column=self.prediction, 
                                  price_column='last')
            if self.verbose > 1:
              print(obs)
            action, _states = self._model_rl.predict(obs)  # 0: HOLD, 1: BUY, 2: SELL
            pprint(ticker_orderbook)
            # Act
            # Evaluate order
            print(f'Position: {self.position}')
            print(f'Active order: {order_id}')
            if order_id is not None:
              order = self.query_order(self._ws, symbol, order_id, 
                                       timeout=self.timeout, 
                                       exchange=exchange)
              pprint(order)
              order_side = order['side']
              order_status = order['status']
              if order_status in ['FILLED']:
                if order_side == 'BUY':
                  if self.position == 'none':
                    self.position = 'long'
                  elif self.position == 'short':
                    self.position = 'none'
                  else:
                    raise ValueError(f"Position can't be {self.position}!")
                else:  # SELL
                  if self.position == 'none':
                    self.position = 'short'
                  elif self.position == 'long':
                    self.position = 'none'
                  else:
                    raise ValueError(f"Position can't be {self.position}!")
                quote_quantity = float(order.get('cummulativeQuoteQty', 0.0))  # origQty
                base_quantity = float(order.get('executedQty', 0.0))
                if order_side == 'BUY':
                  quote_quantity = -quote_quantity
                  base_quantity = -base_quantity
                base_balance += base_quantity
                quote_balance += quote_quantity
                if self.position == 'none':
                  base_delta = base_balance - last_base_balance
                  quote_delta = quote_balance - last_quote_balance
                  last_base_balance = base_balance
                  last_quote_balance = quote_balance
                else:
                  base_delta = None
                  quote_delta = None
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
                  'model_rl': self.model_name_rl,
                  'version_rl': self.cur_model_version_rl,
                  'base_balance': base_balance,
                  'quote_balance': quote_balance,
                  'base_delta': base_delta,
                  'quote_delta': quote_delta,
                  'base_quantity': base_quantity,
                  'quote_quantity': quote_quantity,
                  'quantity': self.quantity,
                  'is_base_quantity': self.is_base_quantity,
                  'position': self.position,
                  'action': order_side.lower(),
                  'api_url': self.api_url,
                  'api_key': self.api_key
                }
                # Get status
                account_status = self.get_account_status(
                  self._ws, timeout=self.timeout, exchange=exchange)
                if self.verbose:
                  print(account_status)
                for b in account_status.get('balances'):
                  asset = b['asset']
                  if asset in [base_symbol, quote_symbol]:
                    value[f'free_{asset}'] = float(b['free'])
                    value[f'locked_{asset}'] = float(b['locked'])
                if self.verbose:
                  pprint(value)
                if not self.is_test:
                  self._producer.send(topic=self.metric_topic, key=self.key, value=value)
                order, order_id = None, None
              elif order_status in ['NEW', 'PARTIALLY_FILLED']:  # Process
                print(f'Order {order_id} in process with status: {order_status}')
                if action in [1, 2]:  # act
                  print(f'Cancel order {order_id}')
                  client_order_id = order['clientOrderId']
                  self.cancel_order(self._ws, symbol, client_order_id, 
                                    timeout=self.timeout, exchange=exchange)
                  order, order_id = None, None
                else:  # hold
                  print(f'Check order {order_id}')
                  order_type = order['type']
                  if order_type == 'LIMIT':
                    order_side = order['side']
                    client_order_id = order['clientOrderId']
                    order_price = float(order['price'])
                    current_bid = ticker_orderbook['b_p_0']
                    current_ask = ticker_orderbook['a_p_0']
                    if order_side == 'BUY' and order_price != current_bid:
                      self.cancel_order(self._ws, symbol, client_order_id, 
                                        timeout=self.timeout, exchange=exchange)
                      order, order_id = None, None
                      print(f'Cancel {order_side} order with price {order_price} != {current_bid} top bid')
                    elif order_side == 'SELL' and order_price != current_ask:
                      self.cancel_order(self._ws, symbol, client_order_id, 
                                        timeout=self.timeout, exchange=exchange)
                      order, order_id = None, None
                      print(f'Cancel {order_side} order with price {order_price} != {current_ask} top ask')  
              else:  # End
                print(f'Order {order_id} ends with status: {order_status}')
                order, order_id = None, None
            print(f'Position: {self.position}')
            print(f'Active order: {order_id}')
            print(f'Action: {action}')
            do_buy_taker = action == 1 and self.position != 'long'
            do_sell_taker = action == 2 and self.position != 'short'
            if any([do_buy_taker, do_sell_taker]) and order_id is None:
              if do_buy_taker:
                print('buy_taker')
                result = self.buy_taker(self._ws, 
                                        symbol, 
                                        quantity=self.quantity, 
                                        is_base_quantity=self.is_base_quantity, 
                                        is_test=self.is_test,
                                        timeout=self.timeout, 
                                        exchange=exchange)
              elif do_sell_taker:
                print('sell_taker')
                result = self.sell_taker(self._ws,
                                          symbol, 
                                          quantity=self.quantity, 
                                          is_base_quantity=self.is_base_quantity, 
                                          is_test=self.is_test,
                                          timeout=self.timeout,
                                          exchange=exchange)
              else:
                raise NotImplementedError()
              # elif do_b_m:
              #   print('buy_maker')
              #   result = self.buy_maker(self._ws, 
              #                           symbol,
              #                           price=ticker_orderbook['b_p_0'],
              #                           time_in_force=self.time_in_force,
              #                           quantity=self.quantity, 
              #                           is_base_quantity=self.is_base_quantity, 
              #                           is_test=self.is_test,
              #                           timeout=self.timeout, 
              #                           exchange=exchange)
              # elif do_s_m:
              #   print('sell_maker')
              #   result = self.sell_maker(self._ws,
              #                            symbol, 
              #                            price=ticker_orderbook['a_p_0'],
              #                            time_in_force=self.time_in_force,
              #                            quantity=self.quantity, 
              #                            is_base_quantity=self.is_base_quantity, 
              #                            is_test=self.is_test,
              #                            timeout=self.timeout,
              #                            exchange=exchange)
              order_id = result.get('orderId', None)
              print(f'New order {order_id}')
              pprint(result)
        last_timestamp = timestamp
      except Exception as e:
        print(e)
        if self.position != 'none':
          print('Closing open positions after exception')
          ws = websocket.WebSocket()
          ws.connect(self.api_url, timeout=self.timeout)
          if self.position == 'short':
            result = self.buy_taker(ws, 
                                    symbol, 
                                    quantity=self.quantity, 
                                    is_base_quantity=self.is_base_quantity, 
                                    is_test=self.is_test,
                                    timeout=self.timeout, 
                                    exchange=exchange)
            pprint(result)
          elif self.position == 'long':
            result = self.sell_taker(ws,
                                     symbol, 
                                     quantity=self.quantity, 
                                     is_base_quantity=self.is_base_quantity, 
                                     is_test=self.is_test,
                                     timeout=self.timeout,
                                     exchange=exchange)
            pprint(result)
          self.position = 'none'
          ws.close()
        if self._consumer is not None:
          self._consumer.close()
        if self._ws is not None:
          self._ws.close()
        raise e
    if self._consumer is not None:
      self._consumer.close()
    if self._ws is not None:
      self._ws.close()
      