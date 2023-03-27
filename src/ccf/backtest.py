import random
from pathlib import Path

from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
import numpy as np
import pandas as pd
from ccf.agents import InfluxDB
from ccf.utils import initialize_time


class SimpleStrategy(Strategy):
  """"
  pd/t = from 1e-3 to 1/from 1e-4 to 8e-4 without fees
  pd/t = 9e-3/9e-3 with fees or 5e-2+/5e-3+  TODO needs more statistics...
  num_forecasts/num_d_forecasts/num_d2_forecasts = 0/2/3 or 0/2/4 or 0/1/7 ...
  """
  price_delta = 1e-1  # Price delta used in SL and TP
  sl_tp_ratio = 1.0  # Stop Loss / Take Profit or Risk / Reward
  size = 1.0
  max_orders = 10
  threshold = 2e-4  # Decision threshold
  num_forecasts = 3
  num_d_forecasts = 2
  num_d2_forecasts = 0

  def init(self):
    last_column = 'last'
    prediction_column = 'value' if 'value' in self.data.df else 'quantile_0.5'
    prediction_name = 'prediction'
    forecast_name = 'forecast'
    d_forecast_name = 'd_forecast'
    d2_forecast_name = 'd2_forecast'
    # Data
    self.data.df[forecast_name] = self.data.df[prediction_column] / self.data.df[last_column]
    self.data.df[d_forecast_name] = self.data.df[forecast_name] - self.data.df[forecast_name].shift(1)
    self.data.df[d2_forecast_name] = self.data.df[d_forecast_name] - self.data.df[d_forecast_name].shift(1)
    # Indicators
    self.predictions = self.I(lambda: self.data.df[prediction_column], name=prediction_name)
    self.forecasts = self.I(lambda: self.data.df[forecast_name], name=forecast_name)
    self.d_forecasts = self.I(lambda: self.data.df[d_forecast_name], name=d_forecast_name)
    self.d2_forecasts = self.I(lambda: self.data.df[d2_forecast_name], name=d2_forecast_name)
    
  def next(self):
    open, high, low, close = self.data.Open, self.data.High, self.data.Low, self.data.Close
    current_time = self.data.index[-1]

    # forecast = self.forecasts[-1]
    # if len(self.forecasts) > 1:
    #   forecast_2 = self.forecasts[-2]
    # else:
    #   forecast_2 = forecast
    # if len(self.forecasts) > 2:
    #   forecast_3 = self.forecasts[-3]
    # else:
    #   forecast_3 = forecast 
    # if len(self.forecasts) > 3:
    #   forecast_4 = self.forecasts[-4]
    # else:
    #   forecast_4 = forecast 
    # d_forecast = forecast - forecast_2
    # d_forecast_2 = forecast_2 - forecast_3
    # d_forecast_3 = forecast_3 - forecast_4
    # d2_forecast = d_forecast - d_forecast_2
    # d2_forecast_2 = d_forecast_2 - d_forecast_3
    
    k_tp = np.clip(self.price_delta, 0.0, 1.0) 
    k_sl = np.clip(self.sl_tp_ratio*self.price_delta, 0.0, 1.0)
    long_tp = close[-1] * (1 + k_tp)
    long_sl = close[-1] * (1 - k_sl)
    short_tp = close[-1] * (1 - k_tp)
    short_sl = close[-1] * (1 + k_sl)
    
    threshold_up = 1.0 + self.threshold
    threshold_down = 1.0 - self.threshold     
    # print(forecast, forecast_2)
    # If our forecast is upwards and we don't already hold a long position
    # place a long order for 20% of available account equity. Vice versa for short.
    # Also set target take-profit and stop-loss prices to be one price_delta
    # away from the current closing price.
    # upper = close * (1 + self.price_delta)
    # lower = close * (1 - self.price_delta)
    
    # Base
    # print(forecast, close[-1], long_tp, long_sl, short_tp, short_sl)
    # print(self.trades)
    # print(self.closed_trades)
    # print(self.orders)
    # print(self.position.is_long)
    # print(self.position.is_short)
    # print(self.equity)
    
    # if forecast > 1:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif forecast < 1:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
    
    # print(self.position.is_long, self.position.is_short)
    # if forecast > 1 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif forecast < 1 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
      
    # Cross
    # if forecast > 1 and forecast_2 < 1 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif forecast < 1 and forecast_2 > 1 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl) 
      
    # Cross 2
    # if forecast > 1 and forecast_2 > 1 and forecast_3 < 1 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif forecast < 1 and forecast_2 < 1 and forecast_3 > 1 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)  
     
    # Cross 3
    # if forecast > 1 and forecast_2 > 1 and forecast - forecast_2 > 0 and forecast_3 < 1 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif forecast < 1 and forecast_2 < 1 and forecast - forecast_2 < 0 and forecast_2_2 > 1 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)


    
    # Derivative
    # if d_forecast > 0 and d_forecast_2 > 0 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
      
    # Derivative 2
    # if d_forecast > 0 and d_forecast_2 > 0 and d2_forecast > 0 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and d2_forecast < 0 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
   
    # Derivative 3
    # if d_forecast > 0 and d_forecast_2 > 0 and forecast_3 < 1 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and forecast_3 > 1 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
    
    
    # if d_forecast > 0 and d_forecast_2 > 0 and forecast_3 < threshold_down and forecast_2 < threshold_down and forecast < threshold_down and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and forecast_3 > threshold_up and forecast_2 > threshold_up and forecast > threshold_up and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
 
    # k_forecast_3 = (abs(forecast_3 - 1) + 1)**1000
    # print(self.size*k_forecast_3)
        # print()
    # print(current_time)
    # print(open[-1], high[-1], low[-1], close[-1])
    # print(self.position.is_long, self.position.is_short)
    # print(self.orders)
    # print(self.trades)
    # print(self.closed_trades)
    # print(threshold_down, threshold_up)
    # print(forecast_3, forecast_2, forecast)
    # print(d_forecast_2, d_forecast)
    # print(d2_forecast)
    
    # k_forecast_3 = 1

    # if forecast > 0:
    #   if len(self.trades) < self.max_orders:
    #     self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    # elif forecast < 0:
    #   if len(self.trades) < self.max_orders:
    #     self.sell(size=int(self.size*k_forecast_3), tp=short_tp, sl=short_sl)
    # if d_forecast > 0 and d_forecast_2 > 0 and forecast_3 < threshold_down and forecast_2 < threshold_down and forecast < threshold_down:
    #   # if not self.position.is_long:
    #   if len(self.trades) < self.max_orders:
    #     self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    #     # for trade in self.trades:
    #     #   if trade.is_short:
    #     #     self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    #     # self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    #     # if self.position.is_short and len(self.trades) < self.max_orders + 1:
    #     #   self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and forecast_3 > threshold_up and forecast_2 > threshold_up and forecast > threshold_up:
    #   # if not self.position.is_short:
    #   if len(self.trades) < self.max_orders:
    #     self.sell(size=int(self.size*k_forecast_3), tp=short_tp, sl=short_sl)
        # for trade in self.trades:
        #   if trade.is_long:
        #     self.sell(size=int(self.size*k_forecast_3), tp=short_tp, sl=short_sl)
        # if self.position.is_long and len(self.trades) < self.max_orders + 1:
        #   self.sell(size=int(self.size*k_forecast_3), tp=short_tp, sl=short_sl)
    # print(self.orders)
    
    # if d_forecast > 0 and d_forecast_2 > 0 and forecast_3 < threshold_down and forecast_2 < threshold_down and forecast < threshold_down:
    #   # if not self.position.is_long:
    #   if len(self.trades) < self.max_orders:
    #     self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    #     for trade in self.trades:
    #       if trade.is_short:
    #         self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    #     # self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    #     # if self.position.is_short and len(self.trades) < self.max_orders + 1:
    #     #   self.buy(size=int(self.size*k_forecast_3), tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and forecast_3 > threshold_up and forecast_2 > threshold_up and forecast > threshold_up:
    #   # if not self.position.is_short:
    #   if len(self.trades) < self.max_orders:
    #     self.sell(size=int(self.size*k_forecast_3), tp=short_tp, sl=short_sl)
    #     for trade in self.trades:
    #       if trade.is_long:
    #         self.sell(size=int(self.size*k_forecast_3), tp=short_tp, sl=short_sl)
    #     # if self.position.is_long and len(self.trades) < self.max_orders + 1:
    #     #   self.sell(size=int(self.size*k_forecast_3), tp=short_tp, sl=short_sl)
    # print(self.orders)
    
    # Simple  
    # if forecast > threshold_up:
    #   if not self.position.is_long:  # Open long
    #     self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif forecast < threshold_down:
    #   if not self.position.is_short:  # Open short
    #     self.sell(size=self.size, tp=short_tp, sl=short_sl)    
        
    # TODO optimize number of d_forecasts and forecasts
    # Extremum with threshold
    prev_forecasts = self.forecasts[-self.num_forecasts:] if self.num_forecasts > 0 else np.array([])
    prev_d_forecasts = self.d_forecasts[-self.num_d_forecasts:] if self.num_d_forecasts > 0 else np.array([])
    prev_d2_forecasts = self.d2_forecasts[-self.num_d2_forecasts:] if self.num_d2_forecasts > 0 else np.array([])
    # print(prev_forecasts)
    # print(prev_d_forecasts)
    # print(prev_d2_forecasts)
    if all(prev_d2_forecasts > 0) and all(prev_d_forecasts > 0) and all(prev_forecasts < threshold_down):
      # print('buy')
      # if len(self.trades) < self.max_orders:
      #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
      if not self.position.is_long:  # Open long
        self.buy(size=self.size, tp=long_tp, sl=long_sl)
    elif all(prev_d2_forecasts < 0) and all(prev_d_forecasts < 0) and all(prev_forecasts > threshold_up):
      # print('sell')
      # if len(self.trades) < self.max_orders:
      #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
      if not self.position.is_short:  # Open short
        self.sell(size=self.size, tp=short_tp, sl=short_sl)
  
    # Extremum with threshold and close position
    # if d_forecast > 0 and d_forecast_2 > 0 and forecast_3 < threshold_down and forecast_2 < threshold_down and forecast < threshold_down:
    #   if not self.position.is_long:  # Open long
    #     self.buy(size=self.size, tp=long_tp, sl=long_sl)
    #   if self.position.is_short:  # Close short
    #     self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and forecast_3 > threshold_up and forecast_2 > threshold_up and forecast > threshold_up:
    #   if not self.position.is_short:  # Open short
    #     self.sell(size=self.size, tp=short_tp, sl=short_sl)
    #   if self.position.is_long:  # Close long
    #     self.sell(size=self.size, tp=short_tp, sl=short_sl)  
    
    # if d_forecast > 0 and d_forecast_2 > 0 and d_forecast_3 > 0 and forecast_4 < threshold_down and forecast_3 < threshold_down and forecast_2 < threshold_down and forecast < threshold_down:
    #   # if len(self.trades) < self.max_orders:
    #   if not self.position.is_long:
    #     self.buy(size=self.size, tp=long_tp, sl=long_sl)
    #   # if self.position.is_short:
    #   #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and d_forecast_3 < 0 and forecast_4 > threshold_up and forecast_3 > threshold_up and forecast_2 > threshold_up and forecast > threshold_up:
    #   # if len(self.trades) < self.max_orders:
    #   if not self.position.is_short:
    #     self.sell(size=self.size, tp=short_tp, sl=short_sl)
      # if self.position.is_long:
      #   self.sell(size=self.size, tp=short_tp, sl=short_sl)  
  
  
    # if d2_forecast > 0 and d_forecast > 0 and d_forecast_2 > 0 and forecast_3 < 1 and forecast_2 < 1 and forecast < 1 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d2_forecast < 0 and d_forecast < 0 and d_forecast_2 < 0 and forecast_3 > 1 and forecast_2 > 1 and forecast > 1 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
  
    #  if d_forecast > 0 and d_forecast_2 > 0 and forecast_3 < 1 and forecast_2 < 1 and forecast < 1 and not self.position.is_long:
    #   self.buy(size=self.size, tp=long_tp, sl=long_sl)
    # elif d_forecast < 0 and d_forecast_2 < 0 and forecast_3 > 1 and forecast_2 > 1 and forecast > 1 and not self.position.is_short:
    #   self.sell(size=self.size, tp=short_tp, sl=short_sl)
      
    # Double
    # if forecast > 1 and forecast_2 > 1 and not self.position.is_long:
    #   self.buy(size=.2, tp=upper, sl=lower)
    # elif forecast < 1 and forecast_2 < 1 and not self.position.is_short:
    #   self.sell(size=.2, tp=lower, sl=upper)
    
    # Peak
    # if forecast_2 < forecast and forecast_2 < forecast_3 and not self.position.is_long:
    #   self.buy(size=.2, tp=upper, sl=lower)
    # elif forecast_2 > forecast and forecast_2 > forecast_3 and not self.position.is_short:
    #   self.sell(size=.2, tp=lower, sl=upper)
      
    # Peak 3
    # if forecast > forecast_2 and forecast > forecast_3 and forecast_2 < forecast_3 and not self.position.is_long:
    #   self.buy(size=.2, tp=upper, sl=lower)
    # elif forecast < forecast_2 and forecast < forecast_3 and forecast_2 > forecast_3 and not self.position.is_short:
    #   self.sell(size=.2, tp=lower, sl=upper)
      
    # Cross
    # if forecast == 1 and forecast_2 == -1:  # Up
    #   if self.position.is_short:  # Cancel short
    #     trade.sl = min(trade.sl, close)
    #   if not self.position.is_long:
    #     self.buy(size=.2, tp=upper, sl=lower)
    #   elif self.position.is_short:  # Cancel short
    #     trade.sl = min(trade.sl, close)
    # elif forecast == -1 and forecast_2 == 1:  # Down 
    #   if not self.position.is_short:
    #     self.sell(size=.2, tp=lower, sl=upper)
    #   elif self.position.is_long:  # Cancel long
    #     trade.sl = max(trade.sl, close)
    
    # Additionally, set aggressive stop-loss on trades that have been open 
    # for more than two days
    # for trade in self.trades:
    #   if current_time - trade.entry_time > pd.Timedelta('2 days'):
    #     if trade.is_long:
    #       trade.sl = max(trade.sl, low)
    #     else:
    #       trade.sl = min(trade.sl, high)
    
    
if __name__ == "__main__":
  # Data
  # model = 'influxdb-mid-rat-ema_vwap_base_10_1_1-mel_600-mpl_20-no_cat-no_group'    
  model = 'influxdb-mid-rat-ema_vwap_base_10_1_1-mel_20-mpl_20-no_cat-no_group'
  version = 4
  data_filename = f'{model}-{version}.csv'
  data_path = Path(data_filename)
  if not data_path.exists():
    start = '2023-03-19T12:00:00+00:00'
    stop = '2023-03-27T12:00:00+00:00'
    size = None
    quant = None
    batch_size = 3600e9
    bucket = 'ccf'
    exchange = 'binance'
    base = 'btc'
    quote = 'usdt'
    feature = None
    horizon = 20
    target = None
    verbose = True
    start, stop, size, quant = initialize_time(start, stop, size, quant)
    client_kwargs = {'verify_ssl': False}
    client = InfluxDB(client=client_kwargs)
    client_ = client.init_client(client.client)
    query_api = client.get_query_api(client_, client.query_api)
    df = client.read_prediction_dataframe(
      query_api=query_api, 
      batch_size=batch_size, 
      bucket=bucket, 
      start=start, 
      stop=stop,
      exchange=exchange,
      base=base, 
      quote=quote, 
      feature=feature, 
      quant=quant, 
      model=model, 
      version=version, 
      horizon=horizon, 
      target=target,
      verbose=verbose)
    client_.close()
    df = df[[x for x in ['last', 'quantile_0.5', 'value'] if x in df]]
    df.to_csv(data_path)
  else:
    df = pd.read_csv(data_path)

  # Preprocess
  if 'timestamp' in df:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
  print(df)
  unit_scale = 1e-3  # one unit = price*unit_scale
  cols = [x for x in ['last', 'quantile_0.5', 'value'] if x in df]
  # FIXME Filter < 2 due to database errors
  for c in cols:
    df[c] = df[c].mask(lambda x: x < 2, np.nan)
  df = df.fillna(method='pad')
  df[cols] *= unit_scale
  df['Open'] = df['High'] = df['Low'] = df['Close'] = df['last']          
  # df = df.head(36000)
  # df = df.tail(36000)
  # df = df.loc['2023-03-24T17:00:00+00:00':]
  print(df)

  # Backtest
  bt = Backtest(df, 
                SimpleStrategy, 
                cash=30,  # min ~ one unit
                # commission=0.001,  # 0.1%
                # commission=0.00075,  # 0.75%
                commission=1e-6,  # 1e-6 ~ spread
                # commission=0.0, 
                margin=1.0, 
                exclusive_orders=False)
  
  # Optimize
  do_optimize = True
  if do_optimize:
    ncols = 3  # 3
    plot_width = 1200  # 1200
    opt_stats, heatmap = bt.optimize(
      method='grid',
      max_tries=None,
      random_state=None,
      constraint=None,  # constraint=lambda p: p.sma1 < p.sma2
      maximize='Return [%]',  # SQN, Win Rate [%], Avg. Trade [%], Expectancy [%], Return [%]
      threshold=[
        0.0, 1e-6, 
        1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 
        1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 
        1e-3, 
        # 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,  
        # 1e-2, 
        # 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        # 1e-1, 
        # 5e-1
      ],
      price_delta=[
        # 1e-6, 1e-5, 
        # 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 
        1e-3, 
        2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,  
        1e-2, 2e-2, 
        # 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        1e-1, 
        # 5e-1, 
        1.0
      ],
      # num_forecasts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      #                  10, 11, 12, 13, 14, 15, 16, 
      #                  32],
      # num_d_forecasts = [0, 1, 2, 3, 4, 8, 16, 32],
      # # num_d_forecasts = [0, 1, 2, 3],
      # num_d2_forecasts = [0, 1, 2, 3, 4, 8, 16, 32],
      # num_d2_forecasts = [0, 1, 2, 3],
      # threshold=[0.0, 1e-6, 1e-5, 
      #            1e-4, 2e-4, 5e-4, 8e-4, 
      #            1e-3, 2e-3, 5e-3, 8e-3,
      #            1e-2],
      # price_delta=[1e-6, 1e-5, 
      #              1e-4, 2e-4, 4e-4, 8e-4,
      #              1e-4, 2e-3, 4e-3, 8e-3, 
      #              1e-2, 2e-2, 4e-2, 8e-2, 
      #              1e-1, 0.5, 1.0],
      # price_delta=[1e-3, 2e-3, 3e-3, 1e-2, 1e-1],
      # price_delta=[1e-2, 1e-1, 1.0],
      # sl_tp_ratio=[1.0, 2.0, 3.0],
      # sl_tp_ratio=[0.1, 0.2, 0.333, 0.5, 0.667,
      #              1.0, 
      #              1.5, 2.0, 3.0, 5.0, 10.0],
      # sl_tp_ratio=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
      return_heatmap=True,
      return_optimization=False)
    print(opt_stats)
    print(heatmap)
    opt_stats.to_json(data_path.with_stem(f'{data_path.stem}-opt').with_suffix('.json'))
    plot_heatmaps(heatmap, agg='max', ncols=ncols, plot_width=plot_width, open_browser=False,
                  filename=str(data_path.with_stem(f'{data_path.stem}-heatmap').with_suffix('.html')))
  
  # Run
  stats = bt.run()
  print(stats)
  stats.to_json(data_path.with_stem(f'{data_path.stem}-run').with_suffix('.json'))
  bt.plot(filename=str(data_path.with_stem(f'{data_path.stem}-run').with_suffix('.html')))
