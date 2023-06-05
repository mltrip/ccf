from collections import deque
from copy import deepcopy
import concurrent.futures
from pprint import pprint
import random
from pathlib import Path
import sys
import threading
import time
import copy

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from backtesting import Backtest, Strategy
from sklearn import preprocessing
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ccf.create_dataset import Dataset


colormap = {
  'orange': "#ffab40",
  'green': "#a2fca2",
  'blue': "#4dd0e1",
  'yellow': "#eeff41",
  'magenta': "#ff90c6",
  'lightgrey': "#eeeeee",
  'grey': "#adadad",
  'darkgrey': "#303030",
  'darkestgrey': "#212121",
  'white': '#ffffff',
  'black': '#000000'
}


class Trades(Strategy):
    def init(self):
      self.size = 1.0
      self.base_cumsum = self.I(lambda x: x, self.data.base_cumsum)
      self.quote_cumsum = self.I(lambda x: x, self.data.quote_cumsum)

    def next(self):
      open, high, low, close = self.data.Open, self.data.High, self.data.Low, self.data.Close
      current_time = self.data.index[-1]
      action = self.data.action[-1]
      # print(current_time, action, close[-1], self.position)
      if action == 'buy':
        # assert not self.position.is_long
        if not self.position.is_long:
          self.buy(size=self.size)
        else:
          print(f'Warning: long buy {current_time}!')
      elif action == 'sell':
        # assert not self.position.is_short
        if not self.position.is_short:
          self.sell(size=self.size)
        else:
          print(f'Warning: short sell {current_time}!')  
          
          
class RL(Strategy):
    def init(self):
      self.size = 1.0

    def next(self):
      open, high, low, close = self.data.Open, self.data.High, self.data.Low, self.data.Close
      current_time = self.data.index[-1]
      action = self.data.action[-1]
      if action == 1:  # buy
        if not self.position.is_long:
          self.buy(size=self.size)
      elif action == 2:  # sell
        if not self.position.is_short:
          self.sell(size=self.size)

          
class TS(Strategy):
    horizon = 20
    
    def init(self):
      self.size = 1.0
      
    def next(self):
      current_time = self.data.index[-1]
      open, high, low, close = self.data.Open, self.data.High, self.data.Low, self.data.Close
      # Forecast
      if self.horizon is not None:
        last = self.data[f'last-{self.horizon}'][-1]
        value = self.data[f'value-{self.horizon}'][-1]
        forecast = value / last
        if forecast > 1:
          action = 'buy'
        elif forecast < 1:
          action = 'sell'
        else:
          action = 'hold'
      else:
        values, lasts = {}, {}
        for c in self.data.df.columns:
          if c.startswith('value'):
            _, h = c.split('-')
            values[h] = self.data[c][-1]
          if c.startswith('last'):
            _, h = c.split('-')
            lasts[h] = self.data[c][-1]
        forecasts = {}
        for h, v in values.items():
          l = lasts[h]
          forecasts[h] = v / l
        if all(x > 1 for x in forecasts.values()):
          action = 'buy'
        elif all(x < 1 for x in forecasts.values()):
          action = 'sell'
        else:
          action = 'hold'
      # Action
      if action == 'buy':
        if not self.position.is_long:
          self.buy(size=self.size)
      elif action == 'sell':
        if not self.position.is_short:
          self.sell(size=self.size)
          
    
def plot_bars(df, prefix, column, layout_kwargs):
  metrics = [x for x in df.columns if x not in [column]]
  for m in metrics:
    print(m)
    df2 = df.sort_values(by=m, ascending=False)
    if 'Duration' in m:
      df2[m] = pd.to_timedelta(df2[m]).dt.total_seconds()
    fig = px.bar(df2, 
                 x=m,
                 y=column, 
                 color=column, 
                 text_auto='.2f',
                 orientation='h',
                 title=f"{prefix} {m}")
    fig.update_layout(**layout_kwargs)
    fig.write_html(f'{prefix}_{m}.html')

    
def plot_curves(df, prefix, column, layout_kwargs, template='default', plot_html=False):
  fig = go.Figure()
  for n, g in df.groupby(column):
    print(n)
    print(g)
    scatter_kwargs = {'x': g.index, 'y': g['Equity'], 'name': n}
    scatter_kwargs['line_width'] = 1
    if template == 'default':
      pass
    elif 'color_by_type' in template:
      if n.startswith('rl-'):
        if n.endswith('forward'):
          scatter_kwargs['line'] = {'dash': 'dot'}
          if '1w' in n:
            scatter_kwargs['line_color'] = colormap['magenta']
          elif '30m' in n:
            scatter_kwargs['line_color'] = colormap['blue']
          else:
            scatter_kwargs['line_color'] = colormap['yellow']
        else:
          continue
          scatter_kwargs['line_color'] = colormap['lightgrey']
      elif n.startswith('influxdb-'):
        if 'skip_ts' in template:
          continue
        if not 'split_ts' in template:
          scatter_kwargs['line_color'] = colormap['green']
      elif n.startswith('ppo-'):
        if '1w' in n:
          scatter_kwargs['line_color'] = colormap['magenta']
        elif '30m' in n:
          scatter_kwargs['line_color'] = colormap['blue']
        else:
          scatter_kwargs['line_color'] = colormap['yellow']
    print(scatter_kwargs)
    scatter = go.Scatter(**scatter_kwargs)
    fig.add_trace(scatter)
  fig.update_traces(mode='lines')
  fig.update_layout(**layout_kwargs)
  if plot_html:
    fig.write_html(f'{prefix}_{template}.html')
  fig.write_image(f'{prefix}_{template}.png')
  
  
def backtest_rl(rl_data_path, rl_curves_path, rl_dataset_kwargs, lob_df_t, unit_size=1.0):
  rl_dataset = Dataset(**rl_dataset_kwargs)
  rl_ds_t, rl_ds_v, rl_df_t, rl_df_v = rl_dataset()
  # Backtest
  rl_model_stats = {}
  for model, rl_df_t_model in rl_df_t.groupby(['model']):
    if len(rl_df_t_model) == 0:
      continue
    print(model)
    bt_df = lob_df_t[['m_p', 'a_p_0', 'b_p_0']].copy() 
    rl_df_t_model = rl_df_t_model.loc[~rl_df_t_model.index.duplicated(keep='first')]
    bt_df = pd.concat([
      bt_df,
      rl_df_t_model[['action']]], axis=1)
    print(bt_df)
    bt_df[['m_p', 'a_p_0', 'b_p_0']] = bt_df[['m_p', 'a_p_0', 'b_p_0']].interpolate('pad').interpolate('bfill')
    print(bt_df)
    bt_df['Open'] = bt_df['High'] = bt_df['Low'] = bt_df['Close'] = bt_df['m_p']*unit_size
    bt = Backtest(bt_df, RL,
                  # commission=.002,
                  cash=12,  # min ~ one unit
                  # commission=1e-5, 
                  margin=1.0, 
                  exclusive_orders=False)
    stats = bt.run()
    print(stats)
    rl_model_stats[model] = stats
    bt.plot(filename=f'backtest_rl_{model}.html')
  # Save
  rl_model_metrics = []
  rl_curves = []
  for model, stats in rl_model_stats.items():
    metrics = {k: v for k, v in stats.items() if k not in ['_strategy', '_equity_curve', '_trades']}
    metrics['agent'] = model
    print(model)
    print(metrics)
    rl_model_metrics.append(metrics)
    # Back curves
    curve = stats['_equity_curve']
    curve['agent'] = model
    rl_curves.append(curve)
  print('RL stats')
  rl_df = pd.DataFrame(rl_model_metrics)
  rl_df.to_csv(rl_data_path)
  print('RL curves')
  rl_df_curves = pd.concat(rl_curves)
  rl_df_curves.to_csv(rl_curves_path)
  return rl_df, rl_df_curves
  

def backtest_trades(trades_data_path, trades_curves_path, trades_dataset_kwargs, lob_df_t, unit_size=1.0):
  trades_dataset = Dataset(**trades_dataset_kwargs)
  trades_ds_t, trades_ds_v, trades_df_t, trades_df_v = trades_dataset()
  print(trades_df_t)
  # print(trades_df_t.describe())
  # Backtest
  trades_strategy_stats = {}
  for strategy, trades_df_t_strategy in trades_df_t.groupby(['strategy']):
    bt_df = lob_df_t[['m_p', 'a_p_0', 'b_p_0']].copy()    
    print(strategy)
    n_trades = trades_df_t_strategy['quote_delta'].count()
    if n_trades > 0:
      if trades_df_t_strategy.head(1)['quote_delta'].item() != 0:
        trades_df_t_strategy['quote_delta'].iloc[0] = 0
    win_trades = sum(trades_df_t_strategy['quote_delta'] > 0)
    win_rate = win_trades / n_trades if n_trades > 0 else None
    mean_quote_delta = trades_df_t_strategy['quote_delta'].mean()
    mean_base_delta = trades_df_t_strategy['base_delta'].mean()
    trades_df_t_strategy['quote_cumsum'] = trades_df_t_strategy['quote_delta'].cumsum()
    trades_df_t_strategy['base_cumsum'] = trades_df_t_strategy['base_delta'].cumsum()
    print(trades_df_t_strategy.isna().sum())
    print(trades_df_t_strategy.count())
    bt_df = pd.concat([
      bt_df,
      trades_df_t_strategy[['action', 'base_cumsum', 'quote_cumsum']]], axis=1)
    print(bt_df)
    bt_df[['m_p', 'a_p_0', 'b_p_0', 'base_cumsum', 'quote_cumsum']] = bt_df[['m_p', 'a_p_0', 'b_p_0', 'base_cumsum', 'quote_cumsum']].interpolate('pad').interpolate('bfill')
    # print(bt_df)
    # print(bt_df.describe())
    # print(zeros['quote_delta'])
    print(bt_df.isna().sum())
    print(bt_df.count())
    bt_df['Open'] = bt_df['High'] = bt_df['Low'] = bt_df['Close'] = bt_df['m_p']*unit_size
    cash = 12
    bt = Backtest(bt_df, Trades,
                  # commission=.002,
                  cash=cash,  # min ~ one unit
                  # commission=1e-5, 
                  margin=1.0, 
                  exclusive_orders=False)
    stats = bt.run()
    print(stats)
    # quote_balances 
    stats['last_quote_cumsum'] = bt_df.tail(1)['quote_cumsum'].item()
    stats['last_base_cumsum'] = bt_df.tail(1)['base_cumsum'].item()
    stats['max_quote_cumsum'] = bt_df['quote_cumsum'].max().item()
    stats['min_quote_cumsum'] = bt_df['quote_cumsum'].min().item()
    stats['max_base_cumsum'] = bt_df['base_cumsum'].max().item()
    stats['min_base_cumsum'] = bt_df['base_cumsum'].min().item()
    stats['mean_base_delta'] = mean_base_delta
    stats['mean_quote_delta'] = mean_quote_delta
    stats['n_trades'] = n_trades
    stats['win_rate'] = win_rate
    stats['quote_cumsum'] = bt_df[['quote_cumsum']]
    trades_strategy_stats[strategy] = stats
    bt.plot(filename=f'backtest_trades_{strategy}.html')
  # Save
  trades_strategy_metrics = []
  trades_curves = [] 
  for strategy, stats in trades_strategy_stats.items():
    metrics = {k: v for k, v in stats.items() if k not in ['_strategy', '_equity_curve', '_trades', 'quote_cumsum']}
    metrics['agent'] = strategy
    print(strategy)
    print(metrics)
    trades_strategy_metrics.append(metrics)
    # Back curves
    curve = stats['_equity_curve']
    curve['agent'] = strategy
    trades_curves.append(curve)
    # Forward curves
    forward_curve = stats['quote_cumsum']
    forward_curve = forward_curve.rename(columns={"quote_cumsum": "Equity"})
    forward_curve['Equity'] += cash
    forward_curve['agent'] = f'{strategy}_forward'
    trades_curves.append(forward_curve)
  print('Stats')
  trades_df = pd.DataFrame(trades_strategy_metrics)
  trades_df.to_csv(trades_data_path)
  print('Curves')
  trades_df_curves = pd.concat(trades_curves)
  trades_df_curves.to_csv(trades_curves_path)
  return trades_df, trades_df_curves  
  

def backtest_ts(ts_data_path, ts_curves_path, ts_dataset_kwargs, lob_df_t, unit_size=1.0):
  ts_dataset = Dataset(**ts_dataset_kwargs)
  ts_ds_t, ts_ds_v, ts_df_t, ts_df_v = ts_dataset()
  # Backtest
  ts_model_stats = {}
  for model, ts_df_t_model in ts_df_t.groupby(['model']):
    if len(ts_df_t_model) == 0:
      continue
    print(model)
    bt_df = lob_df_t[['m_p', 'a_p_0', 'b_p_0']].copy() 
    ts_df_t_model = ts_df_t_model.loc[~ts_df_t_model.index.duplicated(keep='first')]
    bt_df = pd.concat([bt_df, ts_df_t_model], axis=1)
    print(bt_df)
    bt_df[['m_p', 'a_p_0', 'b_p_0']] = bt_df[['m_p', 'a_p_0', 'b_p_0']].interpolate('pad').interpolate('bfill')
    print(bt_df)
    bt_df['Open'] = bt_df['High'] = bt_df['Low'] = bt_df['Close'] = bt_df['m_p']*unit_size
    bt = Backtest(bt_df, TS,
                  # commission=.002,
                  cash=12,  # min ~ one unit
                  # commission=1e-5, 
                  margin=1.0, 
                  exclusive_orders=False)
    for horizon in [5, 10, 15, 20, None]:
      stats = bt.run(horizon=horizon)
      print(stats)
      ts_model_stats[f'{model}_{horizon}'] = stats
      bt.plot(filename=f'backtest_ts_{model}_{horizon}.html')
  # Save
  ts_model_metrics = []
  ts_curves = []
  for model, stats in ts_model_stats.items():
    metrics = {k: v for k, v in stats.items() 
               if k not in ['_strategy', '_equity_curve', '_trades']}
    metrics['agent'] = model
    # print(model)
    # print(metrics)
    ts_model_metrics.append(metrics)
    # Back curves
    curve = stats['_equity_curve']
    curve['agent'] = model
    ts_curves.append(curve)
  print('TS stats')
  ts_df = pd.DataFrame(ts_model_metrics)
  ts_df.to_csv(ts_data_path)
  print('TS curves')
  ts_df_curves = pd.concat(ts_curves)
  ts_df_curves.to_csv(ts_curves_path)
  return ts_df, ts_df_curves 
  
  
def main(
  hydra_config, 
  start, stop, 
  bt_kwargs, lob_dataset_kwargs, 
  ts_dataset_kwargs, rl_dataset_kwargs, trades_dataset_kwargs,
  layout_kwargs, curves_layout_kwargs,
  update_trades=False, plot_trades=False, plot_trades_curves=False, 
  trades_data_path='trades_data.csv', trades_curves_path='trades_curves.csv',
  update_rl=False, plot_rl=False, plot_rl_curves=False, 
  rl_data_path='rl_data.csv', rl_curves_path='rl_curves.csv',
  update_ts=False, plot_ts=False, plot_ts_curves=False, 
  ts_data_path='ts_data.csv', ts_curves_path='ts_curves.csv',
  plot_all=False, plot_all_curves=False
):
  lob_dataset_kwargs.setdefault('start', start)
  lob_dataset_kwargs.setdefault('stop', stop)
  ts_dataset_kwargs.setdefault('start', start)
  ts_dataset_kwargs.setdefault('stop', stop)
  rl_dataset_kwargs.setdefault('start', start)
  rl_dataset_kwargs.setdefault('stop', stop)
  trades_dataset_kwargs.setdefault('start', start)
  trades_dataset_kwargs.setdefault('stop', stop)
  # LOB
  if any([update_trades, update_rl, update_ts]):
    lob_dataset = Dataset(**lob_dataset_kwargs)
    lob_ds_t, lob_ds_v, lob_df_t, lob_df_v = lob_dataset()
    print(lob_df_t)
  # RL
  if not update_rl:
    if plot_rl or plot_all:
      rl_df = pd.read_csv(rl_data_path)
    if plot_rl_curves or plot_all_curves:
      rl_df_curves = pd.read_csv(rl_curves_path)
      if 'Unnamed: 0' in rl_df_curves:
        rl_df_curves = rl_df_curves.rename(columns={'Unnamed: 0': 'timestamp'})
      rl_df_curves = rl_df_curves.set_index('timestamp')
  else:
    rl_df, rl_df_curves = backtest_rl(rl_data_path, rl_curves_path, 
                                      rl_dataset_kwargs, lob_df_t, unit_size=4e-4)
  # Trades
  if not update_trades:
    if plot_trades or plot_all:
      trades_df = pd.read_csv(trades_data_path)
    if plot_trades_curves or plot_all_curves:
      trades_df_curves = pd.read_csv(trades_curves_path)
      if 'Unnamed: 0' in trades_df_curves:
        trades_df_curves = trades_df_curves.rename(columns={'Unnamed: 0': 'timestamp'})
      trades_df_curves = trades_df_curves.set_index('timestamp')
  else:
    trades_df, trades_df_curves = backtest_trades(trades_data_path, trades_curves_path, 
                                                  trades_dataset_kwargs, lob_df_t, unit_size=4e-4)
  # TS
  if not update_ts:
    if plot_ts or plot_all:
      trades_df = pd.read_csv(ts_data_path)
    if plot_ts_curves or plot_all_curves:
      ts_df_curves = pd.read_csv(ts_curves_path)
      if 'Unnamed: 0' in ts_df_curves:
        ts_df_curves = ts_df_curves.rename(columns={'Unnamed: 0': 'timestamp'})
      ts_df_curves = ts_df_curves.set_index('timestamp')
  else:
    ts_df, ts_df_curves = backtest_ts(ts_data_path, ts_curves_path, 
                                      ts_dataset_kwargs, lob_df_t, unit_size=4e-4)
  if plot_ts:
    plot_bars(ts_df, 'ts', 'agent', layout_kwargs)
  if plot_ts_curves:
    plot_curves(ts_df_curves, 'ts_curves', 'agent', curves_layout_kwargs)
  if plot_trades:
    plot_bars(trades_df, 'trades', 'agent', layout_kwargs)
  if plot_trades_curves:
    plot_curves(trades_df_curves, 'trades_curves', 'agent', curves_layout_kwargs)
  if plot_rl:
    plot_bars(rl_df, 'rl', 'agent', layout_kwargs)
  if plot_rl_curves:
    plot_curves(rl_df_curves, 'rl_curves', 'agent', curves_layout_kwargs)
  if plot_all:
    total_df = pd.concat([trades_df, rl_df, ts_df])
    plot_bars(total_df, 'total', 'agent', layout_kwargs)
  if plot_all_curves:
    total_df_curves = pd.concat([trades_df_curves, rl_df_curves, ts_df_curves])
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 'default')
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 'color_by_type')
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 'color_by_type_split_ts')
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 'color_by_type_skip_ts')
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  hydra_cfg = HydraConfig.get()
  main(hydra_config=hydra_cfg, **OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()   
  