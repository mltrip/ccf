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
import os

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
from plotly_resampler import FigureResampler, FigureWidgetResampler
import matplotlib.pyplot as plt
from tqdm import tqdm

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
  size = 1.0
  unit_size = 1.0

  def init(self):
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
  size = 1.0
  unit_size = 1.0
  
  def init(self):
    pass

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
  size = 1.0
  unit_size = 1.0
  horizon = 20

  def init(self):
    pass

  def next(self):
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

        
def init_backtest(bt_kwargs, strategy_kwargs):
  unit_size = strategy_kwargs.get('unit_size', 1.0)
  bt_kwargs['data']['Open'] *= unit_size
  bt_kwargs['data']['High'] *= unit_size
  bt_kwargs['data']['Low'] *= unit_size
  bt_kwargs['data']['Close'] *= unit_size
  bt = Backtest(**bt_kwargs)
  return bt
    
  
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

    
def plot_curves(df, prefix, group_column=None, layout_kwargs=None, tags=None, 
                plot_html=False, plot_png=True, y_column='Equity'):
  tags = [] if tags is None else tags
  layout_kwargs = {} if layout_kwargs is None else layout_kwargs
  print(len(df))
  if len(df) < 100000:
    fig = go.Figure()
  else:  # Broken Pipe error
    fig = FigureResampler(go.Figure())
  groups = df.groupby(group_column) if group_column is not None else [('all', df)]
  for n, g in groups:
    print(n)
    print(g)
    scatter_kwargs = {'x': g.index, 'y': g[y_column], 'name': n}
    scatter_kwargs['line_width'] = 1
    if 'total' in tags:
      if n.startswith('rl-'):
        if 'skip_rl' in tags:
          continue        
        if n.endswith('forward'):
          if 'skip_rl_forward' in tags:
            continue
          scatter_kwargs['line'] = {'dash': 'dot'}
          if not 'split_rl_forward' in tags:
            if '1w' in n:
              scatter_kwargs['line_color'] = colormap['white']
            elif '30m' in n:
              scatter_kwargs['line_color'] = colormap['yellow']
            else:
              scatter_kwargs['line_color'] = colormap['magenta']
        else:  # backtest
          if 'skip_rl_backtest' in tags:
            continue
          scatter_kwargs['line'] = {'dash': 'dash'}
          if not 'split_rl_backtest' in tags:
            if '1w' in n:
              scatter_kwargs['line_color'] = colormap['white']
            elif '30m' in n:
              scatter_kwargs['line_color'] = colormap['yellow']
            else:
              scatter_kwargs['line_color'] = colormap['magenta']
      elif n.startswith('influxdb-'):
        if 'skip_ts' in tags:
          continue
        if not 'split_ts' in tags:
          scatter_kwargs['line_color'] = colormap['green']
      elif n.startswith('ppo-'):
        if 'skip_ppo' in tags:
          continue
        if not 'split_ppo' in tags:
            if '1w' in n:
              scatter_kwargs['line_color'] = colormap['white']
            elif '30m' in n:
              scatter_kwargs['line_color'] = colormap['yellow']
            else:
              scatter_kwargs['line_color'] = colormap['magenta']
    print(scatter_kwargs)
    scatter = go.Scatter(**scatter_kwargs)
    fig.add_trace(scatter)
  fig.update_traces(mode='lines')
  fig.update_layout(**layout_kwargs)
  suffix = '-' + '-'.join(tags) if len(tags) > 0 else ''
  if plot_html:
    fig.write_html(f'{prefix}{suffix}.html')
  if plot_png:
    fig.write_image(f'{prefix}{suffix}.png')
    
  
def backtest_rl(rl_data_path, rl_curves_path, rl_dataset_kwargs, lob_df_t,
                plot_html=False, bt_kwargs=None, strategy_kwargs=None):
  bt_kwargs = {} if bt_kwargs is None else bt_kwargs
  strategy_kwargs = {} if strategy_kwargs is None else strategy_kwargs
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
    bt_df['Open'] = bt_df['High'] = bt_df['Low'] = bt_df['Close'] = bt_df['m_p']
    bt_kwargs['data'] = bt_df
    bt_kwargs['strategy'] = RL
    bt = init_backtest(bt_kwargs, strategy_kwargs)
    stats = bt.run(**strategy_kwargs)
    print(stats)
    rl_model_stats[model] = stats
    if plot_html:
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
  
  
def plot_histogram_windows(windows, filename, x_col='index', y_col='m_p',
                           figsize=(8,6), bins=100, cmap='viridis', kinds=None,
                           lines=None, plot_histogram=True, plot_lines=True, 
                           plot_html=True, plot_png=True, layout_kwargs=None, 
                           add_y_center_bin=True):
  layout_kwargs = {} if layout_kwargs is None else layout_kwargs
  lines = ['mean'] if lines is None else lines
  kinds = ['diff'] if kinds is None else kinds
  if len(windows) == 0:
    print('Warning! No windows')
    return
  for kind in kinds:
    points = []
    for w in windows:
      if x_col == 'index':
        xs = w.index.to_series().diff().dt.total_seconds().cumsum()
      else:
        xs = w[x_col].to_series().diff().dt.total_seconds().cumsum()
      xs[0] = 0
      y_center_i = len(w) // 2
      y_center = w[y_col].iloc[y_center_i]
      if kind == 'diff':
        ys = w[y_col] - y_center
      elif kind == 'ratio':
        ys = w[y_col] / y_center
      elif kind == 'relative':
        ys = w[y_col] / y_center - 1.0
      elif kind == 'lograt':
        ys = np.log(w[y_col] / y_center)
      # print(y_center_i, y_center)
      # print(xs)
      # print(ys)
      for x, y in zip(xs, ys):
        if np.isfinite(x) and np.isfinite(y):
          points.append([x, y])
    points = np.array(points)
    try:
      min_x, max_x = min(points[:,0]), max(points[:,0])
      min_y, max_y = min(points[:,1]), max(points[:,1])
      bins_x = bins
      if add_y_center_bin:  # Add "0" bin to y
        y_center = 1 if kind == 'ratio' else 0
        y_tol = 1e-9
        bins_y = np.concatenate([np.linspace(min_y, y_center - y_tol, bins//2), 
                                 np.linspace(y_center + y_tol, max_y, bins//2)], 
                                axis=0)
      else:
        bins_y = bins
      H, xedges, yedges = np.histogram2d(points[:,0], points[:,1], 
                                         bins=[bins_x, bins_y])
      # print(H.shape)
      # print(xedges)
      # print(yedges)
    except Exception as e:
      print(e)
      return
    # Hist
    if plot_histogram:
      H_norm = H / H.sum(axis=1, keepdims=True)
      plt.clf()
      plt.figure(figsize=figsize)
      X, Y = np.meshgrid(xedges, yedges)
      plt.pcolormesh(X, Y, H_norm.T, cmap=plt.cm.get_cmap(cmap))
      plt.savefig(f'{filename}-histogram-{kind}.png')
    # Lines
    if plot_lines:
      lines_ys = {x: [] for x in lines}
      for ys in H:
        vs = []
        for yi, y in enumerate(ys):
          vy0, vy1 = yedges[yi], yedges[yi+1]
          vy = 0.5*(vy0 + vy1)
          for _ in range(int(y)):
            vs.append(vy)
        for line in lines_ys.keys():
          if line.isalpha():
            lines_ys[line].append(getattr(np, line)(vs))
          else:  # quantile
            lines_ys[line].append(np.quantile(vs, float(line)))
      fig = go.Figure()
      # if len(df) < 100000:
      # else:  # Broken Pipe error
      #   fig = FigureResampler(go.Figure())
      for line, ys in lines_ys.items():
        y_center_i = len(ys) // 2
        xs = np.arange(len(ys)) - y_center_i 
        scatter_kwargs = {'x': xs, 'y': ys, 'name': line}
        scatter = go.Scatter(**scatter_kwargs)
        fig.add_trace(scatter)
      fig.update_traces(mode='lines')
      fig.update_layout(**layout_kwargs)
      if plot_html:
        fig.write_html(f'{filename}-lines-{kind}.html')
      if plot_png:
        fig.write_image(f'{filename}-lines-{kind}.png')  
  

def backtest_signature(
  actions_df, lob_df, prefix='-', window='T', g_col='strategy', a_col='action',
  plot_strategy=True, layout_kwargs=None, histogram_kwargs=None, resample_kwargs=None):
  histogram_kwargs = {} if histogram_kwargs is None else histogram_kwargs
  histogram_kwargs['layout_kwargs'] = layout_kwargs
  trades_strategy_stats = {}
  buys_total, sells_total = [], []
  lob_df = lob_df[~lob_df.index.duplicated(keep='first')]
  for strategy, actions_df_strategy in actions_df.groupby(g_col):
    print(strategy)
    actions_df_strategy = actions_df_strategy[~actions_df_strategy.index.duplicated(keep='first')]
    df = pd.concat([
      lob_df[['m_p', 'a_p_0', 'b_p_0']],
      actions_df_strategy[[a_col]]], 
      axis=1)
    df[['m_p', 'a_p_0', 'b_p_0']] = df[['m_p', 'a_p_0', 'b_p_0']].interpolate('time')
    if resample_kwargs is not None:
      df = df.resample(**resample_kwargs).last()
    buys, sells = [], []
    # df = df[2000000:]
    for w in tqdm(df.rolling(window=window), total=len(df)):
      center = len(w) // 2
      action = w.iloc[center][a_col]
      if action == 'buy' or action == 1:
        buys.append(w)
        buys_total.append(w)
      elif action == 'sell' or action == 2:
        sells.append(w)
        sells_total.append(w)
    print(f'buys {len(buys)}')
    print(f'sells {len(sells)}')
    if plot_strategy:
      plot_histogram_windows(buys, f'signature{prefix}{strategy}-buy', **histogram_kwargs)
      plot_histogram_windows(sells, f'signature{prefix}{strategy}-sell', **histogram_kwargs)
  print(f'buys_total {len(buys_total)}')
  print(f'sells_total {len(sells_total)}')
  plot_histogram_windows(buys_total, f'signature{prefix}buy_total', **histogram_kwargs)
  plot_histogram_windows(sells_total, f'signature{prefix}sell_total', **histogram_kwargs)

    
def backtest_trades(trades_data_path, trades_curves_path, trades_df_t, lob_df_t, 
                    plot_html=False, bt_kwargs=None, strategy_kwargs=None):
  bt_kwargs = {} if bt_kwargs is None else bt_kwargs
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
    bt_df['Open'] = bt_df['High'] = bt_df['Low'] = bt_df['Close'] = bt_df['m_p']
    bt_kwargs['data'] = bt_df
    bt_kwargs['strategy'] = Trades
    bt = init_backtest(bt_kwargs, strategy_kwargs)
    stats = bt.run(**strategy_kwargs)
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
    if plot_html:
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
    forward_curve['Equity'] += bt_kwargs.get('cash', 0)
    forward_curve['agent'] = f'{strategy}_forward'
    trades_curves.append(forward_curve)
  print('Stats')
  trades_df = pd.DataFrame(trades_strategy_metrics)
  trades_df.to_csv(trades_data_path)
  print('Curves')
  trades_df_curves = pd.concat(trades_curves)
  trades_df_curves.to_csv(trades_curves_path)
  return trades_df, trades_df_curves  
  

def backtest_ts(ts_data_path, ts_curves_path, ts_dataset_kwargs, lob_df_t,
                plot_html=False, bt_kwargs=None, strategy_kwargs=None):
  bt_kwargs = {} if bt_kwargs is None else bt_kwargs
  strategy_kwargs = {} if strategy_kwargs is None else strategy_kwargs
  ts_dataset = Dataset(**ts_dataset_kwargs)
  ts_ds_t, ts_ds_v, ts_df_t, ts_df_v = ts_dataset()
  # Shift predictions to the past
  for c in ts_df_t.columns:
    if 'last' in c or 'value' in c:
      print(ts_df_t[c])
      periods = int(c.split('-')[-1])  # metric-horizon (e.g value-15 or last-20)
      ts_df_t[c] = ts_df_t[c].shift(periods=-periods)
      print(ts_df_t[c])
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
    bt_df['Open'] = bt_df['High'] = bt_df['Low'] = bt_df['Close'] = bt_df['m_p']
    bt_kwargs['data'] = bt_df
    bt_kwargs['strategy'] = TS
    bt = init_backtest(bt_kwargs, strategy_kwargs)
    for horizon in [5, 10, 15, 20, None]:
      stats = bt.run(horizon=horizon, **strategy_kwargs)
      print(stats)
      ts_model_stats[f'{model}_{horizon}'] = stats
      if plot_html:
        bt.plot(filename=f'backtest_ts_{model}_{horizon}.html')
  # Save
  ts_model_metrics = []
  ts_curves = []
  for model, stats in ts_model_stats.items():
    metrics = {k: v for k, v in stats.items() 
               if k not in ['_strategy', '_equity_curve', '_trades']}
    metrics['agent'] = model
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
  start, stop, benchmark_dir,
  bt_kwargs, strategy_kwargs, 
  lob_dataset_kwargs, ts_dataset_kwargs, rl_dataset_kwargs, trades_dataset_kwargs,
  layout_kwargs, curves_layout_kwargs, signature_layout_kwargs, histogram_kwargs,
  lob_dataset_path='lob.csv', update_lob=False,
  update_trades_dataset=False, update_trades_backtest=False, update_trades_signature=False,
  plot_trades=False, plot_trades_curves=False, 
  trades_dataset_path='trades.csv', trades_signature_path='trades_signature.csv',
  trades_data_path='trades_data.csv', trades_curves_path='trades_curves.csv',
  trades_signature_kwargs=None, rl_signature_kwargs=None, ts_signature_kwargs=None,
  plot_rl=False, plot_rl_curves=False, 
  rl_data_path='rl_data.csv', rl_curves_path='rl_curves.csv',
  update_rl_dataset=False, update_rl_backtest=False, update_rl_signature=False, 
  rl_dataset_path='rl.csv', rl_signature_path='rl_signature.csv', 
  update_ts_dataset=False, update_ts_backtest=False, update_ts_signature=False,
  ts_dataset_path='ts.csv', ts_signature_path='ts_signature.csv', 
  plot_ts=False, plot_ts_curves=False, 
  ts_data_path='ts_data.csv', ts_curves_path='ts_curves.csv',
  plot_all=False, plot_all_curves=False, update_all=False,
  plot_backtest_html=False
):
  trades_signature_kwargs = {} if trades_signature_kwargs is None else trades_signature_kwargs
  rl_signature_kwargs = {} if rl_signature_kwargs is None else rl_signature_kwargs
  ts_signature_kwargs = {} if ts_signature_kwargs is None else ts_signature_kwargs
  lob_dataset_kwargs.setdefault('start', start)
  lob_dataset_kwargs.setdefault('stop', stop)
  ts_dataset_kwargs.setdefault('start', start)
  ts_dataset_kwargs.setdefault('stop', stop)
  rl_dataset_kwargs.setdefault('start', start)
  rl_dataset_kwargs.setdefault('stop', stop)
  trades_dataset_kwargs.setdefault('start', start)
  trades_dataset_kwargs.setdefault('stop', stop)
  benchmark_path = Path(benchmark_dir)
  benchmark_path.mkdir(parents=True, exist_ok=True)
  os.chdir(benchmark_path)
  # LOB
  lob_dataset_path = Path(lob_dataset_path)
  if update_lob or update_all:
    print(f'Reading lob from db')
    lob_dataset = Dataset(**lob_dataset_kwargs)
    lob_ds_t, lob_ds_v, lob_df_t, lob_df_v = lob_dataset()
    plot_curves(lob_df_t, prefix='m_p', group_column=None, 
                layout_kwargs=curves_layout_kwargs, tags=None, 
                plot_html=False, plot_png=True, y_column='m_p')
    lob_df_t.to_csv(lob_dataset_path)
  else:
    print(f'Reading lob from {lob_dataset_path}')
    lob_df_t = pd.read_csv(lob_dataset_path,
                           index_col='timestamp', 
                           parse_dates=['timestamp'])
  print(lob_df_t)
  # RL
  rl_dataset_path = Path(rl_dataset_path)
  rl_signature_path = Path(rl_signature_path)
  if update_all or update_rl_dataset:
    print(f'Reading rl from db')
    rl_dataset = Dataset(**rl_dataset_kwargs)
    rl_ds_t, rl_ds_v, rl_df_t, rl_df_v = rl_dataset()
    rl_df_t.to_csv(rl_dataset_path)
  elif rl_dataset_path.exists():
    print(f'Reading rl from {rl_dataset_path}')
    rl_df_t = pd.read_csv(rl_dataset_path,
                          index_col='timestamp', 
                          parse_dates=['timestamp'])
  else:
    rl_df_t = None
  print(rl_df_t)
  if update_rl_backtest or update_all:
    rl_df, rl_df_curves = backtest_rl(rl_data_path, rl_curves_path, 
                                      rl_dataset_kwargs, lob_df_t,
                                      plot_html=plot_backtest_html,
                                      bt_kwargs=bt_kwargs, 
                                      strategy_kwargs=strategy_kwargs)
  else:
    if plot_rl or plot_all:
      rl_df = pd.read_csv(rl_data_path)
    if plot_rl_curves or plot_all_curves:
      rl_df_curves = pd.read_csv(rl_curves_path)
      if 'Unnamed: 0' in rl_df_curves:
        rl_df_curves = rl_df_curves.rename(columns={'Unnamed: 0': 'timestamp'})
      rl_df_curves = rl_df_curves.set_index('timestamp')
  if update_rl_signature or update_all:
    print(f'Calculating rl signatures')
    rl_signature_kwargs['histogram_kwargs'] = histogram_kwargs
    rl_signature_kwargs['layout_kwargs'] = signature_layout_kwargs
    rl_signature_kwargs['g_col'] = 'model'
    rl_signature_kwargs['prefix'] = '-rl-'
    rl_df_sign = backtest_signature(rl_df_t, lob_df_t, **rl_signature_kwargs)
    # rl_df_sign.to_csv(trades_signature_path)
  else:
    print(f'Reading rl signatures from {rl_signature_path}')
    # rl_df_sign = pd.read_csv(trades_signature_path)
  # TRADES
  trades_dataset_path = Path(trades_dataset_path)
  trades_signature_path = Path(trades_signature_path)
  if update_all or update_trades_dataset:
    print(f'Reading trades from db')
    trades_dataset = Dataset(**trades_dataset_kwargs)
    trades_ds_t, trades_ds_v, trades_df_t, trades_df_v = trades_dataset()
    trades_df_t.to_csv(trades_dataset_path)
  else:
    print(f'Reading trades from {trades_dataset_path}')
    trades_df_t = pd.read_csv(trades_dataset_path,
                              index_col='timestamp', 
                              parse_dates=['timestamp'])
  print(trades_df_t)
  if update_trades_backtest or update_all:
    trades_df, trades_df_curves = backtest_trades(
      trades_data_path, trades_curves_path, 
      trades_df_t, lob_df_t, 
      plot_html=plot_backtest_html,
      bt_kwargs=bt_kwargs, strategy_kwargs=strategy_kwargs)
  else:
    if plot_trades or plot_all:
      trades_df = pd.read_csv(trades_data_path)
    if plot_trades_curves or plot_all_curves:
      trades_df_curves = pd.read_csv(trades_curves_path)
      if 'Unnamed: 0' in trades_df_curves:
        trades_df_curves = trades_df_curves.rename(columns={'Unnamed: 0': 'timestamp'})
      trades_df_curves = trades_df_curves.set_index('timestamp')
  if update_trades_signature or update_all:
    print(f'Calculating trades signatures')
    trades_signature_kwargs['histogram_kwargs'] = histogram_kwargs
    trades_signature_kwargs['layout_kwargs'] = signature_layout_kwargs
    trades_signature_kwargs['prefix'] = '-trades-'
    trades_df_sign = backtest_signature(trades_df_t, lob_df_t, **trades_signature_kwargs)
    # trades_df_sign.to_csv(trades_signature_path)
  else:
    print(f'Reading trades signatures from {trades_signature_path}')
    # trades_df_sign = pd.read_csv(trades_signature_path)
  # TS
  ts_dataset_path = Path(ts_dataset_path)
  ts_signature_path = Path(ts_signature_path)
  if update_all or update_ts_dataset:
    print(f'Reading ts from db')
    ts_dataset = Dataset(**ts_dataset_kwargs)
    ts_ds_t, ts_ds_v, ts_df_t, ts_df_v = ts_dataset()
    ts_df_t.to_csv(ts_dataset_path)
  else:
    print(f'Reading ts from {ts_dataset_path}')
    ts_df_t = pd.read_csv(ts_dataset_path,
                          index_col='timestamp', 
                          parse_dates=['timestamp'])
  if update_ts_backtest or update_all:
    ts_df, ts_df_curves = backtest_ts(ts_data_path, ts_curves_path, 
                                      ts_dataset_kwargs, lob_df_t,
                                      plot_html=plot_backtest_html,
                                      bt_kwargs=bt_kwargs, strategy_kwargs=strategy_kwargs)
  else:
    if plot_ts or plot_all:
      ts_df = pd.read_csv(ts_data_path)
    if plot_ts_curves or plot_all_curves:
      ts_df_curves = pd.read_csv(ts_curves_path)
      if 'Unnamed: 0' in ts_df_curves:
        ts_df_curves = ts_df_curves.rename(columns={'Unnamed: 0': 'timestamp'})
      ts_df_curves = ts_df_curves.set_index('timestamp')
  if update_ts_signature or update_all:
    print(f'Calculating ts signatures')
    ts_signature_kwargs['histogram_kwargs'] = histogram_kwargs
    ts_signature_kwargs['layout_kwargs'] = signature_layout_kwargs
    ts_signature_kwargs['g_col'] = 'model'
    horizons = []
    for c in ts_df_t.columns:
      if 'last' in c or 'value' in c:
        periods = int(c.split('-')[-1])  # metric-horizon (e.g value-15 or last-20)
        ts_df_t[c] = ts_df_t[c].shift(periods=-periods)
        horizons.append(periods)
    for h in horizons:
      ts_df_t[f'forecast-{h}'] = ts_df_t[f'value-{h}'] / ts_df_t[f'last-{h}'] - 1
      ts_df_t[f'action-{h}'] = (ts_df_t[f'forecast-{h}'] < 0).astype(int) + 1  # 1 buy, 2 sell
      ts_signature_kwargs['a_col'] = f'action-{h}'
      ts_signature_kwargs['prefix'] = f'-ts-action-{h}-'
      ts_df_sign = backtest_signature(ts_df_t, lob_df_t, **ts_signature_kwargs)
    # ts_df_sign.to_csv(ts_signature_path)
  else:
    print(f'Reading ts signatures from {ts_dataset_path}')
    # ts_df_sign = pd.read_csv(ts_signature_path)
  # PLOT
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
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs)
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 
                tags=['total', 'split_ts', 'split_rl', 'split_ppo', 'skip_rl_backtest'])
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 
                tags=['total', 'split_ts', 'skip_rl', 'skip_ppo'])
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 
                tags=['total', 'skip_ts', 'skip_ppo'])
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 
                tags=['total', 'split_ts', 'skip_rl_backtest'])
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 
                tags=['total', 'skip_ts', 'skip_rl_backtest'])
    plot_curves(total_df_curves, 'total_curves', 'agent', curves_layout_kwargs, 
                tags=['total', 'skip_ts', 'split_rl_forward', 'split_ppo', 'skip_rl_backtest'])
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  main(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()   
  