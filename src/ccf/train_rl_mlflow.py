from copy import deepcopy
from pprint import pprint
import random
from pathlib import Path
import sys
import threading
import time
import copy
from threading import (Event, Thread)
from typing import Union, Callable, Dict, Tuple, Any
from abc import *

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import gym
from tqdm import tqdm
from backtesting import Backtest, Strategy
import gym
from gym import spaces
from gym.core import Env
from sklearn import preprocessing
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient

from ccf.agents import InfluxDB
from ccf.utils import initialize_time
from ccf.model_mlflow import CCFRLModel, load_model


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class __FULL_EQUITY(float):
    def __repr__(self): return '.9999'
_FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)


class TradingStrategy(Strategy):
    window_size: int
    callback: Callable
    
    def init(self):
        pass

    def next(self):
        self.callback(self)

        
class BacktestingThread(Thread):
    def __init__(self, 
            data, 
            window_size,
            cash: float = 10_000,
            commission: float = .0,
            margin: float = 1.,
            trade_on_close=False,
            hedging=False,
            exclusive_orders=False
        ):
        # print('__init__')
        threading.Thread.__init__(self)
        self.step_event = Event()  # False
        self.callback_event = Event()  # False
        self.result_event = Event()  # False
        TradingStrategy.callback = self._callback
        TradingStrategy.window_size = window_size
        self.kill_flag = False
        self.bt = Backtest(
            data, 
            TradingStrategy,
            cash=cash,
            commission=commission,
            margin=margin,
            trade_on_close=trade_on_close,
            hedging=hedging,
            exclusive_orders=exclusive_orders,
        )
        self.strategy = None
        self.result = None
        self.window_size = window_size
        
    def run(self):
        """Run (i.e. invoke _callback) then start result"""
        # print('run')
        time.sleep(0.01) # for sync first data
        self.result = self.bt.run()
        self.result_event.set()  # Result to True (Stop blocking)

    def kill(self):
        """Start step, start callback"""
        # print('kill')
        self.kill_flag = True
        self.step_event.set()
        self.callback_event.set()   
        
    def get_strategy(self):
        """Start step, wait callback and return strategy from callback"""
        # print('get_strategy')
        if not self.kill_flag:
            self.step_event.set()
            self.callback_event.wait()
            self.callback_event.clear()
        return self.strategy

    def _callback(self, strategy: Strategy):
        """Get strategy from TradingStrategy, start callback and wait step"""
        # print(f'_callback')
        self.strategy = strategy
        if self.kill_flag:
            sys.exit(0)
        # print(f'len(self.strategy.data): {len(self.strategy.data)}, self.window_size: {self.window_size}')
        if len(self.strategy.data) >= self.window_size:  # Start then window is accumulated
            self.callback_event.set()
            self.step_event.wait()
            self.step_event.clear()

    def stats(self):
        """Kill then wait result if result is not start"""
        # print('stats')
        self.kill()
        if not self.result_event.is_set():  # If result is False
            self.result_event.wait()
        return self.result

    def plot(self,
        results: pd.Series = None,
        filename=None,
        plot_width=None,
        plot_equity=True,
        plot_return=False,
        plot_pl=True,
        plot_volume=True,
        plot_drawdown=False,
        smooth_equity=False,
        relative_equity=True,
        superimpose: Union[bool,str] = True,
        resample=True,
        reverse_indicators=False,
        show_legend=True,
        open_browser=True
    ):
        # print('plot')
        self.bt.plot(
            results=results,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
        )
        
        
class EnvParameter:
    def __init__(self, 
        df: pd.DataFrame, 
        window_size: int,
        mode: str = "sequential", 
        add_feature: bool = False,
        step_length: int = 1,
        cash: float = 10_000,
        commission: float = .0,
        margin: float = 1.,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False
    ):
        self.df = df
        self.add_feature = add_feature
        self.window_size = window_size
        self.step_length = step_length
        self.mode = mode  # "sequential": sequential episode, "random": episode start is random, "backtest": for eval
        self.cash = cash
        self.commission = commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.hedging = hedging
        self.exclusive_orders = exclusive_orders
        self._check_param()

    def _check_param(self):
        self._check_column()
        self._check_length()
        self._check_mode()

    def _check_column(self):
        # column check
        if not all([item in self.df.columns for item in ['Open', 'High', 'Low', 'Close']]):
            raise RuntimeError(("Required column is not exist."))

    def _check_length(self):
        if self.window_size > len(self.df):
            raise RuntimeError("df length is not enough.")

    def _check_mode(self):
        if self.mode not in ["sequential", "random"]:
            raise RuntimeError(("Parameter mode is invalid. You should be 'random' or 'sequential'"))

            
class TradingEnv(gym.Env):
    def __init__(self, param: EnvParameter):
        self.param = param
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL
        self.factory = EpisodeFactory(self.param)
        self.episode = None
        self.observation_space = spaces.Box(
          low=-np.inf, high=np.inf, shape=(self.param.window_size, 2), dtype=np.float64)
        

    def step(self, action, size: float = _FULL_EQUITY,
            limit: float = None,
            stop: float = None,
            sl: float = None,
            tp: float = None):
        if self.episode == None:
            print("NotEpisodeInitError: Please run env.reset() at first.")
            sys.exit(1)

        if action == 1:
            # print('BUY')
            self.episode.strategy.buy(
                size=size,
                limit=limit,
                stop=stop,
                sl=sl,
                tp=tp,
            )
        elif action == 2:
            # print('SELL')
            self.episode.strategy.sell(
                size=size,
                limit=limit,
                stop=stop,
                sl=sl,
                tp=tp,
            )
        
        self.episode.forward()
        obs, reward, done, info = self.episode.status()

        return obs, reward, done, info

    def reset(self):
        if self.episode != None:
            self.episode.clear()
        # self.factory = EpisodeFactory(self.param)
        self.episode = self.factory.create()
        obs, _, _, _ = self.episode.status()
        return obs

    def stats(self):
        result = self.episode.bt.stats()
        return result

    def plot(self,
        results: pd.Series = None,
        filename=None,
        plot_width=None,
        plot_equity=True,
        plot_return=False,
        plot_pl=True,
        plot_volume=True,
        plot_drawdown=False,
        smooth_equity=False,
        relative_equity=True,
        superimpose: Union[bool,str] = True,
        resample=True,
        reverse_indicators=False,
        show_legend=True,
        open_browser=True
    ):
        self.episode.bt.plot(
            results=results,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
        )


class Preprocessor:
  def __init__(self, scalers, window_size):
    self.scalers = {}
    for column, scaler in scalers.items():
      scaler_class = scaler.pop('class')
      self.scalers[column] = getattr(preprocessing, scaler_class)(**scaler)
    self.window_size = window_size
    
  def __call__(self, df):
    columns = list(self.scalers.keys())
    df = df[columns][-self.window_size:]
    df = df.sort_index(axis=1)
    vs = []
    for column, scaler in self.scalers.items():
      v = df[column].values
      v_ = scaler.fit_transform(v)
      vs.append(v_)
    vs = np.column_stack(vs)
    return vs
  
        
def preprocess_data(df, window_size, forecast_column='Forecast', price_column='Close'):
  cols = [forecast_column, price_column]
  df2 = df[cols][-window_size:]
  df2[forecast_column] = df2[forecast_column] / df2[price_column]
  x = df2.values  # returns a numpy array
  scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
  x_scaled = scaler.fit_transform(x)
  return x_scaled
        
        
class Episode:
    def __init__(self, episode_data, param: EnvParameter):
        if param.add_feature:
            self.features = pd.DataFrame({"Reward": np.zeros(len(episode_data)), "Position": np.zeros(len(episode_data))}, index=episode_data.index)
        self.episode_data = episode_data
        self.episode_step = 0
        self.timestamp = len(param.df) - len(episode_data)  # index of all data
        self.param = param
        self.finished = False
        self.bt = BacktestingThread(
            self.episode_data, 
            self.param.window_size,
            cash=self.param.cash,
            commission=self.param.commission,
            margin=self.param.margin,
            trade_on_close=self.param.trade_on_close,
            hedging=self.param.hedging,
            exclusive_orders=self.param.exclusive_orders,
        )
        self.bt.daemon = True
        self.bt.start()
        self.strategy = self.bt.get_strategy()

    def forward(self):
        if not self.finished:
            self.episode_step += 1
            self.timestamp += 1
            # print(self.episode_step, self.timestamp, len(self.episode_data), len(self.strategy.data.df))
            self.strategy = self.bt.get_strategy()
            self.finished = True if len(self.strategy.data.df) >= len(self.episode_data) else False
            
    def status(self):
        # calc info
        obs = self.observation()
        info = self.info()
        reward = self.reward()
        done = self.done()
        # add feature to obs
        # if self.param.add_feature:
        #     self.features.at[self.strategy.data.df.index[-1], "Reward"] = reward
        #     self.features.at[self.strategy.data.df.index[-1], "Position"] = info["position"].size
        #     obs = pd.merge(obs, self.features[:len(self.strategy.data.df)], left_index=True, right_index=True)
        return obs, reward, done, info

    def clear(self):
        self.bt.kill()
        self.bt.join()

    def observation(self):
        df = self.strategy.data.df
        window_size = self.param.window_size
        return preprocess_data(df, window_size)

    def reward(self):
        # sum of profit percentage
        # TODO Short term reward?
        # pprint(self.strategy.trades)
        # pprint([trade.pl_pct for trade in self.strategy.trades])
        # TODO Long term reward?
        # pprint(self.strategy.closed_trades)
        # pprint([trade.pl_pct for trade in self.strategy.closed_trades])
        return sum([trade.pl_pct for trade in self.strategy.trades])

    def done(self):
        return self.finished

    def info(self):
        return {
            "date": self.strategy.data.df.index[-1],
            "episode_step": self.episode_step,
            "timestamp": self.timestamp,
            "orders": self.strategy.orders, 
            "trades": self.strategy.trades, 
            "position": self.strategy.position, 
            "closed_trades": self.strategy.closed_trades, 
        }

      
class EpisodeFactory:
    def __init__(self, param: EnvParameter):
        self.param = param
        self.episode = None

    def create(self):
        next_timestamp = self.get_next_episode_timestamp()
        data = self.get_next_episode_data(next_timestamp)
        self.episode = Episode(data, self.param)
        return self.episode

    def get_next_episode_data(self, timestamp):
        return self.param.df[timestamp:]

    def get_next_episode_timestamp(self):
        if self.param.mode == "random": 
            return random.choice(range(len(self.param.df) - 2*self.param.window_size))  # with margin
        else:  # "sequential" or "backtest"
            if self.episode == None or self.episode.timestamp + 2*self.param.window_size > len(self.episode.episode_data):
                return 0
            else:
                return self.episode.timestamp + 1

              
class OneEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter, stop_on_none=False):
        env = TradingEnv(param)
        super().__init__(env)
        self.env = env
        self.side = 'NONE'
        self.stop_on_none = stop_on_none

    def step(self, 
             action,  # 0: HOLD, 1: BUY, 2: SELL
             size: float = _FULL_EQUITY,
             limit: float = None,
             stop: float = None,
             sl: float = None,
             tp: float = None):
        
        if self.side == "LONG" and action == 1:
            action = 0  # HOLD
        if self.side == "SHORT" and action == 2:
            action = 0  # HOLD

        obs, reward, done, info = self.env.step(action, size=1)
        
        if self.stop_on_none:
          if self.side == "LONG" and action == 2 or self.side == "SHORT" and action == 1:
              done = True
             
        if info["position"].size == 0:
            self.side = 'NONE'
        elif info["position"].size > 0:
            self.side = "LONG"
        elif info["position"].size < 0:
            self.side = "SHORT"

        return obs, reward, done, info
    
    
def run_backtest(env, model, filename):
  print('Start')
  obs = env.reset()
  done = False
  pbar = tqdm(total=len(env.episode.episode_data))
  cum_reward = 0.0
  while not done:
    action, _states = model.predict(obs)
    next_obs, reward, done, info = env.step(action, size=1)
    obs = next_obs
    cum_reward += reward
    pbar.set_description(f"reward: {reward:+.6f}%, cum_reward: {cum_reward:+.6f}%")
    pbar.update(1)
  pbar.close()
  print('Done\n')
  stats = env.stats()
  print(stats)
  env.plot(filename=filename, open_browser=False)    
    

def get_data(start, exchange, base, quote,
             model_name, model_version, unit_scale, 
             reload=False, test_size=None, do_dump=True,
             stop=None, horizon=None, feature=None, target=None, bucket='ccf', 
             batch_size=3600e9, verbose=True, quant=None, size=None):
  start, stop, size, quant = initialize_time(start, stop, size, quant)
  data_filename = f'{model_name}-{model_version}-{start}-{stop}.csv'
  data_path = Path(data_filename)
  if not data_path.exists() or reload:
    print('Loading data from DB')
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
      model=model_name, 
      version=model_version, 
      horizon=horizon, 
      target=target,
      verbose=verbose)
    client_.close()
    df = df[[x for x in ['last', 'quantile_0.5', 'value'] if x in df]]
    if do_dump:
      df.to_csv(data_path)
  else:
    print('Loading data from file')
    df = pd.read_csv(data_path)
  # Preprocess
  print(df)
  if 'timestamp' in df:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
  cols = [x for x in ['last', 'quantile_0.5', 'value'] if x in df]
  # FIXME Filter < 2 due to database errors
  for c in cols:
    df[c] = df[c].mask(lambda x: x < 2, np.nan)
  df = df.fillna(method='pad')
  df[cols] *= unit_scale
  df['Open'] = df['High'] = df['Low'] = df['Close'] = df['last']
  df = df.drop(columns=['last'])
  df = df.rename(columns={'quantile_0.5': 'Forecast', 'value': 'Forecast'})
  if test_size is not None:
    df = df.head(test_size)
  print(df)
  return df
    
  
def train(hydra_config, 
          model_name, model_version=None, model_stage=None,
          data_kwargs=None, model_kwargs=None, learn_kwargs=None, env_kwargs=None, 
          split=0.75, n_envs=4, seed=None, verbose=0,
          is_tune=False, parent_name=None, parent_version=None, parent_stage=None,
          do_train=True, do_backtest=True):
  data_kwargs = {} if data_kwargs is None else data_kwargs
  env_kwargs = {} if env_kwargs is None else env_kwargs
  model_kwargs = {} if model_kwargs is None else model_kwargs
  learn_kwargs = {} if learn_kwargs is None else learn_kwargs
  mlflow_params = {**{f'data-{k}': v for k, v in data_kwargs.items()},
                   **{f'env-{k}': v for k, v in env_kwargs.items()},
                   **{f'model-{k}': v for k, v in model_kwargs.items()},
                   **{f'learn-{k}': v for k, v in learn_kwargs.items()},
                   **{'split': split, 'n_envs': n_envs, 'seed': seed,
                      'model_name': model_name, 
                      'model_version': model_version,
                      'model_stage': model_stage,
                      'parent_name': parent_name, 
                      'parent_version': parent_version,
                      'parent_stage': parent_stage,
                      'is_tune': is_tune, 
                      'do_train': do_train,
                      'do_backtest': do_backtest}}
  # Model
  model_class_name = model_kwargs.pop('class')
  model_class = getattr(stable_baselines3, model_class_name)
  
  # Get data
  print('Get data')
  df = get_data(**data_kwargs)
  
  # Split dataset
  print('Split data')
  if split is not None:
    split_idx = int(len(df)*split)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    print(f'train from {df_train.index.min()} to {df_train.index.max()} with {len(df_train)} rows and {len(df_train.columns)} columns')
    print(f'test from {df_test.index.min()} to {df_test.index.max()} with {len(df_test)} rows and {len(df_test.columns)} columns')
  else:
    df_train = df
    print(f'train from {df_train.index.min()} to {df_train.index.max()} with {len(df_train)} rows and {len(df_train.columns)} columns')
    df_test = None
  
  # Train
  print('Train')
  if do_train:
    print('Create environment')
    train_env_kwargs = deepcopy(env_kwargs)
    train_env_kwargs['df'] = df_train
    train_env_kwargs['mode'] = 'random'
    train_param = EnvParameter(**train_env_kwargs)
    if n_envs == 1:
      print('Create one environment')
      train_env = OneEnv(train_param)
      check_env(train_env)
    else:
      print('Create many environments')
      train_env_kwargs = {'param': train_param}
      train_env = make_vec_env(OneEnv, env_kwargs=train_env_kwargs,
                               n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    print('Create loggers')
    loggers = Logger(folder=None,
                     output_formats=[HumanOutputFormat(sys.stdout), 
                                     MLflowOutputFormat()])
    experiment_name = model_name
    print('Create experiment')
    experiment = mlflow.set_experiment(experiment_name)
    print('Start experiment')
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
      print('Log parameters')
      mlflow.log_params(mlflow_params)
      if not is_tune:
        print('Create model')
        model_kwargs['env'] = train_env
        model_kwargs['verbose'] = verbose 
        model = model_class(**model_kwargs)
      else:
        print('Load model')
        last_model, last_version, last_stage = load_model(parent_name, parent_version, parent_stage)
        if last_model is not None:
          # model = model_class.load(path=model_name, env=train_env)
          model = last_model.unwrap_python_model().model
          model.set_env(train_env)
        else:
          print('Model is not found: creating new model')
          model_kwargs['env'] = train_env
          model_kwargs['verbose'] = verbose 
          model = model_class(**model_kwargs)
      model.set_logger(loggers)
      print('Start train')  
      model.learn(**learn_kwargs)
      print('Save model')
      model.save(model_name)
      cwd = Path(hydra_config.runtime.cwd)
      conf_path = cwd / 'conf'
      config_name = hydra_config.job.config_name
      model_path = Path(f'{model_name}.zip')
      mlflow_model = CCFRLModel(config_name=config_name)
      print('Log model')
      model_info = mlflow.pyfunc.log_model(artifact_path=model_name, 
                                           registered_model_name=model_name,
                                           python_model=mlflow_model,
                                           artifacts={'conf': str(conf_path), 
                                                      'model': str(model_path)})
  # Backtest
  if do_backtest:
    print('Backtesting')
    print('Loading model')
    # model = model_class.load(model_name) 
    last_model, last_version, last_stage = load_model(model_name, model_version, model_stage)
    model = last_model.unwrap_python_model().model
    print('Backtesting Train')
    bt_env_kwargs = deepcopy(env_kwargs)
    bt_env_kwargs['df'] = df_train
    bt_env_kwargs['mode'] = 'sequential'
    bt_param = EnvParameter(**bt_env_kwargs)
    bt_env = OneEnv(bt_param, stop_on_none=False)
    run_backtest(env=bt_env, model=model, filename=f'{model_name}-bt-train.html')
    if df_test is not None:
      print('Backtesting Test')
      bt_env_kwargs = deepcopy(env_kwargs)
      bt_env_kwargs['df'] = df_test
      bt_env_kwargs['mode'] = 'sequential'
      bt_param = EnvParameter(**bt_env_kwargs)
      bt_env = OneEnv(bt_param, stop_on_none=False)
      run_backtest(env=bt_env, model=model, filename=f'{model_name}-bt-test.html')
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  hydra_cfg = HydraConfig.get()
  train(hydra_config=hydra_cfg, **OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()   
  