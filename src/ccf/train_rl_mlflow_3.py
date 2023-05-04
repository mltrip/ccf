from copy import deepcopy
import concurrent.futures
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

import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
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
from ccf.create_dataset import Dataset


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


class TradingStrategy(Strategy):
  callback: Callable
    
  def init(self):
    pass

  def next(self):
    self.callback(self)

        
class BacktestingThread(Thread):
  def __init__(self, dataset, data, bt_kwargs, price_column, price_shift=0):
    # print('__init__')
    threading.Thread.__init__(self)
    self.step_event = Event()  # False
    self.callback_event = Event()  # False
    self.result_event = Event()  # False
    TradingStrategy.callback = self._callback
    self.kill_flag = False
    self.dataset = dataset
    self.data = data
    df = pd.DataFrame(data[price_column].shift(price_shift).ffill())
    df['Open'] = df['High'] = df['Low'] = df['Close'] = df[price_column]
    df = df.drop(columns=[price_column])
    # pprint('Open')
    # pprint(list(df['Open']))
    df = df[self.dataset.min_encoder_length-4:-self.dataset.min_prediction_length-1]  # from first dataset timestep (FIXME workaround -4 steps) order filled at next price
    # pprint(list(df['Open']))
    self.bt = Backtest(df, TradingStrategy, **bt_kwargs)
    self.strategy = None
    self.result = None
        
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
    # print('get_strategy start')
    if not self.kill_flag:
      self.step_event.set()
      self.callback_event.wait()
      self.callback_event.clear()
    # print(f'get_strategy stop')
    return self.strategy

  def _callback(self, strategy: Strategy):
    """Get strategy from TradingStrategy, start callback and wait step"""
    # print(f'_callback start')
    self.strategy = strategy
    # print(self.strategy.data.df)
    if self.kill_flag:
      sys.exit(0)
    # if len(self.strategy.data) >= self.dataset.min_encoder_length:  # Start then window is accumulated
    self.callback_event.set()
    self.step_event.wait()
    self.step_event.clear()
    # print(f'_callback stop')

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
      open_browser=True):
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
        open_browser=open_browser)
        
      
class Episode:
    def __init__(self, dataset, data, bt_kwargs, price_column, price_shift=0, 
                 max_len=None, add_reward=False, add_position=False):
        self.dataset = dataset
        self.dataloader = iter(self.dataset.to_dataloader(
          train=False, batch_size=1, batch_sampler='synchronized'))
        self.data = data
        self.bt_kwargs = bt_kwargs
        self.episode_step = 0
        self.terminated = False
        self.truncated = False
        self.bt = BacktestingThread(self.dataset, self.data, self.bt_kwargs, price_column, price_shift)
        self.bt.daemon = True
        self.bt.start()
        # self.strategy = self.bt.get_strategy()  # One step
        self.strategy = None
        self.max_len = max_len
        self.reward_buffer = []
        self.position_buffer = []
        self.add_reward = add_reward
        self.add_position = add_position

    def forward(self):
      # print('forward start')
      if not self.terminated and not self.truncated:
        self.episode_step += 1
        self.strategy = self.bt.get_strategy()
        self.terminated = True if self.episode_step >= len(self.dataset) else False
        if self.max_len is not None:
          if self.episode_step >= self.max_len:
            self.truncated = True
      # print('forward stop')

    def status(self):
        observation = self.observation()
        reward = self.reward()
        terminated = self.terminated
        truncated = self.truncated
        info = self.info()
        self.reward_buffer.append(reward)
        self.position_buffer.append(self.strategy.position.size)
        if self.add_reward:
          n_samples = observation.shape[0]
          len_buffer = len(self.reward_buffer)
          if len_buffer >= n_samples:
            feature = self.reward_buffer[-n_samples:]
          else:
            pad = [0.0 for _ in range(n_samples - len_buffer)]
            feature = pad + self.reward_buffer
          observation = np.concatenate((observation, [[x] for x in feature]), axis=1)
        if self.add_position:
          n_samples = observation.shape[0]
          len_buffer = len(self.position_buffer)
          if len_buffer >= n_samples:
            feature = self.position_buffer[-n_samples:]
          else:
            pad = [0.0 for _ in range(n_samples - len_buffer)]
            feature = pad + self.position_buffer
          observation = np.concatenate((observation, [[x] for x in feature]), axis=1)
        return observation, reward, terminated, truncated, info

    def clear(self):
      self.bt.kill()
      self.bt.join()

    def observation(self):
      x, y = next(self.dataloader)
      # pprint(x)
      # pprint(y)
      # print(self.strategy.data)
      # print(self.strategy.data.df)
      obs = x['encoder_cont'][0]
      # print(obs)
      # print(obs)
      return obs

    def reward(self):
      # sum of profit percentage
      # TODO Short term reward?
      # pprint(self.strategy.trades)
      # pprint([trade.pl_pct for trade in self.strategy.trades])
      # TODO Long term reward?
      # pprint(self.strategy.closed_trades)
      # pprint([trade.pl_pct for trade in self.strategy.closed_trades])
      return sum([trade.pl_pct for trade in self.strategy.trades])

    def info(self):
      return {
        "date": self.strategy.data.df.index[-1],
        "data": self.strategy.data,
        "episode_step": self.episode_step,
        "orders": self.strategy.orders, 
        "trades": self.strategy.trades, 
        "position": self.strategy.position, 
        "closed_trades": self.strategy.closed_trades}


class TradingEnv(gym.Env):
  def __init__(self, dataset=None, data=None, bt_kwargs=None,
               price_column=None, kind='sequential', price_shift=0, 
               max_len=None, scaler=None, add_reward=False, add_position=False,
               reward_scaler=None, position_scaler=None):
    self.dataset = dataset
    self.action_space = spaces.Discrete(3)  # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL
    n_features = len(dataset.static_reals) + len(dataset.time_varying_known_reals) + len(dataset.time_varying_unknown_reals)
    if add_reward:
      n_features += 1
    if add_position:
      n_features += 1
    shape = (dataset.min_encoder_length, n_features)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)
    print(f'action space: {self.action_space.shape}, observation space: {self.observation_space.shape}')
    self.price_column = price_column
    self.price_shift = price_shift
    self.max_len = max_len
    # Episode factory
    self.data = data
    self.kind = kind
    self.bt_kwargs = bt_kwargs
    self.episode = None
    self.scaler = scaler
    self.add_reward = add_reward
    self.add_position = add_position
    self.reward_scaler = reward_scaler
    self.position_scaler = position_scaler
    
  def create_episode(self):
    if self.kind == "random":
      next_timestamp = random.choice(range(len(self.data) - self.dataset.min_encoder_length - self.dataset.min_prediction_length))
    else:  # "sequential" or "backtest"
      if self.episode == None:
        next_timestamp = 0
      elif self.episode.episode_step + self.dataset.min_encoder_length > len(self.episode.dataset):
        next_timestamp = 0
      else:
        next_timestamp = self.episode.episode_step + 1
    # print(next_timestamp, len(self.data), self.data.index.min(), self.data.index.max())
    data = self.data[next_timestamp:]
    # print(len(data), data.index.min(), data.index.max())
    dataset = self.dataset.from_dataset(self.dataset, data)
    if self.kind == "backtest":
      episode = Episode(dataset, data, self.bt_kwargs, self.price_column, self.price_shift, 
                        max_len=None, add_reward=self.add_reward, add_position=self.add_position)
    else:
      episode = Episode(dataset, data, self.bt_kwargs, self.price_column, self.price_shift, 
                        max_len=self.max_len,
                        add_reward=self.add_reward, add_position=self.add_position)
    return episode
    
  @staticmethod
  def scale_array(array, scaler):
    if array is None or scaler is None:
      return array
    scaler = deepcopy(scaler)
    scaler_class = scaler.pop('class')
    kind = scaler.pop('kind', 'feature')
    s = getattr(preprocessing, scaler_class)(**scaler)
    if kind == 'feature':
      array = s.fit_transform(array)
    elif kind == 'sample':
      array = np.transpose(s.fit_transform(np.transpose(array)))
    elif kind in ['all', 'global']:
      shape = array.shape
      array = s.fit_transform(array.reshape(-1, 1)).reshape(shape)
    else:
      raise NotImplementedError(f'scaler kind: {kind}')
    return array      
    
  def scale_observation(self, observation):
    if self.add_reward and self.add_position:
      obs, rew, pos = observation[:,:-2], observation[:,-2:-1], observation[:,-1:]
      obs = self.scale_array(obs, self.scaler)
      rew = self.scale_array(rew, self.reward_scaler)
      pos = self.scale_array(pos, self.position_scaler)
      observation = np.concatenate((obs, rew, pos), axis=1)
    elif self.add_reward:
      obs, rew = observation[:,:-1], observation[:,-1:]
      obs = self.scale_array(obs, self.scaler)
      rew = self.scale_array(rew, self.reward_scaler)
      observation = np.concatenate((obs, rew), axis=1)
    elif self.add_position:
      obs, pos = observation[:,:-1], observation[:,-1:]
      obs = self.scale_array(obs, self.scaler)
      pos = self.scale_array(pos, self.position_scaler)
      observation = np.concatenate((obs, pos), axis=1)
    else:
      observation = self.scale_array(observation, self.scaler)
    return observation
    
  def step(self, 
           action, 
           size: float = 1.0,  
           limit: float = None,
           stop: float = None,
           sl: float = None,
           tp: float = None):
    if self.episode == None:
        print("NotEpisodeInitError: Please run env.reset() at first.")
        sys.exit(1)
    if action == 1:
        self.episode.strategy.buy(
            size=size,
            limit=limit,
            stop=stop,
            sl=sl,
            tp=tp)
    elif action == 2:
        self.episode.strategy.sell(
            size=size,
            limit=limit,
            stop=stop,
            sl=sl,
            tp=tp)
    self.episode.forward()
    observation, reward, terminated, truncated, info = self.episode.status()
    observation = self.scale_observation(observation)
    return observation, reward, terminated or truncated, info

  def reset(self):
    # print('reset')
    if self.episode != None:
      self.episode.clear()
    self.episode = self.create_episode()
    self.episode.forward()
    observation, _, _, _, _ = self.episode.status()
    # print(observation)
    observation = self.scale_observation(observation)
    # print(observation)
    return observation

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
    open_browser=True):
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
        open_browser=open_browser)
    
  
class PositionEnv(gym.Wrapper):
    def __init__(self, dataset=None, data=None, bt_kwargs=None, 
                 price_column=None, kind='sequential', price_shift=0, max_len=None,
                 terminate_on_none=False, 
                 size=1.0, limit=None, stop=None, sl=None, tp=None, scaler=None,
                 add_reward=False, add_position=False, reward_scaler=None, position_scaler=None):
        env = TradingEnv(dataset=dataset, data=data, bt_kwargs=bt_kwargs, 
                         price_column=price_column, kind=kind, price_shift=price_shift,
                         max_len=max_len, scaler=scaler, 
                         add_reward=add_reward, add_position=add_position,
                         reward_scaler=reward_scaler, position_scaler=position_scaler)
        super().__init__(env)
        self.env = env
        self.side = 'NONE'
        self.terminate_on_none = terminate_on_none
        self.size = size
        self.limit = limit
        self.stop = stop
        self.sl = sl
        self.tp = tp

    def step(self, action,   # 0: HOLD, 1: BUY, 2: SELL
             size: float = 1.0,  
             limit: float = None,
             stop: float = None,
             sl: float = None,
             tp: float = None):
      
      if self.side == "LONG" and action == 1:
        action = 0  # HOLD
      if self.side == "SHORT" and action == 2:
        action = 0  # HOLD

      observation, reward, done, info = self.env.step(
        action, size=self.size, limit=self.limit, stop=self.stop, 
        sl=self.sl, tp=self.tp)

      if self.terminate_on_none:
        if self.side == "LONG" and action == 2 or self.side == "SHORT" and action == 1:
          self.terminated = True
          done = True

      if info["position"].size == 0:
        self.side = 'NONE'
      elif info["position"].size > 0:
        self.side = "LONG"
      elif info["position"].size < 0:
        self.side = "SHORT"
            
      info['side'] = self.side
        
      return observation, reward, done, info
    
    
def run_backtest(env, model, filename):
  print('Start')
  observation = env.reset()
  pbar = tqdm(total=len(env.episode.dataset))
  pbar.update(1)
  cum_reward = 0.0
  done = False
  while not done:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    cum_reward += reward
    pbar.set_description(f"{info['date']}, {info['episode_step']}, reward: {reward:+.6f}%, cum_reward: {cum_reward:+.6f}%")
    pbar.update(1)
  pbar.close()
  print('Done\n')
  stats = env.stats()
  print(stats)
  env.plot(filename=filename, open_browser=False)    
    
    
def train(
  hydra_config, model_name, model_version=None, model_stage=None,
  create_dataset_kwargs=None, bt_kwargs=None,
  model_kwargs=None, learn_kwargs=None, env_kwargs=None, 
  n_envs=1, seed=None, verbose=0,
  is_tune=False, parent_name=None, parent_version=None, parent_stage=None,
  do_train=True, do_backtest=True, do_mlflow=True
):
  # Params
  create_dataset_kwargs = {} if create_dataset_kwargs is None else create_dataset_kwargs
  bt_kwargs = {} if bt_kwargs is None else bt_kwargs
  env_kwargs = {} if env_kwargs is None else env_kwargs
  model_kwargs = {} if model_kwargs is None else model_kwargs
  learn_kwargs = {} if learn_kwargs is None else learn_kwargs
  mlflow_params = {**{f'dataset-{k}': v for k, v in create_dataset_kwargs.items()},
                   **{f'bt-{k}': v for k, v in bt_kwargs.items()},
                   **{f'env-{k}': v for k, v in env_kwargs.items()},
                   **{f'model-{k}': v for k, v in model_kwargs.items()},
                   **{f'learn-{k}': v for k, v in learn_kwargs.items()},
                   **{'n_envs': n_envs, 
                      'seed': seed,
                      'model_name': model_name, 
                      'model_version': model_version,
                      'model_stage': model_stage,
                      'parent_name': parent_name, 
                      'parent_version': parent_version,
                      'parent_stage': parent_stage,
                      'is_tune': is_tune, 
                      'do_train': do_train,
                      'do_backtest': do_backtest,
                      'do_mlflow': do_mlflow}}
  
  print('Dataset')
  dataset = Dataset(**create_dataset_kwargs)
  ds_t, ds_v, df_t, df_v = dataset()
  if ds_t is None:
    raise ValueError('Bad dataset!')
  
  # Model  
  model_class_name = model_kwargs.pop('class')
  model_class = getattr(stable_baselines3, model_class_name)
  print(f'Model: {model_class_name}')

  # Environment
  env_class_name = env_kwargs.pop('class')
  env_class = globals()[env_class_name]
  print(f'Environment: {env_class_name}')
  
  # Train
  if do_train:
    print('Train')
    train_env_kwargs = deepcopy(env_kwargs)
    train_env_kwargs['dataset'] = ds_t
    train_env_kwargs['data'] = df_t
    train_env_kwargs['bt_kwargs'] = deepcopy(bt_kwargs)
    train_env_kwargs['kind'] = 'random'
    train_env_kwargs['terminate_on_none'] = True
    if n_envs == 1:
      print('Create one environment')
      train_env = env_class(**train_env_kwargs)
      # check_env(train_env)
    else:
      print('Create many environments')
      train_env = make_vec_env(env_class, env_kwargs=train_env_kwargs,
                               n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    print('Create loggers')
    loggers = Logger(folder=None,
                     output_formats=[HumanOutputFormat(sys.stdout), 
                                     MLflowOutputFormat()])
    train_model_kwargs = deepcopy(model_kwargs)
    train_model_kwargs['env'] = train_env
    train_model_kwargs['verbose'] = verbose
    if do_mlflow:
      print('MLflow')
      experiment_name = model_name
      print('Create experiment')
      experiment = mlflow.set_experiment(experiment_name)
      print('Start experiment')
      with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        print('Log parameters')
        mlflow.log_params(mlflow_params)
        if not is_tune:
          print('Create model')
          model = model_class(**train_model_kwargs)
        else:
          print('Load model')
          last_model, last_version, last_stage = load_model(parent_name, parent_version, parent_stage)
          if last_model is not None:
            model = last_model.unwrap_python_model().model
            try:
              model.set_env(train_env)
            except AssertionError as e:
              print(e)
              print('Loading local model if exists')
              model = model_class.load(path=model_name, env=train_env)
          else:
            print('Model is not found: creating new model')
            model = model_class(**train_model_kwargs)
        print('Set logger')  
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
    else:
      print('Local')
      if not is_tune:
        print('Create model')
        model = model_class(**train_model_kwargs)
      else:
        print('Load model')
        try:
          model = model_class.load(path=model_name, env=train_env)
        except Exception as e:
          print(e)
          print('Create model')
          model = model_class(**train_model_kwargs)
      print('Set logger')
      model.set_logger(loggers)
      print('Start train')  
      model.learn(**learn_kwargs)
      print('Save model')
      model.save(model_name)
      
  # Backtest
  if do_backtest:
    print('Backtesting')
    print('Loading model')
    if do_mlflow:
      print('MLflow')
      model = model_class.load(model_name) 
      last_model, last_version, last_stage = load_model(model_name, model_version, model_stage)
      model = last_model.unwrap_python_model().model
    else:
      print('Local')
      model = model_class.load(path=model_name)
    print('Backtesting Train')
    bt_env_kwargs = deepcopy(env_kwargs)
    bt_env_kwargs['dataset'] = ds_t
    bt_env_kwargs['data'] = df_t
    bt_env_kwargs['bt_kwargs'] = deepcopy(bt_kwargs)
    bt_env_kwargs['kind'] = 'backtest'
    bt_env_kwargs['terminate_on_none'] = False
    bt_train_env = env_class(**bt_env_kwargs)
    run_backtest(env=bt_train_env, model=model, filename=f'{model_name}-bt-train.html')
    if ds_v is not None and df_v is not None:
      print('Backtesting Test')
      bt_env_kwargs = deepcopy(env_kwargs)
      bt_env_kwargs['dataset'] = ds_v
      bt_env_kwargs['data'] = df_v
      bt_env_kwargs['bt_kwargs'] = deepcopy(bt_kwargs)
      bt_env_kwargs['kind'] = 'backtest'
      bt_env_kwargs['terminate_on_none'] = False
      bt_test_env = env_class(**bt_env_kwargs)
      run_backtest(env=bt_test_env, model=model, filename=f'{model_name}-bt-test.html')
    
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  hydra_cfg = HydraConfig.get()
  train(hydra_config=hydra_cfg, **OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()   
  