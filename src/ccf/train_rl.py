import random
from pathlib import Path

import gym
from backtesting.test import GOOG
from tqdm import tqdm
from backtesting import Backtest, Strategy
from abc import *
import gym
from gym import spaces
from threading import (Event, Thread)
import random
from typing import Union, Callable
from gym.core import Env
import pandas as pd
import sys
import threading
import time
import numpy as np
import copy
import gym
import pandas as pd
from sklearn import preprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
import numpy as np
import pandas as pd
from ccf.agents import InfluxDB
from ccf.utils import initialize_time


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
        threading.Thread.__init__(self)
        self.step_event = Event()
        self.callback_event = Event()
        self.result_event = Event()
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
        time.sleep(0.01) # for sync first data
        self.result = self.bt.run()
        self.result_event.set()

    def get_strategy(self):
        # print(self.kill_flag)
        if not self.kill_flag:
            self.step_event.set()
            # print('step')
            self.callback_event.wait(timeout=10)
            # print('wait')
            self.callback_event.clear()
        #     print('clear')
        # print('HERE')
        return self.strategy

    def kill(self):
        self.kill_flag = True
        self.step_event.set()
        self.callback_event.set()

    def _callback(self, strategy: Strategy):
        self.strategy = strategy
        if self.kill_flag:
            sys.exit(0)

        if len(self.strategy.data) >= self.window_size:
            self.callback_event.set()
            self.step_event.wait(timeout=10)
            self.step_event.clear()

    def stats(self):
        # print('HERE')
        self.kill()
        # print('HERE')
        if not self.result_event.is_set():
            # print('HERE')
            self.result_event.wait(timeout=10)
        # print('HERE')
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
            self.episode.strategy.buy(
                size=size,
                limit=limit,
                stop=stop,
                sl=sl,
                tp=tp,
            )
        elif action == 2:
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
            # print(self.episode_step, self.timestamp, len(self.strategy.data.df), len(self.episode_data))
            self.finished = True if len(self.strategy.data.df) >= len(self.episode_data) else False
            if not self.finished:
              self.strategy = self.bt.get_strategy()
            
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
        # Columns
        # cols = [x for x in self.strategy.data.df.columns if x.startswith('Norm')]  # Normalized
        cols = ['Forecast', 'Close']
        # Normalized
        df = self.strategy.data.df[cols][-self.param.window_size:]
        df['Forecast'] = df['Forecast'] / df['Close']
        # print(df)
        x = df.values # returns a numpy array
        # min_max_scaler = preprocessing.MinMaxScaler()
        scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)
        # df = pd.DataFrame(x_scaled)
        # print(df)
        # x = scaler.inverse_transform(df.values)
        # print(x)
        # print(x_scaled)
        return x_scaled

    def reward(self):
        # sum of profit percentage
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
        # print(next_timestamp)
        data = self.get_next_episode_data(next_timestamp)
        self.episode = Episode(data, self.param)
        return self.episode

    def get_next_episode_data(self, timestamp):
        return self.param.df[timestamp:]

    def get_next_episode_timestamp(self):
        if self.param.mode == "random":
            return random.choice(range(len(self.param.df)-self.param.window_size))
        else: # "sequential" or "backtest"
            if self.episode == None or self.episode.timestamp + self.param.window_size > len(self.episode.episode_data):
                return 0
            else:
                return self.episode.timestamp + 1

              
########################################
# Trading environment that takes only 
# one position and ends the episode 
# when the position reaches 0
#######################################
class CustomEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        env = TradingEnv(param)
        super().__init__(env)
        self.env = env
        self.side = None # None: No Position, LONG: Long Position, SHORT: Short Position

    def step(self, action):

        if self.side == "LONG" and action == 1:
            action = 0
        if self.side == "SHORT" and action == 2:
            action = 0

        obs, reward, done, info = self.env.step(action, size=1)

        if self.side == "LONG" and action == 2 or self.side == "SHORT" and action == 1:
            done = True

        if info["position"].size == 0:
            if self.side != 0:
                self.side = 0
        elif info["position"].size > 0:
            self.side = "LONG"
        elif info["position"].size < 0:
            self.side = "SHORT"

        return obs, reward, done, info
      
      
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

      
class TradingEnvOne(gym.Env):
    def __init__(self, param: EnvParameter):
        super().__init__()
        self.param = param
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL
        self.factory = EpisodeFactory(self.param)
        self.episode = None
        self.observation_space = spaces.Box(
          low=-np.inf, high=np.inf, shape=(self.param.window_size, 2), dtype=np.float64)
        self.side = 'NONE' 
       
    def step(self, action, size: float = _FULL_EQUITY,
            limit: float = None,
            stop: float = None,
            sl: float = None,
            tp: float = None):
        if self.episode == None:
            print("NotEpisodeInitError: Please run env.reset() at first.")
            sys.exit(1)
        
        if self.side == "LONG" and action == 1:
            action = 0  # HOLD
        if self.side == "SHORT" and action == 2:
            action = 0  # HOLD
        
        if action == 1:
            self.episode.strategy.buy(
                size=size,
                limit=limit,
                stop=stop,
                sl=sl,
                tp=tp,
            )
        elif action == 2:
            self.episode.strategy.sell(
                size=size,
                limit=limit,
                stop=stop,
                sl=sl,
                tp=tp,
            )
        
        if self.side == "LONG" and action == 2 or self.side == "SHORT" and action == 1:
            done = True
        
        self.episode.forward()
        obs, reward, done, info = self.episode.status()

        if info["position"].size == 0:
            self.side = 'NONE'
        elif info["position"].size > 0:
            self.side = "LONG"
        elif info["position"].size < 0:
            self.side = "SHORT"
        
        return obs, reward, done, info

    def reset(self):
      if self.episode != None:
          self.episode.clear()
      self.factory = EpisodeFactory(self.param)
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

      
      
def run_custom_env():
    param = EnvParameter(df=GOOG[:200], mode="sequential", add_feature=True, window_size=10)
    print(GOOG)
    env = CustomEnv(param)
    for i in range(20):
        print("Episode: ", i)
        obs = env.reset()
        done = False
        while not done:
            action = random.choice([0,1,2])
            next_obs, reward, done, info = env.step(action)
            print("obs", obs.tail())
            print("action: ", action)
            print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
            print("next_obs", next_obs.tail())
            print("-"*10)
            obs = next_obs

    stats = env.stats()
    print(stats)
      
def run_simple_env():
    param = EnvParameter(df=GOOG[:200], mode="sequential", window_size=10)
    print(GOOG)
    env = TradingEnv(param)
    
    for i in range(20):
        print("episode: ", i)
        obs = env.reset()
        for k in range(10):
            action = random.choice([0,1,2])
            next_obs, reward, done, info = env.step(action)
            print("obs", obs.tail())
            print("action: ", action)
            print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
            print("next_obs", next_obs.tail())
            print("-"*10)
            obs = next_obs
    print("finished")
    stats = env.stats()
    print(stats)

    
def run_backtest_env(df):
    param = EnvParameter(df, mode="sequential", window_size=20)
    env = OneEnv(param)
    obs = env.reset()
    done = False
    while not done:
        action = random.choice([0,1,2])
        next_obs, reward, done, info = env.step(action, size=1)
        print("obs", obs.tail(20))
        print("action: ", action)
        print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
        print("next_obs", next_obs.tail(20))
        print("-"*10)
        obs = next_obs
    print("finished")
    stats = env.stats()
    print(stats)
    env.plot(filename='test.html', open_browser=False)    

    
def run_backtest(param, window_size, model_name, name_suffix):
  print(f'Backtesting {model_name}')
  env = OneEnv(param, stop_on_none=False)
  print('Loading model')
  model = PPO.load(model_name)
  print('Reset environment')
  obs = env.reset()
  print('Start')
  done = False
  cnt = 0
  pbar = tqdm(total=len(param.df))
  while not done:
    action, _states = model.predict(obs)
    next_obs, reward, done, info = env.step(action, size=1)
    obs = next_obs
    cnt += 1
    pbar.update(cnt)
  pbar.close()
  print('Done\n')
  stats = env.stats()
  print(stats)
  env.plot(filename=f'{model_name}{name_suffix}.html', open_browser=False)    
    
    
def train_ppo(df, model_name, window_size=20, commission=0.0,
              split=0.75, total_timesteps=300000, n_envs=4, is_tune=False):
  # Split dataset
  split_idx = int(len(df)*split)
  df_train = df.iloc[:split_idx]
  df_test = df.iloc[split_idx:]
  print(f'train: {len(df_train)} {len(df_train.columns)} {df_train.index.min()} {df_train.index.max()}')
  print(f'test: {len(df_test)} {len(df_test.columns)} {df_test.index.min()} {df_test.index.max()}')
  # Create environment
  train_param = EnvParameter(df_train, 
                             cash=30.0,
                             commission=commission,
                             mode="random", 
                             window_size=window_size)
  train_env = OneEnv(train_param)
  # wrapper_kwargs TODO vectorized SubprocVecEnv, DummyVecEnv
  # train_env_kwargs = {'param': train_param}
  # train_env = make_vec_env(OneEnv, env_kwargs=train_env_kwargs,
  #                          n_envs=n_envs, seed=0, vec_env_cls=DummyVecEnv)
  # env = OneEnv(train_param)
  # train_env = DummyVecEnv([lambda: env])
  # Check environment
  # check_env(train_env)
  # Create model
  if not is_tune:
    model = PPO(
      "MlpPolicy",
      train_env,
      verbose=2, 
      learning_rate=0.0003, 
      n_steps=2048,  # The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
      batch_size=64, 
      n_epochs=10,  # Number of epoch when optimizing the surrogate loss
      gamma=0.99, 
      gae_lambda=0.95,
      clip_range=0.2,
      seed=42,
      device='cuda',
      clip_range_vf=None, 
      normalize_advantage=True,
      ent_coef=0.0, 
      vf_coef=0.5,
      max_grad_norm=0.5, 
      use_sde=False, 
      sde_sample_freq=-1, 
      target_kl=None, 
      tensorboard_log="./tensorboard/ppo",
      policy_kwargs=None, 
     )
  else:
    model = PPO.load(model_name)
  # Train
  print('Training')
  reset_num_timesteps = False if is_tune else True
  model.learn(
    total_timesteps=total_timesteps,  # n_steps * n_envs
    # callback=None, 
    log_interval=1,  #  The number of episodes before logging.
    tb_log_name=model_name, 
    reset_num_timesteps=reset_num_timesteps, 
    progress_bar=True,
   )

  # Save
  print('Saving model')
  model.save(model_name)
  
  print('Backtesting')
  # Backtest Train
  train_param = EnvParameter(df_train, 
                             cash=30.0,
                             commission=commission,
                             mode="sequential", 
                             window_size=window_size)
  train_env = OneEnv(train_param)
  run_backtest(train_param, window_size=window_size, model_name=model_name, name_suffix='-train')
  # Backtest Test
  test_param = EnvParameter(df_test, 
                            cash=30.0,
                            commission=commission,
                            mode="sequential", 
                            window_size=window_size)
  run_backtest(test_param, window_size=window_size, model_name=model_name, name_suffix='-test')
    
    
if __name__ == "__main__":
  # Data
  # model = 'influxdb-mid-rat-ema_vwap_base_10_1_1-mel_600-mpl_20-no_cat-no_group'    
  model = 'influxdb-mid-rat-ema_vwap_base_10_1_1-mel_20-mpl_20-no_cat-no_group'
  version = 4
  data_filename = f'{model}-{version}.csv'
  data_path = Path(data_filename)
  if not data_path.exists():
    # start = '2022-10-01T00:00:00+00:00'
    # stop = '2022-10-15T00:00:00+00:00'
    start = '2023-03-20T00:00:00+00:00'
    stop = '2023-04-01T00:00:00+00:00'
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
  print(df)
  if 'timestamp' in df:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
  unit_scale = 1e-3  # one unit = price*unit_scale
  cols = [x for x in ['last', 'quantile_0.5', 'value'] if x in df]
  # FIXME Filter < 2 due to database errors
  for c in cols:
    df[c] = df[c].mask(lambda x: x < 2, np.nan)
  df = df.fillna(method='pad')
  df[cols] *= unit_scale
  df['Open'] = df['High'] = df['Low'] = df['Close'] = df['last']
  df = df.drop(columns=['last'])
  df = df.rename(columns={'quantile_0.5': 'Forecast', 'value': 'Forecast'})
  # df = df.head(100)
  # df = df.tail(10000)
  # df = df.loc['2023-03-28T13:28:12+00:00':'2023-03-31T07:47:57+00:00']
  print(df)

  train_ppo(df, model_name='march', n_envs=4, total_timesteps=int(1e6), is_tune=False)
  # param = EnvParameter(
  #   df, 
  #   cash=30.0,
  #   commission=0.0,
  #   mode="sequential", 
  #   window_size=20)
  # run_backtest(param, window_size=20, model_name='march', name_suffix='-test')
  
