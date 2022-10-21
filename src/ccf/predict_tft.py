import sys
from datetime import datetime, timedelta, timezone
import shutil
from pathlib import Path
import time

import yaml
from sqlalchemy import create_engine
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss

from ccf.read_data import read_data


def predict_tft(model_path, verbose=False, delay=1,
                engine_kwargs=None):
  target = 'a_p_0'
  max_encoder_length = 15
  max_prediction_length = 5
  print(f'Loading model from {model_path}')
  model = TemporalFusionTransformer.load_from_checkpoint(model_path)
  print('Starting prediction')
  while True:
    t0 = time.time()
    # TODO add rowid to DB
    now = datetime.utcnow()
    before = now - timedelta(seconds=20)
    read_kwargs = {}
    read_kwargs['sql'] = f"SELECT * FROM 'data' WHERE time >= '{before}'"
    df = read_data(engine_kwargs=engine_kwargs,
                   read_kwargs=read_kwargs)
    if len(df) < max_encoder_length:
      print(f'{now} not enough data for prediction: {len(df)}')
      continue
    df = df[-max_encoder_length:]
    df.time = pd.to_datetime(df.time)
    last_time = df.iloc[[-1]].time
    last_idx = len(df) - 1
    df = pd.concat([df, df.iloc[[-1]*max_prediction_length]])  # repeat last row
    df = df.reset_index(drop=True)
    for i in range (1, max_prediction_length + 1):
      df.iloc[last_idx + i, df.columns.get_loc('time')] = last_time + timedelta(seconds=i)
    df = df.reset_index()  # index to column
    df['group'] = 0
    ds = TimeSeriesDataSet(
      df,
      time_idx='index',
      add_relative_time_idx=True,
      target=target,
      group_ids=['group'],
      max_encoder_length=max_encoder_length,
      max_prediction_length=5,
      time_varying_unknown_reals=[target],
    )
    dl = ds.to_dataloader(train=False, batch_size=1, num_workers=1)
    predictions = model.predict(dl)  # one value
    for i in range (1, max_prediction_length + 1):
      df.iloc[last_idx + i, df.columns.get_loc(target)] = float(predictions[0][i - 1])
    # raw_predictions, x = model.predict(dl, mode="raw", return_x=True)
    # print(x)
    # print(raw_predictions['prediction'])
    df = df[-max_prediction_length:]
    df = df[['time', target]]
    con = create_engine(url='sqlite:///btcusdt@prediction@tft.db')
    df.to_sql('data', con, index=False, if_exists='append')
    if verbose:
      print(f'time spent: {time.time() - t0}')
    time.sleep(max(0, delay - (time.time() - t0)))
    
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'predict_tft.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  predict_tft(**kwargs)