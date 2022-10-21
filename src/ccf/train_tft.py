import sys
from datetime import datetime, timedelta, timezone
import shutil
from pathlib import Path

import yaml
from sqlalchemy import create_engine
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss

from ccf.read_data import read_data


def train_tft(engine_kwargs=None, read_kwargs=None):
  df = read_data(engine_kwargs=engine_kwargs,
                 read_kwargs=read_kwargs)
  df = df.reset_index()
  df['group'] = 0
  split_idx = int(0.8*len(df))
  training = TimeSeriesDataSet(
    df[:split_idx],
    time_idx='index',
    add_relative_time_idx=True,
    target='a_p_0',
    # weight="weight",
    group_ids=['group'],
    max_encoder_length=15,
    max_prediction_length=5,
    # static_categoricals=[ ... ],
    # static_reals=[ ... ],
    # time_varying_known_categoricals=[ ... ],
    # time_varying_known_reals=[ ... ],
    # time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=['a_p_0']
  )
  validation = TimeSeriesDataSet.from_dataset(
    training, df[split_idx:], stop_randomization=True)
  batch_size = 32
  train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=2)
  val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=2)
  # define trainer with early stopping
  early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-8, patience=4, verbose=False, mode="min")
  # lr_logger = LearningRateMonitor()
  trainer = pl.Trainer(
      max_epochs=32,
      gpus=0,
      gradient_clip_val=0.1,
      limit_train_batches=30,
      callbacks=[
        # lr_logger, 
                 early_stop_callback],
  )
  model = TemporalFusionTransformer.from_dataset(
      training,
      learning_rate=0.03,
      hidden_size=32,
      attention_head_size=1,
      dropout=0.1,
      hidden_continuous_size=16,
      output_size=7,
      loss=QuantileLoss(),
      log_interval=2,
      reduce_on_plateau_patience=2)
  # res = trainer.tuner.lr_find(
  #   model, train_dataloaders=train_dataloader,
  #   val_dataloaders=val_dataloader, 
  #   early_stop_threshold=1000.0,
  #   max_lr=0.3)
  # print(res.results)
  # print(res.suggestion())
  #   lr_finder.results
  # # Plot with
  # fig = lr_finder.plot(suggest=True)
  # fig.show()
  # # Pick point based on plot, or get suggestion
  # new_lr = lr_finder.suggestion()
  # # fit the model
  trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader,
  )
  best = Path(trainer.checkpoint_callback.best_model_path)
  shutil.copyfile(best, 
                  best.with_name(f'model{best.suffix}').name)
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'train_tft.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  train_tft(**kwargs)