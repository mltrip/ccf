import sys
from datetime import datetime, timedelta, timezone
import shutil
from pathlib import Path

import pandas as pd
import pytorch_forecasting as pf
import pytorch_lightning as pl
from sqlalchemy import create_engine
import yaml

from ccf.make_dataset import make_dataset


def train(dataset_kwargs, dataloader_kwargs,
          model_kwargs, trainer_kwargs, 
          model_path=None, tune=False):
  model_path = Path(model_path) if model_path is not None else None
  # Dataset
  ds_t, ds_v, df_t, df_v = make_dataset(**dataset_kwargs)
  if ds_t is None:
    raise ValueError('Bad dataset!')
  # Dataloader
  dl_t = ds_t.to_dataloader(**dataloader_kwargs['train'])
  dl_v= ds_v.to_dataloader(**dataloader_kwargs['val'])
  # Model
  if not tune:
    loss_kwargs = model_kwargs.pop('loss')
    l = getattr(pf.metrics, loss_kwargs.pop('class'))
    model_kwargs['loss'] = l(**loss_kwargs)
    model_kwargs['dataset'] = ds_t
    c = getattr(pf.models, model_kwargs.pop('class'))
    model = c.from_dataset(**model_kwargs)
  else:
    if model_path is not None:
      c = getattr(pf.models, model_kwargs.pop('class'))
      model = c.load_from_checkpoint(model_path)
      if isinstance(tune, str):
        model_path = Path(tune)
  # Trainer
  cs = trainer_kwargs.get('callbacks', [])
  for i, c in enumerate(cs):
    cc = getattr(pl.callbacks, c.pop('class'))
    cs[i] = cc(**c)
  trainer_kwargs['callbacks'] = cs
  ls = trainer_kwargs.get('logger', [])
  ls = [ls] if not isinstance(ls, list) else ls
  for i, l in enumerate(ls):
    ll = getattr(pl.loggers, l.pop('class'))
    ls[i] = ll(**l)
  trainer_kwargs['logger'] = ls
  trainer = pl.Trainer(**trainer_kwargs)
  # res = trainer.tuner.lr_find(
  #   model, train_dataloaders=train_dataloader,
  #   val_dataloaders=val_dataloader, 
  #   early_stop_threshold=1000.0,
  #   max_lr=0.3)
  # lr = res.suggestion()
  # Fit
  trainer.fit(model, train_dataloaders=dl_t, val_dataloaders=dl_v)
  if model_path is not None:
    best = Path(trainer.checkpoint_callback.best_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, model_path)
  return trainer
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'train.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  trainer = train(**kwargs)
