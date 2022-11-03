import sys
from datetime import datetime, timedelta, timezone
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytorch_forecasting as pf
import pytorch_lightning as pl
from sqlalchemy import create_engine

import ccf
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
  loss_kwargs = model_kwargs.pop('loss')
  l = getattr(pf.metrics, loss_kwargs.pop('class'))
  loss = l(**loss_kwargs)
  if ds_t.multi_target:
    model_kwargs['loss'] = pf.metrics.MultiLoss(
      metrics=[loss for _ in ds_t.target_names])
  else:
    model_kwargs['loss'] = loss
  model_kwargs['dataset'] = ds_t
  model_name = model_kwargs.pop('class')
  c = getattr(pf.models, model_name, None)
  if c is None:
    c = getattr(ccf.models, model_name, None)
  if c is None:
    raise NotImplementedError(model_name)
  if not tune:
    model = c.from_dataset(**model_kwargs)
  else:
    if model_path is not None:
      if model_path.is_file():
        model = c.load_from_checkpoint(model_path)
      else:
        model = c.from_dataset(**model_kwargs)
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
  

@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  train(**OmegaConf.to_object(cfg))


if __name__ == "__main__":
  app()
