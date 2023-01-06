import sys
from pprint import pprint
from datetime import datetime, timedelta, timezone
import shutil
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient
import hydra
from hydra import compose, initialize, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import pytorch_forecasting as pf
import pytorch_lightning as pl

import ccf
from ccf import models as ccf_models
from ccf import agents as ccf_agents
from ccf.create_dataset import Dataset
from ccf.model_mlflow import CCFModel, load_model


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

    
def main(hydra_config, create_dataset_kwargs, dataloader_kwargs, model_kwargs, 
         trainer_kwargs,
         kind, model_name, parent_name=None, parent_version=None, parent_stage=None):
  # Initialize parameters
  parent_name = model_name if parent_name is None else parent_name
  # Train
  if 'continual' in kind:
    max_cnt = None
  elif 'occasional' in kind:
    max_cnt = 1
  else:
    raise NotImplementedError(kind)
  cnt = 0
  while True:
    if max_cnt is not None and cnt == max_cnt:
      break
    cnt += 1
    print(f'Iteration {cnt}')
    # Dataset
    dataset = Dataset(**create_dataset_kwargs)
    ds_t, ds_v, df_t, df_v = dataset()
    if ds_t is None:
      raise ValueError('Bad dataset!')
    # Model
    parent_model, last_version, last_stage = load_model(parent_name, parent_version, parent_stage)
    if parent_model is None:
      model_kwargs_ = deepcopy(model_kwargs)
      loss_kwargs = model_kwargs_.pop('loss')
      l = getattr(pf.metrics, loss_kwargs.pop('class'))
      loss = l(**loss_kwargs)
      if ds_t.multi_target:
        model_kwargs_['loss'] = pf.metrics.MultiLoss(
          metrics=[loss for _ in ds_t.target_names])
      else:
        model_kwargs_['loss'] = loss
      model_kwargs_['dataset'] = ds_t
      class_name = model_kwargs_.pop('class')
      model_class = getattr(pf.models, class_name, None)
      if model_class is None:
        model_class = getattr(ccf.models, class_name, None)
      if model_class is None:
        raise NotImplementedError(class_name)
      model = model_class.from_dataset(**model_kwargs_)
    else:
      model = parent_model.unwrap_python_model().model
    # Trainer
    trainer_kwargs_ = deepcopy(trainer_kwargs)
    cs = trainer_kwargs_.get('callbacks', [])
    for i, c in enumerate(cs):
      cc = getattr(pl.callbacks, c.pop('class'))
      cs[i] = cc(**c)
    trainer_kwargs_['callbacks'] = cs
    ls = trainer_kwargs_.get('logger', [])
    ls = [ls] if not isinstance(ls, list) else ls
    for i, l in enumerate(ls):
      ll = getattr(pl.loggers, l.pop('class'))
      ls[i] = ll(**l)
    trainer_kwargs_['logger'] = ls
    trainer = pl.Trainer(**trainer_kwargs_)
    mlflow.pytorch.autolog(log_models=False)
    # res = trainer.tuner.lr_find(
    #   model, train_dataloaders=train_dataloader,
    #   val_dataloaders=val_dataloader, 
    #   early_stop_threshold=1000.0,
    #   max_lr=0.3)
    # lr = res.suggestion()
    # Dataloader
    dl_t = ds_t.to_dataloader(**dataloader_kwargs['train'])
    dl_v= ds_v.to_dataloader(**dataloader_kwargs['val'])
    # Fit and Log
    with mlflow.start_run() as run:
      trainer.fit(model, train_dataloaders=dl_t, val_dataloaders=dl_v)
      best_path = Path(trainer.checkpoint_callback.best_model_path)
      cwd = Path(hydra_config.runtime.cwd)
      conf_path = cwd / 'conf'
      config_name = hydra_config.job.config_name
      model_path = best_path.rename('model.ckpt')
      # model_path = cwd / 'model.ckpt'
      # model_path.symlink_to(best_path)
      # shutil.copyfile(best_path, model_path)
      mlflow_model = CCFModel(config_name=config_name)
      model_info = mlflow.pyfunc.log_model(artifact_path=model_name, 
                                           registered_model_name=model_name,
                                           python_model=mlflow_model,
                                           artifacts={'conf': str(conf_path), 
                                                      'model': str(model_path)})
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    
  
@hydra.main()
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  hydra_cfg = HydraConfig.get()
  main(hydra_config=hydra_cfg, **OmegaConf.to_object(cfg))
  
  
if __name__ == "__main__":
  app()
