import os
import sys
from pprint import pprint
import hmac
import hashlib
import requests
import json
import yaml
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import shutil
import os
import subprocess
import numpy as np


def plot(benchmark_name, results):
  for frame, configs in results.items():
    for config, stats in configs.items():
      for kind, runs in stats.items():
        returns = [x['Return [%]'] for x in runs]
        print(frame, config, kind, len(runs), np.mean(returns), returns)
  
  
def run(benchmark_name, n_runs, train_script, train_config, train_configs, frames, 
        benchmarks_dir='benchmarks', conf_dir='conf', pythonpath=None):
  b_dir = Path(benchmarks_dir) / benchmark_name
  print(b_dir)
  b_dir.mkdir(parents=True, exist_ok=True)
  b_conf_dir = b_dir / conf_dir
  if b_conf_dir.exists():
    shutil.rmtree(b_conf_dir)
  shutil.copytree(conf_dir, b_conf_dir)
  t_script = Path(train_script)
  results = {}
  for frame_name, frame in frames.items():
    frame_results = results.setdefault(frame_name, {})
    for config_name, config in train_configs.items():
      config_results = frame_results.setdefault(config_name, {})
      train_results = config_results.setdefault('train', [])
      test_results = config_results.setdefault('test', [])
      for run_i in range(n_runs):
        print(frame_name, config_name, run_i + 1, n_runs)
        name_suffix = '-'.join([frame_name, config_name, str(run_i + 1)])
        print(name_suffix)
        print(config)
        config_path = (b_conf_dir / config).with_suffix('.yaml')
        cfg = OmegaConf.load(config_path)
        print(cfg)
        new_cfg = OmegaConf.merge(cfg, train_config)
        new_cfg['create_dataset_kwargs']['start'] = frame['start']
        new_cfg['create_dataset_kwargs']['stop'] = frame['stop']
        new_cfg['create_dataset_kwargs']['split'] = frame['split']
        new_cfg['model_name'] = new_cfg['model_name'] + f'-{name_suffix}'
        print(new_cfg)
        new_config_path = config_path.with_stem(config_path.stem + f'-{name_suffix}')
        print(new_config_path)
        OmegaConf.save(new_cfg, new_config_path)
        train_stats_path = b_dir / (new_cfg['model_name'] + new_cfg['bt_suffix'] + '-train' + '.json')
        test_stats_path = b_dir / (new_cfg['model_name'] + new_cfg['bt_suffix'] + '-test' + '.json')
        print(train_stats_path)
        print(test_stats_path)
        args = ['python', t_script.resolve(), '-cd', b_conf_dir.resolve(), '-cn', new_config_path.stem]
        print(args)
        env = dict(os.environ)
        if pythonpath is not None:
          env['PYTHONPATH'] = Path(pythonpath).resolve()
        result = subprocess.run(args, cwd=b_dir.resolve(), env=env)
        with open(train_stats_path) as f:
          train_stats = json.load(f)
        with open(test_stats_path) as f:
          test_stats = json.load(f)
        train_results.append(train_stats)
        test_results.append(test_stats)
  return results
  
  
def main(benchmark_name, run_kwargs, plot_kwargs):
  results_path = Path(benchmark_name).with_suffix('.json')
  if results_path.exists():
    print(f'Results of benchmark already exists: {results_path}')
    print('Loading results')
    with open(results_path) as f:
      results = json.load(f)
  else:
    print('Running benchmark')
    results = run(benchmark_name, **run_kwargs)
    print('Saving results')
    with open(results_path, 'w') as f:
      json.dump(results, f)
  print('Plotting results')
  plot(benchmark_name, results, **plot_kwargs)
  
  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  main(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()  

