import os
import sys
from pprint import pprint
import hmac
import hashlib
import requests
from tqdm import tqdm
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
import pandas as pd
import uuid
import plotly.express as px


def run_local(benchmark_path, executable_path, conf_path, config_path, env_path): 
  return 0
  
  
def run_pbs(benchmark_path, executable_path, conf_path, config_path, env_path):
  pbs_script = f"""
    #PBS -q workq
    #PBS -l nodes=1:ppn=1
    #PBS -l mem=3gb
    ##PBS -l walltime=48:00:00
    #PBS -N {benchmark_path.stem}
    #PBS -o {config_path.stem}.out
    #PBS -e {config_path.stem}.err

    echo "Start"
    date
    hostname
    echo PBS_JOBID $PBS_JOBID 
    echo PBS_JOBNAME $PBS_JOBNAME
    echo PBS_O_WORKDIR $PBS_O_WORKDIR
    echo TMPDIR $TMPDIR
    cd $PBS_O_WORKDIR
    pwd
    echo "Env"
    export $(cat {env_path.resolve()} | xargs)
    echo "Tunnel"
    exists=$(lsof -i -n | grep 127.0.0.1:3128 | wc -L)
    if [ "$exists" != "0" ]; then
        echo "Tunnel already exists!"
    else
        echo "Tunnel doesn't exist"
        echo "Creating tunnel"
        ssh -f -L 3128:10.254.50.30:3128 m01 sleep 86400000
        echo "Done"
    fi
    echo "List of all ssh connections"
    lsof -i -n | grep ssh
    export HTTP_PROXY="127.0.0.1:3128"
    export HTTPS_PROXY="127.0.0.1:3128"
    echo HTTP_PROXY $HTTP_PROXY
    echo HTTPS_PROXY $HTTPS_PROXY
    echo "Activate conda"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ccf_ml01
    echo "Run"
    python {executable_path.resolve()} -cd {conf_path.resolve()} -cn {config_path.resolve().stem}
    echo "Deactivate conda"
    conda deactivate
    date
    echo "End"
    """
  pbs_script_path = benchmark_path / config_path.with_suffix('.sh').name
  with open(pbs_script_path, 'w') as f:
    f.write(pbs_script)
  args = ['qsub', pbs_script_path.resolve()]
  print(args)
  result = subprocess.run(args, cwd=benchmark_path.resolve())
  return result
  
  
def run(benchmark_name, n_runs, executable_path, train_config, train_configs, frames, 
        env_path, benchmarks_dir='benchmarks', conf_dir='conf', run_kind='local', 
        reset_benchmark=False, reset_conf=False, delay=0.0):
  benchmark_path = Path(benchmarks_dir) / benchmark_name
  print(benchmark_path)
  if reset_benchmark and benchmark_path.exists():
    shutil.rmtree(benchmark_path)     
  benchmark_path.mkdir(parents=True, exist_ok=True)
  conf_path = benchmark_path / conf_dir
  if reset_conf and conf_path.exists():
    shutil.rmtree(conf_path)
  if reset_benchmark or reset_conf:
    shutil.copytree(conf_dir, conf_path, dirs_exist_ok=True)
  env_path = Path(env_path)
  executable_path = Path(executable_path)
  # Run
  runs_total = len(frames)*len(train_configs)*n_runs
  run_cnt = 0
  for frame_name, frame in frames.items():
    for config_name, config in train_configs.items():
      for run_i in range(n_runs):
        run_cnt += 1
        run_id = str(uuid.uuid4().hex)
        print(f'\n{run_cnt}/{runs_total} {frame_name} {config_name} {run_i + 1}/{n_runs} {run_id}')
        name_suffix = '-'.join([frame_name, config_name, run_id])
        config_path = (conf_path / config).with_suffix('.yaml')
        cfg = OmegaConf.load(config_path)
        new_cfg = OmegaConf.merge(cfg, train_config)
        new_cfg['create_dataset_kwargs']['start'] = frame['start']
        new_cfg['create_dataset_kwargs']['stop'] = frame['stop']
        new_cfg['create_dataset_kwargs']['split'] = frame['split']
        new_cfg['model_name'] = new_cfg['model_name'] + f'-{name_suffix}'
        new_config_path = config_path.with_stem(config_path.stem + f'-{name_suffix}')
        OmegaConf.save(new_cfg, new_config_path)
        if run_kind == 'local':
          result = run_local(benchmark_path, executable_path, 
                             conf_path, new_config_path, env_path)
        elif run_kind == 'pbs':
          result = run_pbs(benchmark_path, executable_path, 
                           conf_path, new_config_path, env_path)
        else:
          raise NotImplementedError(f'run kind: {run_kind}')
        print(result)
        time.sleep(delay)
  
  
def plot(benchmark_name, benchmarks_dir='benchmarks', 
         reset_benchmark=False, reset_conf=False,
         metrics=None, update_results=True, results_name='results',
         layout_kwargs=None):
  metrics = {} if metrics is None else metrics
  layout_kwargs = {} if layout_kwargs is None else layout_kwargs
  benchmark_path = Path(benchmarks_dir) / benchmark_name
  results_path = benchmark_path / f'{results_name}.csv'
  if update_results or not results_path.exists():
    results_paths = list(benchmark_path.glob('*.json'))
    print(f'Number of results: {len(results_paths)}')
    print('Loading results')
    results = {}
    for p in results_paths:
      with open(p) as f:
        result = json.load(f)
      results[p.stem] = result
    print('Processing results')
    samples = []
    for name, results in results.items():
      for metric_name, metric in metrics.items():
        tokens = name.split('-')
        sample = {
          'model': '-'.join(tokens[:-5]),
          'frame': tokens[-5],
          'train': tokens[-4],
          'run': tokens[-3],
          'suffix': tokens[-2],
          'kind': tokens[-1]}
        sample['metric'] = metric_name
        sample['value'] = results[metric]
        samples.append(sample)
    print('Writing results')
    df = pd.DataFrame(samples)
    df.to_csv(results_path)
  elif results_path.exists():
    print('Loading results')
    df = pd.read_csv(results_path)
  print(df)
  print('Trains')
  print(df['train'].value_counts())
  print('Frames')
  print(df['frame'].value_counts())
  print('Metrics')
  print(df['metric'].value_counts())
  print('Plotting box plots')
  pbar = tqdm(metrics.items())
  for metric_name, metric in pbar:
    pbar.set_description(f'{metric_name}')
    for kind in ['test', 'train']:
      df2 = df[(df['metric'] == metric_name) & (df['kind'] == kind)]
      fig = px.box(df2, 
                   x='train',
                   y='value', 
                   color="train",
                   facet_row='frame',
                   points='all',
                   title=f"Box plot of {kind} {metric_name}")
      # fig.update_yaxes(matches=None)
      fig.update_layout(**layout_kwargs)
      fig_path = benchmark_path / f'{metric_name}-{kind}.html'
      fig.write_html(fig_path)
      train_order = list(df2.groupby('train').agg({'value': 'median'}).sort_values(by='value').index)
      fig_all = px.box(df2, 
                   x='train',
                   y='value', 
                   color="train",
                   points='all',
                   category_orders={'train': train_order},
                   title=f"Box plot of {kind} {metric_name} by all frames")
      fig_all.update_layout(**layout_kwargs)
      fig_path = benchmark_path / f'{metric_name}-{kind}-all_frames.html'
      fig_all.write_html(fig_path)
    

def main(benchmark_kwargs, run_kwargs, plot_kwargs,
         do_plot=False, do_run=False):
  if do_run:
    run(**benchmark_kwargs, **run_kwargs)
  if do_plot:
    plot(**benchmark_kwargs, **plot_kwargs)

  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  main(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()  

