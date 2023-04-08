import concurrent.futures
import os
import platform
from pprint import pprint
import psutil
import signal
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from ccf import agents as ccf_agents


def trade(
  agents, executor=None, 
  consumer=None, producer=None, 
  consumer_partitioner=None, producer_partitioner=None
):
  if executor is None:
    executor = {'class': 'ThreadPoolExecutor'}
  if 'max_workers' not in executor:
    executor['max_workers'] = len(agents)
  for name, agent_kwargs in agents.items():
    if consumer is not None and 'consumer' not in agent_kwargs:
      agent_kwargs['consumer'] = consumer
    if producer is not None and 'producer' not in agent_kwargs:
      agent_kwargs['producer'] = producer
    if consumer_partitioner is not None and 'consumer_partitioner' not in agent_kwargs:
      agent_kwargs['consumer_partitioner'] = consumer_partitioner
    if producer_partitioner is not None and 'producer_partitioner' not in agent_kwargs:
      agent_kwargs['producer_partitioner'] = producer_partitioner
  # Initialize
  executor_class = executor.pop('class')
  executor = getattr(concurrent.futures, executor_class)(**executor)
  future_to_name = {}
  for name, agent_kwargs in agents.items():
    agent_class = agent_kwargs.pop('class')
    a = getattr(ccf_agents, agent_class)(**agent_kwargs)
    future_to_name[executor.submit(a)] = name
  # Run
  try:
    for f in concurrent.futures.as_completed(future_to_name):
      n = future_to_name[f]
      try:
        r = f.result()
      except Exception as e:
        print(f'Exception of {n}: {e}')
        raise e
      else:
        # print(f'Result of {n}: {r}')
        raise RuntimeError(f'Result of {n}: {r}')
  except KeyboardInterrupt:
    print("Keyboard interrupt: trade")
    print(f'Executor shutdown with wait')
    executor.shutdown(wait=True, cancel_futures=True)
  except Exception as e:
    print("Exception: trade")
    print(e)
    print(f'Executor shutdown without wait')
    executor.shutdown(wait=False, cancel_futures=True)
    pid = os.getpid()
    print(f'Parent PID: {pid}')
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    print(f'Number of children: {len(children)}')
    if len(children) > 0:
      print(f'Kill child processes')
      if platform.system() == 'Windows':
        sig = signal.CTRL_C_EVENT
      else:
        sig = signal.SIGINT
      for process in children:
        print(process)
        try:
          process.send_signal(sig)  # Send KeyboardInterrupt
        except psutil.NoSuchProcess:
          pass
      print(f'Wait child processes')
      def on_terminate(proc):
        print(f'Process {proc} terminated with exit code {proc.returncode}')
      gone, alive = psutil.wait_procs(children, 
                                      timeout=None,
                                      callback=on_terminate)
      print('Children gone:')
      pprint(gone)
      print('Children alive:')
      pprint(alive)
  finally:
    print('Done: trade')
    sys.exit(0)

    
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  trade(**OmegaConf.to_object(cfg))

  
def handler(signum, frame):
  signame = signal.Signals(signum).name
  raise RuntimeError(f'Signal handler called with signal {signame} ({signum})')
  
  
if __name__ == "__main__":
  signal.signal(signal.SIGTERM, handler)
  signal.signal(signal.SIGINT, handler)
  app()
