import concurrent.futures
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from ccf import agents as ccf_agents
from ccf.utils import wait_first_future


def get_data(agents, executor=None):
  if executor is None:
    executor = {'class': 'ThreadPoolExecutor'}
  executor_class = executor.pop('class')
  executor = getattr(concurrent.futures, executor_class)(**executor)
  futures = []
  for name, agent_kwargs in agents.items():
    agent_class = agent_kwargs.pop('class')
    a = getattr(ccf_agents, agent_class)(**agent_kwargs)
    future = executor.submit(a)
    futures.append(future)
  wait_first_future(executor, futures)

      
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  get_data(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
