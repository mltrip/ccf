import concurrent.futures

import hydra
from omegaconf import DictConfig, OmegaConf

from ccf import agents as ccf_agents


def get_data(agents, executor_kwargs):
  if executor_kwargs is None:
    executor_kwargs = {'class': 'ThreadPoolExecutor'}
  executor_class = executor_kwargs.pop('class')
  executor = getattr(concurrent.futures, executor_class)(**executor_kwargs)
  futures = []
  for name, agent_kwargs in agents.items():
    agent_class = agent_kwargs.pop('class')
    a = getattr(ccf_agents, agent_class)(**agent_kwargs)
    future = executor.submit(a)
    futures.append(future)
  concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
            

@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  get_data(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
