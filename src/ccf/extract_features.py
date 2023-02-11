import concurrent.futures

import hydra
from omegaconf import DictConfig, OmegaConf

from ccf import agents as ccf_agents
from ccf.utils import wait_first_future


def extract_features(agents, consumer=None, producer=None, quant=None, executor=None):
  if executor is None:
    executor = {'class': 'ThreadPoolExecutor'}
  quant = int(quant) if quant is not None else quant
  # Initialize agents
  for name, kwargs in agents.items():
    if consumer is not None:
      for c in kwargs['consumers']:
        kwargs['consumers'][c].update(consumer)
    if producer is not None:
      for p in kwargs['producers']:
        kwargs['producers'][p].update(producer)
    if quant is not None:
      kwargs['quant'] = quant
    class_name = kwargs.pop('class')
    agents[name] = getattr(ccf_agents, class_name)(**kwargs)
  # Run agents
  if len(agents) == 1:
    for name, agent in agents.items():
      print(name)
      agent()
  else:
    executor_class = executor.pop('class')
    executor = getattr(concurrent.futures, executor_class)(**executor)
    futures = []
    for name, agent in agents.items():
      print(name)
      future_kwargs = {}
      future = executor.submit(agent, **future_kwargs)
      futures.append(future)
    wait_first_future(executor, futures)

  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  extract_features(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
