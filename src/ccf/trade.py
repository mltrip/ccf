import concurrent.futures

import hydra
from omegaconf import DictConfig, OmegaConf

from ccf import agents as ccf_agents
from ccf.utils import wait_first_future


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
  for f in concurrent.futures.as_completed(future_to_name):
    n = future_to_name[f]
    try:
      r = f.result()
    except Exception as e:
      print(f'Exception of {n}: {e}')
      # raise e
    else:
      print(f'Result of {n}: {r}')

    
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  trade(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
