import sys
from datetime import datetime, timedelta, timezone
import time
from copy import deepcopy
import gc
import json
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
# from sqlalchemy import create_engine
# from pyspark.sql import SparkSession
# from pyspark.conf import SparkConf

from ccf.read_data import read_data
from ccf.utils import expand_columns
# import talib as ta
# import pandas_ta as ta
import numpy as np
# from pyspark.sql import functions as F
# from pyspark.sql import types as T

from ccf.utils import expand_columns, loop
from ccf import agents as ccf_agents


def extract_features(agents, consumer=None, producer=None, quant=None, executor=None):
  executor = {} if executor is None else executor
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
    print(name)
    pprint(kwargs)
    class_name = kwargs.pop('class')
    agents[name] = getattr(ccf_agents, class_name)(**kwargs)
  # Run agents
  if len(agents) == 1:
    for name, agent in agents.items():
      print(name)
      agent()
  else:
    e = ThreadPoolExecutor(**executor)
    f2c = {}
    for name, agent in agents.items():
      print(name)
      future_kwargs = {}
      future = e.submit(agent, **future_kwargs)
      f2c[future] = [agent, future_kwargs]
    loop(e, f2c)

  
@hydra.main(version_base=None)
def app(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  extract_features(**OmegaConf.to_object(cfg))

  
if __name__ == "__main__":
  app()
