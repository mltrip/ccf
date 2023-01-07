import sys
import time
from datetime import datetime, timedelta
from copy import deepcopy
from pathlib import Path
from pprint import pprint
import json
from collections import deque

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
from sqlalchemy import create_engine
import streamlit as st
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from ccf import partitioners as ccf_partitioners


def main(partitioner, consumer, topics, keys, maxsize, horizon):
  st.set_page_config(
    page_title="Crypto Currency Forecasting",
    page_icon="₿",
    layout="wide")
  partitioner_class = partitioner.pop('class')
  partitioner = getattr(ccf_partitioners, partitioner_class)(**partitioner)
  partitioner.update()
  consumer['key_deserializer'] = partitioner.deserialize_key
  consumer['value_deserializer'] = partitioner.deserialize_value
  consumer = KafkaConsumer(**consumer)
  partitions = list(set([y for x in keys for y in partitioner[x]]))
  topic_partitions = [TopicPartition(x, y) for x in topics for y in partitions]
  consumer.assign(topic_partitions)
  threshold = 0.01*st.number_input('Decision threshold, %', min_value=0., value=0., format='%.4f')
  placeholder = st.empty()
  data = deque()
  for message in consumer:
    value = message.value
    if message.topic == 'prediction' and value.get('horizon', None) == horizon:
      value['prediction'] = value['value']
      value['actual'] = None
      if len(data) >= horizon:
        data[-horizon]['actual'] = value['last']
      data.append(value)  
      df = pd.DataFrame(data)
      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
      df = df.set_index('timestamp', drop=False)
      with placeholder.container():
        # st.json(json.dumps(value))
        # st.dataframe(df)
        # https://www.w3.org/wiki/CSS/Properties/color/keywords
        change = value['prediction'] / value['last'] - 1
        if change > threshold:
          st.success('BUY')
        elif change < -threshold:
          st.error('SELL')
        else:
          st.warning('HOLD')
        df_melt = df.melt(value_name='price', var_name='type', 
                          id_vars=['timestamp'], value_vars=['last', 'prediction', 'actual'])
        time_horizon = timedelta(seconds=horizon*value['quant']/1e9)
        col_last, col_prediction = st.columns(2)
        col_last.metric(f'{value["exchange"]}-{value["base"]}-{value["quote"]} last', 
                        value=f'{value["last"]:.4f}',
                        delta=None)
        col_prediction.metric(f'{value["exchange"]}-{value["base"]}-{value["quote"]} with horizon {time_horizon}', 
                              value=f'{value["prediction"]:.4f}',
                              delta=f'{change*100:.4f}%')
        fig = px.line(df_melt, x='timestamp', y='price', 
                      color='type', template='plotly_dark')        
        st.plotly_chart(fig, use_container_width=True,
                        sharing="streamlit", theme="streamlit")
      if len(df) == maxsize:
        data.popleft()

  
if __name__ == "__main__":
  config_path = sys.argv[1] if len(sys.argv) > 1 else 'conf/ui.yaml'
  #   with open(config_path) as f:
  #     config = yaml.safe_load(f)
  config_path = Path(config_path)
  config_name = config_path.stem
  config_dir = Path(config_path.parts[0]).resolve()
  print(f'config_name: {config_name}\nconfig_dir: {config_dir}')
  # GlobalHydra.instance().clear()
  with initialize_config_dir(config_dir=str(config_dir)):
    config = compose(config_name=config_name)
  kwargs = OmegaConf.to_object(config)
  pprint(kwargs)
  main(**kwargs)
