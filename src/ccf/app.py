import sys
import time
from datetime import datetime, timedelta
from copy import deepcopy

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
from sqlalchemy import create_engine
import streamlit as st
import yaml


def get_data(past, engine_kwargs, read_kwargs, group=None):
  ek = deepcopy(engine_kwargs)
  rk = deepcopy(read_kwargs)
  rk['index_col'] = 'time'
  rk['parse_dates'] = ['time']
  name = rk.pop('name')
  cols = rk.pop('columns', None)
  now = datetime.utcnow()
  start = now + timedelta(seconds=-past)
  sql_cols = ','.join([f'`{x}`' for x in cols]) if cols is not None else '*'
  if group is None:
    rk['sql'] = f"SELECT {sql_cols} FROM '{name}' WHERE time > '{start}'"
  else:
    rk['sql'] = f"SELECT {sql_cols} FROM '{name}' WHERE `group` = '{group}' AND time > '{start}'"
  rk['con'] = create_engine(**ek)
  df = pd.read_sql(**rk)
  return df


def get_prices_with_prediction_chart(past, engine_kwargs, read_kwargs, resample_kwargs):
  charts = []
  for g, ek in engine_kwargs.items():
    rk = read_kwargs[g]
    dfo = get_data(past, ek['orderbook'], rk['orderbook'])
    dfo = dfo.resample(**resample_kwargs).last()
    dfp = get_data(past, ek['prediction'], rk['prediction'], group=g)
    dfp = dfp.groupby('time').first()
    if 'pred-m_p' in dfp:
      dfo['m_p'] = 0.5*(dfo['a_p_0'] + dfo['b_p_0'])
    df = pd.concat([dfo, dfp], axis=1)
    # df = df.interpolate(method='pad')
    value_vars = df.columns
    df['time'] = df.index
    df = df.melt(id_vars=['time'], 
                 value_vars=value_vars, 
                 var_name='price', 
                 value_name='value')
    lines = (
      alt.Chart(df, title=f"{g}")
      .mark_line()
      .encode(
          x="time",
          y=alt.Y('value', 
                  scale=alt.Scale(
                    zero=False,
                  ),
                 ),
          color="price",
      )
    )
    charts.append(lines)
  return charts


def app(past, freq, engine_kwargs, read_kwargs, resample_kwargs):
  st.set_page_config(
      page_title="CryptoCurrency Forecasting",
      page_icon="₿",
      layout="wide",
  )
  st.title("CryptoCurrency Forecasting")

  placeholder = st.empty()

  while True:
    charts = get_prices_with_prediction_chart(past, 
                                              engine_kwargs, 
                                              read_kwargs,
                                              resample_kwargs)
    with placeholder.container():
      for chart in charts:
        st.altair_chart(chart, use_container_width=True)
      time.sleep(freq)
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'app.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  app(**kwargs)