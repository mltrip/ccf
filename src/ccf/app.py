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


def get_data(past, engine_kwargs, read_kwargs):
  ek = deepcopy(engine_kwargs)
  rk = deepcopy(read_kwargs)
  rk['index_col'] = 'time'
  rk['parse_dates'] = ['time']
  name = rk.pop('name')
  cols = rk.pop('columns', None)
  now = datetime.utcnow()
  start = now + timedelta(seconds=-past)
  sql_cols = ','.join(cols) if cols is not None else '*'
  rk['sql'] = f"SELECT {sql_cols} FROM '{name}' WHERE time > '{start}'"
  rk['con'] = create_engine(**ek)
  df = pd.read_sql(**rk)
  return df


def get_prices_chart(past, engine_kwargs, read_kwargs, resample_kwargs):
  dfo = get_data(past, engine_kwargs['orderbook'], read_kwargs['orderbook'])
  dfo = dfo.resample(**resample_kwargs).last()
  if 'a_p_0' in dfo and 'b_p_0' in dfo:
    dfo['m_p'] = 0.5*(dfo['a_p_0'] + dfo['b_p_0'])
  df = dfo
  value_vars = df.columns
  df['time'] = df.index
  df = df.melt(id_vars=['time'], 
               value_vars=['m_p', 'a_p_0', 'b_p_0'], 
               var_name='price', value_name='value')
  lines = (
    alt.Chart(df, title="prices")
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
  return lines


def get_prices_with_prediction_chart(past, engine_kwargs, read_kwargs, resample_kwargs):
  dfo = get_data(past, engine_kwargs['orderbook'], read_kwargs['orderbook'])
  dfo = dfo.resample(**resample_kwargs).last()
  if 'a_p_0' in dfo and 'b_p_0' in dfo:
    dfo['m_p'] = 0.5*(dfo['a_p_0'] + dfo['b_p_0'])
  dfp = get_data(past, engine_kwargs['prediction'], read_kwargs['prediction'])
  dfp = dfp.groupby('time').first()
  df = pd.concat([dfo, dfp], axis=1)
  # df = df.interpolate(method='pad')
  value_vars = df.columns
  df['time'] = df.index
  df = df.melt(id_vars=['time'], 
               value_vars=value_vars, 
               var_name='price', 
               value_name='value')
  lines = (
    alt.Chart(df, title="prices")
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
  return lines


def app(past, freq, engine_kwargs, read_kwargs, resample_kwargs):
  st.set_page_config(
      page_title="CryptoCurrency Forecasting",
      page_icon="₿",
      layout="wide",
  )
  st.title("CryptoCurrency Forecasting")

  placeholder = st.empty()

  while True:
    # ps = get_prices_chart(past, engine_kwargs, read_kwargs, resample_kwargs)
    pps = get_prices_with_prediction_chart(past, 
                                           engine_kwargs, 
                                           read_kwargs,
                                           resample_kwargs)
    with placeholder.container():
      # st.dataframe(df)  
      st.altair_chart(pps, use_container_width=True)
      time.sleep(freq)
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'app.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  app(**kwargs)