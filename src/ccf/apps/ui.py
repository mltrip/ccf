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

from ccf.read_data import read_data


def make_time_charts(data, accumulate=False):
  charts = []
  for name, df in data.items():
    value_vars = df.columns
    if accumulate:
      for c in df:
        if 'pct_' in c:
          df[c] = df[c].cumsum()
        elif 'lograt_' in c:
          df[c] = np.exp(df[c].cumsum())
        elif 'rat_' in c:
          df[c] = df[c].cumprod()
    df['time'] = df.index
    df = df.melt(id_vars=['time'], 
                 value_vars=value_vars, 
                 var_name='variable', 
                 value_name='value')
    lines = (
      alt.Chart(df, title=f'{name}')
      .mark_line()
      .encode(
          x='time',
          y=alt.Y('value', 
                  scale=alt.Scale(
                    zero=False,
                  ),
                 ),
          color=alt.Color('variable', scale=alt.Scale(scheme='category20')),
      )
    )
    charts.append(lines)
  return charts


def ui(read_data_kwargs, delay=0, accumulate=False):
  st.set_page_config(
      page_title="CryptoCurrency Forecasting",
      page_icon="₿",
      layout="wide")
  st.title("CryptoCurrency Forecasting")
  placeholder = st.empty()
  while True:
    data = read_data(**read_data_kwargs)
    charts = make_time_charts(data, accumulate)
    with placeholder.container():
      for chart in charts:
        st.altair_chart(chart, use_container_width=True)
    time.sleep(delay)
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'ui.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  ui(**kwargs)