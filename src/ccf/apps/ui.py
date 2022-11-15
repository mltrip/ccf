import sys
import time
from datetime import datetime, timedelta
from copy import deepcopy

import numpy as np
import pandas as pd
import altair as alt
# import plotly.express as px
from sqlalchemy import create_engine
import streamlit as st
import yaml
# import plotly.express as px

from ccf.read_data import read_data
from ccf.utils import rat2val


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


def make_metrics_charts(data):
  charts = []
  for n, d in data.items():
    for nn, df in d.items():
      # print(n, nn)
      # print(df.describe(include='all'))
      # ['horizon', 'group', 'metric', 'label', 'value', 'model', 'prediction', 'kind', 'target']
      # for (label, group, horizon), dff in df.groupby(['label', 'group', 'horizon']):
      # for (label, group, kind), dff in df.groupby(['label', 'group', 'kind']):
      for (label, group), dff in df.groupby(['label', 'group']):
        dff['time'] = dff.index
        # dff['color'] = dff['model'] + ' ' + dff['target']
        # lines = (
        #   alt.Chart(dff, title=f'{n} {label} {group} {horizon}')
        #   .mark_line()
        #   .encode(
        #       x='time',
        #       y=alt.Y('value', 
        #               scale=alt.Scale(
        #               zero=False,
        #               ),
        #              ),
        #       color=alt.Color('color', scale=alt.Scale(scheme='category20')),
        #   )
        # )
        chart = alt.Chart(dff, title=f'{n} {label} {group}').mark_boxplot().encode(
          x='model:N',
          y='value:Q',
          row='kind:N',
          column='horizon:O',
          # color=alt.Color('model:N', scale=alt.Scale(scheme='category20'))
          color='model:N').properties(
          width=100,
          height=100).resolve_scale(y='independent')
        charts.append(chart)
  return charts


def ui(read_data_kwargs=None, read_metrics_kwargs=None, delay=0, accumulate=False):
  st.set_page_config(
      page_title="CryptoCurrency Forecasting",
      page_icon="₿",
      layout="wide")
  st.title("CryptoCurrency Forecasting")
  placeholder = st.empty()
  while True:
    if read_data_kwargs is not None:
      data = read_data(**read_data_kwargs)
      charts = make_time_charts(data, accumulate)
      with placeholder.container():
          for chart in charts:
            st.altair_chart(chart, use_container_width=True)
    if read_metrics_kwargs is not None:
      metrics_data = read_data(**read_metrics_kwargs)
      metrics_charts = make_metrics_charts(metrics_data)
      with placeholder.container():
        for chart in metrics_charts:
          st.altair_chart(chart, use_container_width=False)
    time.sleep(delay)
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'ui.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  ui(**kwargs)