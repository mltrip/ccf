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
from ccf.utils import rat2val


def make_time_charts(data, accumulate=False):
  charts = []
  for name, df in data.items():
    value_vars = df.columns
    if accumulate:
      now = datetime.utcnow()
      future_mask = df.index > now
      past_mask = ~future_mask
      for c in df:
        parts = []
        if past_mask.sum():
          past = rat2val(df[c][past_mask])
          parts.append(past)
        if future_mask.sum():
          future = rat2val(df[c][future_mask])
          parts.append(future)
        if len(parts) > 0:
          df[c] = pd.concat(parts)
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


def make_metrics_box(data):
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
        chart = alt.Chart(dff, title=f'{n} {label} {group} {len(dff)}').mark_boxplot().encode(
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


def make_metrics_heatmap(data):
  charts = []
  for n, d in data.items():
    for nn, df in d.items():
      # ['horizon', 'group', 'metric', 'label', 'value', 'model', 'prediction', 'kind', 'target']
      for (kind, group, label), dff in df.groupby(['kind', 'group', 'label']):
        dff['name'] = dff['model']
        # base = alt.Chart(dff, title=f'{n} {group} {label} {kind} {len(dff)}').transform_joinaggregate(
        #   mean_value='mean(value)',
        #   count_value='count(value)',
        #   groupby=['horizon', 'name']
        base = alt.Chart(dff, title=f'{n} {group} {label} {kind} {len(dff)}').transform_aggregate(
          mean_value='mean(value)',
          groupby=['horizon', 'name']
        ).encode(
          alt.X('horizon:O', scale=alt.Scale(paddingInner=0)),
          alt.Y('name:N', scale=alt.Scale(paddingInner=0)),
        )
        heatmap = base.mark_rect().encode(
          color=alt.Color('mean_value:Q',
            scale=alt.Scale(scheme='redyellowgreen', reverse=True),
            legend=alt.Legend(direction='vertical')
          )
        )
        text = base.mark_text(baseline='middle').encode(
          text='mean_value:Q',
        )
        # heatmap2 = base.mark_rect().encode(
        #   color=alt.Color('count_value:Q',
        #     scale=alt.Scale(scheme='redyellowgreen', reverse=True),
        #     legend=alt.Legend(direction='vertical')
        #   )
        # )
        # text2 = base.mark_text(baseline='middle').encode(
        #   text='count_value:O',
        # )
        c1 = (heatmap + text).properties(width=800, height=400)
        # c2 = (heatmap2 + text2).properties(width=400, height=400)
        # chart = c1 | c2
        charts.append(c1)
  return charts


def ui(read_data_kwargs=None, read_metrics_kwargs=None, 
       delay=0, accumulate=False, metrics_kind='heatmap'):
  st.set_page_config(
    page_title="CryptoCurrency Forecasting",
    page_icon="₿",
    layout="wide")
  st.title("CryptoCurrency Forecasting")
  placeholder = st.empty()
  while True:
    print(datetime.utcnow())
    t0 = time.time()
    # placeholder.empty()
    if read_data_kwargs is not None:
      data = read_data(**read_data_kwargs)
      charts = make_time_charts(data, accumulate)
      with placeholder.container():
          for chart in charts:
            st.altair_chart(chart, use_container_width=True)
    if read_metrics_kwargs is not None:
      metrics_data = read_data(**read_metrics_kwargs)
      if metrics_kind == 'box':
        metrics_charts = make_metrics_box(metrics_data)
      elif metrics_kind == 'heatmap':
        metrics_charts = make_metrics_heatmap(metrics_data)
      else:
        raise NotImplementedError(metrics_kind)
      with placeholder.container():
        for chart in metrics_charts:
          st.altair_chart(chart, use_container_width=False)
    dt = time.time() - t0
    wt = max(0, delay - dt)
    print(f'dt: {dt:.3f}, wt: {wt:.3f}')
    time.sleep(wt)
  
  
if __name__ == "__main__":
  cfg = sys.argv[1] if len(sys.argv) > 1 else 'ui.yaml'
  with open(cfg) as f:
    kwargs = yaml.safe_load(f)
  ui(**kwargs)