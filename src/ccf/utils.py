import concurrent.futures
from datetime import datetime, timedelta
import os
import psutil
import re
import signal
import time
import warnings

import numpy as np
import pandas as pd
from pytorch_forecasting.metrics import base_metrics as pf_base_metrics
from pytorch_forecasting import metrics as pf_metrics

from ccf import metrics as ccf_metrics


def expand_columns(ref_columns, columns):
  new_columns = []
  for c in columns:
    n = len(new_columns)
    for cc in ref_columns:
      if re.match(c, cc):
        new_columns.append(cc)
    if len(new_columns) == n:
      warnings.warn(f'Warning! No columns found with pattern "{c}"') 
  if len(new_columns) == 0 and len(columns) != 0:
    warnings.warn(f'Warning! No columns found with patterns: {columns}')
  new_columns = list(set(new_columns))  # remove duplicates
  return new_columns


def delta2value(deltas, kind, initial_value=1):
  if kind == 'lograt':
    deltas = np.exp(np.cumsum(deltas))
  elif kind == 'rat':
    deltas = np.cumprod(deltas)
  elif kind == 'rel':
    deltas = np.cumprod(1 + deltas)
  elif kind == 'value':
    return deltas
  else:
    raise NotImplementedError(kind)
  return deltas * initial_value


def kill_child_processes(parent_pid, sig=signal.SIGTERM, kill_parent=True):
  try:
    parent = psutil.Process(parent_pid)
  except psutil.NoSuchProcess:
    return
  children = parent.children(recursive=True)
  for process in children:
    process.send_signal(sig)
  if kill_parent:
    parent.send_signal(sig)

    
def wait_first_future(executor, futures):
  for f in concurrent.futures.as_completed(futures):
    try:
      r = f.result()
    except Exception as e:
      print(f'Exception of {f}: {e}')
    else:
      print(f'Result of {f}: {r}')
    finally:  # Shutdown pool with all futures and kill all children processes
      executor.shutdown(wait=False, cancel_futures=True)
      kill_child_processes(os.getpid())

      
def initialize_time(start, stop, size, quant):
  """Convert different combinations of start/stop/size/quant to Unix time in nanoseconds"""
  if start is not None:
    if isinstance(start, str):
      try:
        start = datetime.fromisoformat(start)
      except Exception:
        try:
          start = int(float(start))
        except Exception:
          raise NotImplementedError(start)
    else:
      start = int(start)
  if stop is not None:
    if isinstance(stop, str):
      try:
        stop = datetime.fromisoformat(stop)
      except Exception:
        try:
          stop = int(float(stop))
        except Exception:
          raise NotImplementedError(stop)
    else:
      stop = int(stop)
  if size is not None:
    if isinstance(size, str):
      size = int(float(size))
    size = int(size)
  if quant is not None:
    if isinstance(quant, str):
      quant = int(float(quant))
    quant = int(quant)
  if start is not None and stop is None and size is None and quant is None:
    if isinstance(start, int) and start < 0:
      start = time.time_ns() + start
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
    stop = time.time_ns() 
  elif start is not None and stop is None and size is not None and quant is not None:
    if isinstance(start, int) and start < 0:
      start = time.time_ns() + start*quant
      stop = start + size*quant
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
      stop = start + size*quant
  elif start is not None and stop is None and size is None and quant is not None:
    if isinstance(start, int) and start < 0:
      start = time.time_ns() + start*quant
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
  elif start is not None and stop is None and size is None and quant is None:
    if isinstance(start, int) and start < 0:
      start = time.time_ns() + start
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
  elif start is not None and stop is not None and size is None and quant is not None:
    if isinstance(start, int) and start < 0:
      start = time.time_ns() + start
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
    if isinstance(stop, int) and stop < 0:
      stop = time.time_ns() + stop
    elif isinstance(stop, datetime):
      stop = int(stop.timestamp())*int(10**9)
  # else:
  #   raise NotImplementedError(start, stop, size, quant)
  return start, stop, size, quant


def initialize_metric(metric):
  if isinstance(metric, dict):
    metric_class_name = metric.pop('class')
  elif isinstance(metric, str):
    metric_class_name = metric
    metric = {}
  else:
    raise NotImplementedError(metric)
  if metric_class_name == 'AggregationMetric':
    metric['metric'] = initialize_metric(metric['metric'])
  if metric_class_name in ['CompositeMetric', 'CompositeMetricFix']:
    metric['metrics'] = [initialize_metric(x) for x in metric['metrics']]
  metric_class = getattr(pf_metrics, metric_class_name, None)
  if metric_class is None:
    metric_class = getattr(pf_base_metrics, metric_class_name, None)
  if metric_class is None:
    metric_class = getattr(ccf_metrics, metric_class_name, None)
  if metric_class is None:
    raise NotImplementedError(metric_class_name)
  metric = metric_class(**metric)
  if metric_class_name == 'AggregationMetric':  # TODO PR to PF?
    metric.name = f'Agg{metric.metric.name}'
  return metric
