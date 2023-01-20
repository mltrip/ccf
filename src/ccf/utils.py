import concurrent.futures
from datetime import datetime, timedelta
import time
import os
import psutil
import re
import signal
import warnings

import pandas as pd
import numpy as np


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
  if start is not None:
    if isinstance(start, str):
      try:
        start = datetime.fromisoformat(start)
      except Exception:
        try:
          start = int(start)
        except Exception:
          raise NotImplementedError(start)
  if stop is not None:
    if isinstance(stop, str):
      try:
        stop = datetime.fromisoformat(stop)
      except Exception:
        try:
          stop = int(stop)
        except Exception:
          raise NotImplementedError(stop)
  if size is not None:
    if isinstance(start, str):
      size = int(size)
    size = int(size)
  if quant is not None:
    if isinstance(quant, str):
      quant = int(float(quant))
    quant = int(quant)
  if start is not None and stop is None and size is not None and quant is not None:
    if isinstance(start, int):
      start = time.time_ns() + start*quant
      stop = start + size*quant
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
      stop = start + size*quant
  elif start is not None and stop is None and size is None and quant is not None:
    if isinstance(start, int):
      start = time.time_ns() + start*quant
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
  elif start is not None and stop is None and size is None and quant is None:
    if isinstance(start, int):
      start = time.time_ns() + start
    elif isinstance(start, datetime):
      start = int(start.timestamp())*int(10**9)
  # else:
  #   raise NotImplementedError(start, stop, size, quant)
  return start, stop, size, quant
