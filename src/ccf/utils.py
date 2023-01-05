import concurrent.futures
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
