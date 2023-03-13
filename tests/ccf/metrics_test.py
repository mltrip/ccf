import pytest
import torch

import ccf.metrics as metrics


def test_ror_value_qs():
  m = metrics.ROR(quantiles=[0.25, 0.5, 0.75], 
                  target_kind='value',
                  direction='qs',
                  fees=0.,
                  threshold=0.,
                  mask_kind='none',
                  stop_loss=None,
                  take_profit=None)
  print(m.name)
  # n_samples x prediction_horizon (3 x 4)
  target = torch.tensor([
    [100., 50., 150., 100.],  
    [100., 50., 150., 100.],
    [100., 50., 150., 100.]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  y_pred = torch.tensor([
    [[100., 100., 100.],
     [50., 50., 50.],
     [150., 150., 150.],
     [100., 100., 100.]],
    [[100., 100., 100.],
     [150., 150., 150.],
     [50., 50., 50.],
     [100., 100., 100.]],
    [[100., 100., 100.],
     [100., 100., 100.],
     [100., 100., 100.],
     [100., 100., 100.]]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  losses_actual = m.loss(y_pred, target)
  print(losses_actual)
  losses_expected = torch.tensor([
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.5000, 0.5000],
     [0.5000, 0.5000, 0.0000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [-0.5000, -0.5000, 0.0000],
     [0.0000, -0.5000, -0.5000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000]]])
  torch.testing.assert_close(losses_actual, losses_expected)

  
def test_ror_value_qs_last():
  m = metrics.ROR(quantiles=[0.25, 0.5, 0.75], 
                  target_kind='value',
                  direction='qs',
                  fees=0.,
                  threshold=0.,
                  mask_kind='last',
                  stop_loss=None,
                  take_profit=None)
  print(m.name)
  # n_samples x prediction_horizon (3 x 4)
  target = torch.tensor([
    [100., 50., 150., 100.],  
    [100., 50., 150., 100.],
    [100., 50., 150., 100.],
    [100., 50., 150., 100.],
    [100., 50., 150., 100.]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  y_pred = torch.tensor([
    [[100., 100., 100.],
     [50., 50., 50.],
     [150., 150., 150.],
     [100., 100., 100.]],
    [[100., 100., 100.],
     [150., 150., 150.],
     [50., 50., 50.],
     [100., 100., 100.]],
    [[100., 100., 100.],
     [100., 100., 100.],
     [100., 100., 100.],
     [100., 100., 100.]],
    [[100., 100., 100.],
     [50., 50., 50.],
     [150., 150., 150.],
     [150., 50., 50.]],
    [[100., 100., 100.],
     [50., 50., 50.],
     [150., 150., 150.],
     [50., 150., 150.]]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  losses_actual = m.loss(y_pred, target)
  print(losses_actual)
  losses_expected = torch.tensor([
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [-0.5000, 0.5000, 0.5000],
     [0.5000, -0.5000, -0.5000],
     [0.0000, 0.0000, 0.000]],
    [[0.0000, 0.0000, 0.0000],
     [0.0000, -0.5000, 0.0000],
     [0.0000, 0.5000, 0.0000],
     [0.0000, 0.0000, 0.0000]]])
  torch.testing.assert_close(losses_actual, losses_expected)  
  
  
def test_ror_rat_qs():
  m = metrics.ROR(quantiles=[0.25, 0.5, 0.75], 
                  target_kind='rat',
                  direction='qs',
                  fees=0.,
                  threshold=0.,
                  mask_kind='none',
                  stop_loss=None,
                  take_profit=None)
  print(m.name)
  # n_samples x prediction_horizon (3 x 4)
  # 100., 50., 150., 100.
  target = torch.tensor([
    [1.0, 0.5, 3.0, 0.666667],  
    [1.0, 0.5, 3.0, 0.666667],
    [1.0, 0.5, 3.0, 0.666667]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  y_pred = torch.tensor([
    [[1.0, 1.0, 1.0],
     [0.5, 0.5, 0.5],
     [3.0, 3.0, 3.0],
     [0.666667, 0.666667, 0.666667]],
    [[1.0, 1.0, 1.0],
     [1.5, 1.5, 1.5],
     [0.333333, 0.333333, 0.333333],
     [2.0, 2.0, 2.0]],
    [[1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0]]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  losses_actual = m.loss(y_pred, target)
  print(losses_actual)
  losses_expected = torch.tensor([
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.5000, 0.5000],
     [0.5000, 0.5000, 0.0000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [-0.5000, -0.5000, 0.0000],
     [0.0000, -0.5000, -0.5000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000]]])
  torch.testing.assert_close(losses_actual, losses_expected)

  
def test_ror_rel_qs():
  m = metrics.ROR(quantiles=[0.25, 0.5, 0.75], 
                  target_kind='rel',
                  direction='qs',
                  fees=0.,
                  threshold=0.,
                  mask_kind='none',
                  stop_loss=None,
                  take_profit=None)
  print(m.name)
  # n_samples x prediction_horizon (3 x 4)
  # 100., 50., 150., 100.
  target = torch.tensor([
    [1.0, -0.5, 2.0, -0.333333],  
    [1.0, -0.5, 2.0, -0.333333],
    [1.0, -0.5, 2.0, -0.333333]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  y_pred = torch.tensor([
    [[1.0, 1.0, 1.0],
     [-0.5, -0.5, -0.5],
     [2.0, 2.0, 2.0],
     [-0.333333, -0.333333, -0.333333]],
    [[1.0, 1.0, 1.0],
     [0.5, 0.5, 0.5],
     [-0.666667, -0.666667, -0.666667],
     [1.0, 1.0, 1.0]],
    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]]])
  # n_samples x prediction_horizon x n_outputs (3 x 4 x 3)
  losses_actual = m.loss(y_pred, target)
  print(losses_actual)
  losses_expected = torch.tensor([
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.5000, 0.5000],
     [0.5000, 0.5000, 0.0000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [-0.5000, -0.5000, 0.0000],
     [0.0000, -0.5000, -0.5000],
     [0.0000, 0.0000, 0.0000]],
    [[0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000],
     [0.0000, 0.0000, 0.0000]]])
  torch.testing.assert_close(losses_actual, losses_expected)