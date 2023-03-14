import pytest
import torch

import ccf.transformations as transformations


def test_cumprod():
  forward = transformations.CumProd()
  inverse = transformations.InvCumProd()
  # 1
  x = torch.tensor([1.1, 0.909090, 2.0, 0.5, 1.2, 0.833333])
  y = torch.tensor([1.1, 1.0, 2.0, 1.0, 1.2, 1.0])
  bx = x[None, ...]  # batch x horizon
  by = y[None, ...]  # batch x horizon
  bxq = bx[..., None]  # batch x horizon x quantile
  byq = by[..., None]  # batch x horizon x quantile
  y2 = forward(x)
  torch.testing.assert_close(y, y2)
  bx2 = inverse(by)
  torch.testing.assert_close(bx, bx2)
  bxq2 = inverse(byq) 
  torch.testing.assert_close(bxq, bxq2)
  # Ones
  x = torch.tensor([1.0, 1.0, 1.0])
  y = torch.tensor([1.0, 1.0, 1.0])
  bx = x[None, ...]  # batch x horizon
  by = y[None, ...]  # batch x horizon
  bxq = bx[..., None]  # batch x horizon x quantile
  byq = by[..., None]  # batch x horizon x quantile
  y2 = forward(x)
  torch.testing.assert_close(y, y2)
  bx2 = inverse(by)
  torch.testing.assert_close(bx, bx2)
  bxq2 = inverse(byq) 
  torch.testing.assert_close(bxq, bxq2)
  # Zeros
  x = torch.tensor([0.0, 0.0, 0.0])
  xx = torch.tensor([0.0, torch.nan, torch.nan])
  y = torch.tensor([0.0, 0.0, 0.0])
  bx = xx[None, ...]  # batch x horizon
  by = y[None, ...]  # batch x horizon
  bxq = bx[..., None]  # batch x horizon x quantile
  byq = by[..., None]  # batch x horizon x quantile
  y2 = forward(x)
  torch.testing.assert_close(y, y2)
  bx2 = inverse(by)
  torch.testing.assert_close(bx, bx2, equal_nan=True)
  bxq2 = inverse(byq) 
  torch.testing.assert_close(bxq, bxq2, equal_nan=True)
  # Mix
  x = torch.tensor([0.0, 1.0, 0.0])
  xx = torch.tensor([0.0, torch.nan, torch.nan])
  y = torch.tensor([0.0, 0.0, 0.0])
  bx = xx[None, ...]  # batch x horizon
  by = y[None, ...]  # batch x horizon
  bxq = bx[..., None]  # batch x horizon x quantile
  byq = by[..., None]  # batch x horizon x quantile
  y2 = forward(x)
  torch.testing.assert_close(y, y2)
  bx2 = inverse(by)
  torch.testing.assert_close(bx, bx2, equal_nan=True)
  bxq2 = inverse(byq) 
  torch.testing.assert_close(bxq, bxq2, equal_nan=True)
  # Empty
  x = torch.tensor([])
  y = torch.tensor([])
  bx = x[None, ...]  # batch x horizon
  by = y[None, ...]  # batch x horizon
  bxq = bx[..., None]  # batch x horizon x quantile
  byq = by[..., None]  # batch x horizon x quantile
  y2 = forward(x)
  torch.testing.assert_close(y, y2)
  bx2 = inverse(by)
  torch.testing.assert_close(bx, bx2)
  bxq2 = inverse(byq) 
  torch.testing.assert_close(bxq, bxq2)
  
  
def test_minus_one():
  pass


def test_plus_one():
  pass


def test_power_minus_one():
  pass


def test_power_plus_one():
  pass