import torch
import torch.nn.functional as F

torch.set_printoptions(precision=8)
      
  
class Identity:
  def __init__(self):
    super().__init__()
  
  def __call__(self, x):
    """
    Args:
      x: with shape prediction_horizon
    """
    return x
  
  
class Prod:
  def __init__(self, p):
    super().__init__()
    self.p = p
  
  def __call__(self, x):
    """
    Args:
      x: with shape prediction_horizon
    """
    return x*self.p
  
  
class Div:
  def __init__(self, p):
    super().__init__()
    self.p = p
  
  def __call__(self, x):
    """
    Args:
      x: with shape prediction_horizon
    """
    return x / self.p
  
  
class CumProd:
  def __init__(self):
    super().__init__()   
  
  def __call__(self, x):
    """
    Args:
      x: with shape prediction_horizon
    """
    return x.cumprod(dim=0)

  
class InvCumProd:
  def __init__(self):
    super().__init__()   
  
  def __call__(self, x):
    """
    Args:
      x: with shape n_samples x prediction_horizon x n_outputs or n_samples x prediction_horizon
    """
    prev_x = x.roll(shifts=1, dims=1)
    prev_x[:, :1, ...] = 1
    return x / prev_x
  
  
class PlusOne:
  def __init__(self):
    super().__init__()   
  
  def __call__(self, x):
    return x + 1

  
class MinusOne:
  def __init__(self):
    super().__init__()   
  
  def __call__(self, x):
    return x - 1

  
class PowerMinusOne:
  def __init__(self, p, min, max):
    super().__init__()
    self.p = p
    self.min = min
    self.max = max
    self.xs = {}
  
  def __call__(self, x):
    p = self.p
    x = F.relu(x)  # x >= 0
    x = 2.0 - F.relu(-x + 2.0)  # x <= 2
    x = F.relu(torch.pow(x, p) - 1.0) - F.relu(torch.pow(x - 2.0, p) - 1.0)
    x = self.min + F.relu(x - self.min)  # x > min
    x = self.max - F.relu(-x + self.max)  # x < max
    return x
  

class PowerPlusOne:
  def __init__(self, p, min, max):
    super().__init__()
    self.p = p
    self.min = min
    self.max = max
  
  def __call__(self, x):
    p = self.p
    x = -1.0 + F.relu(x + 1.0)  # x >= -1
    x = 1.0 - F.relu(-x + 1.0)  # x <= 1
    x = F.relu(torch.pow(x + 1.0, p) - 1.0) - F.relu(torch.pow(F.relu(-x) + 1.0, p) - 1.0) + 1.0
    x = self.min + F.relu(x - self.min)  # x > min
    x = self.max - F.relu(-x + self.max)  # x < max  
    return x