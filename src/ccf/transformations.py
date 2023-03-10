import torch


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
