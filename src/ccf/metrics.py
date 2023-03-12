from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

 
class ROR(MultiHorizonMetric):
  """Rate of return
  
  See Also:
    * https://en.wikipedia.org/wiki/Rate_of_return
  """

  def __init__(self, 
               quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
               fees: float = 0,
               direction: str = 'all',
               kind: float = 1,
               th: str = 'any',
               sl: float = None,
               tp: float = None,
               target_kind: str = None,
               target_scale: float = None,
               **kwargs):
      """Rate of return
      Args:
          quantiles: quantiles for metric
          fees: fees
          dir: pos, neg, all, max or none
          kind: one, two
          target_kind: none, rel, rat, lograt
          scale: scale
          sl: stop loss
          tp: take profit
          th: threshold: item, any, all or float from 0 to 1
      """
      name = '-'.join(['ROR', 
                       f'f_{fees*100:.4g}%', 
                       f'd_{direction}'])
      if isinstance(kind, str):
        name += f'-k_{kind}'
      else:  # float/int
        name += f'-k_{kind:.2g}'
      if isinstance(th, str):
        name += f'-th_{th}'
      else:  # float/int
        name += f'-th_{th*100:.2g}%' 
      if sl is not None:
        name += f'-sl_{sl*100:.4g}%'
      if tp is not None:
        name += f'-tp_{tp*100:.4g}%'
      super().__init__(name=name, quantiles=quantiles, **kwargs)
      self.fees = fees
      self.kind = kind
      self.direction = direction
      self.target_kind = target_kind
      self.target_scale = target_scale
      self.sl = sl
      self.tp = tp
      self.th = th
      
  def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    is_quantile = True
    if y_pred.ndim == 2:
      is_quantile = False
      y_pred = y_pred.unsqueeze(-1)
    # print('target', target.shape)
    # print(target[0, ..., 0])
    # print('y_pred', y_pred.shape)
    # print(y_pred[0, ..., 0])
    # ROR
    if self.target_scale is not None:
      y_pred = y_pred / self.target_scale
      target = target / self.target_scale
    if self.target_kind is None:
      target_0 = target[..., :1]
      y_pred_0 = y_pred[..., :1, :]
      dy_target = target - target_0
      ror_target = dy_target / target_0
      dy_pred = y_pred - y_pred_0
      ror_pred = dy_pred / y_pred_0
    elif self.target_kind == 'rel':
      target_0 = target[..., :1]
      y_pred_0 = y_pred[..., :1, :]
      ror_target = target.cumsum(1) - target_0
      ror_pred = y_pred.cumsum(1) - y_pred_0
    elif self.target_kind == 'rat':
      y_pred = y_pred - 1
      target = target - 1
      target_0 = target[..., :1]
      y_pred_0 = y_pred[..., :1, :]
      ror_target = target.cumsum(1) - target_0
      ror_pred = y_pred.cumsum(1) - y_pred_0
    elif self.target_kind == 'lograt':
      y_pred = y_pred.exp() - 1
      target = target.exp() - 1
      target_0 = target[..., :1]
      y_pred_0 = y_pred[..., :1, :]
      ror_target = target.cumsum(1) - target_0
      ror_pred = y_pred.cumsum(1) - y_pred_0  
    else:
      raise NotImplementedError(self.target_kind)
    # SL/TP
    if self.sl is not None or self.tp is not None:
      if self.direction == 'pos':
        tp_mask = ror_target > self.tp if self.tp is not None else torch.full_like(ror_target, False)
        sl_mask = ror_target < -self.sl if self.sl is not None else torch.full_like(ror_target, False)
        ror_target[tp_mask] = self.tp
        ror_target[sl_mask] = -self.sl
      elif self.direction == 'neg':
        tp_mask = ror_target < -self.tp if self.tp is not None else torch.full_like(ror_target, False)
        sl_mask = ror_target > self.sl if self.sl is not None else torch.full_like(ror_target, False)
        ror_target[tp_mask] = -self.tp
        ror_target[sl_mask] = self.sl
      else:
        raise NotImplementedError(self.direction)
      # Mask RORs after first SL/TP
      sltp_mask = tp_mask | sl_mask
      sltp_mask[..., -1] = True  # Force close order at last horizon
      _, idxs = sltp_mask.max(1, keepdim=True)
      firsts = ror_target.gather(1, idxs)
      firsts = firsts.expand(-1, ror_target.size(1))
      firsts_mask = torch.stack([torch.cat([torch.full((x[0],), False), torch.full((ror_target.size(1) - x[0],), True)]) for x in idxs])
      ror_target[firsts_mask] = firsts[firsts_mask]
    ror_target = ror_target.unsqueeze(-1).expand(-1, -1, ror_pred.size(dim=2))
    # print('ror_pred', ror_pred.shape)
    # print(ror_pred[0, ..., 0])
    # print('ror_target', ror_target.shape)
    # print(ror_target[0, ..., 0])
    # Fees
    if isinstance(self.kind, str):
      raise ValueError(self.kind)
    elif isinstance(self.kind, (float, int)):
      kfees = self.kind*self.fees
      pred_fees = kfees + kfees*(1 + ror_pred)
      target_fees = self.fees + self.fees*(1 + ror_target)
    else:
      raise ValueError(self.kind)
    # Decision threshold
    pos_mask = ror_pred > pred_fees
    neg_mask = -ror_pred > pred_fees
    horizon = ror_pred.size(1)
    if self.th == 'item':
      pass
    elif self.th == 'any':
      pos_mask = pos_mask.any(1, keepdim=True).expand(-1, horizon, -1)
      neg_mask = neg_mask.any(1, keepdim=True).expand(-1, horizon, -1)
    elif self.th == 'all':
      pos_mask = pos_mask.all(1, keepdim=True).expand(-1, horizon, -1)
      neg_mask = neg_mask.all(1, keepdim=True).expand(-1, horizon, -1)
    elif isinstance(self.th, float):
      pos_sum = pos_mask.sum(1, keepdim=True).expand(-1, horizon, -1)
      neg_sum = neg_mask.sum(1, keepdim=True).expand(-1, horizon, -1)
      pos_th = pos_sum / horizon
      neg_th = neg_sum / horizon
      pos_mask = pos_th > self.th
      neg_mask = neg_th > self.th
    else:
      raise ValueError(self.th)
    # Results
    losses = torch.zeros_like(y_pred)
    if self.direction == 'all':
      losses[pos_mask] = ror_target[pos_mask] - target_fees[pos_mask]
      losses[neg_mask] = -ror_target[neg_mask] - target_fees[neg_mask]
    elif self.direction == 'pos': 
      losses[pos_mask] = ror_target[pos_mask] - target_fees[pos_mask]
    elif self.direction == 'neg': 
      losses[neg_mask] = -ror_target[neg_mask] - target_fees[neg_mask]
    elif self.direction == 'none':
      losses = ror_target
    elif self.direction == 'max':
      pos_mask = ror_target > 0
      neg_mask = ror_target < 0
      losses[pos_mask] = ror_target[pos_mask] - target_fees[pos_mask]
      losses[neg_mask] = -ror_target[neg_mask] - target_fees[neg_mask]
      losses[losses < 0] = 0
    else:
      raise ValueError(self.direction)
    return losses if is_quantile else losses[..., 0]
  
  def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Convert network prediction into a point prediction.
    Args:
        y_pred: prediction output of network
    Returns:
        torch.Tensor: point prediction
    """
    if y_pred.ndim == 3:
      idx = self.quantiles.index(0.5)
      y_pred = y_pred[..., idx]
    return y_pred

  def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Convert network prediction into a quantile prediction.
    Args:
        y_pred: prediction output of network
    Returns:
        torch.Tensor: prediction quantiles
    """
    return y_pred

  
class RORLoss(MultiHorizonMetric):
  """Rate Of Return Loss
  
  See Also:
    * https://en.wikipedia.org/wiki/Rate_of_return
  """

  def __init__(self, 
               quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
               fees: float = 0,
               direction: str = 'all',
               **kwargs):
      """Rate of return
      Args:
          quantiles: quantiles for metric
          fees: fees
          dir: pos, neg or all
      """
      name = '_'.join(['ROR', f'{fees*100}%', direction])
      super().__init__(name=name, quantiles=quantiles, **kwargs)
      self.fees = fees
      self.direction = direction
      
  def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    is_quantile = True
    if y_pred.ndim == 2:
      is_quantile = False
      y_pred = y_pred.unsqueeze(-1)
    # print('target', target.shape)
    # print('y_pred', y_pred.shape)
    target_0 = target[..., :1]
    # print('target_0', target_0.shape)
    y_pred_0 = y_pred[..., :1, :]
    # print('y_pred_0', y_pred_0.shape)
    dy_target = target - target_0
    # print('dy_target', dy_target.shape)
    ror_target = dy_target / target_0
    # print('ror_target', ror_target.shape)
    dy_pred = y_pred - y_pred_0
    # print('dy_pred', dy_pred.shape)
    ror_pred = dy_pred / y_pred_0
    # print('ror_pred', ror_pred.shape)
    # print(ror_pred[0, ..., 3])
    # print(ror_target[0, ..., 3])
    ror_target = ror_target.unsqueeze(-1).expand(-1, -1, ror_pred.size(dim=2))  # to quantiles
    pos_pred_delta = nn.functional.relu(ror_pred - self.fees)
    neg_pred_delta = nn.functional.relu(-ror_pred - self.fees)
    pos_target_delta = nn.functional.relu(ror_target - self.fees)
    neg_target_delta = nn.functional.relu(-ror_target - self.fees)
    # pos_target_delta = ror_target - self.fees
    # neg_target_delta = -ror_target - self.fees
    if self.direction == 'all':
      losses_pos = nn.functional.relu(pos_pred_delta - pos_target_delta)
      # print(losses_pos[0, ..., 3])
      losses_neg = nn.functional.relu(neg_pred_delta - neg_target_delta)
      # print(losses_neg[0, ..., 3])
      losses = losses_pos + losses_neg
    elif self.direction == 'pos': 
      losses = nn.functional.relu(pos_pred_delta - pos_target_delta)
    elif self.direction == 'neg':
      losses = nn.functional.relu(neg_pred_delta - neg_target_delta)
    else:
      raise ValueError(self.direction)
    return losses if is_quantile else losses[..., 0]
  
  def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Convert network prediction into a point prediction.
    Args:
        y_pred: prediction output of network
    Returns:
        torch.Tensor: point prediction
    """
    if y_pred.ndim == 3:
      idx = self.quantiles.index(0.5)
      y_pred = y_pred[..., idx]
    return y_pred

  def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Convert network prediction into a quantile prediction.
    Args:
        y_pred: prediction output of network
    Returns:
        torch.Tensor: prediction quantiles
    """
    return 