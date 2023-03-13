from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

 
class ROR(MultiHorizonMetric):
  def __init__(self, 
               quantiles: Optional[List[float]] = None,
               target_kind: str = 'value',
               direction: str = 'qs',
               fees: float = 0.,
               threshold: float = 0.,
               mask_kind: Union[str, float] = 'none',
               stop_loss: Optional[float] = None,
               take_profit: Optional[float] = None,
               **kwargs):
      """Rate of return
      
      Actually it's a reward - opposite to loss.
      
      Args:
          quantiles: quantiles for metric
          target_kind: value, rel, rat, lograt
          direction: pos, neg, all, max or qs
          fees: fees
          threshold: decision threshold 
          mask_kind: none, any, all or float from 0 to 1
          stop_loss: stop loss
          take_profit: take profit
      
      See Also:
          * https://en.wikipedia.org/wiki/Rate_of_return
      """
      name = 'ROR'
      name += f'-d_{direction}'
      name += f'-f_{fees:.6g}'
      name += f'-th_{threshold:.6g}'
      if isinstance(mask_kind, str):
        name += f'-m_{mask_kind}'
      else:  # float/int
        name += f'-m_{mask_kind:.6g}'
      if stop_loss is not None:
        name += f'-sl_{stop_loss:.6g}'
      if take_profit is not None:
        name += f'-tp_{take_profit:.6g}'
      super().__init__(name=name, quantiles=quantiles, **kwargs)
      self.target_kind = target_kind
      self.direction = direction
      self.threshold = threshold
      self.mask_kind = mask_kind
      self.fees = fees
      self.sl = stop_loss
      self.tp = take_profit

      
  def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Evaluate loss
    
    Args:
      y_pred: prediction output of network 
          with shape - n_samples x prediction_horizon x n_outputs or n_samples x prediction_horizon
      target: actual values with shape - n_samples x prediction_horizon
      
    Returns:
      losses: shape like y_pred
    """
    is_quantile = True
    if y_pred.ndim == 2:
      is_quantile = False
      y_pred = y_pred.unsqueeze(-1)
    # print(f'name: {self.name}')
    # print(f'is_quantile: {is_quantile}')
    # print(f'quantiles: {self.quantiles}')
    # print('target', target.shape)
    # print(target[0, ..., 0])
    # print('y_pred', y_pred.shape)
    # print(y_pred[0, ..., 0])
    # ROR (relative to first horizon)
    if self.target_kind == 'value':
      target_0 = target[..., :1]
      y_pred_0 = y_pred[..., :1, :]
      dy_target = target - target_0
      ror_target = dy_target / target_0
      dy_pred = y_pred - y_pred_0
      ror_pred = dy_pred / y_pred_0
    elif self.target_kind == 'rel':
      ror_target = target + 1
      ror_pred = y_pred + 1
      ror_target[..., :1] = 1
      ror_pred[..., :1, :] = 1
      ror_target = ror_target.cumprod(1) - 1
      ror_pred = ror_pred.cumprod(1) - 1
    elif self.target_kind == 'rat':
      ror_target = target.clone().detach()
      ror_pred = y_pred.clone().detach()
      ror_target[..., :1] = 1
      ror_pred[..., :1, :] = 1
      ror_target = ror_target.cumprod(1) - 1
      ror_pred = ror_pred.cumprod(1) - 1
    elif self.target_kind == 'lograt':
      y_pred = y_pred.exp()
      target = target.exp()
      ror_target[..., :1] = 1
      ror_pred[..., :1, :] = 1
      ror_target = ror_target.cumprod(1) - 1
      ror_pred = ror_pred.cumprod(1) - 1
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
    pred_fees = self.fees + self.fees*(1 + ror_pred)
    target_fees = self.fees + self.fees*(1 + ror_target)
    # Decision
    pos_mask = ror_pred > pred_fees + self.threshold
    neg_mask = -ror_pred > pred_fees + self.threshold
    horizon = ror_pred.size(1)
    if self.mask_kind == 'none':
      pass
    elif self.mask_kind == 'any':
      pos_mask = pos_mask.any(1, keepdim=True).expand(-1, horizon, -1)
      neg_mask = neg_mask.any(1, keepdim=True).expand(-1, horizon, -1)
    elif self.mask_kind == 'all':  # skip first horizon (always False)
      pos_mask = pos_mask[:, 1:, :].all(1, keepdim=True).expand(-1, horizon, -1)
      neg_mask = neg_mask[:, 1:, :].all(1, keepdim=True).expand(-1, horizon, -1)
    elif self.mask_kind == 'last':
      pos_mask = pos_mask[:, -1:, :].expand(-1, horizon, -1)
      neg_mask = neg_mask[:, -1:, :].expand(-1, horizon, -1)
    elif self.mask_kind == 'max':
      pos_mask = ror_target > 0
      neg_mask = ror_target < 0
    elif isinstance(self.mask_kind, float):  
      pos_sum = pos_mask[:, 1:, :].sum(1, keepdim=True).expand(-1, horizon, -1)
      neg_sum = neg_mask[:, 1:, :].sum(1, keepdim=True).expand(-1, horizon, -1)
      pos_th = pos_sum / (horizon - 1)
      neg_th = neg_sum / (horizon - 1)
      pos_mask = pos_th > self.mask_kind
      neg_mask = neg_th > self.mask_kind
    else:
      raise NotImplementedError(self.mask_kind)
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
    elif self.direction == 'qs':
      for i, q in enumerate(self.quantiles):
        if q > 0.5:
          pos_mask[..., i] = False
        elif q < 0.5:
          neg_mask[..., i] = False
        else:  # q == 0.5
          pass
      losses[pos_mask] = ror_target[pos_mask] - target_fees[pos_mask]
      losses[neg_mask] = -ror_target[neg_mask] - target_fees[neg_mask]
    else:
      raise NotImplementedError(self.direction)
    # print(losses)
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
