"""
Naive model that always predicts 0
"""
from typing import Any, Dict

import torch

from pytorch_forecasting.models import BaseModel


class NaiveZero(BaseModel):
  """
  Naive model that always predicts 0
  """

  def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Network forward pass.

    Args:
        x (Dict[str, torch.Tensor]): network input

    Returns:
        Dict[str, torch.Tensor]: netowrk outputs
    """
    if isinstance(x["encoder_target"], (list, tuple)):  # multiple targets
      prediction = [
        self.forward_one_target(
          encoder_lengths=x["encoder_lengths"],
          decoder_lengths=x["decoder_lengths"],
          encoder_target=encoder_target)
        for encoder_target in x["encoder_target"]]
    else:  # one target
      prediction = self.forward_one_target(
        encoder_lengths=x["encoder_lengths"],
        decoder_lengths=x["decoder_lengths"],
        encoder_target=x["encoder_target"])
    return self.to_network_output(prediction=prediction)

  def forward_one_target(self, encoder_lengths: torch.Tensor, 
                         decoder_lengths: torch.Tensor, encoder_target: torch.Tensor):
    # max_prediction_length = decoder_lengths.max()
    # assert encoder_lengths.min() > 0, "Encoder lengths of at least 1 required to obtain last value"
    # last_values = encoder_target[torch.arange(encoder_target.size(0)), encoder_lengths - 1]
    # prediction = last_values[:, None].expand(-1, max_prediction_length)
    prediction = torch.zeros(decoder_lengths.size(0), decoder_lengths.max())
    return prediction

  def to_prediction(self, out: Dict[str, Any], use_metric: bool = True, **kwargs):
    return out.prediction

  def to_quantiles(self, out: Dict[str, Any], use_metric: bool = True, **kwargs):
    return out.prediction[..., None]
