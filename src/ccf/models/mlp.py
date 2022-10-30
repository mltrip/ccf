import os
import warnings
from typing import Dict

import torch
from torch import nn
from pytorch_forecasting.models import BaseModel
from pytorch_forecasting.data import TimeSeriesDataSet

warnings.filterwarnings("ignore")


class Module(nn.Module):
  def __init__(self, input_size: int, output_size: int,
               hidden_size: int, n_hidden_layers: int):
    super().__init__()

    # input layer
    module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
    # hidden layers
    for _ in range(n_hidden_layers):
        module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
    # output layer
    module_list.append(nn.Linear(hidden_size, output_size))

    self.sequential = nn.Sequential(*module_list)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x of shape: batch_size x n_timesteps_in
    # output of shape batch_size x n_timesteps_out
    return self.sequential(x)


# # test that network works as intended
# network = FullyConnectedModule(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)
# x = torch.rand(20, 5)
# network(x).shape

class MLP(BaseModel):
  def __init__(self, input_size: int, output_size: int, 
               hidden_size: int, n_hidden_layers: int, **kwargs):
    # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
    self.save_hyperparameters()
    # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
    super().__init__(**kwargs)
    self.network = Module(
        input_size=self.hparams.input_size,
        output_size=self.hparams.output_size,
        hidden_size=self.hparams.hidden_size,
        n_hidden_layers=self.hparams.n_hidden_layers,
    )

  def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # x is a batch generated based on the TimeSeriesDataset
    network_input = x["encoder_cont"].squeeze(-1)
    prediction = self.network(network_input)

    # rescale predictions into target space
    prediction = self.transform_output(prediction, target_scale=x["target_scale"])

    # We need to return a dictionary that at least contains the prediction
    # The parameter can be directly forwarded from the input.
    # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
    return self.to_network_output(prediction=prediction)
  
  @classmethod
  def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
    new_kwargs = {
        "output_size": dataset.max_prediction_length,
        "input_size": dataset.max_encoder_length,
    }
    new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
    # example for dataset validation
    assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
    assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
    assert (
        len(dataset.time_varying_known_categoricals) == 0
        and len(dataset.time_varying_known_reals) == 0
        and len(dataset.time_varying_unknown_categoricals) == 0
        and len(dataset.static_categoricals) == 0
        and len(dataset.static_reals) == 0
        and len(dataset.time_varying_unknown_reals) == 1
        # and dataset.time_varying_unknown_reals[0] == dataset.target
    ), "Only covariate should be the target in 'time_varying_unknown_reals'"

    return super().from_dataset(dataset, **new_kwargs)