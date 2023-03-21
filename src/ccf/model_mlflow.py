from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import mlflow
from mlflow import MlflowClient
from omegaconf import OmegaConf
import pytorch_forecasting as pf

from ccf import models as ccf_models


class CCFModel(mlflow.pyfunc.PythonModel):
  def __init__(self, config_name, config=None, model=None):
    self.config_name = config_name
    self.config = config
    self.model = model
    
  def load_context(self, context):
    config_path = Path(context.artifacts['conf'])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_path)):
      cfg = compose(config_name=self.config_name)
    self.config = OmegaConf.to_object(cfg)
    model_path = Path(context.artifacts['model'])
    class_name = self.config['model_kwargs']['class']
    model_class = getattr(pf.models, class_name, None)  # PyTorch Forecasting
    if model_class is None:
      model_class = getattr(ccf_models, class_name, None)  # CCF
    if model_class is None:
      raise NotImplementedError(class_name)
    self.model = model_class.load_from_checkpoint(model_path)
    
  def predict(self, context, model_input):
    return self.predict_model(model_input)

  def predict_model(self, model_input):
    return self.model.predict(**model_input)

  
def load_model(name, version=None, stage=None, metadata_only=False):
  """
    Required environment variables:
      MLFLOW_TRACKING_URI=http://mlflow:5000
      MLFLOW_TRACKING_USERNAME=ccf
      MLFLOW_TRACKING_PASSWORD=
  """
  stage = 'None' if stage is None else stage
  version = str(version) if version is not None else version
  client = MlflowClient()
  model = None
  is_found = False
  if model is None:
    parent_registered_model = None
    for rm in client.search_registered_models():
      if rm.name == name:
        parent_registered_model = rm
    if parent_registered_model is not None:    
      print(f'Registered model "{name}" is found')
      model_version = None
      if version is None:
        for mv in parent_registered_model.latest_versions:
          if mv.current_stage == stage:
            model_version = mv
      else:
        for mv in client.search_model_versions(f"name='{name}'"):
          if mv.version == version:
            model_version = mv
      if model_version is not None:
        version = model_version.version
        stage = model_version.current_stage
        print(f'Model stage "{stage}" with version "{version}" is found')
        model_uri = f'models:/{name}/{version}'
        if not metadata_only:
          model = mlflow.pyfunc.load_model(model_uri=model_uri)
        is_found = True
      else:
        print(f'Model stage "{stage}" and/or version "{version}" is not found!')
    else:
      print(f'Registered model "{name}" is not found!')
  else:  # TODO
    for mv in client.search_model_versions(f"name='{model_name}'"):
      pprint(dict(mv), indent=4)
  if is_found:
    print(f'Model "{name}" is found')
  else:
      print(f'Model "{name}" is not found!')
  return model, version, stage
  