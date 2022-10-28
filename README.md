# ccf
CryptoCurrency Forecasting App for [ML System Design Course on ODS.ai](https://ods.ai/tracks/ml-system-design-22)

# Install
* Install python 3.9
* Install requirements for Data part:
```
pip install -r src/ccf/requirements_data.txt
``` 
* Install requirements for ML part:
```
pip install -r src/ccf/requirements_ml.txt
```
* Install requirements dor App part:
```
pip install -r src/ccf/requirements_app.txt
```

# Run
Go to working directory
```sh
cd work
```
Get data
* Linux (by default)
```sh
PYTHONPATH=../src/ python ../src/ccf/get_data.py
```
* Windows
```sh
cmd /C  "set PYTHONPATH=../src && python ../src/ccf/get_data.py"
```
Train [TFT](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html) model
```sh
PYTHONPATH=../src/ python ../src/ccf/train.py train_tft.yaml
```
Predict with [TFT](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html) model
* With memory leak:(
```sh
PYTHONPATH=../src/ python ../src/ccf/predict.py predict_tft.yaml
```
* Workaround of memory leak;) 
```sh
while true; do PYTHONPATH=../src/ timeout 1800 python ../src/ccf/predict.py predict_tft.yaml; done
```
Run Streamlit app
```sh
streamlit run ../src/ccf/app.py app_tft.yaml
```

# Configs
Data `work/get_data.yaml`
```yaml
executor_kwargs:
  max_workers: 3
# run_kwargs:
#   http_proxy_host: 127.0.0.1
#   http_proxy_port: 3128
#   proxy_type: http
market_kwargs:
  verbose: false
feeds_kwargs:
  split: 1
  delay: 3
  before: 3600
  verbose: false
markets:
- btcusdt
streams:
- depth5@1000ms
- trade
feeds:
  https://cointelegraph.com/rss: Yes
  https://www.newsbtc.com/feed: Yes
  https://www.cryptoninjas.net/feed/: Yes
```
Train `work/train_tft.yaml`
```yaml
model_path: tft.ckpt
tune: false
dataset_kwargs:
  start: -86400
  end: ~
  split: 0.8
  engine_kwargs:
    btcusdt:
      orderbook:
        url: sqlite:///btcusdt@depth5@1000ms.db
      trades:
        url: sqlite:///btcusdt@trade.db  
      news:
        url: sqlite:///news.db
  read_kwargs:
    btcusdt:
      orderbook:
        name: data
      trades:
        name: data
      news:
        name: data
  features_kwargs:
    post_features:
    - time_idx
    - group
    pre_features:
    - m_p
    resample_kwargs:
      rule: 1S
    aggregate_kwargs:
      orderbook:
        func: last
      trades:
        func: last
      news:
        func: last
    interpolate_kwargs:
      orderbook:
        method: pad
      trades:
        method: pad
      news: ~
  dataset_kwargs:
    time_idx: time_idx
    allow_missing_timesteps: true
    add_relative_time_idx: true
    target: m_p
    group_ids:
    - group
    max_encoder_length: 10
    max_prediction_length: 5
    time_varying_unknown_reals:
    - m_p
    - a_p_0
    - b_p_0
    target_normalizer:
      class: EncoderNormalizer
    scalers:
      class: EncoderNormalizer
dataloader_kwargs:
  train:
    train: true
    num_workers: 2
    batch_size: 32
  val:
    train: false
    num_workers: 2
    batch_size: 32
model_kwargs:
  class: TemporalFusionTransformer
  learning_rate: 0.03
  hidden_size: 32
  attention_head_size: 1
  dropout: 0.1
  hidden_continuous_size: 16
  output_size: 7
  loss:
    class: QuantileLoss
  log_interval: 0
  reduce_on_plateau_patience: 2
trainer_kwargs:
  max_epochs: 1
  accelerator: cpu  # gpu
  # devices: [ 0 ]
  gradient_clip_val: 0.1
  log_every_n_steps: 50
  # limit_train_batches: 2
  logger:
  - class: TensorBoardLogger
    save_dir: tensorboard
    name: tft
  callbacks:
  - class: LearningRateMonitor
    logging_interval: step
  - class: ModelCheckpoint
    monitor: val_loss
    filename: '{epoch}-{step}-{val_loss:.3f}'
    save_last: true
    save_top_k: 1
  - class: EarlyStopping
    monitor: val_loss
    min_delta: 0
    patience: 4
    verbose: false
    mode: min
```
Predict `work/predict_tft.yaml`
```yaml 
model_path: tft.ckpt
train_kwargs: train_tft.yaml
verbose: true
past: 3600
predict_kwargs:
  return_index: true
  mode: prediction  # prediction or quantiles
engine_kwargs:
  url: sqlite:///tft@prediction.db
write_kwargs:
  name: data
  if_exists: append
```
App `work/app_tft.yaml`
```yaml 
past: 30
freq: 1
engine_kwargs:
  btcusdt:
    orderbook:
      url: sqlite:///btcusdt@depth5@1000ms.db
    trades:
      url: sqlite:///btcusdt@trade.db  
    news:
      url: sqlite:///news.db
    prediction:
      url: sqlite:///tft@prediction.db
read_kwargs:
  btcusdt:
    orderbook:
      name: data
      columns: [time, a_p_0, b_p_0]
    trades:
      name: data
    news:
      name: data
    prediction:
      name: data
      columns: [time, pred]
resample_kwargs:
  rule: 1S
```
