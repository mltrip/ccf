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
## Go to working directory
```sh
cd work
```
## Get data
* Linux (by default)
```sh
PYTHONPATH=../src/ python ../src/ccf/get_data.py -cd conf -cn get_data_multi
```
* Windows
```sh
cmd /C  "set PYTHONPATH=../src && python ../src/ccf/get_data.py -cd conf -cn get_data_multi"
```
## Train [TFT](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html) model
* Once
```sh
PYTHONPATH=../src/ python ../src/ccf/train.py -cd conf -cn train_multi_tgt_tft
```
* Every ~hour
```sh
while true; do PYTHONPATH=../src/ python ../src/ccf/train.py -cd conf -cn train_multi_tgt_tft; sleep 3600; done
```
## Predict with [TFT](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html) model
* With memory leak:(
```sh
PYTHONPATH=../src/ python ../src/ccf/predict.py -cd conf -cn predict_multi_tgt_tft
```
* Workaround of memory leak;) 
```sh
while true; do PYTHONPATH=../src/ timeout 1800 python ../src/ccf/predict.py -cd conf -cn predict_multi_tgt_tft; done
```
* Naive
```sh
while true; do PYTHONPATH=../src/ timeout 1800 python ../src/ccf/predict.py -cd conf -cn predict_multi_tgt_tft_naive; done
```
## Run Streamlit UI
```sh
PYTHONPATH=../src/ streamlit run ../src/ccf/apps/ui.py conf/ui_multi_tgt.yaml
```
## Run Evidently Monitor
* Raw data
```sh
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_raw
```
* Target
```sh
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_a.yaml
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_b.yaml
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_naive_a.yaml
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_naive_b.yaml
```

# Configs
Data `work/conf/get_data_multi.yaml`
```yaml
defaults:
 - feeds: all
 - markets: three
 - streams: 1000_d5_t
 - run_kwargs: none

executor_kwargs:
  max_workers: 7
market_kwargs:
  verbose: false
feeds_kwargs:
  split: 1
  delay: 3
  before: 3600
  verbose: false
```
Train `work/conf/train_multi_tgt_tft.yaml`
```yaml
model_path: multi_tgt_tft.ckpt
tune: true
dataset_kwargs:
  start: -3600
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
    ethusdt:
      orderbook:
        url: sqlite:///ethusdt@depth5@1000ms.db
      trades:
        url: sqlite:///ethusdt@trade.db  
      news:
        url: sqlite:///news.db
    ethbtc:
      orderbook:
        url: sqlite:///ethbtc@depth5@1000ms.db
      trades:
        url: sqlite:///ethbtc@trade.db  
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
    ethusdt:
      orderbook:
        name: data
      trades:
        name: data
      news:
        name: data
    ethbtc:
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
    target: [a_p_0, b_p_0]
    group_ids:
    - group
    static_categoricals:
    - group
    max_encoder_length: 10
    max_prediction_length: 5
    time_varying_unknown_reals:
    - a_p_0
    - b_p_0
    target_normalizer:
      class: GroupNormalizer
      groups: [ group ]
    scalers:
      class: GroupNormalizer
      groups: [ group ]
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
  learning_rate: 0.003
  hidden_size: 32
  attention_head_size: 1
  dropout: 0.1
  hidden_continuous_size: 16
  # output_size: 7  # Inferred from loss
  loss:
    class: QuantileLoss
  log_interval: 0
  reduce_on_plateau_patience: 2
trainer_kwargs:
  max_epochs: 32
  accelerator: cpu  # gpu
  # devices: [ 0 ]
  gradient_clip_val: 0.1
  log_every_n_steps: 50
  # limit_train_batches: 10
  logger:
  - class: TensorBoardLogger
    save_dir: tensorboard
    name: multi_tgt_tft
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
Predict `conf/work/predict_multi_tgt_tft.yaml`
```yaml 
defaults:
 - ./@train_kwargs: train_multi_tgt_tft

model_path: multi_tgt_tft.ckpt
verbose: true
past: 3600
predict_kwargs:
  return_index: true
  mode: prediction  # prediction or quantiles
engine_kwargs:
  url: sqlite:///multi_tgt_tft@prediction.db
write_kwargs:
  name: data
  if_exists: append
```
UI `conf/work/ui_multi_tgt.yaml`
```yaml 
delay: 1
read_data_kwargs:
  start: -30
  end: ~
  query:
    btcusdt:
      target:
        engine_kwargs:
          url: sqlite:///btcusdt@depth5@1000ms.db
        read_kwargs:
          name: data
          columns: [ time, a_p_0, b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      tft:
        engine_kwargs:
          url: sqlite:///multi_tgt_tft@prediction.db
        read_kwargs:
          name: data
          group: btcusdt
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      rnn:
        engine_kwargs:
          url: sqlite:///multi_tgt_rnn@prediction.db
        read_kwargs:
          name: data
          group: btcusdt
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      naive:
        engine_kwargs:
          url: sqlite:///multi_tgt_tft_naive@prediction.db
        read_kwargs:
          name: data
          group: btcusdt
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
    ethusdt:
      target:
        engine_kwargs:
          url: sqlite:///ethusdt@depth5@1000ms.db
        read_kwargs:
          name: data
          columns: [ time, a_p_0, b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      tft:
        engine_kwargs:
          url: sqlite:///multi_tgt_tft@prediction.db
        read_kwargs:
          name: data
          group: ethusdt
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      rnn:
        engine_kwargs:
          url: sqlite:///multi_tgt_rnn@prediction.db
        read_kwargs:
          name: data
          group: ethusdt
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      naive:
        engine_kwargs:
          url: sqlite:///multi_tgt_tft_naive@prediction.db
        read_kwargs:
          name: data
          group: ethusdt
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
    ethbtc:
      target:
        engine_kwargs:
          url: sqlite:///ethbtc@depth5@1000ms.db
        read_kwargs:
          name: data
          columns: [ time, a_p_0, b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      tft:
        engine_kwargs:
          url: sqlite:///multi_tgt_tft@prediction.db
        read_kwargs:
          name: data
          group: ethbtc
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      rnn:
        engine_kwargs:
          url: sqlite:///multi_tgt_rnn@prediction.db
        read_kwargs:
          name: data
          group: ethbtc
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
      naive:
        engine_kwargs:
          url: sqlite:///multi_tgt_tft_naive@prediction.db
        read_kwargs:
          name: data
          group: ethbtc
          columns: [ time, pred-a_p_0, pred-b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
```
UI `conf/work/monitor_raw.yaml`
```yaml 
delay: 600
log_dir: monitor/raw
report_kwargs:
  metrics: 
  - class: DataDriftPreset
  - class: DataQualityPreset
test_kwargs:
  tests:
  - class: DataQualityTestPreset
  - class: DataStabilityTestPreset
column_mapping_kwargs:
  datetime: time
read_data_kwargs:
  start: -600
  end: ~
  query:
    btcusdt_orderbook:
      o:
        engine_kwargs:
          url: sqlite:///btcusdt@depth5@1000ms.db
        read_kwargs:
          name: data
    ethusdt_orderbook:
      o:
        engine_kwargs:
          url: sqlite:///ethusdt@depth5@1000ms.db
        read_kwargs:
          name: data
    ethbtc_orderbook:
      o:
        engine_kwargs:
          url: sqlite:///ethbtc@depth5@1000ms.db
        read_kwargs:
          name: data
    btcusdt_trades:
      t:
        engine_kwargs:
          url: sqlite:///btcusdt@trade.db
        read_kwargs:
          name: data
    ethusdt_trades:
      t:
        engine_kwargs:
          url: sqlite:///ethusdt@trade.db
        read_kwargs:
          name: data
    ethbtc_trades:
      t:
        engine_kwargs:
          url: sqlite:///ethbtc@trade.db
        read_kwargs:
          name: data
    news:
      n:
        engine_kwargs:
          url: sqlite:///news.db
        read_kwargs:
          name: data
```
