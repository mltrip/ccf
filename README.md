# CCF
Crypto Currency Forecasting App for [ML System Design Course on ODS.ai](https://ods.ai/tracks/ml-system-design-22)

## ARCHITECTURE (TODO implement it!)
![architecture](docs/architecture.png)

App consists of 6 main parts
> We could install and run different parts of the App independently
### $$\textcolor{#4dd0e1}{\text{DATA}}$$
In this part data from `exchanges` and `news rss` are collected in the `raw data` database
### $$\textcolor{#a2fca2}{\text{FEATURES}}$$
Here are continiuos feature extracting from `raw data` database and saving in the `feature store` (it could be the same database as raw). `Dataset` creation occurs dynamically at the ML and PREDICTIONS parts, but the logic is described here
### $$\textcolor{#eeff41}{\text{ML}}$$
Here we create `datasets`, train/tune `models` and add/update them in the `models registry`
### $$\textcolor{#ffab40}{\text{PREDICTIONS}}$$
This part is for making `predictions` based on `models` from `models registry` and `datasets`
### $$\textcolor{#adadad}{\text{METRICS}}$$
There are metrics collectors and monitors with techincal information about `raw data`, `features`, training/tuning, `models`, `predictions`, etc
### $$\textcolor{#eeeeee}{\text{UI}}$$
We show `users`: `predictions`, performance `metrics`, `raw data`, etc. This part uses some information from METRICS part
## INSTALL
### Python 3.9
### $$\textcolor{#4dd0e1}{\text{DATA}}$$ 
```sh
pip install -r src/ccf/requirements_data.txt
```
### $$\textcolor{#a2fca2}{\text{FEATURES}}$$
```sh
pip install -r src/ccf/requirements_features.txt
``` 
### $$\textcolor{#eeff41}{\text{ML}}$$ 
```sh
pip install -r src/ccf/requirements_ml.txt
```
### $$\textcolor{#ffab40}{\text{PREDICTIONS}}$$ 
```sh
pip install -r src/ccf/requirements_predictions.txt
```
### $$\textcolor{#adadad}{\text{METRICS}}$$
```sh
pip install -r src/ccf/requirements_metrics.txt
```
### $$\textcolor{#eeeeee}{\text{UI}}$$
```sh
pip install -r src/ccf/requirements_ui.txt
```
## RUN
```sh
cd work
```
### $$\textcolor{#4dd0e1}{\text{GET DATA}}$$ 
* Linux (by default)
```sh
PYTHONPATH=../src/ python ../src/ccf/get_data.py -cd conf -cn get_data_multi
```
* Windows (as example)
```sh
cmd /C  "set PYTHONPATH=../src && python ../src/ccf/get_data.py -cd conf -cn get_data_multi"
```
### $$\textcolor{#a2fca2}{\text{EXTRACT FEATURES}}$$
```sh
PYTHONPATH=../src/ python ../src/ccf/extract_features.py -cd conf -cn extract_features
```
> There could be a memeory leak, workaround to avoid it;)
```sh
while true; do PYTHONPATH=../src/ timeout 1800 python ../src/ccf/extract_features.py -cd conf -cn extract_features; done
```
### $$\textcolor{#eeff41}{\text{TRAIN/TUNE MODEL}}$$ 
* Train once
```sh
PYTHONPATH=../src/ python ../src/ccf/train.py -cd conf -cn train_multi_tgt_tft
```
* Tune every ~1 hour
```sh
while true; do PYTHONPATH=../src/ python ../src/ccf/train.py -cd conf -cn train_multi_tgt_tft; sleep 3600; done
```
### $$\textcolor{#ffab40}{\text{MAKE PREDICTIONS}}$$
* With memory leak:(
```sh
PYTHONPATH=../src/ python ../src/ccf/predict.py -cd conf -cn predict_multi_tgt_tft
```
* Workaround of memory leak;) (just restart script every ~0.5 hour)
```sh
while true; do PYTHONPATH=../src/ timeout 1800 python ../src/ccf/predict.py -cd conf -cn predict_multi_tgt_tft; done
```
* Naive model (predict last known target)
```sh
while true; do PYTHONPATH=../src/ timeout 1800 python ../src/ccf/predict.py -cd conf -cn predict_multi_tgt_tft_naive; done
```
### $$\textcolor{#adadad}{\text{MONITOR METRICS}}$$ 
* Access metrics reports directory by 8000 port
```sh
cd work/monitor
python -m http.server 8000
```
#### $$\textcolor{#4dd0e1}{\text{DATA}}$$
```sh
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_raw
```
#### $$\textcolor{#a2fca2}{\text{FEATURES}}$$
```sh
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_features
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_features_lograt
```
#### $$\textcolor{#eeff41}{\text{TRAIN/TUNE}}$$
```sh
tensorboard --logdir tensorboard/ --host 0.0.0.0 --port 6007
```
#### $$\textcolor{#eeff41}{\text{TODO MODELS with MLFLOW}}$$
#### $$\textcolor{#ffab40}{\text{PREDICTIONS}}$$
```sh
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_a.yaml
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_b.yaml
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_naive_a.yaml
PYTHONPATH=../src/ python ../src/ccf/apps/monitor.py -cd conf -cn monitor_multi_tgt_tft_naive_b.yaml
```
* collect predictions metrics
```sh
PYTHONPATH=../src/ python ../src/ccf/collect_metrics.py -cd conf -cn collect_metrics_lograt
```
### $$\textcolor{#eeeeee}{\text{RUN UI}}$$
* predictions
```sh
PYTHONPATH=../src/ streamlit run ../src/ccf/apps/ui.py conf/ui_multi_tgt_tft.yaml
```
* collected prediction metrics
```sh
PYTHONPATH=../src/ streamlit run ../src/ccf/apps/ui.py conf/ui_mid_tft_lograt_min_metrics.yaml
```
## CONFIGS EXAMPLES
### $$\textcolor{#4dd0e1}{\text{GET DATA}}$$
`work/conf/get_data_multi.yaml`
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
### $$\textcolor{#a2fca2}{\text{EXTRACT FEATURES}}$$
`work/conf/extract_features.yaml`
```yaml
verbose: true
delay: 1
remove_pre_features: false
pre_features:
- feature: m_p
  depth: ~
- feature: get
  columns: [ a_p_*, b_p_* ]
post_features:
- feature: relative
  kind: rat
  shift: 1
- feature: relative
  shift: 0
  kind: rat
  columns: [ "o_m_p_[1-9]*", "o_a_p_[1-9]*", "o_b_p_[1-9]*" ]
  column: o_m_p_0
- feature: relative
  shift: 0
  kind: rat
  columns: [ "o_a_p_[1-9]*" ]
  column: o_a_p_0
- feature: relative
  shift: 0
  kind: rat
  columns: [ "o_b_p_[1-9]*" ]
  column: o_b_p_0
resample_kwargs:
  rule: 1S
aggregate_kwargs:
  func: last
interpolate_kwargs:
  func: pad
feature_data_kwargs:
  start: -5
  end: ~
  concat: false
  query:
    feature:
      btcusdt:
        engine_kwargs:
          url: sqlite:///btcusdt@feature.db
        read_kwargs:
          name: data
          columns: ~
        write_kwargs:
          name: data
          if_exists: append
      ethusdt:
        engine_kwargs:
          url: sqlite:///ethusdt@feature.db
        read_kwargs:
          name: data
          columns: ~
        write_kwargs:
          name: data
          if_exists: append
      ethbtc:
        engine_kwargs:
          url: sqlite:///ethbtc@feature.db
        read_kwargs:
          name: data
          columns: ~
        write_kwargs:
          name: data
          if_exists: append
raw_data_kwargs:
  start: -5
  end: ~
  concat: False
  query:
    btcusdt:
      o:
        engine_kwargs:
          url: sqlite:///btcusdt@depth5@1000ms.db
        read_kwargs:
          name: data
          columns: ~
    ethusdt:
      o:
        engine_kwargs:
          url: sqlite:///ethusdt@depth5@1000ms.db
        read_kwargs:
          name: data
          columns: ~
    ethbtc:
      o:
        engine_kwargs:
          url: sqlite:///ethbtc@depth5@1000ms.db
        read_kwargs:
          name: data
          columns: ~
```
### $$\textcolor{#eeff41}{\text{TRAIN/TUNE MULTI TARGET TFT}}$$ 
`work/conf/train_multi_tgt_tft.yaml`
```yaml
model_path: multi_tgt_tft.ckpt
tune: true
create_dataset_kwargs:
  split: 0.8
  feature_data_kwargs:
    start: -360
    end: ~
    concat: false
    query:
      feature:
        btcusdt:
          engine_kwargs:
            url: sqlite:///btcusdt@feature.db
          read_kwargs:
            name: data
            columns: ~
          write_kwargs:
            name: data
            if_exists: append
        ethusdt:
          engine_kwargs:
            url: sqlite:///ethusdt@feature.db
          read_kwargs:
            name: data
            columns: ~
          write_kwargs:
            name: data
            if_exists: append
        ethbtc:
          engine_kwargs:
            url: sqlite:///ethbtc@feature.db
          read_kwargs:
            name: data
            columns: ~
          write_kwargs:
            name: data
            if_exists: append
  dataset_kwargs:
    time_idx: time_idx
    allow_missing_timesteps: true
    add_relative_time_idx: true
    target: [o_a_p_0, o_b_p_0]
    group_ids:
    - group
    static_categoricals:
    - group
    max_encoder_length: 10
    max_prediction_length: 5
    time_varying_unknown_reals:
    - o_a_p_0
    - o_b_p_0
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
### $$\textcolor{#ffab40}{\text{MAKE PREDICTIONS WITH MULTI TARGET TFT}}$$
`work/conf/predict_multi_tgt_tft.yaml`
```yaml 
defaults:
 - ./@train_kwargs: train_multi_tgt_tft

model_path: multi_tgt_tft.ckpt
verbose: true
past: 360
rule: 1S
predict_kwargs:
  return_index: true
  mode: prediction  # prediction or quantiles
engine_kwargs:
  url: sqlite:///multi_tgt_tft@prediction.db
write_kwargs:
  name: data
  if_exists: append
```
### $$\textcolor{#adadad}{\text{MONITOR RAW DATA METRICS}}$$
`conf/work/monitor_raw.yaml`
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
### $$\textcolor{#adadad}{\text{MONITOR FEATURES METRICS}}$$
`conf/work/monitor_reatures.yaml`
```yaml 
delay: 600
log_dir: monitor/features
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
  concat: false
  query:
    feature:
      btcusdt:
        engine_kwargs:
          url: sqlite:///btcusdt@feature.db
        read_kwargs:
          name: data
          columns: ~
        write_kwargs:
          name: data
          if_exists: append
      ethusdt:
        engine_kwargs:
          url: sqlite:///ethusdt@feature.db
        read_kwargs:
          name: data
          columns: ~
        write_kwargs:
          name: data
          if_exists: append
      ethbtc:
        engine_kwargs:
          url: sqlite:///ethbtc@feature.db
        read_kwargs:
          name: data
          columns: ~
        write_kwargs:
          name: data
          if_exists: append
```
### $$\textcolor{#eeeeee}{\text{RUN MULTI TARGET TFT UI}}$$
`conf/work/ui_multi_tgt_tft.yaml`
```yaml 
delay: 1
read_data_kwargs:
  start: -30
  end: ~
  query:
    btcusdt:
      target:
        engine_kwargs:
          url: sqlite:///btcusdt@feature.db
        read_kwargs:
          name: data
          columns: [ time, o_a_p_0, o_b_p_0 ]
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
          columns: [ time, pred-o_a_p_0, pred-o_b_p_0 ]
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
          columns: [ time, pred-o_a_p_0, pred-o_b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
    ethusdt:
      target:
        engine_kwargs:
          url: sqlite:///ethusdt@feature.db
        read_kwargs:
          name: data
          columns: [ time, o_a_p_0, o_b_p_0 ]
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
          columns: [ time, pred-o_a_p_0, pred-o_b_p_0 ]
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
          columns: [ time, pred-o_a_p_0, pred-o_b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
    ethbtc:
      target:
        engine_kwargs:
          url: sqlite:///ethbtc@feature.db
        read_kwargs:
          name: data
          columns: [ time, o_a_p_0, o_b_p_0 ]
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
          columns: [ time, pred-o_a_p_0, pred-o_b_p_0 ]
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
          columns: [ time, pred-o_a_p_0, pred-o_b_p_0 ]
          resample_kwargs:
            rule: 1S
          aggregate_kwargs:
            func: last
```
