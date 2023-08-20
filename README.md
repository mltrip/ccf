# CCF
Crypto Currency Forecasting App for [ML System Design Course on ODS.ai](https://ods.ai/tracks/ml-system-design-22)

Also read the article ["Yet another architecture of ML crypto trading system."](https://medium.com/@romanzes/yet-another-architecture-of-ml-crypto-trading-system-381544d32c30) on Medium

More visualizations and metrics are available in the [presentation](https://docs.google.com/presentation/d/1HI-84QSIph-YY7kKMD4bdUZ_YA4oXOgUfFcUXcie3Pg/edit?usp=sharing)

## Architecture
![architecture](docs/architecture.png)

App consists of 6 main parts
> We could install and run different parts of the App independently
### $$\textcolor{#4dd0e1}{\text{MINING}}$$
In this part data `RAW DATA` is mined from `LOB`, `TRADE` and `NEWS`.
### $$\textcolor{#a2fca2}{\text{DATA}}$$
Here `FEATURES` are extracted from `RAW DATA` and `DATASET` is created from `FEATURES` and/or `PREDICTIONS`.
### $$\textcolor{#eeff41}{\text{MODEL}}$$
Here `MODELS` are trained/tuned using `DATASETS` and then stored in the `MODELS REGISTRY`.
### $$\textcolor{#ffab40}{\text{PREDICTIONS}}$$
`MODELS` loaded from the `MODELS REGISTRY` make `PREDICTIONS` using `DATASETS`.
### $$\textcolor{#eeeeee}{\text{METRICS}}$$
`METRICS` are collected from pipeline.
### $$\textcolor{#adadad}{\text{TRADING}}$$
Here the `AGENT` trades using `PREDICTIONS`.
## Process
![process](docs/process.png)
## Deployment
![deployment](docs/deployment.png)

## RUN DOCKER WITH MLFLOW
### Go to docker directory
```sh
cd docker
```
### Generate self-signed certificate for InfluxDB
```sh
sudo openssl req -x509 -nodes -newkey rsa:2048 -keyout influxdb-selfsigned.key -out influxdb-selfsigned.crt -days 365
```
### Set sensitive environment variables for InfluxDB
```sh
cp .env.secret.db.example .env.secret.db
```
### Run Kafka
```sh
docker compose -f docker-compose.kafka.yaml up -d
```
### Run InfluxDB
```sh
docker compose -f docker-compose.db.yaml up -d
```
### Build CCF Image
```sh
docker compose -f docker-compose.get_data.yaml build
```
### Run get_data, extract_features and collect_metrics
```sh
docker compose -f docker-compose.get_data.yaml up -d
docker compose -f docker-compose.extract_features.yaml up -d
docker compose -f docker-compose.collect_metrics.yaml up -d
```
### Set sensitive environment variables for MLflow
```sh
cp .env.secret.mlflow.example .env.secret.mlflow
```
### Generate password for user "ccf" for NGINX proxy of MLflow
```sh
htpasswd -c .htpasswd ccf
```
### Run MLflow
```sh
docker compose -f docker-compose.mlflow.yaml up -d
```
### Set sensitive environment variables for models (password from .htpasswd, influxdb token from .env.secret.db)
```sh
cp .env.secret.model.example .env.secret.model
```
### Train model from influxdb
```sh
docker compose -f docker-compose.train.mlflow.influxdb.yaml up -d
```
### Predict model to kafka
```sh
docker compose -f docker-compose.predict.mlflow.kafka.influxdb up -d
```
### Run Streamlit UI (localhost:8501)
```sh
docker compose -f docker-compose.ui.yaml up -d
```
### Optionally collect system metrics to indluxdb
```sh
docker compose -f docker-compose.system.yaml up -d
```
### Monitor Streamlit (host: localhost:8501)
![streamlit](docs/streamlit.png)
### Monitor InfluxDB (host: localhost:8086, user: ccf, password: see .env.secret.db)
![influxdb](docs/influxdb.png)
### Monitor MLflow (host: localhost:5000, user: ccf, password: see .env.secret.model)
![mlflow](docs/mlflow.png)

## RUN DOCKER WITHOUT MLFLOW
### Go to docker directory
```sh
cd docker
```
### Generate self-signed certificate for InfluxDB
```sh
sudo openssl req -x509 -nodes -newkey rsa:2048 -keyout influxdb-selfsigned.key -out influxdb-selfsigned.crt -days 365
```
### Set sensitive environment variables for InfluxDB
```sh
cp .env.secret.db.example .env.secret.db
```
### Run Kafka
```sh
docker compose -f docker-compose.kafka.yaml up -d
```
### Run InfluxDB
```sh
docker compose -f docker-compose.db.yaml up -d
```
### Build CCF Image
```sh
docker compose -f docker-compose.data.feature.metric.yaml build
```
### Run get_data, extract_features and collect_metrics
```sh
docker compose -f docker-compose.get_data.yaml up -d
docker compose -f docker-compose.extract_features.yaml up -d
docker compose -f docker-compose.collect_metrics.yaml up -d
```
### Train model from influxdb
```sh
docker compose -f docker-compose.train.local.influxdb.yaml up -d
```
### Predict model to kafka
```sh
docker compose -f docker-compose.predict.local.kafka.influxdb up -d
```
### Run Streamlit UI (host: localhost:8501)
```sh
docker compose -f docker-compose.ui.yaml up -d
```
### Monitor Streamlit (host: localhost:8501)
### Monitor InfluxDB (host: localhost:8086, user: ccf, password: see .env.secret.db)

## RUN MANUALLY
### Install Python 3.9
### $$\textcolor{#ffffff}{\text{ALL}}$$ 
```sh
pip install -r requirements.txt
```
### $$\textcolor{#4dd0e1}{\text{MINING}}$$ 
```sh
pip install -r src/ccf/requirements_data.txt
```
### $$\textcolor{#a2fca2}{\text{FEATURES}}$$
```sh
pip install -r src/ccf/requirements_features.txt
``` 
### $$\textcolor{#eeff41}{\text{MODEL}}$$ 
```sh
pip install -r src/ccf/requirements_ml.txt
```
### $$\textcolor{#ffab40}{\text{PREDICTIONS}}$$ 
```sh
pip install -r src/ccf/requirements_predictions.txt
```
### $$\textcolor{#eeeeee}{\text{METRICS}}$$
```sh
pip install -r src/ccf/requirements_metrics.txt
```
### $$\textcolor{#adadad}{\text{TRADING}}$$
```sh
pip install -r src/ccf/requirements_trade.txt
```
## RUN
```sh
cd work
```
### $$\textcolor{#4dd0e1}{\text{GET DATA}}$$ 
* Linux (by default)
```sh
PYTHONPATH=../src/ python ../src/ccf/get_data.py -cd conf -cn get_data-kafka-binance-btc-usdt
```
* Windows (as example)
```sh
cmd /C  "set PYTHONPATH=../src && python ../src/ccf/get_data.py -cd conf -cn get_data-kafka-binance-btc-usdt"
```
### $$\textcolor{#a2fca2}{\text{EXTRACT FEATURES}}$$
```sh
PYTHONPATH=../src/ python ../src/ccf/extract_features.py -cd conf -cn extract_features-kafka-binance-btc-usdt
```
### $$\textcolor{#eeff41}{\text{TRAIN/TUNE MODEL}}$$ 
* Train once
```sh
PYTHONPATH=../src/ python ../src/ccf/train.py -cd conf -cn  train-mid-lograt-tft-kafka-binance-btc-usdt
```
* Tune every ~1 hour
```sh
while true; do PYTHONPATH=../src/ python ../src/ccf/train.py -cd conf -cn train-mid-lograt-tft-kafka-binance-btc-usdt; sleep 3600; done
```
### $$\textcolor{#ffab40}{\text{MAKE PREDICTIONS}}$$
```sh
PYTHONPATH=../src/ python ../src/ccf/predict.py -cd conf -cn predict-mid-lograt-tft-kafka-binance-btc-usdt
```
### $$\textcolor{#ffab40}{\text{COLLECT PREDICTIONS METRICS}}$$
```sh
PYTHONPATH=../src/ python ../src/ccf/collect_metrics.py -cd conf -cn collect_metrics-kafka-binance-btc-usdt
```
### $$\textcolor{#eeeeee}{\text{MONITOR METRICS}}$$ 
* Monitor metrics with InfluxDB (host: localhost:8086, user: ccf, password: see .env.secret.db)
* Monitor metrics with MLflow (host: localhost:5000, user: ccf, password: see .env.secret.model)
* Tensorboard (localhost:6007)
```sh
cd work
tensorboard --logdir tensorboard/ --host 0.0.0.0 --port 6007
```
#### $$\textcolor{#adadad}{\text{RUN AGENTS}}$$
```sh
PYTHONPATH=../src/ ../src/ccf/trade.py, -cd, conf, -cn, trade-kafka-binance-btc-tusd-fast-rl-1
PYTHONPATH=../src/ ../src/ccf/trade.py, -cd, conf, -cn, trade-kafka-binance-btc-tusd-fast-rl-2
...
```
