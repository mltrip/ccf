import time
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from kafka import KafkaProducer
from kafka import KafkaAdminClient
from kafka.admin.new_partitions import NewPartitions


KEY2PART = {
  'binance/btc/usdt': 0, 
  'binance/eth/usdt': 1,
  'binance/eth/btc': 2
}


def get_lob(depth=5):
  data = {}
  for i in range(depth):
    data[f'a_p_{i}'] = random.uniform(0, 1)  # ask price
    data[f'a_q_{i}'] = random.uniform(0, 1)  # ask quantity
    data[f'b_p_{i}'] = random.uniform(0, 1)  # bid price
    data[f'b_q_{i}'] = random.uniform(0, 1)  # bid quantity
  data['m_p'] = 0.5*(data[f'a_p_0'] + data[f'b_p_0'])  # mid price
  key = random.choice(list(KEY2PART.keys()))
  exchange, base, quote = key.split('/')
  data['exchange'] = exchange
  data['base'] = base
  data['quote'] = quote
  data['timestamp'] = time.time_ns()
  return key, data


def get_trade():
  data = {}
  data['t_p'] = random.uniform(0, 1)  # trade price
  data['t_q'] = random.uniform(0, 1)  # trade quantity
  data['t_s'] = random.randint(0, 1)  # trade side: 0 - maker_sell/taker_buy or 1 - maker_buy/taker_sell
  key = random.choice(list(KEY2PART.keys()))
  exchange, base, quote = key.split('/')
  data['exchange'] = exchange
  data['base'] = base
  data['quote'] = quote
  data['timestamp'] = time.time_ns()
  return key, data


def get_news():
  data = {}
  data['id'] = uuid.uuid4().hex
  data['title'] = random.choice(['Title', None])
  data['summary'] = random.choice(['Summary', None])
  data['links'] = random.choice(['link_1|link_2|link_3', None])
  data['authors'] = random.choice(['author_1|author_2|author_3', None])
  data['tags'] = random.choice(['tag_1|tag_2|tag_3', None])
  data['timestamp'] = time.time_ns()
  return None, data


def get_feature():
  data = {}
  data['f_0'] = random.uniform(0, 1)  # or any meaningful name
  data['f_1'] = random.uniform(0, 1)  # or any meaningful name
  data['f_2'] = random.uniform(0, 1)  # or any meaningful name
  key = random.choice(list(KEY2PART.keys()))
  exchange, base, quote = key.split('/')
  data['exchange'] = exchange
  data['base'] = base
  data['quote'] = quote
  data['quant'] = int(1e9)
  data['feature'] = random.choice(['group_1', 'group_2', 'group_3'])
  data['timestamp'] = time.time_ns()
  return key, data


def get_prediction():
  data = {}
  if random.randint(0, 1):
    data['quantile_0.25'] = random.uniform(0, 1)
    data['quantile_0.5'] = random.uniform(0, 1)
    data['quantile_0.75'] = random.uniform(0, 1)
  else:
    data['value'] = random.uniform(0, 1)
  key = random.choice(list(KEY2PART.keys()))
  exchange, base, quote = key.split('/')
  data['exchange'] = exchange
  data['base'] = base
  data['quote'] = quote
  data['quant'] = int(1e9)
  data['feature'] = random.choice(['group_1', 'group_2', 'group_3'])
  data['model'] = random.choice(['name_1', 'name_2', 'name_3'])
  data['version'] = random.randint(1, 3)
  data['target'] = random.choice(['m_p', 'a_p_0', 'b_p_0'])
  data['horizon'] = random.randint(1, 60)
  data['timestamp'] = time.time_ns()
  return key, data


def get_metric():
  data = {}
  data['MAE'] = random.uniform(0, 1)
  data['RMSE'] = random.uniform(0, 1)
  data['ROR'] = random.uniform(0, 1)
  key = random.choice(list(KEY2PART.keys()))
  exchange, base, quote = key.split('/')
  data['exchange'] = exchange
  data['base'] = base
  data['quote'] = quote
  data['quant'] = int(1e9)
  data['feature'] = random.choice(['group_1', 'group_2', 'group_3'])
  data['model'] = random.choice(['name_1', 'name_2', 'name_3'])
  data['version'] = random.randint(1, 3)
  data['target'] = random.choice(['m_p', 'a_p_0', 'b_p_0'])
  data['prediction'] = random.choice(['quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'value'])
  data['horizon'] = random.randint(1, 60)
  data['timestamp'] = time.time_ns()
  return key, data


def partitioner(key, all_partitions, available):
  """
  Customer Kafka partitioner to get the partition corresponding to key
  :param key: partitioning key
  :param all_partitions: list of all partitions sorted by partition ID
  :param available: list of available partitions in no particular order
  :return: one of the values from all_partitions or available
  """
  if key is None:
    if len(available) > 0:
      return random.choice(available)
    return random.choice(all_partitions)
  else:
    idx = KEY2PART[key.decode('ascii')]
    return all_partitions[idx]
  
  
def run(topic, data_function, server='kafka:9092', delay=1):
  producer = KafkaProducer(bootstrap_servers=server,
                           partitioner=partitioner,
                           key_serializer=lambda x: x.encode('ascii') if isinstance(x, str) else x,
                           value_serializer=lambda x: json.dumps(x).encode('ascii'))
  while True:
    t = time.time()
    key, data = data_function()
    if key is not None:
      if len(producer.partitions_for(topic)) != len(KEY2PART):
        client = KafkaAdminClient(bootstrap_servers=server)
        res = client.create_partitions({topic: NewPartitions(len(KEY2PART))})
        print(res)
      producer.send(topic, key=key, value=data)
    else:
      producer.send(topic, value=data)
    dt = time.time() - t
    wt = max(0, delay - dt)
    print(f'{topic}/{key}/{dt:.3f}/{wt:.3f}: {data}')
    time.sleep(wt)
    
    
def loop(executor, future2callable):
  for future in as_completed(future2callable):
    try:
      r = future.result()
    except Exception as e:
      print(f'Exception: {future} - {e}')
    else:
      print(f'Done: {future} - {r}')
    finally:  # Resubmit
      c, kwargs = future2callable[future]
      new_future = executor.submit(c, **kwargs)
      new_future2callable = {new_future: [c, kwargs]}
      loop(executor, new_future2callable)
  
  
def main():
  run_kwargs = [
    {'topic': 'lob', 'data_function': get_lob, 'delay': 1},
    {'topic': 'trade', 'data_function': get_trade, 'delay': 0.1},
    {'topic': 'news', 'data_function': get_news, 'delay': 60},
    {'topic': 'feature', 'data_function': get_feature, 'delay': 1},
    {'topic': 'prediction', 'data_function': get_prediction, 'delay': 1},
    {'topic': 'metric', 'data_function': get_metric, 'delay': 1},
  ]
  executor = ThreadPoolExecutor()
  f2c = {}  # future: [callable, kwargs]
  for kwargs in run_kwargs:
    future = executor.submit(run, **kwargs)
    f2c[future] = [run, kwargs]
  loop(executor, f2c)
  

if __name__ == "__main__":
  main()
