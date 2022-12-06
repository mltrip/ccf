from pprint import pprint

from ccf import partitioners as ccf_partitioners
from kafka import KafkaProducer, KafkaConsumer, TopicPartition


class Agent:
  def __init__(self):
    super().__init__()    


class Kafka(Agent):
  def __init__(self, consumers=None, producers=None, verbose=False):
    super().__init__()
    consumers = {} if consumers is None else consumers
    producers = {} if producers is None else producers
    self.verbose = verbose
    # Initialize consumers
    consumers_topic_keys = {}
    consumers_partitioners = {}
    for name, consumer in consumers.items():
      topics, topics_partitions = [], []
      partitioners = consumer.pop('partitioners', {})
      topic_keys = consumer.pop('topic_keys', {})
      for topic, keys in topic_keys.items():
        partitioner = partitioners.get(topic, None)
        if isinstance(partitioner, dict):
          class_name = partitioner.pop('class', 'Partitioner')
          c = getattr(ccf_partitioners, class_name)
          partitioner = c(**partitioner)
          partitioners[topic] = partitioner
        consumer['key_deserializer'] = partitioner.deserialize_key
        consumer['value_deserializer'] = partitioner.deserialize_value
        if keys is None:
          topics.add(topic)
        else:
          for key in keys:
            partitions = partitioner[key]
            for partition in partitions:
              topics_partitions.append(TopicPartition(topic, partition))
      consumer = KafkaConsumer(**consumer)
      if len(topics_partitions) > 0:
        consumer.assign(topics_partitions)
      else:
        consumer.subsribe(topics)
      consumers[name] = consumer
      consumers_topic_keys[name] = topic_keys
      consumers_partitioners[name] = partitioners
    self.consumers = consumers
    self.consumers_topic_keys = consumers_topic_keys
    self.consumers_partitioners = consumers_partitioners
    if verbose:
      pprint(consumers_topic_keys)
      pprint(consumers_partitioners)
      pprint(consumers)
    # Initialize producers
    producers_topic_keys = {}
    producers_partitioners = {}
    for name, producer in producers.items():
      partitioners = producer.pop('partitioners', {})
      topic_keys = producer.pop('topic_keys', {})
      for topic, keys in topic_keys.items():
        partitioner = partitioners.get(topic, None)
        if isinstance(partitioner, dict):
          class_name = partitioner.pop('class', 'Partitioner')
          c = getattr(ccf_partitioners, class_name)
          partitioner = c(**partitioner)
          partitioners[topic] = partitioner
        producer['key_serializer'] = partitioner.serialize_key
        producer['value_serializer'] = partitioner.serialize_value
      producer = KafkaProducer(**producer)
      producers[name] = producer
      producers_topic_keys[name] = topic_keys
      producers_partitioners[name] = partitioners
    self.producers_topic_keys = producers_topic_keys
    self.producers_partitioners = producers_partitioners
    self.producers = producers
    if verbose:
      pprint(producers_topic_keys)
      pprint(producers_partitioners)
      pprint(producers)
  
  @staticmethod
  def init_consumer(consumer):
    print(consumer)
    topics, topics_partitions = [], []
    partitioners = consumer.pop('partitioners', {})
    topic_keys = consumer.pop('topic_keys', {})
    for topic, keys in topic_keys.items():
      partitioner = partitioners.get(topic, {})
      if isinstance(partitioner, dict):
        class_name = partitioner.pop('class', 'Partitioner')
        c = getattr(ccf_partitioners, class_name)
        partitioner = c(**partitioner)
        partitioners[topic] = partitioner
      consumer['key_deserializer'] = partitioner.deserialize_key
      consumer['value_deserializer'] = partitioner.deserialize_value
      if keys is None:
        topics.add(topic)
      else:
        for key in keys:
          partitions = partitioner[key]
          for partition in partitions:
            topics_partitions.append(TopicPartition(topic, partition))
    consumer = KafkaConsumer(**consumer)
    if len(topics_partitions) > 0:
      consumer.assign(topics_partitions)
    else:
      consumer.subsribe(topics)
    return consumer
  
  @staticmethod
  def init_producer(kwargs): 
    if isinstance(kwargs.get('partitioner', {}), dict):
      kwargs['partitioner'] = Kafka.init_partitioner()
    partitioner = kwargs['partitioner']
    kwargs['key_serializer'] = partitioner.serialize_key
    kwargs['value_serializer'] = partitioner.serialize_value
    producer = KafkaProducer(**kwargs)
    return producer