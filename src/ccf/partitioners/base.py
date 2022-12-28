import json
from pprint import pprint
import random

from kafka import KafkaAdminClient
from kafka.admin import NewPartitions
from kafka.errors import InvalidPartitionsError
   
  
class Partitioner:
  def __init__(self, topic, mapping, admin_kwargs, verbose):
    super().__init__()
    self.topic = topic
    self.mapping = mapping
    self.admin_kwargs = admin_kwargs
    self.verbose = verbose
    
  def update(self):
    if self.verbose:
      print(f'Mapping of topic "{self.topic}":')
      pprint(self.mapping)
    mapping_partitions = [i for k, v in self.mapping.items() 
                          if v is not None for i in v]
    assert len(mapping_partitions) == len(set(mapping_partitions))
    if self.verbose:
      print(f'Number of mapping partitions: {len(mapping_partitions)}')
    admin = KafkaAdminClient(**self.admin_kwargs)
    new_partitions = NewPartitions(len(mapping_partitions))
    try:
      r = admin.create_partitions({self.topic: new_partitions})
    except InvalidPartitionsError as e:
      if self.verbose:
        print(e)
    else:
      if self.verbose:
        print(r)
  
  def serialize_key(self, key):
    return key.encode('ascii') if isinstance(key, str) else key
  
  def deserialize_key(self, key):
    return key.decode('ascii')
  
  def serialize_value(self, value):
    return json.dumps(value).encode('ascii')
  
  def deserialize_value(self, value):
    return json.loads(value.decode('ascii'))
    
  def __getitem__(self, item):
    return self.mapping.get(item, [])
  
  def __call__(self, key, all_partitions, available):
    if key is None:
      if len(available) > 0:
        return random.choice(available)
      return random.choice(all_partitions)
    else:
      map_partitions = self.mapping[self.deserialize_key(key)]
      if len(map_partitions) > 0:
        return random.choice(map_partitions)
      return random.choice(all_partitions)
