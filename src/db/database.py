import json
import os
import pickle

import redis

class RedisDB:
    def __init__(self):
        self.conn = redis.Redis(
            host='docker.for.mac.localhost',
            port=6379,
            decode_responses=False
        )

    def get_value(self, key):
        return self.conn.get(key)

    def set_value(self, key, value):
        self.conn.set(key, value)

    def rpush(self, key, value):
        self.conn.rpush(key, json.dumps(value))

    def lrange(self, key, start, stop):
        messages = self.conn.lrange(key, start, stop)
        messages = [json.loads(m) for m in messages]
        return messages