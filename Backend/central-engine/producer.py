from aiokafka import AIOKafkaProducer
import json
from datetime import datetime
from .config import KAFKA_BOOTSTRAP_SERVERS


def json_serializer(obj):
    """Custom serializer to handle datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to string
    raise TypeError(f"Type {type(obj)} not serializable")


class KafkaProducer:
    def __init__(self):
        self.producer = None

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v, default=json_serializer).encode('utf-8')
        )
        await self.producer.start()

    async def send(self, topic: str, message: dict):
        if self.producer is None:
            raise Exception("Producer not started")
        print("here is the topic i am publishing to " , topic)
        await self.producer.send(topic, value=message)

    async def stop(self):
        if self.producer:
            await self.producer.stop()



# Singleton instance of producer
kafka_producer = KafkaProducer()
