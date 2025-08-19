import asyncio
import json
from aiokafka import AIOKafkaConsumer
from ..config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPICS, KAFKA_GROUPS
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
# from ..tDatabase import SessionLocal
from datetime import datetime
from ..websockets import ws_manager
import random


# SessionDep = Annotated[AsyncSession, Depends(get_db)]

async def sensor_classification_data():
    consumer = AIOKafkaConsumer(
        KAFKA_TOPICS["sensor_data"],
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUPS["atkins-2"],
        auto_offset_reset='latest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        enable_auto_commit=True

    )

    print("Starting Kafka Consumer...")
    await consumer.start()

    try:
        async for msg in consumer:
            sensor_data = msg.value
            print(f"Received sensor data 2")

            classList = ["NORMAL" , "BROKEN" , "RECOVERING"]

            # Get a database session and save the data
            response_data = {
                "time": sensor_data["timestamp"],
                "classification": classList[random.randint(0,2)]
            }
            await ws_manager.broadcast(response_data)

    except Exception as e:
        print("the error is " , e)
        
    finally:
        print("stopping the kafka consumer ....")
        await consumer.stop()


def json_serializer(obj):
    """Custom serializer to handle datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to string
    raise TypeError(f"Type {type(obj)} not serializable")