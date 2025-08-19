import asyncio
import json
from aiokafka import AIOKafkaConsumer
from ..config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPICS, KAFKA_GROUPS
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import Base2, get_db2, AsyncSessionLocal
from typing import Annotated
from ..services.accident_data_service import save_accident_data
from datetime import datetime


# SessionDep = Annotated[Session, Depends(get_db2)]

async def consume_accident_data():
    consumer = AIOKafkaConsumer(
        KAFKA_TOPICS["accident_data"],
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUPS["atkins"],
        auto_offset_reset='latest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        enable_auto_commit=True

    )

    print("Starting Kafka Consumer...")
    await consumer.start()

    try:
        async for msg in consumer:
            accident_data = msg.value
            print(f"Received accident data 1")

            # Get a database session and save the data
            async with AsyncSessionLocal() as db:  # Creates an async session
                await save_accident_data(db, accident_data)

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