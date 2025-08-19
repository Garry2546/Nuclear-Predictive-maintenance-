from fastapi import APIRouter, Depends
from ..models.accidentModel import SensorData , SensorDataCreate
from ..producer import kafka_producer
from ..config import KAFKA_TOPICS

accident_router = APIRouter()

@accident_router.post("/accident-data")
async def receive_sensor_data(data: SensorDataCreate):
    """Receive and forward sensor data to Kafka"""
    await kafka_producer.send(KAFKA_TOPICS["accident_data"], data.model_dump())
    return {"message": "Sensor data sent to Kafka"}
