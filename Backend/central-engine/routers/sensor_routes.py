from fastapi import APIRouter, Depends
from ..models.sensorModel import SensorMeasurementsCreate
from ..producer import kafka_producer
from ..config import KAFKA_TOPICS

sensor_router = APIRouter()

@sensor_router.post("/sensor-data")
async def receive_sensor_data(data: SensorMeasurementsCreate):
    """Receive and forward sensor data to Kafka"""

    await kafka_producer.send(KAFKA_TOPICS["sensor_data"], data.model_dump())

    return {"message": "Sensor data sent to Kafka"}
