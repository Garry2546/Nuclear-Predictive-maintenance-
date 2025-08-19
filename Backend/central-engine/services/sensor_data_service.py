# import json
# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.future import select
# from datetime import datetime
# from ..models.sensorModel import SensorData
# from sqlalchemy.ext.asyncio import AsyncSession

# async def save_sensor_data(db: AsyncSession, sensor_data: dict):
#     """Save incoming sensor data to the database"""
#     try:
#         # Create an instance of the SensorDataModel
#         sensor_data["timestamp"] = datetime.fromisoformat(sensor_data["timestamp"])
#         sensor_entry = SensorData(**sensor_data)

#         db.add(sensor_entry)
#         await db.commit()
#         print(f"Saved sensor data for {sensor_data['device_id']} at {sensor_data['timestamp']}")

#     except Exception as e:
#         print(f"Error saving sensor data: {e}")
#         await db.rollback()
