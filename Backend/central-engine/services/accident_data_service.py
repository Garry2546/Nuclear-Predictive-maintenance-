import json
from sqlalchemy import select
from datetime import datetime
from ..models.accidentModel import SensorData
from sqlalchemy.orm import Session
from ..database import Base2, get_db2, AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession


async def save_accident_data(db:AsyncSession, accident_data:dict):
    """Save incoming sensor data to the database"""
    try:
        # Create an instance of the SensorDataModel
        accident_data["TIME"] = datetime.fromisoformat(accident_data["TIME"])
        accident_entry = SensorData(**accident_data)

        db.add(accident_entry)
        await db.commit()
        print(f"Saved accident data at {accident_data['TIME']}")

    except Exception as e:
        print(f"Error saving sensor data: {e}")
        await db.rollback()