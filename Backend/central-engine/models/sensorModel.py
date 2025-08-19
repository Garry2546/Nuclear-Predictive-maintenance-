from pydantic import BaseModel
from typing import Dict
from typing import Optional
from datetime import datetime
from sqlalchemy import Column, String, TIMESTAMP, Float
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
# from ..tDatabase import Base

# class SensorData(Base):
#     __tablename__ = "sensor_data"  # Ensure this matches the existing table

#     device_id = Column(String, primary_key=True)
#     timestamp = Column(TIMESTAMP, primary_key=True)
    
#     sensor_0 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_1 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_2 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_3 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_4 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_5 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_6 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_7 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_8 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_9 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_10 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_11 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_12 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_13 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_14 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_15 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_16 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_17 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_18 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_19 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_20 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_21 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_22 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_23 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_24 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_25 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_26 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_27 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_28 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_29 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_30 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_31 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_32 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_33 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_34 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_35 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_36 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_37 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_38 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_39 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_40 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_41 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_42 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_43 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_44 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_45 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_46 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_47 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_48 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_49 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_50 = Column(DOUBLE_PRECISION, nullable=True)
#     sensor_51 = Column(DOUBLE_PRECISION, nullable=True)
    
#     machine_status = Column(String, nullable=True)

class SensorMeasurementsCreate(BaseModel):
    device_id : str
    timestamp : datetime
    
    # Define 50 sensor columns
    sensor_0: float
    sensor_1 : float
    sensor_2 : float
    sensor_3 : float
    sensor_4 : float
    sensor_5 : float
    sensor_6 : float
    sensor_7 : float
    sensor_8 : float
    sensor_9 : float
    sensor_10 : float
    sensor_11 : float
    sensor_12 : float
    sensor_13 : float
    sensor_14 : float
    sensor_15 : float
    sensor_16 : float
    sensor_17 : float
    sensor_18: float
    sensor_19 : float
    sensor_20 : float
    sensor_21: float
    sensor_22 : float
    sensor_23 : float
    sensor_24 : float
    sensor_25 : float
    sensor_26: float
    sensor_27: float
    sensor_28: float
    sensor_29: float
    sensor_30: float
    sensor_31: float
    sensor_32: float
    sensor_33: float
    sensor_34: float
    sensor_35: float
    sensor_36: float
    sensor_37: float
    sensor_38: float
    sensor_39: float
    sensor_40: float
    sensor_41: float
    sensor_42: float
    sensor_43: float
    sensor_44: float
    sensor_45: float
    sensor_46: float
    sensor_47: float
    sensor_48: float
    sensor_49: float
    sensor_50: float
    sensor_51 : float
    machine_status: str

class SensorDatadb(BaseModel):
    timestamp: datetime
    device_id: str
