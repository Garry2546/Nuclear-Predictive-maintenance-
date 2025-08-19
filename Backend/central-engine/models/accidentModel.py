from typing import Optional
from sqlalchemy import Column, Float, Integer ,DateTime
from ..database import Base2
from pydantic import BaseModel
from datetime import datetime

# ✅ SQLAlchemy Model (Database Table)
class SensorData(Base2):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, autoincrement=True)  # ✅ Auto-Increment Primary Key

    # ✅ Every column is optional (nullable=True allows NULL values)
    TIME = Column(DateTime, nullable=True)
    P = Column(Float, nullable=True)
    TAVG = Column(Float, nullable=True)
    THA = Column(Float, nullable=True)
    THB = Column(Float, nullable=True)
    TCA = Column(Float, nullable=True)
    TCB = Column(Float, nullable=True)
    WRCA = Column(Float, nullable=True)
    WRCB = Column(Float, nullable=True)
    PSGA = Column(Float, nullable=True)
    PSGB = Column(Float, nullable=True)
    WFWA = Column(Float, nullable=True)
    WFWB = Column(Float, nullable=True)
    WSTA = Column(Float, nullable=True)
    WSTB = Column(Float, nullable=True)
    VOL = Column(Float, nullable=True)
    LVPZ = Column(Float, nullable=True)
    VOID = Column(Float, nullable=True)
    WLR = Column(Float, nullable=True)
    WUP = Column(Float, nullable=True)
    HUP = Column(Float, nullable=True)
    HLW = Column(Float, nullable=True)
    WHPI = Column(Float, nullable=True)
    WECS = Column(Float, nullable=True)
    QMWT = Column(Float, nullable=True)
    LSGA = Column(Float, nullable=True)
    LSGB = Column(Float, nullable=True)
    QMGA = Column(Float, nullable=True)
    QMGB = Column(Float, nullable=True)
    NSGA = Column(Float, nullable=True)
    NSGB = Column(Float, nullable=True)
    TBLD = Column(Float, nullable=True)
    WTRA = Column(Float, nullable=True)
    WTRB = Column(Float, nullable=True)
    TSAT = Column(Float, nullable=True)
    QRHR = Column(Float, nullable=True)
    LVCR = Column(Float, nullable=True)
    SCMA = Column(Float, nullable=True)
    SCMB = Column(Float, nullable=True)
    FRCL = Column(Float, nullable=True)
    PRB = Column(Float, nullable=True)
    PRBA = Column(Float, nullable=True)
    TRB = Column(Float, nullable=True)
    LWRB = Column(Float, nullable=True)
    DNBR = Column(Float, nullable=True)
    QFCL = Column(Float, nullable=True)
    WBK = Column(Float, nullable=True)
    WSPY = Column(Float, nullable=True)
    WCSP = Column(Float, nullable=True)
    HTR = Column(Float, nullable=True)
    MH2 = Column(Float, nullable=True)
    CNH2 = Column(Float, nullable=True)
    RHBR = Column(Float, nullable=True)
    RHMT = Column(Float, nullable=True)
    RHFL = Column(Float, nullable=True)
    RHRD = Column(Float, nullable=True)
    RH = Column(Float, nullable=True)
    PWNT = Column(Float, nullable=True)
    PWR = Column(Float, nullable=True)
    TFSB = Column(Float, nullable=True)
    TFPK = Column(Float, nullable=True)
    TF = Column(Float, nullable=True)
    TPCT = Column(Float, nullable=True)
    WCFT = Column(Float, nullable=True)
    WLPI = Column(Float, nullable=True)
    WCHG = Column(Float, nullable=True)
    RM1 = Column(Float, nullable=True)
    RM2 = Column(Float, nullable=True)
    RM3 = Column(Float, nullable=True)
    RM4 = Column(Float, nullable=True)
    RC87 = Column(Float, nullable=True)
    RC131 = Column(Float, nullable=True)
    STRB = Column(Float, nullable=True)
    STSG = Column(Float, nullable=True)
    STTB = Column(Float, nullable=True)
    RBLK = Column(Float, nullable=True)
    SGLK = Column(Float, nullable=True)
    DTHY = Column(Float, nullable=True)
    DWB = Column(Float, nullable=True)
    WRLA = Column(Float, nullable=True)
    WRLB = Column(Float, nullable=True)
    WLD = Column(Float, nullable=True)
    MBK = Column(Float, nullable=True)
    EBK = Column(Float, nullable=True)
    TKLV = Column(Float, nullable=True)
    FRZR = Column(Float, nullable=True)
    TDBR = Column(Float, nullable=True)
    MDBR = Column(Float, nullable=True)
    MCRT = Column(Float, nullable=True)
    MGAS = Column(Float, nullable=True)
    TCRT = Column(Float, nullable=True)
    TSLP = Column(Float, nullable=True)
    PPM = Column(Float, nullable=True)
    RRCA = Column(Float, nullable=True)
    RRCB = Column(Float, nullable=True)
    RRCO = Column(Float, nullable=True)
    WFLB = Column(Float, nullable=True)


# ✅ Pydantic Model for Creating Data (All fields optional)
class SensorDataCreate(BaseModel):
    TIME: Optional[datetime] = None
    P: Optional[float] = None
    TAVG: Optional[float] = None
    THA: Optional[float] = None
    THB: Optional[float] = None
    TCA: Optional[float] = None
    TCB: Optional[float] = None
    WRCA: Optional[float] = None
    WRCB: Optional[float] = None
    PSGA: Optional[float] = None
    PSGB: Optional[float] = None
    WFWA: Optional[float] = None
    WFWB: Optional[float] = None
    WSTA: Optional[float] = None
    WSTB: Optional[float] = None
    VOL: Optional[float] = None
    LVPZ: Optional[float] = None
    VOID: Optional[float] = None
    WLR: Optional[float] = None
    WUP: Optional[float] = None
    HUP: Optional[float] = None
    HLW: Optional[float] = None
    WHPI: Optional[float] = None
    WECS: Optional[float] = None
    QMWT: Optional[float] = None
    LSGA: Optional[float] = None
    LSGB: Optional[float] = None
    QMGA: Optional[float] = None
    QMGB: Optional[float] = None
    NSGA: Optional[float] = None
    NSGB: Optional[float] = None
    TBLD: Optional[float] = None
    WTRA: Optional[float] = None
    WTRB: Optional[float] = None
    TSAT: Optional[float] = None
    QRHR: Optional[float] = None
    LVCR: Optional[float] = None
    SCMA: Optional[float] = None
    SCMB: Optional[float] = None
    FRCL: Optional[float] = None
    PRB: Optional[float] = None
    PRBA: Optional[float] = None
    TRB: Optional[float] = None
    LWRB: Optional[float] = None
    DNBR: Optional[float] = None
    QFCL: Optional[float] = None
    WBK: Optional[float] = None
    WSPY: Optional[float] = None
    WCSP: Optional[float] = None
    HTR: Optional[float] = None
    MH2: Optional[float] = None
    CNH2: Optional[float] = None
    RHBR: Optional[float] = None
    RHMT: Optional[float] = None
    RHFL: Optional[float] = None
    RHRD: Optional[float] = None
    RH: Optional[float] = None
    PWNT: Optional[float] = None
    PWR: Optional[float] = None
    TFSB: Optional[float] = None
    TFPK: Optional[float] = None
    TF: Optional[float] = None
    TPCT: Optional[float] = None
    WCFT: Optional[float] = None
    WLPI: Optional[float] = None
    WCHG: Optional[float] = None
    RM1: Optional[float] = None
    RM2: Optional[float] = None
    RM3: Optional[float] = None
    RM4: Optional[float] = None
    RC87: Optional[float] = None
    RC131: Optional[float] = None
    STRB: Optional[float] = None
    STSG: Optional[float] = None
    STTB: Optional[float] = None
    RBLK: Optional[float] = None
    SGLK: Optional[float] = None
    DTHY: Optional[float] = None
    DWB: Optional[float] = None
    WRLA: Optional[float] = None
    WRLB: Optional[float] = None
    WLD: Optional[float] = None
    MBK: Optional[float] = None
    EBK: Optional[float] = None
    TKLV: Optional[float] = None
    FRZR: Optional[float] = None
    TDBR: Optional[float] = None
    MDBR: Optional[float] = None
    MCRT: Optional[float] = None
    MGAS: Optional[float] = None
    TCRT: Optional[float] = None
    TSLP: Optional[float] = None
    PPM: Optional[float] = None
    RRCA: Optional[float] = None
    RRCB: Optional[float] = None
    RRCO: Optional[float] = None
    WFLB: Optional[float] = None
    # ... Continue with all other fields (same as above) ...


# ✅ Pydantic Model for Response Data
class SensorDataResponse(SensorDataCreate):
    id: int  # ✅ Includes the `id` field for responses

    class Config:
        orm_mode = True
