import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import async_sessionmaker

# ✅ Use an async database connection (Update MySQL URL)
DATABASE_URL = "mysql+aiomysql://atkins:hackafuture@localhost/mydb"

# ✅ Create an async engine
engine2 = create_async_engine(DATABASE_URL, echo=True)

# ✅ Use async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine2, 
    class_=AsyncSession, 
    expire_on_commit=False
)

Base2 = declarative_base()

# ✅ Async database session dependency
async def get_db2():
    async with AsyncSessionLocal() as db:
        yield db
