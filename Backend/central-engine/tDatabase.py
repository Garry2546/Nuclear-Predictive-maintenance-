# import os
# from sqlalchemy.ext.asyncio import create_async_engine

# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.ext.asyncio import async_sessionmaker
# from sqlalchemy.ext.asyncio import AsyncSession
# from dotenv import load_dotenv

# load_dotenv()

# DATABASE_URL = f"postgresql+asyncpg://atkins:hackafuture@localhost:5432/timeseries"

# engine = create_async_engine(DATABASE_URL)
# SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
# Base = declarative_base()

# async def get_db():
#     async with SessionLocal() as db:
#         yield db 