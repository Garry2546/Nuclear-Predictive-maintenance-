import asyncio
import json
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
# from .tDatabase import Base, engine, get_db
from fastapi.middleware.cors import CORSMiddleware
from .database import Base2, engine2, get_db2
from .producer import kafka_producer
from .routers import sensor_routes
from .routers import accident_routes
from .routers import chatbot_routes
from sqlalchemy import text
# from .consumers.sensor_consumer import consume_sensor_data
from .consumers.sensor_classification_consumer import sensor_classification_data
from .consumers.accident_consumer import consume_accident_data
from .consumers.accident_classification_consume import accident_classification_data
from sqlalchemy.ext.asyncio import AsyncSession
from .websockets import websocketRouter
from sqlalchemy.orm import Session
from .config import KAFKA_BOOTSTRAP_SERVERS  # Import Kafka settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan event handling Kafka startup and shutdown."""
    global consumer_tasks

    # strating the timeseries database
    # await init_db()

    # starting the mysql server
    await init_db_2()

    # Initialize Kafka producer
    await kafka_producer.start()
    print("Kafka Producer Started.")

    # Start Kafka consumers
    consumer_tasks = [
        # asyncio.create_task(consume_sensor_data()),
        asyncio.create_task(consume_accident_data()),
        asyncio.create_task(sensor_classification_data()),
        asyncio.create_task(accident_classification_data())
    ]
    print("Kafka Consumers Started.")

    yield  # Run the FastAPI application

    # Cleanup on shutdown
    print("Shutting down Kafka Producer and Consumers...")
    await kafka_producer.stop()

    for task in consumer_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan=lifespan)


# async def init_db():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

async def init_db_2():
    async with engine2.begin() as conn2:
        await conn2.run_sync(Base2.metadata.create_all)



# Include API routes
app.include_router(sensor_routes.sensor_router)
app.include_router(accident_routes.accident_router)
app.include_router(websocketRouter)
app.include_router(chatbot_routes.chatbotRouter)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/test-db/")
# async def test_db_connection(db: AsyncSession = Depends(get_db)):  # ✅ Async function
#     try:
#         result = await db.execute(text("SELECT 1"))  # ✅ Await the DB query
#         return {"message": "Database connected successfully"}
#     except Exception as e:
#         return {"error": str(e)}

@app.get("/test-db-2/")
async def test_db_connectio_2(db: AsyncSession = Depends(get_db2)):  # ✅ Async function
    try:
        result = await db.execute(text("SELECT 1"))  # ✅ Await the DB query
        return {"message": "Database connected successfully"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0000", port=8000)