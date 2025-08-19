import pandas as pd
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class SensorDataSimulator:
    def __init__(self, csv_path, api_url, interval=10):
        """
        Initialize simulator with sensor data
        
        :param csv_path: Path to CSV file with sensor data
        :param api_url: FastAPI endpoint URL
        :param interval: Seconds between sending each row
        """
        self.df = pd.read_csv(csv_path)
        self.api_url = api_url
        self.interval = interval
        
        # Convert sensor columns to float, handling missing values
        sensor_columns = [col for col in self.df.columns if col.startswith("sensor_")]
        self.df[sensor_columns] = self.df[sensor_columns].astype(float)

    def row_to_json(self, row, index):
        """
        Convert a row to JSON with proper column names, replacing NaN values with None
        """
        base_time = datetime.now()
        current_time = base_time + timedelta(seconds=index * self.interval)
        
        data = {
            "device_id": "1",
            "timestamp": current_time.isoformat(),
            "machine_status": row["machine_status"]
        }

        # Add sensors directly (no nested dictionary)
        for col in row.index:
            if col.startswith("sensor_"):
                data[col] = 0 if pd.isna(row[col]) else float(row[col])  # Replace NaN with 0
        return data
    async def send_row(self, row, index):
        """
        Send a single row of data to FastAPI endpoint
        """
        json_data = self.row_to_json(row, index)
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.api_url, json=json_data) as response:
                    if response.status == 200:
                        print(f"✅ Sent data at {json_data['timestamp']}")
                    else:
                        print(f"❌ Failed to send data. Status: {response.status}")
            except Exception as e:
                print(f"⚠️ Error sending data: {e}")

    async def simulate(self):
        """
        Simulate sending sensor data row by row with interval
        """
        print(f"🚀 Starting simulation with {len(self.df)} rows of data...")
        print(f"📡 Data will be sent every {self.interval} seconds")
        
        for index, row in self.df.iterrows():
            print(f"\n📤 Sending row {index + 1}/{len(self.df)}")
            await self.send_row(row, index)
            await asyncio.sleep(self.interval)

def main():
    simulator = SensorDataSimulator(
        csv_path='pump_sensor.csv',
        api_url='http://localhost:8000/sensor-data',
        interval=10
    )
    asyncio.run(simulator.simulate())

if __name__ == "__main__":
    main()
