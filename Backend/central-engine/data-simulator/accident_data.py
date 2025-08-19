import pandas as pd
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class NuclearSensorSimulator:
    def __init__(self, csv_path, api_url, interval=10):
        """
        Initialize simulator with nuclear sensor data

        :param csv_path: Path to CSV file with sensor data
        :param api_url: FastAPI endpoint URL
        :param interval: Seconds between sending each row
        """
        # Read CSV with first row as headers
        self.df = pd.read_csv(csv_path)
        self.api_url = api_url
        self.interval = interval

        # Convert numeric columns to float
        self.df = self.df.astype(float)

    def row_to_json(self, row, index):
        """
        Convert a row to JSON with flat structure and correct column names
        """
        # Simulate timestamp using TIME column as base
        base_time = datetime.now()
        current_time = base_time + timedelta(seconds=index * self.interval)

        # Add TIME as the simulated timestamp
        data = row.to_dict()
        data["TIME"] = current_time.isoformat()  # ✅ Keep TIME consistent
        print(data)

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
                        print(f"✅ Sent data at {json_data['TIME']}")
                    else:
                        print(f"❌ Failed to send data. Status: {response.status}")
            except Exception as e:
                print(f"⚠️ Error sending data: {e}")

    async def simulate(self):
        """
        Simulate sending sensor data row by row with interval
        """
        print(f"🚀 Starting simulation with {len(self.df)} rows of data...")
        print(f"⏳ Data will be sent every {self.interval} seconds")

        for index, row in self.df.iterrows():
            print(f"\n📡 Sending row {index + 1}/{len(self.df)}")
            await self.send_row(row, index)
            await asyncio.sleep(self.interval)

def main():
    simulator = NuclearSensorSimulator(
        csv_path='NRML.csv',
        api_url='http://localhost:8000/accident-data',
        interval=10
    )
    asyncio.run(simulator.simulate())

if __name__ == "__main__":
    main()
