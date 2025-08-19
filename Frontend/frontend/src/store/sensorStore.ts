import { create } from 'zustand';
import { pumpSensorData } from '../utils/mockData';

// Define the type for a single pump's data
type PumpData = {
  timestamp: string;
  sensors: {
    sensor_temperature: number;
    sensor_pressure: number;
    sensor_vibration: number;
  };
  machine_status: string;
};

// Define valid pump names as a union type
type PumpName = keyof typeof pumpSensorData;

interface SensorStore {
  [x: string]: any;
  selectedPump: PumpName | null;
  setSelectedPump: (pump: PumpName | null) => void;
  pumpData: PumpData | null;
}

export const useSensorStore = create<SensorStore>((set) => ({
  selectedPump: null,
  setSelectedPump: (pump) => set({ 
    selectedPump: pump,
    pumpData: pump ? pumpSensorData[pump] : null 
  }),
  pumpData: null,
}));

// Export the types so they can be reused
export type { PumpName, PumpData };