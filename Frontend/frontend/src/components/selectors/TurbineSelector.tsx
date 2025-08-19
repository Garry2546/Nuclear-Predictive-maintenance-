"use client";
import { create } from "zustand";
const turbineSensors = ["Turbine Sensor X", "Turbine Sensor Y", "Turbine Sensor Z"];


interface SensorStore {
    selectedPump: string | null;
    selectedTurbine: string | null;
    setSelectedPump: (pump: string) => void;
    setSelectedTurbine: (turbine: string) => void;
}

export const useSensorStore = create<SensorStore>((set) => ({
    selectedPump: null,
    selectedTurbine: null,
    setSelectedPump: (pump) => set({ selectedPump: pump }),
    setSelectedTurbine: (turbine) => set({ selectedTurbine: turbine }),
}));


interface TurbineSelectorProps {
    selectedTurbine: string | null;
    setSelectedTurbine: (turbine: string) => void;
}

export default function TurbineSelector({ selectedTurbine, setSelectedTurbine }: TurbineSelectorProps) {
    return (
        <select 
            value={selectedTurbine || ""}
            onChange={(e) => setSelectedTurbine(e.target.value)}
            className="w-full p-2 bg-gray-700 text-white rounded-lg"
        >
            <option value="" disabled>Select a Turbine Sensor</option>
            {turbineSensors.map((sensor, index) => (
                <option key={index} value={sensor}>{sensor}</option>
            ))}
        </select>
    );
}
