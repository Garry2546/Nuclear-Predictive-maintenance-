import { useSensorStore } from "../../store/sensorStore";

const ReactorCore = () => {
    const { temperature, pressure, powerOutput } = useSensorStore((state) => state.sensorData);

    return (
        <div className="p-4 bg-gray-800 rounded-lg shadow-md">
            <h2 className="text-lg font-bold">REACTOR CORE MONITORING</h2>
            <p>Temperature: <span className="text-red-400">{temperature}°C</span></p>
            <p>Power Output: <span className="text-yellow-400">{powerOutput} MW</span></p>
            <p>Pressure: <span className="text-blue-400">{pressure} PSI</span></p>
        </div>
    );
};

export default ReactorCore;
