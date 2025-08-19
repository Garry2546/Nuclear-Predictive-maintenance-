import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";
import { useSensorStore } from "../../store/sensorStore";
import { useState, useEffect } from "react";

const LiveChart = () => {
    const { temperature } = useSensorStore((state) => state.sensorData);
    const [data, setData] = useState<{ time: string; temp: number }[]>([]);

    useEffect(() => {
        setData((prev) => [...prev.slice(-20), { time: new Date().toLocaleTimeString(), temp: temperature }]);
    }, [temperature]);

    return (
        <LineChart width={600} height={300} data={data}>
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="temp" stroke="#39FF14" />
        </LineChart>
    );
};

export default LiveChart;
