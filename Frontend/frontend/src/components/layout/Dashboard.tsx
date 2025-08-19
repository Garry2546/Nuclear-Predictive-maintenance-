"use client";

import { useSensorStore } from "../../store/sensorStore";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

export default function Dashboard() {
    const { pumpData } = useSensorStore();

    // Create mock historical data for the graphs
    const generateHistoricalData = (currentValue: number, dataPoints: number = 10) => {
        return Array.from({ length: dataPoints }, (_, i) => {
            const time = new Date();
            time.setMinutes(time.getMinutes() - (dataPoints - i - 1) * 5);
            return {
                timestamp: time.toLocaleTimeString(),
                value: currentValue + (Math.random() - 0.5) * 10
            };
        });
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case "Stable": return "var(--success)";
            case "Warning": return "var(--warning)";
            case "Critical": return "var(--danger)";
            default: return "var(--text-secondary)";
        }
    };

    return (
        <div className="space-y-6">
            <h2 className="text-xl font-semibold mb-4">Sensor Data</h2>
            
            {pumpData ? (
                <>
                    {/* Status Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div className="card">
                            <h3 className="text-sm font-medium mb-2">Temperature</h3>
                            <p className="text-2xl">{pumpData.sensors.sensor_temperature}°C</p>
                        </div>
                        <div className="card">
                            <h3 className="text-sm font-medium mb-2">Pressure</h3>
                            <p className="text-2xl">{pumpData.sensors.sensor_pressure} PSI</p>
                        </div>
                        <div className="card">
                            <h3 className="text-sm font-medium mb-2">Vibration</h3>
                            <p className="text-2xl">{pumpData.sensors.sensor_vibration}</p>
                        </div>
                        <div className="card">
                            <h3 className="text-sm font-medium mb-2">Status</h3>
                            <p className={`text-lg rounded-full px-3 py-1 text-center ${
                                pumpData.machine_status === "Stable" ? "bg-[var(--success)]" : 
                                pumpData.machine_status === "Warning" ? "bg-[var(--warning)]" : "bg-[var(--danger)]"
                            }`}>
                                {pumpData.machine_status}
                            </p>
                        </div>
                    </div>

                    {/* Sensor Graphs */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Temperature Graph */}
                        <div className="card">
                            <h3 className="text-lg font-medium mb-4">Temperature Trends</h3>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={generateHistoricalData(pumpData.sensors.sensor_temperature)}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                        <XAxis 
                                            dataKey="timestamp" 
                                            tickFormatter={(str) => {
                                                const date = new Date(str);
                                                return date.toLocaleTimeString().split(' ')[0];
                                            }}
                                        />
                                        <YAxis />
                                        <Tooltip labelFormatter={(value) => {
                                            const date = new Date(value);
                                            return date.toLocaleTimeString().split(' ')[1];
                                        }} />
                                        <Line 
                                            type="monotone" 
                                            dataKey="value" 
                                            stroke="var(--accent)" 
                                            strokeWidth={2}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        {/* Pressure Graph */}
                        <div className="card">
                            <h3 className="text-lg font-medium mb-4">Pressure Trends</h3>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={generateHistoricalData(pumpData.sensors.sensor_pressure)}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                        <XAxis 
                                            dataKey="timestamp" 
                                            tickFormatter={(str) => {
                                                const date = new Date(str);
                                                return date.toLocaleTimeString().split(' ')[0];
                                            }}
                                        />
                                        <YAxis />
                                        <Tooltip labelFormatter={(value) => {
                                            const date = new Date(value);
                                            return date.toLocaleTimeString().split(' ')[1];
                                        }} />
                                        <Line 
                                            type="monotone" 
                                            dataKey="value" 
                                            stroke="var(--accent)" 
                                            strokeWidth={2}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        {/* Vibration Graph */}
                        <div className="card">
                            <h3 className="text-lg font-medium mb-4">Vibration Trends</h3>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={generateHistoricalData(pumpData.sensors.sensor_vibration)}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                        <XAxis 
                                            dataKey="timestamp" 
                                            tickFormatter={(str) => {
                                                const date = new Date(str);
                                                return date.toLocaleTimeString().split(' ')[0];
                                            }}
                                        />
                                        <YAxis />
                                        <Tooltip labelFormatter={(value) => {
                                            const date = new Date(value);
                                            return date.toLocaleTimeString().split(' ')[1];
                                        }} />
                                        <Line 
                                            type="monotone" 
                                            dataKey="value" 
                                            stroke="var(--accent)" 
                                            strokeWidth={2}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        {/* Status Graph */}
                        <div className="card">
                            <h3 className="text-lg font-medium mb-4">Status Trends</h3>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={[
                                        // Generate static status data points
                                        ...Array.from({ length: 10 }, (_, i) => {
                                            const time = new Date();
                                            time.setMinutes(time.getMinutes() - (10 - i - 1) * 5);
                                            return {
                                                timestamp: time.toLocaleTimeString(),
                                                status: pumpData.machine_status,
                                                value: pumpData.machine_status === "Stable" ? 100 :
                                                    pumpData.machine_status === "Warning" ? 50 : 0
                                            };
                                        })
                                    ]}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                        <XAxis 
                                            dataKey="timestamp" 
                                            tickFormatter={(str) => {
                                                const date = new Date(str);
                                                return date.toLocaleTimeString().split(' ')[0];
                                            }}
                                        />
                                        <YAxis />
                                        <Tooltip 
                                            labelFormatter={(value) => {
                                                const date = new Date(value);
                                                return date.toLocaleTimeString().split(' ')[1];
                                            }}
                                            formatter={(value, name, props) => [props.payload.status, 'Status']}
                                        />
                                        <Line 
                                            type="monotone" 
                                            dataKey="value" 
                                            stroke={getStatusColor(pumpData.machine_status)}
                                            strokeWidth={2}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="text-center py-8 text-[var(--text-secondary)]">
                    Select a pump sensor to view data
                </div>
            )}
        </div>
    );
}