"use client";

import { PumpName } from "@/store/sensorStore";

interface PumpStatusProps {
  selectedPump: PumpName | null;
  sensorData: any;
}

export default function PumpStatus({ selectedPump, sensorData }: PumpStatusProps) {
  const getStatusColor = (classification: string) => {
    switch (classification?.toLowerCase()) {
      case 'normal':
        return 'bg-green-600';
      case 'recovering':
        return 'bg-yellow-500';
      case 'broken':
        return 'bg-red-600';
      default:
        return 'bg-gray-700';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    if (!timestamp) return 'No data';
    return new Date(timestamp).toLocaleString();
  };

  const classification = sensorData?.classification || 'Unknown';
  const statusColor = getStatusColor(classification);
  const timestamp = sensorData?.time || null;

  return (
    <div className={`p-6 rounded-lg text-white bg-gray-800`}>
      <h2 className="text-xl font-semibold mb-4">Pump Status</h2>
      
      <div className="flex flex-col items-center justify-center min-h-[200px] space-y-8">
        {/* Time Display */}
        <div className="text-center">
          <div className="text-sm text-gray-300">Time:</div>
          <div className="font-mono text-xl">{timestamp || 'No data'}</div>
        </div>

        {/* Classification Status */}
        <div className={`px-6 py-3 rounded-lg ${statusColor}`}>
          <div className="font-bold text-2xl">{classification.toUpperCase()}</div>
        </div>
      </div>

      {/* Live Sensor Data */}
      {sensorData && (
        <div className="mt-8">
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(sensorData).map(([key, value]) => (
              !['time', 'classification'].includes(key) && (
                <div key={key} className="bg-black/20 p-3 rounded-lg">
                  <span className="text-gray-300">{key}:</span>
                  <div className="text-lg font-semibold">{String(value)}</div>
                </div>
              )
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
