"use client";

import { PumpName } from "@/store/sensorStore";
import { ExclamationTriangleIcon } from '@heroicons/react/24/solid';

interface EmergencyAlertsProps {
  selectedPump: PumpName | null;
  accidentData: any;
}

export default function EmergencyAlerts({ selectedPump, accidentData }: EmergencyAlertsProps) {
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

  return (
    <div className="p-6 bg-gray-800 rounded-lg text-white">
      <div className="flex items-center gap-2 mb-4">
        <ExclamationTriangleIcon className="h-6 w-6 text-yellow-500" />
        <h2 className="text-xl font-semibold">Emergency Alerts</h2>
      </div>

      <div className="flex items-center justify-center min-h-[200px]">
        {accidentData ? (
          <div className={`w-full flex items-center justify-center p-8 rounded-lg ${getStatusColor(accidentData.classification)}`}>
            <div className="text-4xl font-bold">
              {accidentData.classification?.toUpperCase()}
            </div>
          </div>
        ) : (
          <div className="w-full flex items-center justify-center p-8 bg-gray-700 rounded-lg">
            <div className="text-4xl font-bold text-gray-400">
              NO DATA
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
