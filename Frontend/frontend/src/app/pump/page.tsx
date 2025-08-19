"use client";

import { useState } from "react";
import PumpStatus from "@/components/dashboard/PumpStatus";
import EmergencyAlerts from "@/components/dashboard/EmergencyAlerts";
import { useWebSocket } from "@/components/layout/WebSocketProvider";
import { PumpName } from "@/store/sensorStore";

export default function PumpPage() {
  const [selectedPump, setSelectedPump] = useState<PumpName | null>(null);
  const { sensorData, accidentData } = useWebSocket();

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PumpStatus 
          selectedPump={selectedPump}
          sensorData={sensorData}
        />
        <EmergencyAlerts 
          selectedPump={selectedPump}
          accidentData={accidentData}
        />
      </div>
    </div>
  );
}
