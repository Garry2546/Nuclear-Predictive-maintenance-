import { useSystemStore } from '@/store/systemStore';
import SystemCard from './common/SystemCard';
import StatusIndicator from './common/StatusIndicator';

export default function PumpSystemCard() {
  const { pumpSystems } = useSystemStore();
  
  return (
    <SystemCard title="Pump Systems">
      <div className="space-y-4">
        {Object.entries(pumpSystems).map(([pumpId, data]) => (
          <div key={pumpId} className="p-4 rounded bg-gray-800 border-l-4 border-[var(--accent)]">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-medium">{pumpId}</h3>
              <StatusIndicator status={data.status} />
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
              <div>Temperature: {data.temperature.toFixed(2)}°C</div>
              <div>Pressure: {data.pressure.toFixed(2)} PSI</div>
            </div>
          </div>
        ))}
      </div>
    </SystemCard>
  );
}