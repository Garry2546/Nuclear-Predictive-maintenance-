import { useSystemStore } from '@/store/systemStore';
import SystemCard from './common/SystemCard';
import StatusIndicator from './common/StatusIndicator';

export default function CoolantSystemCard() {
  const coolantSystems = useSystemStore((state) => state.coolantSystems);
  
  return (
    <SystemCard title="Coolant Systems">
      <div className="space-y-4">
        {Object.entries(coolantSystems).map(([coolantId, data]) => (
          <div key={coolantId} className="p-4 rounded bg-gray-800 border-l-4 border-cyan-500">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-medium">{coolantId}</h3>
              <StatusIndicator status={data.status} />
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
              <div>Temperature: {data.temperature.toFixed(2)}°C</div>
              <div>Pressure: {data.pressure.toFixed(2)} PSI</div>
              <div>Flow Rate: {data.flowRate.toFixed(2)} L/min</div>
              <div>Conductivity: {data.conductivity.toFixed(3)} S/m</div>
            </div>
          </div>
        ))}
      </div>
    </SystemCard>
  );
}
