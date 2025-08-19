import { useSystemStore } from '@/store/systemStore';
import SystemCard from './common/SystemCard';
import StatusIndicator from './common/StatusIndicator';

export default function ReactorCoreCard() {
  const reactorSystems = useSystemStore((state) => state.reactorSystems);
  
  return (
    <SystemCard title="Reactor Systems">
      <div className="space-y-4">
        {Object.entries(reactorSystems).map(([reactorId, data]) => (
          <div key={reactorId} className="p-4 rounded bg-gray-800 border-l-4 border-blue-500">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-medium">{reactorId}</h3>
              <StatusIndicator status={data.status} />
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
              <div>Temperature: {data.temperature.toFixed(2)}°C</div>
              <div>Pressure: {data.pressure.toFixed(2)} PSI</div>
              <div>Power Output: {data.powerOutput.toFixed(2)} MW</div>
              <div>Control Rod Level: {data.controlRodLevel.toFixed(1)}%</div>
            </div>
          </div>
        ))}
      </div>
    </SystemCard>
  );
}