'use client';

import ReactorCoreCard from '@/components/systems/ReactorCoreCard';
import PumpSystemCard from '@/components/systems/PumpSystemCard';
import TurbineSystemCard from '@/components/systems/TurbineSystemCard';
import CoolantSystemCard from '@/components/systems/CoolantSystemCard';

export default function OperatorDashboard() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reactor Systems */}
        <div>
          <ReactorCoreCard />
        </div>

        {/* Pump Systems */}
        <div>
          <PumpSystemCard />
        </div>

        {/* Turbine Systems */}
        <div>
          <TurbineSystemCard />
        </div>

        {/* Coolant Systems */}
        <div>
          <CoolantSystemCard />
        </div>
      </div>
    </div>
  );
}