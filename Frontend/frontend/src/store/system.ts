export type SystemStatusType = 'Stable' | 'Warning' | 'Critical';

interface BaseSystem {
  temperature: number;
  pressure: number;
  status: SystemStatusType;
}

export interface PumpSystem extends BaseSystem {
  // Pump-specific properties can be added here
}

export interface TurbineSystem extends BaseSystem {
  rotationSpeed: number;
  powerOutput: number;
}

export interface CoolantSystem extends BaseSystem {
  flowRate: number;
  conductivity: number;
}

export interface ReactorSystem extends BaseSystem {
  powerOutput: number;
  controlRodLevel: number;
  neutronFlux: number;
}

export interface SystemData {
  pumpSystems: Record<string, PumpSystem>;
  turbineSystems: Record<string, TurbineSystem>;
  coolantSystems: Record<string, CoolantSystem>;
  reactorSystems: Record<string, ReactorSystem>;
}