import { SystemData, SystemStatusType } from '@/store/system';

const getRandomStatus = (): SystemStatusType => {
  const rand = Math.random();
  if (rand > 0.9) return 'Critical';
  if (rand > 0.7) return 'Warning';
  return 'Stable';
};

export const generateMockData = (): SystemData => ({
  pumpSystems: {
    'Pump1': {
      temperature: 275 + Math.random() * 10,
      pressure: 85 + Math.random() * 5,
      status: getRandomStatus()
    },
    'Pump2': {
      temperature: 280 + Math.random() * 10,
      pressure: 88 + Math.random() * 5,
      status: getRandomStatus()
    },
    'Pump3': {
      temperature: 278 + Math.random() * 10,
      pressure: 86 + Math.random() * 5,
      status: getRandomStatus()
    }
  },
  turbineSystems: {
    'Turbine1': {
      temperature: 150 + Math.random() * 15,
      pressure: 50 + Math.random() * 10,
      rotationSpeed: 1500 + Math.random() * 100,
      powerOutput: 800 + Math.random() * 50,
      status: getRandomStatus()
    },
    'Turbine2': {
      temperature: 155 + Math.random() * 15,
      pressure: 52 + Math.random() * 10,
      rotationSpeed: 1520 + Math.random() * 100,
      powerOutput: 820 + Math.random() * 50,
      status: getRandomStatus()
    },
    'Turbine3': {
      temperature: 152 + Math.random() * 15,
      pressure: 51 + Math.random() * 10,
      rotationSpeed: 1510 + Math.random() * 100,
      powerOutput: 810 + Math.random() * 50,
      status: getRandomStatus()
    }
  },
  coolantSystems: {
    'Primary Coolant': {
      temperature: 95 + Math.random() * 5,
      pressure: 150 + Math.random() * 10,
      flowRate: 1000 + Math.random() * 50,
      conductivity: 250 + Math.random() * 20,
      status: getRandomStatus()
    },
    'Secondary Coolant': {
      temperature: 85 + Math.random() * 5,
      pressure: 140 + Math.random() * 10,
      flowRate: 950 + Math.random() * 50,
      conductivity: 240 + Math.random() * 20,
      status: getRandomStatus()
    },
    'Emergency Coolant': {
      temperature: 75 + Math.random() * 5,
      pressure: 160 + Math.random() * 10,
      flowRate: 1100 + Math.random() * 50,
      conductivity: 260 + Math.random() * 20,
      status: getRandomStatus()
    }
  },
  reactorSystems: {
    'Reactor Core 1': {
      temperature: 350 + Math.random() * 20,
      pressure: 2000 + Math.random() * 100,
      powerOutput: 1000 + Math.random() * 50,
      controlRodLevel: 75 + Math.random() * 10,
      neutronFlux: 3.5 + Math.random() * 0.5,
      status: getRandomStatus()
    },
    'Reactor Core 2': {
      temperature: 355 + Math.random() * 20,
      pressure: 2050 + Math.random() * 100,
      powerOutput: 1020 + Math.random() * 50,
      controlRodLevel: 78 + Math.random() * 10,
      neutronFlux: 3.6 + Math.random() * 0.5,
      status: getRandomStatus()
    }
  }
});