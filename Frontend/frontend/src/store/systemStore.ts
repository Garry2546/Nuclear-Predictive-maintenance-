import { create } from 'zustand';
import { SystemData } from './system';
import { generateMockData } from '@/utils/mockSystemData';

interface SystemStore extends SystemData {
  updateSystemData: (data: Partial<SystemData>) => void;
  updateSystem: <T extends keyof SystemData>(
    systemType: T,
    systemId: string,
    data: Partial<SystemData[T][string]>
  ) => void;
}

export const useSystemStore = create<SystemStore>((set) => ({
  // Initialize with mock data
  ...generateMockData(),
  
  // Update methods
  updateSystemData: (data) => 
    set((state) => ({ ...state, ...data })),
    
  updateSystem: (systemType, systemId, data) =>
    set((state) => ({
      ...state,
      [systemType]: {
        ...state[systemType],
        [systemId]: {
          ...state[systemType][systemId],
          ...data
        }
      }
    }))
}));