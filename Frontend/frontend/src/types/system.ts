export type PumpStatus = 'RECOVERING' | 'NORMAL' | 'BROKEN';

export interface PumpData {
  time: string;
  classification: PumpStatus;
}

export interface EmergencyAlert {
  time: string;
  message: string;
  severity: 'high' | 'medium' | 'low';
} 