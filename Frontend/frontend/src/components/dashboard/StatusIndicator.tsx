import { PumpStatus } from "@/types/system";

interface StatusIndicatorProps {
  status: PumpStatus;
}

export default function StatusIndicator({ status }: StatusIndicatorProps) {
  const getStatusColor = (status: PumpStatus) => {
    switch (status) {
      case 'NORMAL':
        return 'bg-green-500';
      case 'RECOVERING':
        return 'bg-yellow-500';
      case 'BROKEN':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${getStatusColor(status)}`} />
      <span className="font-medium">{status}</span>
    </div>
  );
} 