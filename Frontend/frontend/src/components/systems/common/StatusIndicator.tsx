interface StatusIndicatorProps {
    status: 'Stable' | 'Warning' | 'Critical';
  }
  
  export default function StatusIndicator({ status }: StatusIndicatorProps) {
    const colors = {
      Stable: 'bg-green-500',
      Warning: 'bg-yellow-500',
      Critical: 'bg-red-500'
    };
  
    return (
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${colors[status]}`} />
        <span>{status}</span>
      </div>
    );
  }