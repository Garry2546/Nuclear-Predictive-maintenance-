'use client';

interface ChecklistProgressProps {
  total: number;
  completed: number;
}

export default function ChecklistProgress({ total, completed }: ChecklistProgressProps) {
  const percentage = Math.round((completed / total) * 100);
  
  return (
    <div className="mb-6">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-[var(--text-secondary)]">
          Completion Progress
        </span>
        <span className="text-sm font-medium text-[var(--text)]">
          {percentage}%
        </span>
      </div>
      <div className="w-full bg-[var(--secondary)] rounded-full h-2.5">
        <div
          className="bg-[var(--accent)] h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}