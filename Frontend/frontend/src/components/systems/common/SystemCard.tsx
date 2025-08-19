interface SystemCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export default function SystemCard({ title, children, className = '' }: SystemCardProps) {
  return (
    <div className={`p-6 rounded-lg bg-[var(--secondary)] bg-opacity-20 backdrop-blur-lg ${className}`}>
      <h2 className="text-xl font-semibold mb-4">{title}</h2>
      {children}
    </div>
  );
}