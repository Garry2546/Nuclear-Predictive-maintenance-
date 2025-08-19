'use client';

import ProtectedRoute from '@/components/auth/ProtectedRoute';

export default function TurbinePage() {
  return (
    <ProtectedRoute>
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Turbine Monitoring</h2>
        <p className="text-[var(--text-secondary)]">
          Turbine monitoring dashboard coming soon.
        </p>
      </div>
    </ProtectedRoute>
  );
}