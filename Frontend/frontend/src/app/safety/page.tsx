'use client';

import ComplianceChecklist from '@/components/safety/ComplianceChecklist';
import RoleProtectedRoute from '@/components/auth/RoleProtectedRoute';

export default function SafetyPage() {
  return (
    <RoleProtectedRoute allowedRoles={['compliance']}>
      <ComplianceChecklist />
    </RoleProtectedRoute>
  );
}