'use client';

import { useRole } from '@/contexts/RoleContext';
import { UserRole } from '@/types/auth';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

interface RoleProtectedRouteProps {
  children: React.ReactNode;
  allowedRoles: UserRole[];
}

export default function RoleProtectedRoute({ 
  children, 
  allowedRoles 
}: RoleProtectedRouteProps) {
  const { role } = useRole();
  const router = useRouter();

  useEffect(() => {
    if (!allowedRoles.includes(role)) {
      router.push('/unauthorized');
    }
  }, [role, router, allowedRoles]);

  if (!allowedRoles.includes(role)) {
    return null;
  }

  return <>{children}</>;
}