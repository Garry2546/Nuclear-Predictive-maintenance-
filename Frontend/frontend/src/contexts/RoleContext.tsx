'use client';

import { createContext, useContext, ReactNode } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { UserRole } from '@/types/auth';
import { getRole } from '@/utils/roles';

interface RoleContextType {
  role: UserRole;
  isOperator: boolean;
  isCompliance: boolean;
}

const RoleContext = createContext<RoleContextType>({
  role: 'unauthorized',
  isOperator: false,
  isCompliance: false
});

export function RoleProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth0();
  const role = user?.email ? getRole(user.email) : 'unauthorized';

  const value = {
    role,
    isOperator: role === 'operator',
    isCompliance: role === 'compliance'
  };

  return (
    <RoleContext.Provider value={value}>
      {children}
    </RoleContext.Provider>
  );
}

export const useRole = () => useContext(RoleContext);