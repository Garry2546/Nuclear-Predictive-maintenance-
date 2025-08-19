import { UserRole } from '@/types/auth';

export const ROLE_MAPPINGS: Record<string, UserRole> = {
  'operator@gmail.com': 'operator',
  'compliance@gmail.com': 'compliance'
};

export const getRole = (email: string): UserRole => {
  return ROLE_MAPPINGS[email] || 'unauthorized';
};

export const ROLE_BASED_ROUTES = {
  operator: ['/', '/pump', '/turbine'],
  compliance: ['/', '/safety-case', '/ai-chat'],
  unauthorized: ['/']
};

export const canAccessRoute = (role: UserRole, path: string): boolean => {
  return ROLE_BASED_ROUTES[role]?.includes(path) || false;
};