export type UserRole = 'operator' | 'compliance' | 'unauthorized';

export interface UserProfile {
  email: string;
  role: UserRole;
}