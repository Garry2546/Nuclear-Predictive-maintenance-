import { useAuth0 } from '@auth0/auth0-react';
import { useRole } from '@/contexts/RoleContext';

export function useAuth() {
  const { 
    isAuthenticated, 
    isLoading, 
    user, 
    loginWithRedirect, 
    logout 
  } = useAuth0();
  
  const { role, isOperator, isCompliance } = useRole();

  const logoutWithRedirect = () => {
    logout({ 
      logoutParams: {
        returnTo: window.location.origin 
      }
    });
  };

  return {
    // Auth0 states
    isAuthenticated,
    isLoading,
    user,
    
    // Role states
    role,
    isOperator,
    isCompliance,
    
    // Auth methods
    login: loginWithRedirect,
    logout: logoutWithRedirect,
    
    // Helper computed properties
    isAuthorized: isAuthenticated && role !== 'unauthorized',
    userEmail: user?.email,
    userName: user?.name,
  };
}
