'use client';

import { useAuth0 } from '@auth0/auth0-react';

export default function UserProfile() {
  const { user, isAuthenticated, isLoading } = useAuth0();

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (!isAuthenticated || !user) {
    return null;
  }

  return (
    <div className="flex items-center space-x-2">
      {user.picture && (
        <img
          src={user.picture}
          alt={user.name || 'User'}
          className="w-8 h-8 rounded-full"
        />
      )}
      <span>{user.name}</span>
    </div>
  );
} 