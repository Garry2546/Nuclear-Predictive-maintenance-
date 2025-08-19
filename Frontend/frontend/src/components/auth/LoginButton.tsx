'use client';

import { useAuth } from '@/hooks/useAuth';

export default function LoginButton() {
  const { isAuthenticated, isLoading, login, logout, userEmail } = useAuth();

  if (isLoading) {
    return (
      <button 
        className="px-4 py-2 rounded bg-[var(--accent)] bg-opacity-50 text-white cursor-not-allowed" 
        disabled
      >
        <span className="opacity-50">Please wait...</span>
      </button>
    );
  }

  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-4">
        <span className="text-sm text-[var(--text-secondary)]">
          {userEmail}
        </span>
        <button
          onClick={logout}
          className="px-4 py-2 rounded bg-red-500 hover:bg-red-600 text-white transition-colors"
        >
          Logout
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={(e) => login()}
      className="px-4 py-2 rounded bg-[var(--accent)] hover:bg-opacity-90 text-white transition-colors"
    >
      Login
    </button>
  );
} 