'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth0 } from '@auth0/auth0-react';

export default function CallbackPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading, error } = useAuth0();

  useEffect(() => {
    if (!isLoading) {
      if (isAuthenticated) {
        router.push('/');
      } else if (error) {
        console.error('Authentication error:', error);
        router.push('/');
      }
    }
  }, [isLoading, isAuthenticated, error, router]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[var(--accent)] mx-auto"></div>
        <p className="mt-4 text-[var(--text-secondary)]">Completing authentication...</p>
      </div>
    </div>
  );
}