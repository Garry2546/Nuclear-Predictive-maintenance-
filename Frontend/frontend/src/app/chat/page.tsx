'use client';

import { useSearchParams } from 'next/navigation';
import SearchBar from '@/components/search/SearchBar';
import RoleProtectedRoute from '@/components/auth/RoleProtectedRoute';

export default function ChatPage() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get('q') || '';

  return (
    <RoleProtectedRoute allowedRoles={['compliance']}>
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="mb-8">
          <SearchBar initialValue={initialQuery} />
        </div>
        
        <div className="text-center py-12">
          <h2 className="text-2xl font-semibold mb-4">
            AI Regulation Chat Bot
          </h2>
          <p className="text-[var(--text-secondary)]">
            Coming soon...
          </p>
        </div>
      </div>
    </RoleProtectedRoute>
  );
}