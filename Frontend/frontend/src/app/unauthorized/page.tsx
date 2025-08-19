'use client';

import { useRouter } from 'next/navigation';

export default function UnauthorizedPage() {
  const router = useRouter();

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh]">
      <h1 className="text-2xl font-bold mb-4">Unauthorized Access</h1>
      <p className="text-[var(--text-secondary)] mb-6">
        You do not have permission to access this page.
      </p>
      <button
        onClick={() => router.push('/')}
        className="px-4 py-2 bg-[var(--accent)] text-white rounded-lg 
                 hover:bg-opacity-90 transition-colors"
      >
        Return to Home
      </button>
    </div>
  );
}
