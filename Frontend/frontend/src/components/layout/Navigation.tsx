'use client';

import Link from 'next/link';
import { useRole } from '@/contexts/RoleContext';
import LoginButton from '@/components/auth/LoginButton';

export default function Navigation() {
  const { isOperator, isCompliance } = useRole();

  return (
    <header className="bg-[var(--secondary)] shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-8">
            <h1 className="text-xl font-bold">Nuclear Safety System</h1>
            <nav className="flex space-x-4">
              <Link href="/" className="text-[var(--text)] hover:text-[var(--accent)] transition-colors">
                Home
              </Link>
              
              {/* Operator-only links */}
              {isOperator && (
                <>
                  <Link href="/pump" className="text-[var(--text)] hover:text-[var(--accent)] transition-colors">
                    Pump Systems
                  </Link>
                  <Link href="/turbine" className="text-[var(--text)] hover:text-[var(--accent)] transition-colors">
                    Turbine Systems
                  </Link>
                </>
              )}

              {/* Compliance-only links */}
              {isCompliance && (
                <>
                  <Link href="/safety" className="text-[var(--text)] hover:text-[var(--accent)] transition-colors">
                    Safety Case Management
                  </Link>
                  <Link href="/chat" className="text-[var(--text)] hover:text-[var(--accent)] transition-colors">
                    AI Chat
                  </Link>
                </>
              )}
            </nav>
          </div>
          <div>
            <LoginButton />
          </div>
        </div>
      </div>
    </header>
  );
}

export const getBaseUrl = () => {
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  // Provide a default for server-side rendering
  return process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';
};