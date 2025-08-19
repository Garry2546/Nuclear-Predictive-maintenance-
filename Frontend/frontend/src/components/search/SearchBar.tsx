'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { FaSearch } from 'react-icons/fa';

interface SearchBarProps {
  initialValue?: string;
  onSearch?: (query: string) => void;
}

export default function SearchBar({ initialValue = '', onSearch }: SearchBarProps) {
  const [query, setQuery] = useState(initialValue);
  const router = useRouter();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      if (onSearch) {
        onSearch(query);
      } else {
        router.push(`/chat?q=${encodeURIComponent(query)}`);
      }
    }
  };

  return (
    <form onSubmit={handleSearch} className="w-full max-w-2xl">
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for AI regulations and compliance guidelines..."
          className="w-full px-4 py-3 pl-12 bg-[var(--secondary)] rounded-lg 
                   text-[var(--text)] placeholder-[var(--text-secondary)]
                   focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
        />
        <FaSearch 
          className="absolute left-4 top-1/2 transform -translate-y-1/2 
                    text-[var(--text-secondary)] text-lg"
        />
      </div>
    </form>
  );
}
