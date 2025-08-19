'use client';

import { useEffect } from 'react';
import { useApi } from '@/hooks/useApi';
import { apiService, Summary } from '@/services/apiService';
import SearchBar from '../search/SearchBar';
import { FaRobot, FaBell } from 'react-icons/fa';

export default function ComplianceHome() {
  const { 
    data: summary,
    error,
    isLoading
  } = useApi<Summary>(apiService.getSummary);

  const renderSummaryContent = () => {
    if (!summary) return null;

    return (
      <div className="space-y-6">
        {summary.title && (
          <h3 className="text-lg font-semibold text-[var(--text-secondary)]">
            {summary.title}
          </h3>
        )}
        
        {summary.sections?.map((section, sectionIndex) => (
          <div key={sectionIndex} className="space-y-2">
            <h4 className="font-semibold text-[var(--text-secondary)]">
              {section.title}
            </h4>
            <ul className="list-disc list-inside space-y-2 text-[var(--text-secondary)] ml-4">
              {section.items.map((item, itemIndex) => (
                <li key={itemIndex} className="whitespace-pre-wrap">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <h1 className="text-3xl font-bold mb-8">
        Compliance Dashboard
      </h1>
      <div className="mb-8">
        <SearchBar />
      </div>
      {/* <p className="text-[var(--text-secondary)] mb-8">
        Search for AI regulations and compliance guidelines
      </p> */}

      {/* AI Generated Summary Block */}
      <div className="card mb-8">
        <div className="flex items-center gap-3 mb-4">
          <FaRobot className="text-2xl text-[var(--accent)]" />
          <h2 className="text-xl font-semibold">AI-Regulation Research Paper Summary</h2>
        </div>
        <div className="space-y-4">
          {isLoading ? (
            <div className="flex justify-center py-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--accent)]" />
            </div>
          ) : error ? (
            <p className="text-red-500">Failed to load summary</p>
          ) : (
            renderSummaryContent()
          )}
        </div>
      </div>

      {/* What's New Block */}
      <div className="card">
        <div className="flex items-center gap-3 mb-4">
          <FaBell className="text-2xl text-[var(--accent)]" />
          <h2 className="text-xl font-semibold">What's New</h2>
        </div>
        <div className="space-y-4">
          <div className="border-l-4 border-[var(--accent)] pl-4">
            <p className="font-semibold">Updated Safety Guidelines</p>
            <p className="text-[var(--text-secondary)] text-sm">
              New safety protocols for nuclear facility AI systems
            </p>
            <p className="text-[var(--text-secondary)] text-xs mt-1">
              2 hours ago
            </p>
          </div>
          
          <div className="border-l-4 border-[var(--accent)] pl-4">
            <p className="font-semibold">Compliance Update</p>
            <p className="text-[var(--text-secondary)] text-sm">
              Revised requirements for AI monitoring systems
            </p>
            <p className="text-[var(--text-secondary)] text-xs mt-1">
              1 day ago
            </p>
          </div>
          
          <div className="border-l-4 border-[var(--accent)] pl-4">
            <p className="font-semibold">New Training Material</p>
            <p className="text-[var(--text-secondary)] text-sm">
              AI safety training modules now available
            </p>
            <p className="text-[var(--text-secondary)] text-xs mt-1">
              2 days ago
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 