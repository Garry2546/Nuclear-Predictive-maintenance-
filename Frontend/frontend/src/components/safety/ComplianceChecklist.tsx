'use client';

import { useState, useEffect } from 'react';
import { useApi } from '@/hooks/useApi';
import { apiService, ChecklistSection } from '@/services/apiService';
import ChecklistProgress from './ChecklistProgress';
import { FaChevronDown, FaChevronRight } from 'react-icons/fa';
import { AxiosError } from 'axios';

export default function ComplianceChecklist() {
  const [localSections, setLocalSections] = useState<ChecklistSection[]>([]);

  const handleError = (error: Error) => {
    let message = 'An error occurred';
    
    if (error instanceof AxiosError) {
      if (!error.response) {
        message = 'Unable to connect to the server. Please check if the API is running.';
      } else {
        message = error.response.data?.message || error.message;
      }
    }
    console.error(message);
  };

  const { 
    data: sections, 
    error, 
    isLoading,
    refetch 
  } = useApi(apiService.getChecklist, {
    onError: handleError
  });

  useEffect(() => {
    if (sections) {
      setLocalSections(sections);
    }
  }, [sections]);

  const toggleSection = (sectionId: string) => {
    setLocalSections(prevSections => 
      prevSections.map(section => 
        section.id === sectionId 
          ? { ...section, isExpanded: !section.isExpanded }
          : section
      )
    );
  };

  const handleToggleItem = async (sectionId: string, itemId: string, checked: boolean) => {
    setLocalSections(prevSections => 
      prevSections.map(section => 
        section.id === sectionId 
          ? {
              ...section,
              items: section.items.map(item =>
                item.id === itemId ? { ...item, checked } : item
              )
            }
          : section
      )
    );

    try {
      await apiService.updateChecklistItem(sectionId, itemId, checked);
    } catch (error) {
      console.error('Failed to update checklist item:', error);
      refetch();
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-[400px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--accent)]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center text-red-500 p-4">
        Error loading checklist: {error.message}
      </div>
    );
  }

  const totalItems = localSections.reduce((acc, section) => acc + section.items.length, 0);
  const completedItems = localSections.reduce(
    (acc, section) => acc + section.items.filter(item => item.checked).length, 
    0
  );

  return (
    <div className="max-w-4xl mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold mb-6">
        Compliance Checklist for Implementing AI Solutions in Nuclear Plant Operations
      </h1>
      
      <ChecklistProgress total={totalItems} completed={completedItems} />

      <div className="space-y-6">
        {localSections.map(section => (
          <div key={section.id} className="card">
            <button
              onClick={() => toggleSection(section.id)}
              className="w-full flex items-center justify-between p-4 hover:bg-opacity-50"
            >
              <h2 className="text-xl font-semibold">{section.title}</h2>
              {section.isExpanded ? <FaChevronDown /> : <FaChevronRight />}
            </button>

            {section.isExpanded && (
              <div className="p-4 pt-0 space-y-3">
                {section.items.map(item => (
                  <div key={item.id} className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={item.checked}
                      onChange={(e) => handleToggleItem(section.id, item.id, e.target.checked)}
                      className="mt-1 h-4 w-4 rounded border-gray-300 
                               text-[var(--accent)] focus:ring-[var(--accent)]"
                    />
                    <label className="text-[var(--text-secondary)]">
                      {item.text}
                    </label>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}