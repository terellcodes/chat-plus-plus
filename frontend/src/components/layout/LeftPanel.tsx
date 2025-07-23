'use client';

import { useState } from 'react';
import { LeftPanelProps, RagStrategy, RagStrategyConfig } from '@/types';
import RagStrategyCard from '@/components/shared/RagStrategyCard';

// RAG Strategy configurations with power levels
const RAG_STRATEGIES: RagStrategyConfig[] = [
  {
    id: RagStrategy.NAIVE_RETRIEVAL,
    name: 'Naive Retrieval',
    description: 'Basic semantic search',
    icon: '⚡',
    enabled: false,
    powerLevel: 1
  },
  {
    id: RagStrategy.BM25_RETRIEVAL,
    name: 'BM25 Retrieval',
    description: 'Keyword-based ranking',
    icon: '🔍',
    enabled: false,
    powerLevel: 2
  },
  {
    id: RagStrategy.CONTEXTUAL_COMPRESSION,
    name: 'Contextual Compression',
    description: 'Context-aware filtering',
    icon: '🎯',
    enabled: false,
    powerLevel: 3
  },
  {
    id: RagStrategy.MULTI_QUERY_RETRIEVAL,
    name: 'Multi Query',
    description: 'Multiple query expansion',
    icon: '🚀',
    enabled: false,
    powerLevel: 4
  },
  {
    id: RagStrategy.PARENT_DOCUMENT_RETRIEVAL,
    name: 'Parent Document',
    description: 'Hierarchical retrieval',
    icon: '🌟',
    enabled: false,
    powerLevel: 5
  }
];

/**
 * Left configuration panel component (15% of screen width)
 * Handles API key, PDF upload, and RAG strategy selection
 */
export default function LeftPanel({ configuration, onConfigurationChange }: LeftPanelProps) {
  const [apiKeyVisible, setApiKeyVisible] = useState(false);
  const [strategies, setStrategies] = useState(RAG_STRATEGIES);

  const handleApiKeyChange = (value: string) => {
    onConfigurationChange({ openaiApiKey: value });
  };

  const handleStrategyToggle = (strategyId: RagStrategy) => {
    const updatedStrategies = strategies.map(s => 
      s.id === strategyId ? { ...s, enabled: !s.enabled } : s
    );
    setStrategies(updatedStrategies);
    
    const enabledStrategies = updatedStrategies
      .filter(s => s.enabled)
      .map(s => s.id);
    
    onConfigurationChange({ enabledStrategies });
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      // Create mock PDF document for now
      const mockPdf = {
        id: Date.now().toString(),
        filename: file.name,
        size: file.size,
        uploadedAt: new Date(),
        processingStatus: 'ready' as const
      };
      onConfigurationChange({ selectedDocument: mockPdf });
    }
  };

  const isEmpty = !configuration.openaiApiKey || !configuration.selectedDocument;

  return (
    <aside className="w-[15%] min-w-[280px] bg-[#252526] border-r border-[#3e3e42] flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-[#3e3e42]">
        <h2 className="text-[#d4d4d4] font-semibold text-sm uppercase tracking-wide">
          Configuration
        </h2>
      </div>

      {/* Content */}
      <div className="flex-1 p-4 space-y-6 overflow-y-auto">
        
        {/* OpenAI API Key Section */}
        <div className="space-y-2">
          <label className="block text-[#d4d4d4] text-sm font-medium">
            OpenAI API Key
          </label>
          {isEmpty ? (
            <div className="space-y-2">
              <input
                type={apiKeyVisible ? 'text' : 'password'}
                value={configuration.openaiApiKey || ''}
                onChange={(e) => handleApiKeyChange(e.target.value)}
                placeholder="sk-..."
                className="w-full px-3 py-2 bg-[#1e1e1e] border border-[#3e3e42] rounded text-[#d4d4d4] text-sm focus:border-[#007acc] focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setApiKeyVisible(!apiKeyVisible)}
                className="text-[#007acc] text-xs hover:underline"
              >
                {apiKeyVisible ? 'Hide' : 'Show'} Key
              </button>
            </div>
          ) : (
            <div className="flex items-center space-x-2">
              <div className="flex-1 px-3 py-2 bg-[#1e1e1e] border border-[#4ec9b0] rounded text-[#4ec9b0] text-sm">
                ••••••••••••••••
              </div>
              <div className="w-2 h-2 bg-[#4ec9b0] rounded-full"></div>
            </div>
          )}
        </div>

        {/* PDF Upload Section */}
        <div className="space-y-2">
          <label className="block text-[#d4d4d4] text-sm font-medium">
            Upload PDF
          </label>
          {!configuration.selectedDocument ? (
            <div className="relative">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="w-full px-3 py-8 border-2 border-dashed border-[#3e3e42] rounded text-center hover:border-[#007acc] transition-colors">
                <div className="text-2xl mb-2">📄</div>
                <p className="text-[#d4d4d4] text-sm">
                  Click to upload PDF
                </p>
                <p className="text-[#6a9955] text-xs mt-1">
                  Max size: 10MB
                </p>
              </div>
            </div>
          ) : (
            <div className="px-3 py-2 bg-[#1e1e1e] border border-[#4ec9b0] rounded">
              <div className="flex items-center space-x-2">
                <span className="text-[#4ec9b0]">📄</span>
                <div className="flex-1 min-w-0">
                  <p className="text-[#d4d4d4] text-sm truncate">
                    {configuration.selectedDocument.filename}
                  </p>
                  <p className="text-[#6a9955] text-xs">
                    {(configuration.selectedDocument.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
                <div className="w-2 h-2 bg-[#4ec9b0] rounded-full"></div>
              </div>
            </div>
          )}
        </div>

        {/* RAG Strategies Section */}
        <div className="space-y-3">
          <label className="block text-[#d4d4d4] text-sm font-medium">
            RAG Strategies
          </label>
          <p className="text-[#6a9955] text-xs">
            Select strategies to power up your retrieval
          </p>
          
          <div className="space-y-2">
            {strategies.map((strategy) => (
              <RagStrategyCard
                key={strategy.id}
                strategy={strategy}
                isSelected={strategy.enabled}
                onToggle={handleStrategyToggle}
              />
            ))}
          </div>

          {/* Power Level Indicator */}
          {configuration.enabledStrategies?.length > 0 && (
            <div className="mt-4 p-3 bg-[#1e1e1e] border border-[#007acc] rounded">
              <div className="flex items-center space-x-2">
                <span className="text-[#007acc]">⚡</span>
                <span className="text-[#d4d4d4] text-sm">
                  Power Level: {configuration.enabledStrategies.length}/5
                </span>
              </div>
              <div className="mt-2 w-full bg-[#3e3e42] rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-[#007acc] to-[#4ec9b0] h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(configuration.enabledStrategies.length / 5) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
} 