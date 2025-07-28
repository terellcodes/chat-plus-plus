'use client';

import { useState } from 'react';
import { LeftPanelProps, RagStrategy, RagStrategyConfig } from '@/types';
import RagStrategyCard from '@/components/shared/RagStrategyCard';
import { documentService } from '@/services/document';

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
    id: RagStrategy.CONTEXTUAL_COMPRESSION_RETRIEVAL,
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
    id: RagStrategy.RAG_FUSION_RETRIEVAL,
    name: 'RAG Fusion',
    description: 'Multi-query with RRF ranking',
    icon: '🚀',
    enabled: false,
    powerLevel: 5
  },
  {
    id: RagStrategy.PARENT_DOCUMENT_RETRIEVAL,
    name: 'Parent Document',
    description: 'Hierarchical retrieval',
    icon: '🌟',
    enabled: false,
    powerLevel: 6
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

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || file.type !== 'application/pdf') {
      onConfigurationChange({ 
        uploadError: 'Please select a valid PDF file' 
      });
      return;
    }

    // if (file.size > 10 * 1024 * 1024) {
    //   onConfigurationChange({ 
    //     uploadError: 'File size must be less than 10MB' 
    //   });
    //   return;
    // }

    try {
      // Clear any previous errors and set uploading state
      onConfigurationChange({ 
        uploadError: undefined,
        isUploading: true,
        selectedDocument: {
          id: Date.now().toString(),
          filename: file.name,
          size: file.size,
          uploadedAt: new Date(),
          processingStatus: 'uploading'
        }
      });

      // Upload the file (no API key needed for upload)
      const response = await documentService.uploadPDF(file);

      if (response.status === 'Success' && response.data) {
        onConfigurationChange({
          isUploading: false,
          currentSessionId: response.data.session_id,
          selectedDocument: {
            id: response.data.document_id,
            filename: response.data.filename,
            size: file.size,
            uploadedAt: new Date(response.data.metadata?.upload_timestamp || new Date().toISOString()),
            processingStatus: 'ready',
            sessionId: response.data.session_id
          }
        });
      } else {
        throw new Error(response.message || 'Failed to upload PDF');
      }
    } catch (error) {
      onConfigurationChange({
        isUploading: false,
        uploadError: error instanceof Error ? error.message : 'Failed to upload PDF',
        selectedDocument: undefined
      });
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
                disabled={configuration.isUploading}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
              />
              <div className={`w-full px-3 py-8 border-2 border-dashed rounded text-center transition-colors ${
                configuration.uploadError 
                  ? 'border-[#f44747] bg-[#f447471a]'
                  : configuration.isUploading
                  ? 'border-[#007acc] bg-[#007acc1a]'
                  : 'border-[#3e3e42] hover:border-[#007acc]'
              }`}>
                <div className="text-2xl mb-2">
                  {configuration.isUploading ? '📤' : configuration.uploadError ? '❌' : '📄'}
                </div>
                <p className={`text-sm ${
                  configuration.uploadError 
                    ? 'text-[#f44747]'
                    : configuration.isUploading
                    ? 'text-[#007acc]'
                    : 'text-[#d4d4d4]'
                }`}>
                  {configuration.isUploading 
                    ? 'Uploading...' 
                    : configuration.uploadError
                    ? configuration.uploadError
                    : 'Click to upload PDF'
                  }
                </p>
                <p className="text-[#6a9955] text-base mt-1">
                  Max size: 10MB
                </p>
              </div>
            </div>
          ) : (
            <div className={`px-3 py-2 bg-[#1e1e1e] border rounded ${
              configuration.selectedDocument.processingStatus === 'ready'
                ? 'border-[#4ec9b0]'
                : 'border-[#dcdcaa]'
            }`}>
              <div className="flex items-center space-x-2">
                <span className={configuration.selectedDocument.processingStatus === 'ready' ? 'text-[#4ec9b0]' : 'text-[#dcdcaa]'}>
                  {configuration.selectedDocument.processingStatus === 'ready' ? '📄' : '⏳'}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-[#d4d4d4] text-sm truncate">
                    {configuration.selectedDocument.filename}
                  </p>
                  <p className="text-[#6a9955] text-xs">
                    {(configuration.selectedDocument.size / 1024 / 1024).toFixed(1)} MB
                    {configuration.selectedDocument.processingStatus !== 'ready' && ' • Processing...'}
                  </p>
                </div>
                <div className={`w-2 h-2 rounded-full ${
                  configuration.selectedDocument.processingStatus === 'ready'
                    ? 'bg-[#4ec9b0]'
                    : 'bg-[#dcdcaa] animate-pulse'
                }`}></div>
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