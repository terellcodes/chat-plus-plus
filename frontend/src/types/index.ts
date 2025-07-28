// Core backend types
export interface ApiResponse<T = unknown> {
  data?: T;
  message: string;
  status: 'success' | 'error';
  code: number;
}

// RAG Strategy types
export enum RagStrategy {
  NAIVE_RETRIEVAL = 'naive_retrieval',
  BM25_RETRIEVAL = 'bm25_retrieval',
  CONTEXTUAL_COMPRESSION_RETRIEVAL = 'contextual_compression_retrieval',
  MULTI_QUERY_RETRIEVAL = 'multi_query_retrieval',
  PARENT_DOCUMENT_RETRIEVAL = 'parent_document_retrieval',
  RAG_FUSION_RETRIEVAL = 'rag_fusion_retrieval'
}

export interface RagStrategyConfig {
  id: RagStrategy;
  name: string;
  description: string;
  icon: string;
  enabled: boolean;
  powerLevel: number; // 1-5 for the "power-up" feel
}

// PDF and Document types
export interface PdfDocument {
  id: string;
  filename: string;
  size: number;
  uploadedAt: Date;
  pageCount?: number;
  processingStatus: 'uploading' | 'processing' | 'ready' | 'error';
  error?: string;
  sessionId: string; // Session ID for this document
}

// Chat types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

// Session types
export interface SessionInfo {
  session_id: string;
  created_at: number;
  last_accessed: number;
  loaded_strategies: string[];
  document_metadata?: {
    filename: string;
    upload_timestamp: string;
  };
}

// Configuration types
export interface AppConfiguration {
  openaiApiKey: string;
  selectedDocument?: PdfDocument;
  enabledStrategies: RagStrategy[];
  isConfigured: boolean;
  isUploading?: boolean;
  uploadError?: string;
  currentSessionId?: string; // Current session for the uploaded document
}

// Component Props types
export interface MenuBarProps {
  onSettingsClick: () => void;
}

export interface LeftPanelProps {
  configuration: AppConfiguration;
  onConfigurationChange: (config: Partial<AppConfiguration>) => void;
}

export interface MainPanelProps {
  configuration: AppConfiguration;
  chatSession: ChatSession;
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
}

export interface RagStrategyCardProps {
  strategy: RagStrategyConfig;
  isSelected: boolean;
  onToggle: (strategy: RagStrategy) => void;
}

// API Error types
export interface ApiError {
  message: string;
  code: number;
  details?: string;
}

// API Health types
export type HealthStatus = 'checking' | 'online' | 'offline' | 'error';

export interface HealthCheckResult {
  status: HealthStatus;
  message: string;
  lastChecked?: Date;
}

// Utility types
export type ConfigurationStep = 'api_key' | 'pdf_upload' | 'rag_strategies';

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
} 