export const logger = {
    info: (message: string, data?: unknown) => {
      if (process.env.NODE_ENV === 'development') {
        console.log(`ℹ️ ${message}`, data ? data : '');
      }
    },
    success: (message: string, data?: unknown) => {
      if (process.env.NODE_ENV === 'development') {
        console.log(`✅ ${message}`, data ? data : '');
      }
    },
    warn: (message: string, data?: unknown) => {
      if (process.env.NODE_ENV === 'development') {
        console.warn(`⚠️ ${message}`, data ? data : '');
      }
    },
  };

// Get API base URL based on environment
const getApiBaseUrl = () => {
    if (process.env.NODE_ENV === 'development') {
      return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    }
    // In production, use the deployed API URL
    return 'https://api-ancient-glade-3016.fly.dev';
  };

export const API_BASE_URL = getApiBaseUrl();

export const API_ENDPOINTS = {
    HEALTH: `${API_BASE_URL}/health`,
    UPLOAD: `${API_BASE_URL}/upload`,
    CHAT: `${API_BASE_URL}/chat`,
    STRATEGIES: `${API_BASE_URL}/strategies`,
    SESSION: (sessionId: string) => `${API_BASE_URL}/session/${sessionId}`,
} as const; 
