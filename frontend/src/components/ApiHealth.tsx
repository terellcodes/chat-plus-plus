'use client';

import { useApiHealth } from '@/hooks/useApiHealth';

interface ApiHealthProps {
  compact?: boolean;
  showRetryButton?: boolean;
}

export default function ApiHealth({ compact = false, showRetryButton = false }: ApiHealthProps) {
  const { health, isChecking, isOnline, isOffline, manualRetry } = useApiHealth();

  // Get appropriate styling based on status
  const getStatusColor = () => {
    if (isChecking) return '#dcdcaa'; // Yellow for checking
    if (isOnline) return '#4ec9b0'; // Green for online
    return '#f44747'; // Red for offline/error
  };

  const getStatusIcon = () => {
    if (isChecking) return '🟡';
    if (isOnline) return '🟢';
    return '🔴';
  };

  if (compact) {
    // Compact version for header display
    return (
      <div className="flex items-center space-x-2">
        <div 
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: getStatusColor() }}
        />
        <span className="text-[#d4d4d4] text-sm">
          {health.message}
        </span>
        {showRetryButton && isOffline && (
          <button
            onClick={manualRetry}
            className="text-[#007acc] hover:text-[#4ec9b0] text-xs transition-colors ml-2"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  // Full version for standalone display
  return (
    <div className="flex items-center justify-between p-3 bg-[#2d2d30] border border-[#3e3e42] rounded">
      <div className="flex items-center space-x-3">
        <span className="text-lg">{getStatusIcon()}</span>
        <div>
          <p className="text-[#d4d4d4] text-sm font-medium">
            {health.message}
          </p>
          {health.lastChecked && (
            <p className="text-[#6a9955] text-xs">
              Last checked: {health.lastChecked.toLocaleTimeString()}
            </p>
          )}
        </div>
      </div>
      
      {showRetryButton && isOffline && (
        <button
          onClick={manualRetry}
          className="px-3 py-1 bg-[#007acc] hover:bg-[#4ec9b0] text-white text-xs rounded transition-colors"
        >
          Retry Connection
        </button>
      )}
    </div>
  );
} 