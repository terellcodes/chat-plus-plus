'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { API_ENDPOINTS } from '@/config/api';
import { HealthCheckResult } from '@/types';

const RETRY_INTERVAL = 10000; // 10 seconds
const MAX_DURATION = 120000; // 2 minutes
const MAX_RETRIES = MAX_DURATION / RETRY_INTERVAL; // 12 attempts

export function useApiHealth() {
  const [health, setHealth] = useState<HealthCheckResult>({
    status: 'checking',
    message: 'Waking up API...'
  });
  
  const retryCount = useRef(0);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const startTime = useRef<number | undefined>(undefined);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(API_ENDPOINTS.HEALTH, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        // Add timeout to prevent hanging requests
        signal: AbortSignal.timeout(8000) // 8 second timeout per request
      });

      if (response.ok) {
        setHealth({
          status: 'online',
          message: 'API Online',
          lastChecked: new Date()
        });
        return true; // Success - stop polling
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      const isTimeout = error instanceof Error && error.name === 'TimeoutError';
      const errorMessage = isTimeout ? 'API timeout' : 'Connection failed';
      
      // Don't update to error state during normal retry process
      if (retryCount.current < MAX_RETRIES) {
        setHealth({
          status: 'checking',
          message: `Waking up API... (${retryCount.current + 1}/${MAX_RETRIES})`,
          lastChecked: new Date()
        });
      } else {
        setHealth({
          status: 'offline',
          message: `API Offline (${errorMessage})`,
          lastChecked: new Date()
        });
      }
      
      return false; // Failed - continue polling if within limits
    }
  }, []);

  const scheduleNextCheck = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    retryCount.current += 1;
    
    // Check if we've exceeded our retry limits
    if (retryCount.current >= MAX_RETRIES) {
      setHealth({
        status: 'offline',
        message: 'API Offline (timeout after 2 minutes)',
        lastChecked: new Date()
      });
      return;
    }

    // Schedule next health check
    timeoutRef.current = setTimeout(async () => {
      const success = await checkHealth();
      if (!success && retryCount.current < MAX_RETRIES) {
        scheduleNextCheck();
      }
    }, RETRY_INTERVAL);
  }, [checkHealth]);

  const startHealthCheck = useCallback(async () => {
    startTime.current = Date.now();
    retryCount.current = 0;
    
    // Perform initial check immediately
    const success = await checkHealth();
    
    // If first check fails, start polling
    if (!success) {
      scheduleNextCheck();
    }
  }, [checkHealth, scheduleNextCheck]);

  const manualRetry = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    
    setHealth({
      status: 'checking',
      message: 'Retrying...'
    });
    
    startHealthCheck();
  }, [startHealthCheck]);

  // Start health check on mount
  useEffect(() => {
    startHealthCheck();

    // Cleanup on unmount
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [startHealthCheck]);

  return {
    health,
    isChecking: health.status === 'checking',
    isOnline: health.status === 'online',
    isOffline: health.status === 'offline' || health.status === 'error',
    manualRetry
  };
}