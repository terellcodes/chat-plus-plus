'use client';

import { useEffect, useState } from 'react';
import { API_ENDPOINTS } from '@/config/api';

interface HealthStatus {
    status: string;
    code: number;
    message: string;
}

export default function ApiHealth() {
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const checkHealth = async () => {
            try {
                const response = await fetch(API_ENDPOINTS.HEALTH);
                const data = await response.json();
                setHealth(data);
                setError(null);
            } catch (err) {
                setError('Failed to connect to API');
                setHealth(null);
            } finally {
                setLoading(false);
            }
        };

        checkHealth();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="animate-pulse">Checking API status...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 rounded-lg">
                {error}
            </div>
        );
    }

    return (
        <div className={`p-4 ${health?.status === 'Success' ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400' : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400'} rounded-lg`}>
            <p className="font-medium">API Status: {health?.status}</p>
            <p className="text-sm mt-1">{health?.message}</p>
        </div>
    );
} 