'use client';

import { useState } from 'react';
import MenuBar from './MenuBar';
import LeftPanel from './LeftPanel';
import MainPanel from './MainPanel';
import { AppConfiguration, ChatSession, ChatMessage } from '@/types';

/**
 * Main application layout component
 * Manages the overall state and layout structure
 */
export default function AppLayout() {
  const [configuration, setConfiguration] = useState<AppConfiguration>({
    openaiApiKey: '',
    selectedDocument: undefined,
    enabledStrategies: [],
    isConfigured: false
  });

  const [chatSession, setChatSession] = useState<ChatSession>({
    id: 'default',
    messages: [],
    createdAt: new Date(),
    updatedAt: new Date()
  });

  const [showSettings, setShowSettings] = useState(false);

  const handleConfigurationChange = (updates: Partial<AppConfiguration>) => {
    const newConfig = { ...configuration, ...updates };
    
    // Check if configuration is complete
    const isConfigured = !!(
      newConfig.openaiApiKey &&
      newConfig.selectedDocument &&
      newConfig.enabledStrategies?.length > 0
    );
    
    setConfiguration({ ...newConfig, isConfigured });
  };

  const handleSendMessage = (content: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date()
    };

    // Add user message
    setChatSession(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      updatedAt: new Date()
    }));

    // TODO: Phase 3 - Send to API and handle streaming response
    // For now, add a mock response
    setTimeout(() => {
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Hello! I'm your AI assistant. How can I help you today?`,
        timestamp: new Date()
      };

      setChatSession(prev => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        updatedAt: new Date()
      }));
    }, 1000);
  };

  const handleClearChat = () => {
    setChatSession({
      id: 'default',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    });
  };

  return (
    <div className="h-screen flex flex-col bg-[#1e1e1e]">
      <MenuBar onSettingsClick={() => setShowSettings(!showSettings)} />
      
      <div className="flex-1 flex overflow-hidden">
        <LeftPanel 
          configuration={configuration}
          onConfigurationChange={handleConfigurationChange}
        />
        
        <MainPanel
          configuration={configuration}
          chatSession={chatSession}
          onSendMessage={handleSendMessage}
          onClearChat={handleClearChat}
        />
      </div>
    </div>
  );
} 