'use client';

import { useState } from 'react';
import MenuBar from './MenuBar';
import LeftPanel from './LeftPanel';
import MainPanel from './MainPanel';
import { AppConfiguration, ChatSession, ChatMessage } from '@/types';
import { chatService } from '@/services/chat';
import { logger } from '@/utils/logger';

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

  const handleSendMessage = async (content: string) => {
    try {
      // Create and add user message
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

      // Get response from API
      const response = await chatService.sendMessage(
        content,
        configuration.openaiApiKey,
        configuration.enabledStrategies || [],
        configuration.currentSessionId || '',
      );

      // Add assistant message
      setChatSession(prev => ({
        ...prev,
        messages: [...prev.messages, {
          ...response,
          id: Date.now().toString(),
          timestamp: new Date()
        }],
        updatedAt: new Date()
      }));

    } catch (error) {
      logger.error('Error sending message:', error);
      // Add error message to chat
      setChatSession(prev => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            id: Date.now().toString(),
            role: 'assistant',
            content: 'Sorry, there was an error processing your message. Please try again.',
            timestamp: new Date()
          }
        ],
        updatedAt: new Date()
      }));
    }
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