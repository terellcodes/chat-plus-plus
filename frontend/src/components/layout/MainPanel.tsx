'use client';

import { MainPanelProps } from '@/types';
import ChatMessage from '@/components/shared/ChatMessage';
import ChatInput from '@/components/shared/ChatInput';
import ApiHealth from '@/components/ApiHealth';

/**
 * Main chat panel component (85% of screen width)
 * Shows initial state or conversation thread
 */
export default function MainPanel({ configuration, chatSession, onSendMessage, onClearChat }: MainPanelProps) {
  const isConfigured = configuration.isConfigured;
  const hasMessages = chatSession.messages.length > 0;

  return (
    <main className="flex-1 bg-[#1e1e1e] flex flex-col">
      {/* API Health Indicator - Always visible to start health check on page load */}
      <div className="bg-[#2d2d30] border-b border-[#3e3e42] px-4 py-2">
        <ApiHealth compact showRetryButton />
      </div>

      {!isConfigured ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="max-w-md text-center space-y-4">
            <div className="text-4xl mb-4">🤖</div>
            <h2 className="text-[#d4d4d4] text-xl font-semibold">
              Welcome to Chat++
            </h2>
            <p className="text-[#6a9955] text-sm leading-relaxed">
              To get started, please configure the following in the left panel:
            </p>
            <ul className="space-y-2 text-sm">
              {!configuration.openaiApiKey && (
                <li className="flex items-center space-x-2 text-[#f44747]">
                  <span>❌</span>
                  <span>OpenAI API Key</span>
                </li>
              )}
              {!configuration.selectedDocument && (
                <li className="flex items-center space-x-2 text-[#f44747]">
                  <span>❌</span>
                  <span>Upload PDF Document</span>
                </li>
              )}
              {(!configuration.enabledStrategies || configuration.enabledStrategies.length === 0) && (
                <li className="flex items-center space-x-2 text-[#f44747]">
                  <span>❌</span>
                  <span>Select RAG Strategies</span>
                </li>
              )}
            </ul>
          </div>
        </div>
      ) : (
        <>
          {/* Chat Header */}
          <div className="h-12 bg-[#2d2d30] border-b border-[#3e3e42] flex items-center justify-between px-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-[#4ec9b0] rounded-full"></div>
              <span className="text-[#d4d4d4] text-sm">
                {configuration.selectedDocument?.filename}
              </span>
            </div>
            {hasMessages && (
              <button
                onClick={onClearChat}
                className="text-[#d4d4d4] hover:text-[#f44747] text-sm transition-colors"
              >
                Clear Chat
              </button>
            )}
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!hasMessages ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-4">
                  <div className="text-4xl">💬</div>
                  <h3 className="text-[#d4d4d4] text-lg">
                    Ready to chat!
                  </h3>
                  <p className="text-[#6a9955] text-sm">
                    Ask questions about your PDF document
                  </p>
                </div>
              </div>
            ) : (
              chatSession.messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))
            )}
          </div>

          {/* Chat Input */}
          <div className="border-t border-[#3e3e42] p-4">
            <ChatInput 
              onSendMessage={onSendMessage}
              disabled={!isConfigured}
              placeholder="Ask a question about your PDF..."
            />
          </div>
        </>
      )}
    </main>
  );
}