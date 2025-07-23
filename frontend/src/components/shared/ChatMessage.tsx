'use client';

import { ChatMessage as ChatMessageType } from '@/types';

interface ChatMessageProps {
  message: ChatMessageType;
}

/**
 * Individual chat message component
 */
export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[70%] rounded-lg p-3 ${
        isUser 
          ? 'bg-[#007acc] text-white' 
          : 'bg-[#252526] text-[#d4d4d4] border border-[#3e3e42]'
      }`}>
        <div className="flex items-start space-x-2">
          <div className="text-sm">
            {isUser ? '👤' : '🤖'}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm whitespace-pre-wrap break-words">
              {message.content}
            </p>
            <p className={`text-xs mt-1 ${
              isUser ? 'text-blue-100' : 'text-[#6a9955]'
            }`}>
              {message.timestamp.toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}