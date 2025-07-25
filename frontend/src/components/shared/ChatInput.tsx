'use client';

import { useState } from 'react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

/**
 * Chat input component with send and clear buttons
 */
export default function ChatInput({ onSendMessage, disabled = false, placeholder = "Type your message..." }: ChatInputProps) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex space-x-2">
      <div className="flex-1 relative">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="w-full px-3 py-2 bg-[#252526] border border-[#3e3e42] rounded text-[#d4d4d4] text-sm focus:border-[#007acc] focus:outline-none resize-none"
          style={{ minHeight: '38px', maxHeight: '120px' }}
        />
      </div>
      <button
        type="button"
        onClick={() => setMessage('')}
        disabled={!message.trim() || disabled}
        className="px-3 py-2 bg-[#252526] border border-[#3e3e42] rounded text-[#d4d4d4] hover:bg-[#2d2d30] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Clear
      </button>
      <button
        type="submit"
        disabled={!message.trim() || disabled}
        className="px-4 py-2 bg-[#007acc] text-white rounded hover:bg-[#005a9e] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Send
      </button>
    </form>
  );
} 