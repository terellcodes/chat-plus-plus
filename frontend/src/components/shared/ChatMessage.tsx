'use client';

import ReactMarkdown from 'react-markdown';
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
            <div className="text-sm prose prose-invert max-w-none break-words">
              <ReactMarkdown
                components={{
                  // Style headings
                  h1: ({children}) => <h1 className="text-xl font-bold mb-2">{children}</h1>,
                  h2: ({children}) => <h2 className="text-lg font-bold mb-2">{children}</h2>,
                  h3: ({children}) => <h3 className="text-base font-bold mb-1">{children}</h3>,
                  // Style code
                  code: ({children}) => (
                    <code className="bg-[#1e1e1e] px-1 py-0.5 rounded text-[#4ec9b0] font-mono text-sm">
                      {children}
                    </code>
                  ),
                  // Style code blocks
                  pre: ({children}) => (
                    <pre className="bg-[#1e1e1e] p-3 rounded-md overflow-x-auto my-2 border border-[#3e3e42]">
                      {children}
                    </pre>
                  ),
                  // Style lists
                  ul: ({children}) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
                  ol: ({children}) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
                  li: ({children}) => <li className="mb-1">{children}</li>,
                  // Style paragraphs
                  p: ({children}) => <p className="mb-2 last:mb-0">{children}</p>,
                  // Style blockquotes
                  blockquote: ({children}) => (
                    <blockquote className="border-l-4 border-[#007acc] pl-4 my-2 italic">
                      {children}
                    </blockquote>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
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