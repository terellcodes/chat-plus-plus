'use client';

import { RagStrategyCardProps } from '@/types';

/**
 * Individual RAG strategy selection card with power-up animations
 */
export default function RagStrategyCard({ strategy, isSelected, onToggle }: RagStrategyCardProps) {
  const handleClick = () => {
    onToggle(strategy.id);
  };

  return (
    <button
      onClick={handleClick}
      className={`w-full p-3 rounded border transition-all duration-200 text-left ${
        isSelected 
          ? 'bg-[#1e1e1e] border-[#007acc] strategy-glow power-up' 
          : 'bg-[#2d2d30] border-[#3e3e42] hover:border-[#007acc] hover:bg-[#1e1e1e]'
      }`}
    >
      <div className="flex items-center space-x-3">
        <div className={`text-xl ${isSelected ? 'animate-pulse' : ''}`}>
          {strategy.icon}
        </div>
        <div className="flex-1 min-w-0">
          <h4 className={`text-lg font-medium truncate ${
            isSelected ? 'text-[#007acc]' : 'text-[#d4d4d4]'
          }`}>
            {strategy.name}
          </h4>
          <p className="text-[#6a9955] text-base truncate">
            {strategy.description}
          </p>
        </div>
        <div className="flex items-center space-x-1">
          {/* Power level indicators */}
          {Array.from({ length: strategy.powerLevel }).map((_, i) => (
            <div
              key={i}
              className={`w-1.5 h-1.5 rounded-full ${
                isSelected ? 'bg-[#007acc]' : 'bg-[#3e3e42]'
              }`}
            />
          ))}
        </div>
      </div>
    </button>
  );
} 