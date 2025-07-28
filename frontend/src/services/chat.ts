import { API_BASE_URL, API_ENDPOINTS } from '@/config/api';
import { ChatMessage } from '@/types';
import { logger } from '@/utils/logger';

export class ChatService {
  private static instance: ChatService;
  private readonly baseUrl: string;

  private constructor() {
    this.baseUrl = API_BASE_URL;
  }

  public static getInstance(): ChatService {
    if (!ChatService.instance) {
      ChatService.instance = new ChatService();
    }
    return ChatService.instance;
  }

  async sendMessage(
    message: string,
    openaiApiKey: string,
    enabledStrategies: string[],
    sessionId: string,
  ): Promise<ChatMessage> {
    try {
      logger.info('Sending chat message:', { message, sessionId });

      const response = await fetch(API_ENDPOINTS.CHAT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          openai_api_key: openaiApiKey,
          message,
          retrieval_strategies: enabledStrategies,
          session_id: sessionId,
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to send message');
      }

      const data = await response.json();
      
      // Convert API response to ChatMessage format
      return {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.data?.answer || data.answer || 'No response',
        timestamp: new Date(),
      };

    } catch (error) {
      logger.error('Error sending chat message:', error);
      throw error;
    }
  }

  async getAvailableStrategies(): Promise<string[]> {
    try {
      logger.info('Getting available strategies');

      const response = await fetch(`${this.baseUrl}/strategies`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to get strategies');
      }

      const data = await response.json();
      return data.data?.strategies || [];

    } catch (error) {
      logger.error('Error getting strategies:', error);
      throw error;
    }
  }
}

export const chatService = ChatService.getInstance(); 