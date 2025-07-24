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
  ): Promise<ChatMessage> {
    try {
      logger.info('Sending chat message:', { message });

      const response = await fetch(API_ENDPOINTS.CHAT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          openai_api_key: openaiApiKey,
          message,
          retrieval_strategies: enabledStrategies,
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to send message');
      }

      const data = await response.json();
      return data.message;

    } catch (error) {
      logger.error('Error sending chat message:', error);
      throw error;
    }
  }
}

export const chatService = ChatService.getInstance(); 