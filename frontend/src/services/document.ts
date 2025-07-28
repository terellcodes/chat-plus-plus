import { API_BASE_URL, logger } from '@/config/api';

export interface UploadDocumentResponse {
  status: string;
  code: number;
  data?: {
    document_id: string;
    filename: string;
    total_pages: number;
    total_chunks: number;
    message: string;
    session_id: string;
    metadata?: {
      filename: string;
      upload_timestamp: string;
      total_pages: number;
      total_chunks: number;
    };
  };
  message?: string;
}

export class DocumentService {
  private static instance: DocumentService;
  private readonly baseUrl: string;

  private constructor() {
    this.baseUrl = API_BASE_URL;
  }

  public static getInstance(): DocumentService {
    if (!DocumentService.instance) {
      DocumentService.instance = new DocumentService();
    }
    return DocumentService.instance;
  }

  async uploadPDF(file: File, sessionId?: string): Promise<UploadDocumentResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      logger.info('Uploading PDF:', { filename: file.name, size: file.size, sessionId });

      // Build URL with optional session_id parameter
      const url = new URL(`${this.baseUrl}/upload`);
      if (sessionId) {
        url.searchParams.set('session_id', sessionId);
      }

      const response = await fetch(url.toString(), {
        method: 'POST',
        body: formData,
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
        }
      });

      const data = await response.json();
      
      if (!response.ok) {
        logger.warn('PDF upload failed:', data);
        throw new Error(data.message || 'Failed to upload PDF');
      }

      logger.success('PDF uploaded successfully:', data);
      return data;
    } catch (error) {
      logger.warn('Error uploading PDF:', error);
      throw error;
    }
  }

  async getSessionInfo(sessionId: string): Promise<any> {
    try {
      logger.info('Getting session info:', { sessionId });

      const response = await fetch(`${this.baseUrl}/session/${sessionId}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });

      const data = await response.json();
      
      if (!response.ok) {
        logger.warn('Failed to get session info:', data);
        throw new Error(data.message || 'Failed to get session info');
      }

      logger.success('Session info retrieved:', data);
      return data;
    } catch (error) {
      logger.warn('Error getting session info:', error);
      throw error;
    }
  }
}

export const documentService = DocumentService.getInstance(); 