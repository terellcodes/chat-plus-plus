import { API_BASE_URL, logger } from '@/config/api';

export interface UploadDocumentResponse {
  status: string;
  code: number;
  data?: {
    document_id: string;
    filename: string;
    total_chunks: number;
    upload_timestamp: string;
    status: string;
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

  async uploadPDF(file: File, openaiApiKey: string): Promise<UploadDocumentResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      logger.info('Uploading PDF:', { filename: file.name, size: file.size });

      const response = await fetch(`${this.baseUrl}/upload?openai_api_key=${encodeURIComponent(openaiApiKey)}`, {
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
}

export const documentService = DocumentService.getInstance(); 