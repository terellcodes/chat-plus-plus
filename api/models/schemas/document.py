from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata for a document chunk"""
    source: str = Field(description="Original filename")
    page: int = Field(description="Page number in the document")
    total_pages: int = Field(description="Total number of pages in the document")
    document_id: str = Field(description="Unique identifier for the document")

class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    document_id: str = Field(description="Unique identifier for the document")
    filename: str = Field(description="Original filename")
    total_chunks: int = Field(description="Number of chunks created")
    upload_timestamp: datetime = Field(description="When the document was uploaded")
    status: str = Field(description="Processing status (e.g., ready, failed)") 