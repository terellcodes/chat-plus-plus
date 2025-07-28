from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata for a document chunk"""
    source: str = Field(description="Original filename")
    page: int = Field(description="Page number in the document")
    total_pages: int = Field(description="Total number of pages in the document")
    document_id: str = Field(description="Unique identifier for the document")

class UploadDocumentMetadata(BaseModel):
    """Metadata for upload response"""
    filename: str = Field(description="Original filename")
    upload_timestamp: str = Field(description="Upload timestamp")
    total_pages: int = Field(description="Total number of pages")
    total_chunks: int = Field(description="Number of chunks created")

class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    document_id: str = Field(description="Unique identifier for the document")
    filename: str = Field(description="Original filename")
    total_pages: int = Field(description="Total number of pages")
    total_chunks: int = Field(description="Number of chunks created")
    message: str = Field(description="Success message")
    session_id: Optional[str] = Field(description="Session ID for this upload", default=None)
    metadata: Optional[UploadDocumentMetadata] = Field(description="Document metadata", default=None) 