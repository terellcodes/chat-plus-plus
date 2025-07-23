from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata for a document stored in the vector store"""
    source: str
    page: Optional[int] = None
    total_pages: Optional[int] = None

class Document(BaseModel):
    """Document model for vector store entries"""
    id: str = Field(description="Unique identifier for the document")
    text: str = Field(description="Text content of the document chunk")
    metadata: DocumentMetadata = Field(description="Metadata about the document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the text")

class UploadDocumentResponse(BaseModel):
    """Response model for document upload endpoint"""
    document_id: str = Field(description="ID of the uploaded document")
    filename: str = Field(description="Original filename")
    total_chunks: int = Field(description="Number of chunks created")
    upload_timestamp: datetime = Field(description="When the document was uploaded")
    status: str = Field(description="Status of the upload process") 