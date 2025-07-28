import uuid
import tempfile
import os
from datetime import datetime
from typing import BinaryIO, Optional, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from core.text_cache import text_cache
from core.session_manager import session_manager
from models.schemas.document import UploadDocumentResponse, UploadDocumentMetadata

class DocumentService:
    """Service for handling document operations with session-based text caching"""

    def __init__(self):
        pass  # No longer need text splitter - strategies handle their own chunking

    async def process_pdf(
        self,
        file: BinaryIO,
        filename: str,
        session_id: Optional[str] = None
    ) -> UploadDocumentResponse:
        """Process a PDF file using LangChain PyPDFLoader and cache text for lazy retriever creation.
        
        Args:
            file: PDF file object
            filename: Original filename
            session_id: Optional session ID, will create new session if not provided
            
        Returns:
            Response with session_id and document metadata
        """
        print(f"\n📄 Processing PDF file: {filename}")
        
        # Create session if not provided
        if not session_id:
            session_id = session_manager.create_session()
            print(f"🆔 Created new session: {session_id}")
        
        # Create temporary file for PyPDFLoader (it needs a file path)
        temp_path = None
        try:
            # Write uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                temp_path = tmp_file.name
            
            print("📚 Loading PDF with LangChain PyPDFLoader...")
            
            # Use LangChain's optimized PDF loader
            loader = PyPDFLoader(
                file_path=temp_path,
            )
            
            # Load document - this is much faster than manual parsing
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content extracted from PDF")
                
            # Get the single document (in "single" mode)
            document = documents[0]
            
            # Update metadata with our information
            document.metadata.update({
                "source": filename,
                "document_id": str(uuid.uuid4()),
                "upload_timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id
            })
            
            print(f"✅ PDF loaded successfully. Content length: {len(document.page_content)} characters")
            
            # Cache the document text for lazy retriever creation
            doc_id = text_cache.add_document(
                session_id=session_id,
                document=document,
                metadata={
                    "filename": filename,
                    "upload_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            print(f"💾 Document cached in session {session_id}")
            
            # Return response with session info - no vector store operations yet!
            return UploadDocumentResponse(
                document_id=doc_id,
                filename=filename,
                total_pages=1,  # Single document mode
                total_chunks=0,  # Chunks created lazily when strategies are used
                message=f"PDF processed and cached successfully. Session: {session_id}",
                session_id=session_id,  # Include session_id in response
                metadata=UploadDocumentMetadata(
                    filename=filename,
                    upload_timestamp=datetime.utcnow().isoformat(),
                    total_pages=1,
                    total_chunks=0
                )
            )
            
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            # Clean up session on failure
            if session_id:
                try:
                    text_cache.remove_document(session_id)
                    session_manager.cleanup_session(session_id)
                except:
                    pass
            raise ValueError(f"Failed to process PDF: {str(e)}")
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass 