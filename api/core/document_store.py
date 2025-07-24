"""Document store singleton for storing original documents."""

from typing import Dict, Any, Optional
from langchain.storage import InMemoryStore
from langchain.schema import Document

class DocumentStore:
    """Singleton class for managing document storage."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._store = InMemoryStore()
            cls._instance._metadata: Dict[str, Dict[str, Any]] = {}
        return cls._instance
    
    def add_document(
        self,
        doc_id: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a document and its metadata.
        
        Args:
            doc_id: Unique identifier for the document
            content: Raw document content (e.g., PDF bytes)
            metadata: Optional metadata about the document
        """
        self._store.mset([(doc_id, content)])
        if metadata:
            self._metadata[doc_id] = metadata
            
    def get_document(
        self,
        doc_id: str
    ) -> Optional[bytes]:
        """Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document content if found, None otherwise
        """
        return self._store.mget([doc_id])[0]
        
    def get_metadata(
        self,
        doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get document metadata.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document metadata if found, None otherwise
        """
        return self._metadata.get(doc_id)
        
    def delete_document(
        self,
        doc_id: str
    ) -> None:
        """Delete a document and its metadata.
        
        Args:
            doc_id: Document identifier
        """
        self._store.mdelete([doc_id])
        self._metadata.pop(doc_id, None)
        
    def list_documents(self) -> Dict[str, Dict[str, Any]]:
        """List all stored documents and their metadata.
        
        Returns:
            Dictionary mapping document IDs to their metadata
        """
        return self._metadata.copy()

# Create singleton instance
document_store = DocumentStore() 