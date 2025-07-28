"""Text cache for storing processed PDF content without raw PDF storage."""

import time
import uuid
from typing import Dict, Optional, Any
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older versions
    from langchain.schema import Document


class TextCache:
    """Cache for storing extracted PDF text and metadata without raw PDF bytes."""
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour default TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
    
    def add_document(
        self,
        session_id: str,
        document: Document,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store processed document text for a session.
        
        Args:
            session_id: Session identifier
            document: LangChain Document with extracted text
            metadata: Additional metadata about the upload
            
        Returns:
            Document ID for reference
        """
        doc_id = str(uuid.uuid4())
        
        self._cache[session_id] = {
            'document': document,
            'document_id': doc_id,
            'metadata': metadata or {},
            'created_at': time.time()
        }
        
        return doc_id
    
    def get_document(self, session_id: str) -> Optional[Document]:
        """Get cached document for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Document if found and not expired, None otherwise
        """
        if session_id not in self._cache:
            return None
            
        entry = self._cache[session_id]
        
        # Check if expired
        if time.time() - entry['created_at'] > self.ttl_seconds:
            self.remove_document(session_id)
            return None
            
        return entry['document']
    
    def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Metadata if found and not expired, None otherwise
        """
        if session_id not in self._cache:
            return None
            
        entry = self._cache[session_id]
        
        # Check if expired
        if time.time() - entry['created_at'] > self.ttl_seconds:
            self.remove_document(session_id)
            return None
            
        return entry['metadata']
    
    def remove_document(self, session_id: str) -> None:
        """Remove cached document for a session.
        
        Args:
            session_id: Session identifier
        """
        self._cache.pop(session_id, None)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, entry in self._cache.items()
            if current_time - entry['created_at'] > self.ttl_seconds
        ]
        
        for session_id in expired_sessions:
            self.remove_document(session_id)
            
        return len(expired_sessions)
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active sessions and their metadata.
        
        Returns:
            Dictionary mapping session IDs to their metadata
        """
        return {
            session_id: {
                'document_id': entry['document_id'],
                'metadata': entry['metadata'],
                'created_at': entry['created_at']
            }
            for session_id, entry in self._cache.items()
        }


# Global text cache instance
text_cache = TextCache()