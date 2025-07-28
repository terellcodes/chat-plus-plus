"""Session manager for handling session-based retrievers and cleanup."""

import time
import uuid
import asyncio
from typing import Dict, Optional, Any, List
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older versions
    from langchain.schema import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

from chains.strategies import INDIVIDUAL_STRATEGY_REGISTRY, META_STRATEGY_REGISTRY
from chains.strategies.base import BaseRetrievalStrategy


class SessionManager:
    """Manages session lifecycle and lazy-loaded retrievers."""
    
    def __init__(self, session_ttl: int = 3600, cleanup_interval: int = 300):
        """Initialize session manager.
        
        Args:
            session_ttl: Session time-to-live in seconds (default 1 hour)
            cleanup_interval: Background cleanup interval in seconds (default 5 minutes)
        """
        self.session_ttl = session_ttl
        self.cleanup_interval = cleanup_interval
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
    def create_session(self) -> str:
        """Create a new session.
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            'created_at': time.time(),
            'retrievers': {},  # Strategy name -> Strategy instance
            'vector_stores': {},  # Strategy name -> Vector store instance
            'last_accessed': time.time()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data if it exists and is not expired.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if valid, None if expired or not found
        """
        if session_id not in self._sessions:
            return None
            
        session = self._sessions[session_id]
        
        # Check if expired
        if time.time() - session['created_at'] > self.session_ttl:
            self.cleanup_session(session_id)
            return None
            
        # Update last accessed time
        session['last_accessed'] = time.time()
        return session
    
    async def get_or_create_retriever(
        self,
        session_id: str,
        strategy_name: str,
        document: Document,
        openai_api_key: str,
        **kwargs
    ) -> BaseRetrievalStrategy:
        """Get existing retriever or create new one for a strategy.
        
        Args:
            session_id: Session identifier
            strategy_name: Name of the retrieval strategy
            document: Document to use for retriever setup
            openai_api_key: OpenAI API key
            **kwargs: Additional strategy parameters
            
        Returns:
            Configured retrieval strategy instance
            
        Raises:
            ValueError: If strategy name is not recognized or session is invalid
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")
            
        # Return cached retriever if it exists
        if strategy_name in session['retrievers']:
            return session['retrievers'][strategy_name]
            
        # Check if this is a meta-strategy that cannot be created individually
        if strategy_name in META_STRATEGY_REGISTRY:
            raise ValueError(f"Meta-strategy '{strategy_name}' cannot be created individually. Use service-level ensemble creation.")
            
        # Create new retriever - only allow individual strategies
        if strategy_name not in INDIVIDUAL_STRATEGY_REGISTRY:
            raise ValueError(f"Unknown individual strategy: {strategy_name}")
            
        strategy_class = INDIVIDUAL_STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_class()
        
        # Check if we already have a vector store for this strategy
        if strategy_name in session['vector_stores']:
            vector_store = session['vector_stores'][strategy_name]
            print(f"🔄 Reusing existing vector store for {strategy_name} in session {session_id}")
        else:
            # Create vector store from document for this strategy
            print(f"🔧 Creating vector store for strategy {strategy_name} in session {session_id}")
            
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=getattr(strategy, 'chunk_size', 1000),
                chunk_overlap=getattr(strategy, 'chunk_overlap', 200)
            )
            chunks = text_splitter.split_documents([document])
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Create vector store
            vector_store = Qdrant.from_documents(
                chunks,
                embeddings,
                location=":memory:",  # In-memory for session-based usage
                collection_name=f"session_{session_id}_{strategy_name}",
            )
            
            # Cache the vector store
            session['vector_stores'][strategy_name] = vector_store
            print(f"✅ Vector store created with {len(chunks)} chunks for {strategy_name}")
        
        # Setup strategy with the vector store
        await strategy.setup(
            vector_store=vector_store,
            openai_api_key=openai_api_key,
            **kwargs
        )
        
        # Cache the retriever for future use
        session['retrievers'][strategy_name] = strategy
        
        return strategy
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up a specific session and its retrievers.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            session = self._sessions[session_id]
            
            # Clean up any resources held by retrievers
            for strategy in session['retrievers'].values():
                if hasattr(strategy, 'cleanup'):
                    try:
                        strategy.cleanup()
                    except Exception as e:
                        print(f"Error cleaning up strategy: {e}")
            
            # Clean up vector stores
            for vector_store in session['vector_stores'].values():
                try:
                    # Qdrant in-memory stores don't need explicit cleanup
                    # but we could add it here if needed
                    pass
                except Exception as e:
                    print(f"Error cleaning up vector store: {e}")
            
            del self._sessions[session_id]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if current_time - session['created_at'] > self.session_ttl
        ]
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
            
        return len(expired_sessions)
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active sessions.
        
        Returns:
            Dictionary mapping session IDs to session info
        """
        current_time = time.time()
        return {
            session_id: {
                'created_at': session['created_at'],
                'last_accessed': session['last_accessed'],
                'age_seconds': current_time - session['created_at'],
                'retrievers': list(session['retrievers'].keys()),
                'vector_stores': list(session['vector_stores'].keys())
            }
            for session_id, session in self._sessions.items()
        }
    
    async def start_background_cleanup(self):
        """Start background task for cleaning expired sessions."""
        while True:
            try:
                cleaned_count = self.cleanup_expired_sessions()
                if cleaned_count > 0:
                    print(f"Cleaned up {cleaned_count} expired sessions")
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error


# Global session manager instance
session_manager = SessionManager()