"""Naive retrieval strategy implementation using vector similarity search."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.vectorstores import VectorStore

from .base import BaseRetrievalStrategy
from .utils import create_rag_chain

class NaiveRetrieval(BaseRetrievalStrategy):
    """Naive retrieval strategy using simple vector similarity search.
    
    This strategy performs a basic similarity search using the provided
    vector store without any additional processing or reranking.
    """
    
    def __init__(
        self,
        k: int = 4,  # Default to 4 for naive strategy
        chunk_size: int = 1000,  # Larger chunks for naive strategy
        chunk_overlap: int = 200  # More overlap for context preservation
    ):
        """Initialize naive retrieval.
        
        Args:
            k: Number of documents to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        super().__init__(k=k)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    async def setup(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        **kwargs
    ) -> None:
        """Initialize naive retrieval with vector store.
        
        Args:
            vector_store: Vector store for similarity search
            openai_api_key: OpenAI API key for response generation
            callbacks: Optional callbacks for monitoring
            **kwargs: Additional setup parameters
        """
        if not isinstance(vector_store, VectorStore):
            raise ValueError("vector_store must be an instance of VectorStore")
            
        # Create retriever from vector store
        self._retriever = vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )
        
        # Setup RAG chain
        self._chain = create_rag_chain(
            self._retriever,
            openai_api_key,
            callbacks=callbacks,
            **kwargs
        )
        
    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using similarity search.
        
        Args:
            query: Search query
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieved documents
        """
        self._validate_setup()
        return await self._safe_retrieve(query, **kwargs)
        
    async def run(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the complete retrieval and answer generation chain.
        
        Args:
            query: User question
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
            - answer: Generated response
            - strategy: Name of the strategy
        """
        self._validate_setup()
        
        # Get response using RAG chain with proper input format
        result = await self._chain.ainvoke({"question": query})
        
        return {
            "answer": result,
            "strategy": self.name
        }
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query.
        
        This method is required by the LangChain retriever interface.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        return self._retriever.get_relevant_documents(query)
        
    @property
    def name(self) -> str:
        """Return strategy name."""
        return "naive_retrieval" 