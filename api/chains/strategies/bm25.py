"""BM25 retrieval strategy implementation using LangChain's BM25Retriever."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever

from .base import BaseRetrievalStrategy
from .utils import create_rag_chain

class BM25Retrieval(BaseRetrievalStrategy):
    """BM25 retrieval strategy using LangChain's implementation.
    
    This strategy uses the Best-Matching 25 algorithm for document retrieval,
    which is based on a probabilistic model and uses term frequency and
    document frequency to rank documents.
    """
    
    def __init__(self, k: int = 5):
        """Initialize BM25 retrieval.
        
        Args:
            k: Number of documents to retrieve
        """
        super().__init__(k=k)
        
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        **kwargs
    ) -> None:
        """Initialize BM25 with documents from vector store.
        
        Args:
            vector_store: Vector store containing documents
            openai_api_key: OpenAI API key for response generation
            callbacks: Optional callbacks for monitoring
            **kwargs: Additional setup parameters
        """
        # Get all documents from vector store
        results = vector_store.get_client().scroll(
            collection_name=vector_store._collection_name,
            limit=10000,  # Adjust based on needs
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Convert to LangChain documents
        documents = []
        for point in results:
            text = point.payload.get("page_content", "")
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata=point.payload.get("metadata", {})
                    )
                )
        
        # Initialize BM25 retriever
        self._retriever = LangChainBM25Retriever.from_documents(
            documents,
            k=self.k
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
        """Retrieve documents using BM25 scoring.
        
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
        
        # Get response using RAG chain
        result = await self._chain.ainvoke(query)
        
        return {
            "answer": result,
            "strategy": self.name
        }
        
    @property
    def name(self) -> str:
        """Return strategy name."""
        return "bm25" 