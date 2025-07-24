from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler
import logging

logger = logging.getLogger(__name__)

class BaseRetrievalStrategy(ABC):
    """Base class for all retrieval strategies.
    
    All retrieval strategies must inherit from this class and implement
    its abstract methods. This ensures consistent behavior across different
    retrieval approaches.
    """
    
    def __init__(self, k: int = 5):
        """Initialize the strategy.
        
        Args:
            k (int): Number of documents to retrieve. Defaults to 5.
        """
        self.k = k
        self._retriever = None
        self._chain = None
        self._callback_manager = None
        
    @abstractmethod
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        **kwargs
    ) -> None:
        """Initialize the strategy with necessary components.
        
        Args:
            vector_store: The vector store to use for retrieval
            openai_api_key: OpenAI API key for LLM operations
            callbacks: Optional list of callbacks for monitoring
            **kwargs: Additional strategy-specific parameters
        """
        pass
        
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents for the query.
        
        Args:
            query: The user's question
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieved documents
        """
        pass
        
    @abstractmethod
    async def run(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the complete retrieval and answer generation chain.
        
        Args:
            query: The user's question
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing at least:
            - answer: The generated response
            - strategy: Name of the strategy used
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy's name.
        
        Returns:
            String identifier for the strategy
        """
        pass
        
    async def _safe_retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """Safely execute document retrieval with error handling.
        
        Args:
            query: The user's question
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents, empty list on error
        """
        try:
            return await self.retrieve(query, **kwargs)
        except Exception as e:
            logger.error(f"Error in {self.name} retrieval: {str(e)}")
            return []
            
    def _validate_setup(self) -> None:
        """Validate that the strategy is properly initialized.
        
        Raises:
            ValueError: If required components are not initialized
        """
        if self._retriever is None:
            raise ValueError(f"{self.name} strategy not initialized. Call setup() first.")
            
        if self._chain is None:
            raise ValueError(f"{self.name} chain not initialized. Call setup() first.") 