"""BM25 retrieval strategy implementation."""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler

from .base import BaseRetrievalStrategy
from .utils import create_rag_chain

class BM25Retrieval(BaseRetrievalStrategy):
    """BM25 retrieval strategy using the rank_bm25 implementation."""
    
    def __init__(
        self,
        k: int = 5,
        tokenizer: Optional[Callable[[str], List[str]]] = None
    ):
        """Initialize BM25 retrieval.
        
        Args:
            k: Number of documents to retrieve
            tokenizer: Custom tokenizer function, defaults to NLTK word_tokenize
        """
        super().__init__(k=k)
        self.tokenizer = tokenizer or word_tokenize
        self._bm25 = None
        self._documents: List[Document] = []
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def _preprocess(self, text: str) -> List[str]:
        """Tokenize and preprocess text.
        
        Args:
            text: Text to preprocess
            
        Returns:
            List of tokens
        """
        return self.tokenizer(text.lower())
        
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        **kwargs
    ) -> None:
        """Initialize BM25 with documents from vector store.
        
        Args:
            vector_store: Qdrant vector store
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
        
        # Extract documents and preprocess
        self._documents = []
        corpus = []
        
        for point in results:
            text = point.payload.get("page_content", "")
            if text:
                self._documents.append(
                    Document(
                        page_content=text,
                        metadata=point.payload.get("metadata", {})
                    )
                )
                corpus.append(self._preprocess(text))
                
        # Initialize BM25
        self._bm25 = BM25Okapi(corpus)
        
        # Setup RAG chain
        self._chain = create_rag_chain(
            self,  # BM25Retrieval implements retriever interface
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
        
        # Tokenize query
        tokenized_query = self._preprocess(query)
        
        # Get BM25 scores
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_k_indices = np.argsort(scores)[-self.k:][::-1]
        
        return [self._documents[i] for i in top_k_indices]
        
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
            - scores: Optional BM25 scores for retrieved documents
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