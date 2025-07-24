"""Ensemble retrieval strategy implementation."""

from typing import List, Dict, Any, Optional
import numpy as np
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler

from .base import BaseRetrievalStrategy
from .utils import create_rag_chain

class EnsembleRetrieval(BaseRetrievalStrategy):
    """Ensemble retrieval strategy that combines multiple strategies.
    
    This strategy allows combining multiple retrieval strategies with optional
    weights to create a more robust retrieval system.
    """
    
    def __init__(
        self,
        strategies: List[str],
        weights: Optional[List[float]] = None,
        k: int = 5
    ):
        """Initialize ensemble retrieval.
        
        Args:
            strategies: List of strategy names to combine
            weights: Optional weights for each strategy (normalized if provided)
            k: Number of documents to retrieve per strategy
        """
        super().__init__(k=k)
        self.strategy_names = strategies
        self.weights = self._normalize_weights(weights or [1.0] * len(strategies))
        self._strategies: List[BaseRetrievalStrategy] = []
        
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1.
        
        Args:
            weights: List of weights
            
        Returns:
            Normalized weights
        """
        total = sum(weights)
        if total == 0:
            # If all weights are 0, use equal weights
            return [1.0/len(weights)] * len(weights)
        return [w/total for w in weights]
        
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        **kwargs
    ) -> None:
        """Initialize all component strategies.
        
        Args:
            vector_store: Vector store for retrievals
            openai_api_key: OpenAI API key
            callbacks: Optional callbacks for monitoring
            **kwargs: Additional setup parameters
        """
        from . import STRATEGY_REGISTRY
        
        # Initialize each strategy
        for strategy_name in self.strategy_names:
            if strategy_name not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown strategy: {strategy_name}")
                
            strategy = STRATEGY_REGISTRY[strategy_name](k=self.k)
            await strategy.setup(
                vector_store,
                openai_api_key,
                callbacks=callbacks,
                **kwargs
            )
            self._strategies.append(strategy)
            
        # Setup RAG chain
        self._chain = create_rag_chain(
            self,  # Ensemble implements retriever interface
            openai_api_key,
            callbacks=callbacks,
            **kwargs
        )
        
    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using weighted ensemble of strategies.
        
        Args:
            query: Search query
            **kwargs: Additional retrieval parameters
            
        Returns:
            Combined and deduplicated list of documents
        """
        self._validate_setup()
        
        # Get results from all strategies
        all_docs = {}  # Dict to track unique documents with scores
        
        # Retrieve from each strategy
        for strategy, weight in zip(self._strategies, self.weights):
            try:
                docs = await strategy._safe_retrieve(query, **kwargs)
                
                # Add documents with weighted scores
                for i, doc in enumerate(docs):
                    score = (len(docs) - i) * weight  # Simple rank-based scoring
                    if doc.page_content in all_docs:
                        all_docs[doc.page_content]["score"] += score
                    else:
                        all_docs[doc.page_content] = {
                            "doc": doc,
                            "score": score
                        }
            except Exception as e:
                # Log error but continue with other strategies
                import logging
                logging.error(f"Error in {strategy.name}: {str(e)}")
                continue
                
        # Sort by combined scores and get top k
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [doc["doc"] for doc in sorted_docs[:self.k]]
        
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
            - sub_strategies: List of component strategies
            - weights: Strategy weights used
        """
        self._validate_setup()
        
        # Get response using RAG chain
        result = await self._chain.ainvoke(query)
        
        return {
            "answer": result,
            "strategy": self.name,
            "sub_strategies": self.strategy_names,
            "weights": self.weights
        }
        
    @property
    def name(self) -> str:
        """Return strategy name."""
        return "ensemble" 