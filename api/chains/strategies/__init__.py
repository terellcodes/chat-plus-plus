"""Strategy registry for all available retrieval strategies."""

from typing import Dict, Type
from .base import BaseRetrievalStrategy
from .bm25 import BM25Retrieval
from .contextual_compression import ContextualCompressionRetrieval
from .ensemble import EnsembleRetrieval
from .naive import NaiveRetrieval
from .rag_fusion import RAGFusionRetrieval
# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[BaseRetrievalStrategy]] = {
    "naive_retrieval": NaiveRetrieval,
    "bm25_retrieval": BM25Retrieval,
    "contextual_compression_retrieval": ContextualCompressionRetrieval,
    "ensemble_retrieval": EnsembleRetrieval,
    "rag_fusion_retrieval": RAGFusionRetrieval,  # Add this line
}

def register_strategy(name: str, strategy_class: Type[BaseRetrievalStrategy]) -> None:
    """Register a new retrieval strategy.
    
    Args:
        name: Unique identifier for the strategy
        strategy_class: The strategy class to register
    """
    if not issubclass(strategy_class, BaseRetrievalStrategy):
        raise ValueError(
            f"Strategy class must inherit from BaseRetrievalStrategy. "
            f"Got {strategy_class.__name__}"
        )
    STRATEGY_REGISTRY[name] = strategy_class 