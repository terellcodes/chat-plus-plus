"""Strategy registry for all available retrieval strategies."""

from typing import Dict, Type
from .base import BaseRetrievalStrategy
from .bm25 import BM25Retrieval
from .contextual_compression import ContextualCompressionRetrieval
from .ensemble import EnsembleRetrieval
from .naive import NaiveRetrieval
from .parent_document import ParentDocumentRetrieval
from .rag_fusion import RAGFusionRetrieval
# Individual strategies that can be created independently by session manager
INDIVIDUAL_STRATEGY_REGISTRY: Dict[str, Type[BaseRetrievalStrategy]] = {
    "naive_retrieval": NaiveRetrieval,
    "bm25_retrieval": BM25Retrieval,
    "contextual_compression_retrieval": ContextualCompressionRetrieval,
    "parent_document_retrieval": ParentDocumentRetrieval,
    "rag_fusion_retrieval": RAGFusionRetrieval,
}

# Meta-strategies that require special handling and cannot be created individually
META_STRATEGY_REGISTRY: Dict[str, Type[BaseRetrievalStrategy]] = {
    "ensemble_retrieval": EnsembleRetrieval,
}

# Combined registry for compatibility and lookup
STRATEGY_REGISTRY: Dict[str, Type[BaseRetrievalStrategy]] = {
    **INDIVIDUAL_STRATEGY_REGISTRY,
    **META_STRATEGY_REGISTRY
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