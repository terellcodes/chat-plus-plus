"""Strategy registry for all available retrieval strategies."""

from typing import Dict, Type
from .base import BaseRetrievalStrategy
from .bm25 import BM25Retrieval
from .ensemble import EnsembleRetrieval

# Import strategies as they are implemented
# from .naive import NaiveRetrieval

# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[BaseRetrievalStrategy]] = {
    # Will be populated as strategies are implemented
    # "naive_retrieval": NaiveRetrieval,
    "bm25": BM25Retrieval,
    "ensemble": EnsembleRetrieval,
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