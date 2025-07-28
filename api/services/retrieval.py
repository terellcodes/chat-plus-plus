"""Retrieval service with session-based lazy loading of strategies."""

from typing import List, Dict, Any, Optional, Union
import logging
from core.text_cache import text_cache
from core.session_manager import session_manager
from models.schemas.chat import ChatMessage, StrategyConfig, StrategyResult
from chains.strategies import INDIVIDUAL_STRATEGY_REGISTRY
from chains.strategies.ensemble import EnsembleRetrieval

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for handling different retrieval strategies with session-based lazy loading."""

    def __init__(self):
        pass  # No longer need to store strategies - they're session-scoped

    async def get_response(
        self,
        openai_api_key: str,
        message: str,
        strategy_configs: Union[List[str], List[StrategyConfig]],
        session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get RAG response using specified strategies with lazy loading.
        
        Args:
            openai_api_key: OpenAI API key
            message: User's question
            strategy_configs: List of strategy names or configs to use
            session_id: Session ID for accessing cached document
            **kwargs: Additional parameters
            
        Returns:
            Response with answer and strategy information
        """
        logger.info(f"Processing chat request for session {session_id}")
        
        # Get cached document for this session
        document = text_cache.get_document(session_id)
        if not document:
            raise ValueError(f"No document found for session {session_id}")
        
        # Convert string names to configs if needed
        configs = [
            StrategyConfig(name=s) if isinstance(s, str) else s
            for s in strategy_configs
        ]
        
        if not configs:
            raise ValueError("At least one retrieval strategy must be specified")
        
        logger.info(f"Using strategies: {[config.name for config in configs]}")
        
        # Handle single strategy vs ensemble
        if len(configs) == 1:
            # Single strategy - get or create it lazily
            config = configs[0]
            strategy = await session_manager.get_or_create_retriever(
                session_id=session_id,
                strategy_name=config.name,
                document=document,
                openai_api_key=openai_api_key,
                **(config.parameters or {}),
                **kwargs
            )
            
            # Run the strategy
            result = await strategy.run(message, **kwargs)
            
            return {
                "answer": result["answer"],
                "strategy": config.name,
                "session_id": session_id
            }
        
        else:
            # Multiple strategies - create ensemble at service level
            logger.info(f"Creating ensemble for {len(configs)} strategies")
            individual_strategies = []
            
            # Get or create each individual strategy
            for config in configs:
                strategy = await session_manager.get_or_create_retriever(
                    session_id=session_id,
                    strategy_name=config.name,
                    document=document,
                    openai_api_key=openai_api_key,
                    **(config.parameters or {}),
                    **kwargs
                )
                individual_strategies.append(strategy)
            
            # Create ensemble strategy manually (not through session manager)
            strategy_names = [config.name for config in configs]
            weights = [config.weight for config in configs if config.weight is not None]
            if not weights:
                weights = None  # Use default equal weights
            
            ensemble = EnsembleRetrieval(
                strategies=strategy_names,
                weights=weights,
                k=kwargs.get('k', 5)
            )
            
            # Setup ensemble with already-initialized strategies
            await ensemble.setup_with_strategies(
                initialized_strategies=individual_strategies,
                vector_store=None,  # Not needed for setup_with_strategies
                openai_api_key=openai_api_key,
                **kwargs
            )
            
            # Run ensemble
            result = await ensemble.run(message, **kwargs)
            
            return {
                "answer": result["answer"],
                "strategy": "ensemble",
                "strategies_used": strategy_names,
                "weights": ensemble.weights,
                "session_id": session_id
            }

    async def list_available_strategies(self) -> List[str]:
        """Get list of all available retrieval strategies.
        
        Returns:
            List of strategy names
        """
        # Return only individual strategies (exclude meta-strategies like ensemble)
        return list(INDIVIDUAL_STRATEGY_REGISTRY.keys())

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session and its loaded strategies.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information including loaded strategies
        """
        session = session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")
        
        # Get cached document info
        cached_metadata = text_cache.get_metadata(session_id)
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_accessed": session["last_accessed"],
            "loaded_strategies": list(session["retrievers"].keys()),
            "document_metadata": cached_metadata
        }