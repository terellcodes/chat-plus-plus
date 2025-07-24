from typing import List, Dict, Any, Optional, Union
import logging
from core.vector_store import vector_store
from models.schemas.chat import ChatMessage, StrategyConfig, StrategyResult
from chains.strategies import STRATEGY_REGISTRY
from chains.strategies.ensemble import EnsembleRetrieval

logger = logging.getLogger(__name__)

class RetrievalService:
    """Service for handling different retrieval strategies"""

    def __init__(self):
        self.strategies = {}

    async def initialize_strategies(
        self,
        strategy_configs: Union[List[str], List[StrategyConfig]],
        openai_api_key: str,
        **kwargs
    ) -> None:
        """Initialize requested retrieval strategies.
        
        Args:
            strategy_configs: List of strategy names or configs
            openai_api_key: OpenAI API key
            **kwargs: Additional strategy parameters
        """
        # Get vector store with embeddings
        logger.info("Getting vector store with embeddings...")
        qdrant = vector_store.get_langchain_store(openai_api_key)
        logger.info("Successfully retrieved vector store")

        # Convert string names to configs if needed
        configs = [
            StrategyConfig(name=s) if isinstance(s, str) else s
            for s in strategy_configs
        ]

        # Initialize each strategy
        for config in configs:
            if config.name not in self.strategies:
                if config.name not in STRATEGY_REGISTRY:
                    raise ValueError(f"Unknown strategy: {config.name}")
                
                logger.info(f"Initializing {config.name} strategy...")
                strategy = STRATEGY_REGISTRY[config.name]()
                
                # Add any strategy-specific parameters
                strategy_kwargs = {**kwargs, **(config.parameters or {})}
                
                await strategy.setup(qdrant, openai_api_key, **strategy_kwargs)
                self.strategies[config.name] = strategy
                logger.info(f"Successfully initialized {config.name} strategy")

    async def get_response(
        self,
        openai_api_key: str,
        message: str,
        retrieval_strategies: Union[List[str], List[StrategyConfig]],
        **kwargs
    ) -> Dict[str, Any]:
        """Get response using specified retrieval strategies.
        
        Args:
            openai_api_key: OpenAI API key
            message: User's message/question
            retrieval_strategies: List of strategies to use
            **kwargs: Additional parameters
            
        Returns:
            Response containing assistant's message and strategy info
        """
        # Validate strategies
        if not retrieval_strategies:
            raise ValueError("No retrieval strategies specified")

        # Initialize strategies if needed
        await self.initialize_strategies(retrieval_strategies, openai_api_key, **kwargs)

        # Convert string names to configs if needed
        configs = [
            StrategyConfig(name=s) if isinstance(s, str) else s
            for s in retrieval_strategies
        ]

        # Ensure all strategies are initialized
        missing = [c.name for c in configs if c.name not in self.strategies]
        if missing:
            raise ValueError(f"Strategies not initialized: {missing}")

        try:
            # If only one strategy, use it directly
            if len(configs) == 1:
                logger.info(f"Using single strategy: {configs[0].name}")
                result = await self.strategies[configs[0].name].run(message, **kwargs)
                strategy_info = {
                    configs[0].name: StrategyResult(
                        status="success",
                        details={"type": "single_strategy"}
                    ).dict()
                }
            else:
                # For multiple strategies, use ensemble
                logger.info("Using ensemble strategy for multiple retrievers")
                
                # Get strategy names and weights
                names = [c.name for c in configs]
                weights = [c.weight for c in configs if c.weight is not None]
                if weights and len(weights) != len(names):
                    raise ValueError("If weights are provided, all strategies must have weights")
                
                # Get the already-initialized strategies
                initialized_strategies = [self.strategies[name] for name in names]
                
                ensemble = EnsembleRetrieval(
                    strategies=names,
                    weights=weights if weights else None,
                    k=kwargs.get("k", 5)
                )
                
                # Setup ensemble with already-initialized strategies
                qdrant = vector_store.get_langchain_store(openai_api_key)
                await ensemble.setup_with_strategies(
                    initialized_strategies, 
                    qdrant, 
                    openai_api_key, 
                    **kwargs
                )
                
                result = await ensemble.run(message, **kwargs)
                
                # Create strategy info for each component
                strategy_info = {
                    name: StrategyResult(
                        status="success",
                        details={
                            "type": "ensemble_component",
                            "weight": weight if weights else 1.0/len(names)
                        }
                    ).dict()
                    for name, weight in zip(names, weights if weights else [1.0/len(names)] * len(names))
                }
                
                # Add ensemble info
                strategy_info["ensemble"] = StrategyResult(
                    status="success",
                    details={
                        "type": "ensemble_controller",
                        "components": names
                    }
                ).dict()

            return {
                "message": ChatMessage(
                    role="assistant",
                    content=result["answer"]
                ).dict(),
                "sources": result.get("sources", []),
                "strategy_info": strategy_info
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval service: {str(e)}")
            raise ValueError(f"Failed to get response: {str(e)}")
