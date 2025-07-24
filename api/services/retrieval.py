from typing import List, Dict, Any
import json
from core.vector_store import vector_store
from chains.retrieval import NaiveRetrievalChain
from models.schemas.chat import ChatMessage

class RetrievalService:
    """Service for handling different retrieval strategies"""

    def __init__(self):
        pass

    async def get_response(
        self,
        openai_api_key: str,
        message: str,
        retrieval_strategies: List[str],
    ) -> Dict[str, Any]:
        """Get response using specified retrieval strategies"""
        # For now, we only support naive retrieval
        if not retrieval_strategies or "naive_retrieval" not in retrieval_strategies:
            raise ValueError("Currently only naive_retrieval strategy is supported")

        # Get vector store with embeddings
        print("Getting vector store with embeddings...")
        qdrant = vector_store.get_langchain_store(openai_api_key)
        print("Successfully retrieved vector store")

        # Initialize naive retrieval chain
        print("Initializing naive retrieval chain...")
        chain = NaiveRetrievalChain(openai_api_key, qdrant)
        print("Successfully initialized naive retrieval chain")

        # Get response
        print(f"Getting response for message: {message}")
        result = await chain.run(message)
        print("Successfully got response from chain")

        return {
            "message": ChatMessage(
                role="assistant",
                content=result["answer"]
            ).dict(),
            # "sources": result["sources"],
            "strategy_info": {
                result["strategy"]: "Successfully applied naive retrieval"
            }
        } 