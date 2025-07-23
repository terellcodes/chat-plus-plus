from typing import List, Dict, Any
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
        chat_history: List[ChatMessage]
    ) -> Dict[str, Any]:
        """Get response using specified retrieval strategies"""
        # For now, we only support naive retrieval
        if not retrieval_strategies or "naive_retrieval" not in retrieval_strategies:
            raise ValueError("Currently only naive_retrieval strategy is supported")

        # Get vector store with embeddings
        qdrant = vector_store.get_langchain_store(openai_api_key)

        # Initialize naive retrieval chain
        chain = NaiveRetrievalChain(openai_api_key, qdrant)

        # Convert chat history to list of dicts
        history = [msg.dict() for msg in chat_history] if chat_history else None

        # Get response
        result = await chain.run(message, history)

        return {
            "message": ChatMessage(
                role="assistant",
                content=result["answer"]
            ).dict(),
            "sources": result["sources"],
            "strategy_info": {
                result["strategy"]: "Successfully applied naive retrieval"
            }
        } 