from typing import List
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(description="Role of the message sender (user/assistant)")
    content: str = Field(description="Content of the message")

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    openai_api_key: str = Field(description="OpenAI API key")
    retrieval_strategies: List[str] = Field(description="List of RAG strategies to apply")
    message: str = Field(description="User's message/question")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Previous chat messages")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    message: ChatMessage = Field(description="Assistant's response")
    sources: List[str] = Field(description="Sources used for the response")
    strategy_info: dict = Field(description="Information about how each strategy was applied") 