from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(description="Role of the message sender (user/assistant)")
    content: str = Field(description="Content of the message")

class StrategyConfig(BaseModel):
    """Configuration for a retrieval strategy"""
    name: str = Field(description="Name of the strategy")
    weight: Optional[float] = Field(None, description="Weight for ensemble combination")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional parameters for the strategy"
    )

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    openai_api_key: str = Field(description="OpenAI API key")
    retrieval_strategies: Union[List[str], List[StrategyConfig]] = Field(
        description="List of RAG strategies to apply. Can be simple strategy names or detailed configs"
    )
    message: str = Field(description="User's message/question")

class StrategyResult(BaseModel):
    """Result information for a retrieval strategy"""
    status: str = Field(description="Status of the strategy execution")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details about the execution")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    message: ChatMessage = Field(description="Assistant's response")
    sources: List[str] = Field(description="Sources used for the response")
    strategy_info: Dict[str, StrategyResult] = Field(
        description="Information about how each strategy was applied"
    )
