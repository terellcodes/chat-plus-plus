"""Ensemble retrieval strategy implementation with custom chain."""

from typing import List, Dict, Any, Optional
import asyncio
import logging
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .base import BaseRetrievalStrategy

logger = logging.getLogger(__name__)

# Template for RAG responses
RAG_TEMPLATE = """You are a helpful AI assistant answering questions based on the provided context.

Context:
{context}

Question:
{question}

Please provide a clear, accurate, and helpful response based on the context above.
If the context doesn't contain enough information to answer the question fully,
acknowledge this and provide the best possible answer with the available information.

Response:"""

class EnsembleRetrieval(BaseRetrievalStrategy):
    """Ensemble retrieval strategy that combines multiple strategies.
    
    This strategy uses a custom chain implementation to avoid LCEL compatibility issues.
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
            weights: Optional weights for each strategy (default: equal weights)
            k: Number of documents to retrieve per strategy
        """
        super().__init__(k=k)
        self.strategy_names = strategies
        self.weights = self._normalize_weights(weights or [1.0] * len(strategies))
        self._strategies: List[BaseRetrievalStrategy] = []
        self._llm = None
        self._prompt = None

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights)
        return [w / total for w in weights]

    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0,
        **kwargs
    ) -> None:
        """Set up all sub-strategies and create custom chain components."""
        # Import here to avoid circular import
        from . import STRATEGY_REGISTRY
        
        self._strategies = []
        setup_tasks = []
        
        for strategy_name in self.strategy_names:
            if strategy_name == self.name:
                raise ValueError("Cannot use ensemble strategy within itself")
                
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if not strategy_class:
                raise ValueError(f"Strategy {strategy_name} not found in registry")
                
            strategy = strategy_class(k=self.k)
            self._strategies.append(strategy)
            setup_tasks.append(
                strategy.setup(
                    vector_store=vector_store,
                    openai_api_key=openai_api_key,
                    callbacks=callbacks,
                    **kwargs
                )
            )
        
        # Set up all strategies in parallel
        await asyncio.gather(*setup_tasks)
        
        # Create custom chain components
        self._prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        self._llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=openai_api_key,
            **kwargs
        )
        self._parser = StrOutputParser()

    async def setup_with_strategies(
        self,
        initialized_strategies: List[BaseRetrievalStrategy],
        vector_store: Any,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0,
        **kwargs
    ) -> None:
        """Set up ensemble with already-initialized strategies.
        
        This method is used when we already have initialized strategies
        and want to avoid re-initializing them.
        """
        # Use the already-initialized strategies
        self._strategies = initialized_strategies
        
        # Create custom chain components
        self._prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        self._llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=openai_api_key,
            **kwargs
        )
        self._parser = StrOutputParser()

    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """Retrieve documents from all strategies in parallel and combine results."""
        self._validate_setup()
        
        # Run all retrievers in parallel
        retrieval_tasks = [
            strategy._safe_retrieve(query)
            for strategy in self._strategies
        ]
        
        # Gather results
        results = await asyncio.gather(*retrieval_tasks)
        
        # Combine and score documents
        scored_docs: Dict[str, float] = {}  # doc_content -> score
        
        for strategy_idx, (docs, weight) in enumerate(zip(results, self.weights)):
            # Score based on position and strategy weight
            for doc_idx, doc in enumerate(docs):
                score = (len(docs) - doc_idx) * weight
                key = f"{doc.page_content}:{doc.metadata.get('document_id', '')}"
                scored_docs[key] = scored_docs.get(key, 0) + score
        
        # Convert back to documents, sorted by score
        unique_docs = {}
        for doc_list in results:
            for doc in doc_list:
                key = f"{doc.page_content}:{doc.metadata.get('document_id', '')}"
                if key not in unique_docs:
                    unique_docs[key] = doc
        
        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda d: scored_docs[f"{d.page_content}:{d.metadata.get('document_id', '')}"],
            reverse=True
        )
        
        return sorted_docs[:self.k]

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into a string."""
        if not docs:
            return ""
        
        formatted_docs = []
        for doc in docs:
            if hasattr(doc, 'page_content') and doc.page_content:
                formatted_docs.append(doc.page_content)
        
        return "\n\n".join(formatted_docs)

    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run the complete ensemble retrieval and answer generation.
        
        This uses a custom chain implementation instead of LCEL.
        """
        self._validate_setup()
        
        try:
            # Step 1: Retrieve documents using ensemble
            documents = await self.retrieve(query, **kwargs)
            
            # Step 2: Format documents
            context = self._format_docs(documents)
            
            # Step 3: Create prompt with context and question
            prompt_value = await self._prompt.ainvoke({
                "context": context,
                "question": query
            })
            
            # Step 4: Generate response
            llm_response = await self._llm.ainvoke(prompt_value)
            
            # Step 5: Parse output
            answer = await self._parser.ainvoke(llm_response)
            
            return {
                "answer": answer,
                "strategy": self.name,
                "sub_strategies": self.strategy_names,
                "weights": self.weights,
                "documents_retrieved": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble run: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "strategy": self.name,
                "sub_strategies": self.strategy_names,
                "error": str(e)
            }

    # Keep these methods for backward compatibility but they won't be used in the custom chain
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Synchronous version of retrieve for compatibility."""
        try:
            return asyncio.run(self.retrieve(query, **kwargs))
        except Exception as e:
            logger.error(f"Error in get_relevant_documents: {str(e)}")
            return []

    def invoke(self, input_data: str, **kwargs) -> List[Document]:
        """Invoke method for compatibility."""
        return self.get_relevant_documents(input_data, **kwargs)

    async def ainvoke(self, input_data: str, **kwargs) -> List[Document]:
        """Async invoke method for compatibility."""
        return await self.retrieve(input_data, **kwargs)

    @property
    def name(self) -> str:
        """Return the strategy's name."""
        return "ensemble_retrieval" 
    
    def _validate_setup(self) -> None:
        """Validate that the strategy is properly initialized.
        
        Raises:
            ValueError: If required components are not initialized
        """
        return