# Retrieval Strategies Implementation Guide

This document outlines how to implement and integrate various retrieval strategies into the existing Chat++ API architecture.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Retrieval Strategies](#retrieval-strategies)
3. [Integration Guide](#integration)
4. [Parallel Execution](#parallel-execution)
5. [Implementation Details](#implementation)

## Architecture Overview <a name="architecture-overview"></a>

The retrieval system builds on the existing components:

- `DocumentService`: Handles PDF processing and vector storage
- `RetrievalService`: Manages retrieval strategies
- `VectorStore`: Manages Qdrant integration
- `NaiveRetrievalChain`: Current chain implementation

### Directory Structure

```
api/
├── chains/
│   ├── __init__.py
│   ├── retrieval.py
│   └── strategies/
│       ├── __init__.py
│       ├── base.py
│       ├── naive.py
│       ├── multi_query.py
│       ├── parent_document.py
│       └── ensemble.py
├── core/
│   └── vector_store.py
└── services/
    ├── document.py
    └── retrieval.py
```

## Base Strategy Implementation <a name="base-strategy"></a>

```python
# api/chains/strategies/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document

class BaseRetrievalStrategy(ABC):
    """Base class for all retrieval strategies"""
    
    @abstractmethod
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        **kwargs
    ) -> None:
        """Initialize the strategy"""
        pass
        
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents"""
        pass
        
    @abstractmethod
    async def run(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the complete chain"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass
```

## Strategy Implementations <a name="strategies"></a>

### 1. Naive Retrieval (Current Implementation)

```python
# api/chains/strategies/naive.py

from .base import BaseRetrievalStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class NaiveRetrieval(BaseRetrievalStrategy):
    def __init__(self, k: int = 5):
        self.k = k
        self._chain = None
        self._retriever = None
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        self._retriever = vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )
        
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        self._chain = create_chain(
            self._retriever,
            rag_prompt,
            chat_model
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        return await self._retriever.ainvoke(query)
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        result = await self._chain.ainvoke({"question": query})
        return {
            "answer": result.content,
            "strategy": self.name
        }
        
    @property
    def name(self) -> str:
        return "naive_retrieval"
```

### 2. Multi-Query Retrieval

```python
# api/chains/strategies/multi_query.py

class MultiQueryRetrieval(BaseRetrievalStrategy):
    def __init__(self, k: int = 3, n_queries: int = 3):
        self.k = k
        self.n_queries = n_queries
        self._retriever = None
        self._chain = None
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        from langchain.retrievers.multi_query import MultiQueryRetriever
        
        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )
        
        self._retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=ChatOpenAI(
                temperature=0,
                openai_api_key=openai_api_key
            ),
            n_queries=self.n_queries
        )
        
        # Setup chain similar to naive retrieval
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        self._chain = create_chain(
            self._retriever,
            rag_prompt,
            chat_model
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        return await self._retriever.ainvoke(query)
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        result = await self._chain.ainvoke({"question": query})
        return {
            "answer": result.content,
            "strategy": self.name
        }
        
    @property
    def name(self) -> str:
        return "multi_query"
```

### 3. Parent Document Retrieval

```python
# api/chains/strategies/parent_document.py

class ParentDocumentRetrieval(BaseRetrievalStrategy):
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        k: int = 5
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.k = k
        self._retriever = None
        self._chain = None
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        from langchain.retrievers import ParentDocumentRetriever
        from langchain.storage import InMemoryStore
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        doc_store = InMemoryStore()
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=50
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=20
        )
        
        self._retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=doc_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": self.k}
        )
        
        # Setup chain similar to naive retrieval
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        self._chain = create_chain(
            self._retriever,
            rag_prompt,
            chat_model
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        return await self._retriever.ainvoke(query)
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        result = await self._chain.ainvoke({"question": query})
        return {
            "answer": result.content,
            "strategy": self.name
        }
        
    @property
    def name(self) -> str:
        return "parent_document"
```

### 4. Ensemble Retrieval

```python
# api/chains/strategies/ensemble.py

class EnsembleRetrieval(BaseRetrievalStrategy):
    def __init__(
        self,
        strategies: List[str],
        weights: List[float] = None,
        k: int = 5
    ):
        self.strategy_names = strategies
        self.weights = weights or [1.0/len(strategies)] * len(strategies)
        self.k = k
        self._retrievers = []
        self._chain = None
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        from langchain.retrievers import EnsembleRetriever
        
        # Initialize each strategy
        for strategy_name in self.strategy_names:
            if strategy_name not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown strategy: {strategy_name}")
                
            strategy = STRATEGY_REGISTRY[strategy_name](k=self.k)
            await strategy.setup(vector_store, openai_api_key, **kwargs)
            self._retrievers.append(strategy._retriever)
            
        # Create ensemble retriever
        self._retriever = EnsembleRetriever(
            retrievers=self._retrievers,
            weights=self.weights
        )
        
        # Setup chain similar to naive retrieval
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        self._chain = create_chain(
            self._retriever,
            rag_prompt,
            chat_model
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        return await self._retriever.ainvoke(query)
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        result = await self._chain.ainvoke({"question": query})
        return {
            "answer": result.content,
            "strategy": self.name,
            "sub_strategies": self.strategy_names
        }
        
    @property
    def name(self) -> str:
        return "ensemble"
```

### 5. Contextual Compression Retrieval

```python
# api/chains/strategies/compression.py

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class ContextualCompressionRetrieval(BaseRetrievalStrategy):
    def __init__(
        self,
        k: int = 5,
        base_strategy: str = "naive_retrieval"
    ):
        self.k = k
        self.base_strategy = base_strategy
        self._retriever = None
        self._chain = None
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        # Get base retriever
        if self.base_strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown base strategy: {self.base_strategy}")
            
        base_strategy = STRATEGY_REGISTRY[self.base_strategy](k=self.k)
        await base_strategy.setup(vector_store, openai_api_key, **kwargs)
        base_retriever = base_strategy._retriever
        
        # Create compressor
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create compression retriever
        self._retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Setup chain
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        self._chain = create_chain(
            self._retriever,
            rag_prompt,
            chat_model
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        return await self._retriever.ainvoke(query)
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        result = await self._chain.ainvoke({"question": query})
        return {
            "answer": result.content,
            "strategy": self.name,
            "base_strategy": self.base_strategy
        }
        
    @property
    def name(self) -> str:
        return "contextual_compression"
```

### 6. BM25 Retrieval

```python
# api/chains/strategies/bm25.py

from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.documents import Document

class BM25Retrieval(BaseRetrievalStrategy):
    def __init__(
        self,
        k: int = 5,
        tokenizer: Optional[callable] = None
    ):
        self.k = k
        self.tokenizer = tokenizer or word_tokenize
        self._bm25 = None
        self._documents = []
        self._chain = None
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def _preprocess(self, text: str) -> List[str]:
        """Tokenize and preprocess text"""
        return self.tokenizer(text.lower())
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        # Get all documents from vector store
        results = vector_store.get_client().scroll(
            collection_name=vector_store._collection_name,
            limit=10000,  # Adjust based on your needs
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Extract documents and preprocess
        self._documents = []
        corpus = []
        
        for point in results:
            text = point.payload.get("page_content", "")
            if text:
                self._documents.append(
                    Document(
                        page_content=text,
                        metadata=point.payload
                    )
                )
                corpus.append(self._preprocess(text))
                
        # Initialize BM25
        self._bm25 = BM25Okapi(corpus)
        
        # Setup chain
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        self._chain = create_chain(
            self,  # BM25 implements retriever interface
            rag_prompt,
            chat_model
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        # Tokenize query
        tokenized_query = self._preprocess(query)
        
        # Get BM25 scores
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_k_indices = np.argsort(scores)[-self.k:][::-1]
        
        return [self._documents[i] for i in top_k_indices]
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        result = await self._chain.ainvoke({"question": query})
        return {
            "answer": result.content,
            "strategy": self.name
        }
        
    @property
    def name(self) -> str:
        return "bm25"
```

### 7. Hybrid BM25-Vector Retrieval

```python
# api/chains/strategies/hybrid.py

class HybridRetrieval(BaseRetrievalStrategy):
    def __init__(
        self,
        k: int = 5,
        alpha: float = 0.5  # Weight for BM25 scores (1-alpha for vector scores)
    ):
        self.k = k
        self.alpha = alpha
        self._bm25_retriever = None
        self._vector_retriever = None
        self._chain = None
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        # Setup BM25
        self._bm25_retriever = BM25Retrieval(k=self.k * 2)  # Get more candidates for reranking
        await self._bm25_retriever.setup(vector_store, openai_api_key, **kwargs)
        
        # Setup vector retrieval
        self._vector_retriever = NaiveRetrieval(k=self.k * 2)
        await self._vector_retriever.setup(vector_store, openai_api_key, **kwargs)
        
        # Setup chain
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(
            model="gpt-4.1-nano",
            openai_api_key=openai_api_key
        )
        
        self._chain = create_chain(
            self,
            rag_prompt,
            chat_model
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        # Get results from both retrievers
        bm25_docs = await self._bm25_retriever.retrieve(query)
        vector_docs = await self._vector_retriever.retrieve(query)
        
        # Combine and deduplicate results
        all_docs = {}  # Dict to track unique documents
        
        # Add BM25 results with scores
        for i, doc in enumerate(bm25_docs):
            score = (len(bm25_docs) - i) * self.alpha  # Simple rank-based scoring
            all_docs[doc.page_content] = {
                "doc": doc,
                "score": score
            }
            
        # Add vector results with scores
        for i, doc in enumerate(vector_docs):
            score = (len(vector_docs) - i) * (1 - self.alpha)
            if doc.page_content in all_docs:
                all_docs[doc.page_content]["score"] += score
            else:
                all_docs[doc.page_content] = {
                    "doc": doc,
                    "score": score
                }
                
        # Sort by combined scores and get top k
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [doc["doc"] for doc in sorted_docs[:self.k]]
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        result = await self._chain.ainvoke({"question": query})
        return {
            "answer": result.content,
            "strategy": self.name
        }
        
    @property
    def name(self) -> str:
        return "hybrid"
```

## Integration with Existing Services <a name="integration"></a>

### Modified RetrievalService

```python
# api/services/retrieval.py

import asyncio
from typing import List, Dict, Any
from chains.strategies import STRATEGY_REGISTRY

class RetrievalService:
    """Service for handling different retrieval strategies"""

    def __init__(self):
        self.strategies: Dict[str, BaseRetrievalStrategy] = {}
        
    async def initialize_strategies(
        self,
        strategy_names: List[str],
        openai_api_key: str,
        **kwargs
    ):
        """Initialize requested retrieval strategies"""
        from core.vector_store import vector_store
        
        # Get vector store with embeddings
        qdrant = vector_store.get_langchain_store(openai_api_key)
        
        # Initialize each strategy
        for name in strategy_names:
            if name not in self.strategies:
                if name not in STRATEGY_REGISTRY:
                    raise ValueError(f"Unknown strategy: {name}")
                    
                strategy = STRATEGY_REGISTRY[name]()
                await strategy.setup(qdrant, openai_api_key, **kwargs)
                self.strategies[name] = strategy
                
    async def get_response(
        self,
        message: str,
        retrieval_strategies: List[str],
        weights: List[float] = None,  # Optional weights for ensemble
        **kwargs
    ) -> Dict[str, Any]:
        """Get response using specified retrieval strategies"""
        
        # Validate strategies
        if not retrieval_strategies:
            raise ValueError("No retrieval strategies specified")
            
        # Ensure all strategies are initialized
        missing = [s for s in retrieval_strategies if s not in self.strategies]
        if missing:
            raise ValueError(f"Strategies not initialized: {missing}")
            
        # If only one strategy, use it directly
        if len(retrieval_strategies) == 1:
            return await self.strategies[retrieval_strategies[0]].run(message, **kwargs)
            
        # For multiple strategies, use ensemble strategy
        ensemble = EnsembleRetrieval(
            strategies=retrieval_strategies,
            weights=weights,
            k=kwargs.get('k', 5)
        )
        
        # Setup ensemble with same vector store and API key as other strategies
        example_strategy = self.strategies[retrieval_strategies[0]]
        vector_store = example_strategy._retriever.vectorstore
        openai_api_key = kwargs.get('openai_api_key')
        
        await ensemble.setup(vector_store, openai_api_key, **kwargs)
        
        # Run ensemble strategy
        result = await ensemble.run(message, **kwargs)
        
        return result
```

### Modified DocumentService

```python
# api/services/document.py

class DocumentService:
    async def process_pdf(
        self,
        file: BinaryIO,
        filename: str,
        openai_api_key: str,
        strategy_names: List[str] = None,
        **kwargs
    ) -> UploadDocumentResponse:
        """Process PDF and initialize requested retrieval strategies"""
        
        # Existing PDF processing code...
        
        # Initialize requested strategies if provided
        if strategy_names:
            from services.retrieval import RetrievalService
            retrieval_service = RetrievalService()
            await retrieval_service.initialize_strategies(
                strategy_names,
                openai_api_key,
                **kwargs
            )
            
        return UploadDocumentResponse(...)
```

## Strategy Registry <a name="registry"></a>

```python
# api/chains/strategies/__init__.py

from .naive import NaiveRetrieval
from .multi_query import MultiQueryRetrieval
from .parent_document import ParentDocumentRetrieval
from .ensemble import EnsembleRetrieval
from .compression import ContextualCompressionRetrieval
from .bm25 import BM25Retrieval
from .hybrid import HybridRetrieval

STRATEGY_REGISTRY = {
    "naive_retrieval": NaiveRetrieval,
    "multi_query": MultiQueryRetrieval,
    "parent_document": ParentDocumentRetrieval,
    "ensemble": EnsembleRetrieval,
    "contextual_compression": ContextualCompressionRetrieval,
    "bm25": BM25Retrieval,
    "hybrid": HybridRetrieval
}
```

## Usage Examples <a name="usage"></a>

1. Using a single strategy:
```python
# Initialize service
retrieval_service = RetrievalService()
await retrieval_service.initialize_strategies(
    ["naive_retrieval"],
    openai_api_key
)

# Get response
result = await retrieval_service.get_response(
    "How does vector search work?",
    ["naive_retrieval"]
)
```

2. Using multiple strategies (automatically uses ensemble):
```python
# Initialize multiple strategies
await retrieval_service.initialize_strategies(
    ["naive_retrieval", "multi_query", "parent_document"],
    openai_api_key
)

# Get responses using ensemble strategy automatically
result = await retrieval_service.get_response(
    "How does vector search work?",
    ["naive_retrieval", "multi_query", "parent_document"],
    weights=[0.4, 0.3, 0.3]  # Optional weights for strategies
)
```

3. Using explicit ensemble strategy:
```python
# Initialize ensemble with specific strategies
await retrieval_service.initialize_strategies(
    ["ensemble"],
    openai_api_key,
    strategies=["naive_retrieval", "multi_query"],
    weights=[0.6, 0.4]
)

# Get ensemble response
result = await retrieval_service.get_response(
    "How does vector search work?",
    ["ensemble"]
)
```

4. Processing PDF with specific strategies:
```python
# Process PDF and initialize strategies
response = await document_service.process_pdf(
    file,
    filename,
    openai_api_key,
    strategy_names=["naive_retrieval", "multi_query"]
)
```

5. Using Contextual Compression:
```python
# Initialize compression with specific base strategy
await retrieval_service.initialize_strategies(
    ["contextual_compression"],
    openai_api_key,
    base_strategy="naive_retrieval"
)

# Get compressed response
result = await retrieval_service.get_response(
    "How does vector search work?",
    ["contextual_compression"]
)
```

6. Using Hybrid Retrieval:
```python
# Initialize hybrid retrieval with custom weights
await retrieval_service.initialize_strategies(
    ["hybrid"],
    openai_api_key,
    alpha=0.7  # More weight to BM25 scores
)

# Get hybrid response
result = await retrieval_service.get_response(
    "How does vector search work?",
    ["hybrid"]
)
```

7. Comparing Multiple Strategies:
```python
# Initialize multiple strategies
await retrieval_service.initialize_strategies(
    [
        "naive_retrieval",
        "bm25",
        "contextual_compression",
        "hybrid"
    ],
    openai_api_key
)

# Get response using ensemble to combine all strategies
result = await retrieval_service.get_response(
    "How does vector search work?",
    [
        "naive_retrieval",
        "bm25",
        "contextual_compression",
        "hybrid"
    ],
    weights=[0.3, 0.3, 0.2, 0.2]  # Optional custom weights
)
```

## Configuration <a name="config"></a>

Strategy configurations can be managed through environment variables or a config file:

```python
# api/config/retrieval.py

RETRIEVAL_CONFIG = {
    "naive_retrieval": {
        "k": 5
    },
    "multi_query": {
        "k": 3,
        "n_queries": 3
    },
    "parent_document": {
        "parent_chunk_size": 2000,
        "child_chunk_size": 500,
        "k": 5
    },
    "ensemble": {
        "k": 5,
        "weights": None  # Auto-calculate equal weights
    },
    "contextual_compression": {
        "k": 5,
        "base_strategy": "naive_retrieval"
    },
    "bm25": {
        "k": 5,
        "tokenizer": None  # Use default NLTK tokenizer
    },
    "hybrid": {
        "k": 5,
        "alpha": 0.5  # Equal weight to BM25 and vector scores
    }
}
```

## Adding New Strategies <a name="new-strategies"></a>

To add a new strategy:

1. Create a new file in `api/chains/strategies/`
2. Implement the strategy class inheriting from `BaseRetrievalStrategy`
3. Add to `STRATEGY_REGISTRY`
4. Add configuration to `RETRIEVAL_CONFIG`

Example:
```python
# api/chains/strategies/custom.py

class CustomRetrieval(BaseRetrievalStrategy):
    def __init__(self, **kwargs):
        self.k = kwargs.get("k", 5)
        self._retriever = None
        self._chain = None
        
    async def setup(self, vector_store: Any, openai_api_key: str, **kwargs):
        # Custom setup logic
        pass
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        # Custom retrieval logic
        pass
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        # Custom run logic
        pass
        
    @property
    def name(self) -> str:
        return "custom"

# Add to registry
STRATEGY_REGISTRY["custom"] = CustomRetrieval
```

## Error Handling <a name="error-handling"></a>

The system includes comprehensive error handling:

1. Strategy initialization errors
2. Retrieval errors
3. Parallel execution errors
4. Resource cleanup

Example error handling wrapper:
```python
async def safe_retrieve(strategy: BaseRetrievalStrategy, query: str):
    try:
        return await strategy.retrieve(query)
    except Exception as e:
        logger.error(f"Error in {strategy.name}: {str(e)}")
        return []
``` 