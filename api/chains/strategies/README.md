# Contextual Compression Strategy

The Contextual Compression strategy uses LangChain's `ContextualCompressionRetriever` with Hugging Face CrossEncoder models to filter and refine retrieved documents based on query relevance.

## Overview

This strategy follows a simple pattern:
1. Retrieves documents using naive retrieval (vector similarity)
2. Uses a CrossEncoder model to rerank documents by relevance
3. Returns the most relevant documents for answer generation

**Implementation**: Uses standard LangChain Expression Language (LCEL) pattern with a custom CrossEncoder reranker built on sentence-transformers.

## Requirements

- **sentence-transformers**: Required for CrossEncoder reranking
- **transformers**: Required by sentence-transformers
- **torch**: Required for model inference

All dependencies are already included in your requirements.txt!

## Usage

### Basic Usage
```python
from chains.strategies.contextual_compression import ContextualCompressionRetrieval

# Initialize strategy
strategy = ContextualCompressionRetrieval(k=5)

# Setup - no additional API keys needed!
await strategy.setup(
    vector_store=your_vector_store,
    openai_api_key="your-openai-key"
)

# Use in queries
result = await strategy.run("What are the benefits of vector databases?")
```

### Integration with RetrievalService

```python
# Initialize in RetrievalService
await retrieval_service.initialize_strategies(
    ["contextual_compression_retrieval"],
    openai_api_key="your-openai-key"
)

# Use in chat
result = await retrieval_service.get_response(
    "How does contextual compression work?",
    ["contextual_compression_retrieval"]
)
```

## Features

- **Simple**: ~80 lines of clean, readable code
- **No API Keys**: Uses open-source Hugging Face models locally
- **Effective**: Uses MS-Marco trained CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Standard**: Follows LangChain best practices with LCEL
- **Fast**: Efficient local reranking without API calls
- **Cost-effective**: No ongoing API costs after initial setup

## Technical Details

The implementation follows the standard LangChain pattern:
```python
chain = (
    {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
```

**CrossEncoder Model**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` - a state-of-the-art model trained on Microsoft's MS MARCO dataset for passage ranking. The model:
- Downloads ~400MB on first use (cached afterwards)
- Runs locally on CPU or GPU
- Provides excellent reranking quality for English text
- No API limits or ongoing costs

This provides clean separation of concerns and leverages LangChain's optimized execution engine. 