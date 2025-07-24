"""Contextual Compression retrieval strategy implementation using LangChain."""

from typing import List, Dict, Any, Optional
from operator import itemgetter
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
import logging
from pydantic import Field

# Custom CrossEncoder implementation using sentence-transformers
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    CrossEncoder = None

from .base import BaseRetrievalStrategy

logger = logging.getLogger(__name__)

class CustomCrossEncoderReranker(BaseDocumentCompressor):
    """Custom CrossEncoder reranker using sentence-transformers."""
    
    # Declare fields explicitly with Pydantic Field
    model: Any = Field(default=None, exclude=True)  # Exclude from serialization  
    top_n: int = Field(default=5)
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-JSON serializable types
        
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 5):
        # Initialize parent with declared fields first
        super().__init__(model=None, top_n=top_n)
        
        if not HAS_CROSS_ENCODER:
            raise ImportError("sentence-transformers is required")
        
        # Now assign the actual CrossEncoder object
        self.model = CrossEncoder(model_name)
        logger.info(f"Initialized CustomCrossEncoderReranker with model {model_name}")
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        """Compress documents using cross-encoder reranking."""
        if not documents:
            return []
        
        # Create query-document pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort documents by score (highest first)
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"Num docs to compress: {len(doc_scores)}")

        # Return top_n documents
        return [doc for doc, score in doc_scores[:self.top_n]]
    
    async def acompress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks = None,
    ) -> List[Document]:
        """Async version of compress_documents."""
        return self.compress_documents(documents, query, callbacks)

class ContextualCompressionRetrieval(BaseRetrievalStrategy):
    """Contextual Compression retrieval strategy using Hugging Face CrossEncoder."""
    
    def __init__(self, k1: int = 20, k2: int = 5):
        super().__init__(k=k2)
        self.k1 = k1
        
    async def setup(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        **kwargs
    ) -> None:
        """Initialize contextual compression with CrossEncoder reranking."""
        
        # Create base retriever
        naive_retriever = vector_store.as_retriever(search_kwargs={"k": self.k1})
        
        # Create CrossEncoder compressor with fallback
        if HAS_CROSS_ENCODER:
            try:
                compressor = CustomCrossEncoderReranker(
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_n=self.k
                )
                logger.info("Using CustomCrossEncoderReranker with ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                logger.warning(f"CustomCrossEncoderReranker failed: {e}, falling back to LLMChainExtractor")
                from langchain_openai import OpenAI
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                compressor = LLMChainExtractor.from_llm(llm)
        else:
            logger.info("CrossEncoder not available, using LLMChainExtractor fallback")
            from langchain_openai import OpenAI
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            compressor = LLMChainExtractor.from_llm(llm)
        
        # Create compression retriever
        self._retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=naive_retriever
        )
        
        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the question based on the context provided.

Context: {context}

Question: {question}

Answer:""")
        
        # Create chat model
        chat_model = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # Create chain following screenshot pattern
        self._chain = (
            {"context": itemgetter("question") | self._retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
        )
        
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """Retrieve documents using contextual compression."""
        return self._retriever.get_relevant_documents(query)
        
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run contextual compression retrieval and answer generation."""
        result = await self._chain.ainvoke({"question": query})
        
        return {
            "answer": result["response"].content,
            "strategy": self.name
        }
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        return self._retriever.get_relevant_documents(query)
        
    @property
    def name(self) -> str:
        """Return strategy name."""
        return "contextual_compression_retrieval" 