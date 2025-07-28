"""BM25 retrieval strategy using LangChain's BM25Retriever."""

from typing import List, Dict, Any, Optional
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older versions
    from langchain.schema import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from chains.retrieval import RAG_TEMPLATE
from .base import BaseRetrievalStrategy


class BM25Retrieval(BaseRetrievalStrategy):
    """
    BM25 retrieval strategy using LangChain's BM25Retriever.
    
    This implementation follows the pattern from Advanced_Retrieval_with_LangChain_Assignment.ipynb:
    1. Uses BM25Retriever.from_documents() directly on full document content
    2. No pre-chunking - BM25 works on the complete document text
    3. Integrates with session-based architecture to get full document from cache
    """
    
    def __init__(self, k: int = 5):
        """Initialize BM25 retrieval strategy.
        
        Args:
            k: Number of documents to retrieve
        """
        self.k = k
        self._retriever = None
        self._chain = None
        print(f"🏗️  Initializing BM25 retrieval with k={k}")
    
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        document: Optional[Document] = None,
        **kwargs
    ) -> None:
        """Set up BM25 retrieval strategy.
        
        Args:
            vector_store: Not used by BM25 (for interface compatibility)
            openai_api_key: OpenAI API key for chat model
            document: Full document from session cache (preferred approach)
            **kwargs: Additional parameters
        """
        print(f"🔧 Setting up BM25 retrieval strategy")
        
        # Get document content - prefer session-based full document
        if document:
            print(f"📚 Using full document from session cache ({len(document.page_content)} characters)")
            
            # Create a list with the single document for BM25Retriever.from_documents()
            # BM25 works best when it can see the full document context
            documents = [document]
            
        if not documents:
            raise ValueError("No documents available for BM25 setup")
        
        # Create BM25 retriever using LangChain's implementation
        # This follows the exact pattern from the notebook: BM25Retriever.from_documents()
        print(f"🔍 Creating BM25Retriever from {len(documents)} documents")
        self._retriever = LangChainBM25Retriever.from_documents(
            documents,
            k=self.k
        )
        
        print(f"✅ BM25 retriever initialized with k={self.k}")
        
        # Set up RAG chain using the same pattern as the notebook
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
        
        self._chain = (
            {"context": itemgetter("question") | self._retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
        )
        
        print("✅ BM25 strategy setup complete")
    
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            List of relevant documents ranked by BM25 score
        """
        if not self._retriever:
            raise ValueError("BM25 strategy not properly initialized")
        
        print(f"🔍 BM25 retrieving documents for: '{query}'")
        
        # Use LangChain BM25Retriever's get_relevant_documents method
        documents = self._retriever.get_relevant_documents(query)
        
        print(f"📄 BM25 retrieved {len(documents)} documents")
        
        return documents
    
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run the complete BM25 retrieval chain.
        
        Args:
            query: User's question
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not self._chain:
            raise ValueError("BM25 chain not initialized")
        
        print(f"🚀 Running BM25 chain for query: '{query}'")
        
        # Invoke the chain exactly like in the notebook
        result = self._chain.invoke({"question": query})
        
        return {
            "answer": result["response"].content,
            "strategy": self.name,
            "retrieval_method": "BM25 with LangChain BM25Retriever",
            "context": result["context"]
        }
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query (synchronous interface).
        
        This method is required for LangChain retriever interface compatibility.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        if not self._retriever:
            raise ValueError("BM25 strategy not properly initialized")
        
        return self._retriever.get_relevant_documents(query)
    
    @property
    def name(self) -> str:
        """Strategy name identifier."""
        return "bm25_retrieval"