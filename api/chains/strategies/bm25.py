"""BM25 retrieval strategy implementation using LangChain's BM25Retriever."""

from typing import List, Dict, Any, Optional
import io
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever

from .base import BaseRetrievalStrategy
from .utils import create_rag_chain
from core.document_store import document_store

class BM25Retrieval(BaseRetrievalStrategy):
    """BM25 retrieval strategy using LangChain's implementation.
    
    This strategy uses the Best-Matching 25 algorithm for document retrieval,
    which is based on a probabilistic model and uses term frequency and
    document frequency to rank documents.
    """
    
    def __init__(
        self,
        k: int = 5,
        chunk_size: int = 500,  # Smaller chunks for BM25
        chunk_overlap: int = 50
    ):
        """Initialize BM25 retrieval.
        
        Args:
            k: Number of documents to retrieve
            chunk_size: Size of text chunks for BM25
            chunk_overlap: Overlap between chunks
        """
        super().__init__(k=k)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        **kwargs
    ) -> None:
        """Initialize BM25 with custom document chunking.
        
        Args:
            vector_store: Not used by BM25, but required by interface
            openai_api_key: OpenAI API key for response generation
            callbacks: Optional callbacks for monitoring
            **kwargs: Additional setup parameters
        """
        # Get all stored documents
        stored_files = document_store.list_documents()
        if not stored_files:
            raise ValueError("No documents found for BM25 indexing")
            
        # Process each document with BM25-specific chunking
        documents = []
        for file_id, metadata in stored_files.items():
            # Get PDF content
            pdf_content = document_store.get_document(file_id)
            if not pdf_content:
                continue
                
            # Read PDF content
            pdf_file = io.BytesIO(pdf_content)
            pdf = PdfReader(pdf_file)
            
            # Extract text from each page
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text.strip():  # Skip empty pages
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={
                                "document_id": file_id,
                                "source": metadata.get("filename", "unknown"),
                                "page": i + 1,
                                "total_pages": len(pdf.pages)
                            }
                        )
                    )
            
            # Split pages into chunks
            if pages:
                doc = self.text_splitter.split_documents(pages)
                documents.extend(doc)
        
        if not documents:
            raise ValueError("No valid chunks found for BM25 indexing")
        
        # Initialize BM25 retriever
        self._retriever = LangChainBM25Retriever.from_documents(
            documents,
            k=self.k
        )
        
        # Setup RAG chain
        self._chain = create_rag_chain(
            self._retriever,
            openai_api_key,
            callbacks=callbacks,
            **kwargs
        )
        
    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieved documents
        """
        self._validate_setup()
        return await self._safe_retrieve(query, **kwargs)
        
    async def run(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the complete retrieval and answer generation chain.
        
        Args:
            query: User question
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
            - answer: Generated response
            - strategy: Name of the strategy
        """
        self._validate_setup()
        
        # Get response using RAG chain with proper input format
        result = await self._chain.ainvoke({"question": query})
        
        return {
            "answer": result,
            "strategy": self.name
        }
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query.
        
        This method is required by the LangChain retriever interface.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        return self._retriever.get_relevant_documents(query)
        
    @property
    def name(self) -> str:
        """Return strategy name."""
        return "bm25" 