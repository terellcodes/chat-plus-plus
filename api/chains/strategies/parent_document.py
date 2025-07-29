"""Parent Document retrieval strategy using LangChain's ParentDocumentRetriever."""
from typing import List, Dict, Any, Optional
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older versions
    from langchain.schema import Document

from qdrant_client import QdrantClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import Distance, VectorParams

try:
    # For older versions (langchain==0.1.0)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for newer versions
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from chains.retrieval import RAG_TEMPLATE
from .base import BaseRetrievalStrategy


class ParentDocumentRetrieval(BaseRetrievalStrategy):
    """
    Parent Document retrieval strategy using LangChain's ParentDocumentRetriever.
    
    This implementation follows the pattern from Advanced_Retrieval_with_LangChain_Assignment.ipynb:
    1. Creates separate Qdrant collection for child chunks
    2. Uses InMemoryStore for parent documents
    3. Search small chunks, return big documents (small-to-big strategy)
    4. Integrates with session-based architecture for infrastructure reuse
    """
    
    def __init__(self, chunk_size: int = 750, k: int = 5):
        """Initialize Parent Document retrieval strategy.
        
        Args:
            chunk_size: Size of child chunks for embedding
            k: Number of parent documents to retrieve
        """
        self.chunk_size = chunk_size
        self.k = k
        self._retriever = None
        self._chain = None
        print(f"🏗️  Initializing Parent Document retrieval with chunk_size={chunk_size}, k={k}")
    
    async def setup(
        self,
        vector_store: Any,
        openai_api_key: str,
        document: Optional[Document] = None,
        session_id: Optional[str] = None,
        parent_infrastructure: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Set up Parent Document retrieval strategy.
        
        Args:
            vector_store: Not used by Parent Document (for interface compatibility)
            openai_api_key: OpenAI API key for embeddings and chat model
            document: Full document from session cache
            session_id: Session identifier for collection naming
            parent_infrastructure: Pre-created infrastructure from session manager
            **kwargs: Additional parameters
        """
        print(f"🔧 Setting up Parent Document retrieval strategy")
        
        if not document:
            raise ValueError("Document is required for Parent Document setup")
        
        if not session_id:
            raise ValueError("Session ID is required for Parent Document setup")
            
        print(f"📚 Using document with {len(document.page_content)} characters for parent document setup")

        # Get or create parent document infrastructure
        if parent_infrastructure:
            print(f"📚 Reusing existing parent document infrastructure for session {session_id}")
            vectorstore = parent_infrastructure['vectorstore']
            docstore = parent_infrastructure['docstore']
        else:
            print(f"🔧 Creating new parent document infrastructure for session {session_id}")
            
            # Create embeddings for child chunks
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
            
            # Create empty Qdrant vector store for child chunks
            # We'll let ParentDocumentRetriever populate it with child chunks
            collection_name = f"parent_docs_{session_id}"
            client = QdrantClient(":memory:")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # size depends on your embedding model
            )

            vectorstore = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings,
            )
            
            # Create InMemoryStore for parent documents
            docstore = InMemoryStore()
            
            print(f"✅ Created parent document infrastructure with collection '{collection_name}'")
        
        # Create child splitter following notebook pattern
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size*4)
        # Create ParentDocumentRetriever
        self._retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        print(f"📄 Adding document to parent document retriever ({len(document.page_content)} characters)")
        
        # Add the document - this will create child chunks and store parent document
        self._retriever.add_documents([document])
        
        print("✅ Document added to parent document retriever")
        
        # Set up RAG chain using the same pattern as the notebook
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        chat_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
        
        self._chain = (
            {"context": itemgetter("question") | self._retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
        )
        
        print("✅ Parent Document strategy setup complete")
    
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """Retrieve parent documents using child chunk similarity search.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            List of parent documents (not child chunks)
        """
        if not self._retriever:
            raise ValueError("Parent Document strategy not properly initialized")
        
        print(f"🔍 Parent Document retrieving documents for: '{query}'")
        
        # Use ParentDocumentRetriever's get_relevant_documents method
        # This searches child chunks but returns parent documents
        documents = self._retriever.get_relevant_documents(query)
        collection_name = self._retriever.vectorstore.collection_name
        print(f"Vector count: {self._retriever.vectorstore.client.get_collection(collection_name).vectors_count}")
        
        print(f"📄 Parent Document retrieved {len(documents)} parent documents")
        print(f"📄 Parent Document retrieved documents: {documents}")
        return documents
    
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run the complete Parent Document retrieval chain.
        
        Args:
            query: User's question
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not self._chain:
            raise ValueError("Parent Document chain not initialized")
        
        print(f"🚀 Running Parent Document chain for query: '{query}'")
        
        # Invoke the chain exactly like in the notebook
        result = self._chain.invoke({"question": query})
        
        return {
            "answer": result["response"].content,
            "strategy": self.name,
            "retrieval_method": "Parent Document with small-to-big strategy",
            "context": result["context"]
        }
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query (synchronous interface).
        
        This method is required for LangChain retriever interface compatibility.
        
        Args:
            query: Search query
            
        Returns:
            List of parent documents
        """
        if not self._retriever:
            raise ValueError("Parent Document strategy not properly initialized")
        
        return self._retriever.get_relevant_documents(query)
    
    @property
    def name(self) -> str:
        """Strategy name identifier."""
        return "parent_document_retrieval"