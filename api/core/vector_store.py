from typing import Optional, List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

class VectorStore:
    """Singleton vector store using QDrant"""
    _instance: Optional['VectorStore'] = None
    _client: Optional[QdrantClient] = None
    _collection_name: str = "documents"
    _dimension: int = 1536  # OpenAI embedding dimension

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            # Initialize in-memory QDrant client
            cls._client = QdrantClient(":memory:")
            
            print("Creating new QDrant collection...")
            # Create collection with proper configuration
            cls._client.recreate_collection(
                collection_name=cls._collection_name,
                vectors_config=VectorParams(
                    size=cls._dimension, 
                    distance=Distance.COSINE,
                ),
                # Optionally configure HNSW index params
                hnsw_config=models.HnswConfigDiff(
                    m=16,  # Number of edges per node in the index graph
                    ef_construct=100,  # Size of the dynamic candidate list
                    full_scan_threshold=10000  # When to switch to full scan
                )
            )
            print("QDrant collection created successfully")
        return cls._instance

    def get_client(self) -> QdrantClient:
        """Get the QDrant client instance"""
        return self._client

    def get_langchain_store(self, openai_api_key: str) -> Qdrant:
        """Get a LangChain Qdrant wrapper with embeddings"""
        print("Creating Qdrant store with embeddings...")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        store = Qdrant(
            client=self._client,
            collection_name=self._collection_name,
            embeddings=embeddings
        )
        print("Qdrant store created successfully")
        return store

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[dict],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Add texts with their embeddings to the vector store"""
        print(f"Adding {len(texts)} texts to vector store")
        
        if not texts or not metadatas or not embeddings:
            print("❌ Error: Empty texts, metadatas, or embeddings")
            raise ValueError("Cannot add empty texts to vector store")
            
        if not (len(texts) == len(metadatas) == len(embeddings)):
            print("❌ Error: Mismatched lengths of texts, metadatas, and embeddings")
            raise ValueError("Texts, metadatas, and embeddings must have the same length")
        
        try:
            # Generate proper UUIDs for the points
            ids = [str(uuid.uuid4()) for _ in texts]
            print(f"Generated {len(ids)} UUIDs")
            
            # Create points with page_content in payload
            points = []
            for id_, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
                # Ensure metadata has required fields
                if 'page' not in metadata:
                    print(f"⚠️  Warning: Missing page number in metadata for document {id_}")
                    metadata['page'] = 1
                
                # Create point with page_content
                point = PointStruct(
                    id=id_,
                    payload={
                        "page_content": text,  # Store as page_content for LangChain compatibility
                        "text": text,          # Keep original text field
                        **metadata
                    },
                    vector=embedding
                )
                points.append(point)
            
            print(f"Created {len(points)} points with proper payload structure")
            
            # Add points to the collection
            print("Upserting points to collection...")
            self._client.upsert(
                collection_name=self._collection_name,
                points=points
            )
            print("Successfully added points to vector store")
            return ids
            
        except Exception as e:
            print(f"❌ Error adding texts to vector store: {str(e)}")
            raise ValueError(f"Failed to add texts to vector store: {str(e)}")

vector_store = VectorStore() 