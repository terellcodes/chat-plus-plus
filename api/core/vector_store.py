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
        return cls._instance

    def get_client(self) -> QdrantClient:
        """Get the QDrant client instance"""
        return self._client

    def get_langchain_store(self, openai_api_key: str) -> Qdrant:
        """Get a LangChain Qdrant wrapper with embeddings"""
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return Qdrant(
            client=self._client,
            collection_name=self._collection_name,
            embeddings=embeddings
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[dict],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Add texts with their embeddings to the vector store"""
        print(f"Adding {len(texts)} texts to vector store")
        
        # Generate proper UUIDs for the points
        ids = [str(uuid.uuid4()) for _ in texts]
        print(f"Generated {len(ids)} UUIDs")
        
        # Add points to the collection
        print("Upserting points to collection...")
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=id_,
                    payload={"text": text, **metadata},
                    vector=embedding
                )
                for id_, text, metadata, embedding 
                in zip(ids, texts, metadatas, embeddings)
            ]
        )
        print("Successfully added points to vector store")
        
        return ids

vector_store = VectorStore() 