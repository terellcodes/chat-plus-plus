import uuid
from datetime import datetime
from typing import BinaryIO
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from core.vector_store import vector_store
from models.schemas.document import UploadDocumentResponse, DocumentMetadata

class DocumentService:
    """Service for handling document operations"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    async def process_pdf(
        self,
        file: BinaryIO,
        filename: str,
        openai_api_key: str
    ) -> UploadDocumentResponse:
        """Process a PDF file and store it in the vector store"""
        # Save temporary file
        temp_path = f"/tmp/{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        # Load and split document
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        # Get total pages
        total_pages = len(pages)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)
        
        # Prepare texts and metadata
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [
            DocumentMetadata(
                source=filename,
                page=chunk.metadata.get("page"),
                total_pages=total_pages
            ).dict()
            for chunk in chunks
        ]
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        vectors = embeddings.embed_documents(documents)
        # Store in vector store
        doc_id = str(uuid.uuid4())
        vector_store.add_texts(documents, metadatas, vectors)
        
        return UploadDocumentResponse(
            document_id=doc_id,
            filename=filename,
            total_chunks=len(chunks),
            upload_timestamp=datetime.utcnow(),
            status="ready"  # Changed from "success" to "ready"
        ) 