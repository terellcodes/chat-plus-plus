import uuid
import io
from datetime import datetime
from typing import BinaryIO, Optional, List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from core.vector_store import vector_store
from core.document_store import document_store
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
        openai_api_key: str,
        strategy_names: Optional[List[str]] = None,
        **kwargs
    ) -> UploadDocumentResponse:
        """Process a PDF file and store it in the vector store
        
        Args:
            file: PDF file object
            filename: Original filename
            openai_api_key: OpenAI API key
            strategy_names: Optional list of strategies to initialize
            **kwargs: Additional strategy parameters
        """
        print(f"\n📄 Processing PDF file: {filename}")
        
        try:
            # Generate unique ID for the document
            doc_id = str(uuid.uuid4())
            
            # Read and store PDF content
            print("💾 Storing PDF content...")
            file_content = file.read()
            document_store.add_document(
                doc_id,
                file_content,
                metadata={
                    "filename": filename,
                    "upload_timestamp": datetime.utcnow().isoformat()
                }
            )
            print("✅ PDF content stored")

            # Create in-memory file and read PDF
            print("📚 Loading PDF content...")
            pdf_file = io.BytesIO(file_content)
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
                                "source": filename,
                                "page": i + 1,
                                "total_pages": len(pdf.pages),
                                "document_id": doc_id
                            }
                        )
                    )
            
            # Get total pages
            total_pages = len(pdf.pages)
            print(f"📋 PDF loaded successfully. Total pages: {total_pages}")
            
            # Split into chunks
            print("✂️  Splitting document into chunks...")
            chunks = self.text_splitter.split_documents(pages)
            print(f"📝 Document split into {len(chunks)} chunks")
            
            # Verify chunks have content
            if not chunks:
                print("❌ Error: No content extracted from PDF")
                document_store.delete_document(doc_id)
                raise ValueError("No content extracted from PDF")
            
            # Prepare texts and metadata
            documents = []
            metadatas = []
            skipped_chunks = 0
            
            print("🔍 Processing document chunks...")
            for i, chunk in enumerate(chunks, 1):
                if not chunk.page_content or not isinstance(chunk.page_content, str):
                    skipped_chunks += 1
                    print(f"⚠️  Skipping invalid chunk {i}")
                    continue
                    
                documents.append(chunk.page_content)
                metadatas.append(
                    DocumentMetadata(
                        source=filename,
                        page=chunk.metadata.get("page", 1),
                        total_pages=total_pages,
                        document_id=doc_id
                    ).dict()
                )
            
            if skipped_chunks:
                print(f"⚠️  Skipped {skipped_chunks} invalid chunks")
                
            if not documents:
                print("❌ Error: No valid content chunks found in PDF")
                document_store.delete_document(doc_id)
                raise ValueError("No valid content chunks found in PDF")
            
            # Create embeddings
            print("🧮 Generating embeddings...")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
            vectors = embeddings.embed_documents(documents)
            print(f"✨ Generated {len(vectors)} embeddings")
            
            # Store in vector store
            print(f"💽 Storing documents in vector store...")
            vector_store.add_texts(documents, metadatas, vectors)
            
            # Initialize requested strategies if provided
            if strategy_names:
                print(f"🔄 Initializing requested strategies: {strategy_names}")
                from services.retrieval import RetrievalService
                retrieval_service = RetrievalService()
                await retrieval_service.initialize_strategies(
                    strategy_names,
                    openai_api_key,
                    **kwargs
                )
            
            print(f"✅ Successfully processed PDF. Total valid chunks: {len(documents)}")
            return UploadDocumentResponse(
                document_id=doc_id,
                filename=filename,
                total_chunks=len(documents),
                upload_timestamp=datetime.utcnow(),
                status="ready"
            )
            
        except Exception as e:
            # Clean up stored document if processing failed
            if 'doc_id' in locals():
                try:
                    document_store.delete_document(doc_id)
                    print(f"🧹 Cleaned up stored document after error")
                except Exception as cleanup_error:
                    print(f"⚠️  Failed to cleanup stored document: {str(cleanup_error)}")
            
            print(f"❌ Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}") 