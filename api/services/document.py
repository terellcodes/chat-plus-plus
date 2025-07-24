import uuid
from datetime import datetime
from typing import BinaryIO
from pathlib import Path
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
        print(f"\n📄 Processing PDF file: {filename}")
        
        try:
            # Save temporary file
            temp_path = f"/tmp/{uuid.uuid4()}.pdf"
            print(f"💾 Saving temporary file: {temp_path}")
            
            with open(temp_path, "wb") as f:
                file_content = file.read()
                f.write(file_content)
            print("✅ Temporary file saved")

            # Load and split document
            print("📚 Loading PDF with PyPDFLoader...")
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            
            # Get total pages
            total_pages = len(pages)
            print(f"📋 PDF loaded successfully. Total pages: {total_pages}")
            
            # Split into chunks
            print("✂️  Splitting document into chunks...")
            chunks = self.text_splitter.split_documents(pages)
            print(f"📝 Document split into {len(chunks)} chunks")
            
            # Verify chunks have content
            if not chunks:
                print("❌ Error: No content extracted from PDF")
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
                        total_pages=total_pages
                    ).dict()
                )
            
            if skipped_chunks:
                print(f"⚠️  Skipped {skipped_chunks} invalid chunks")
                
            if not documents:
                print("❌ Error: No valid content chunks found in PDF")
                raise ValueError("No valid content chunks found in PDF")
            
            # Create embeddings
            print("🧮 Generating embeddings...")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
            vectors = embeddings.embed_documents(documents)
            print(f"✨ Generated {len(vectors)} embeddings")
            
            # Store in vector store
            doc_id = str(uuid.uuid4())
            print(f"💽 Storing documents in vector store...")
            vector_store.add_texts(documents, metadatas, vectors)
            
            print(f"✅ Successfully processed PDF. Total valid chunks: {len(documents)}")
            return UploadDocumentResponse(
                document_id=doc_id,
                filename=filename,
                total_chunks=len(documents),
                upload_timestamp=datetime.utcnow(),
                status="ready"
            )
            
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
        
        finally:
            # Cleanup temporary file
            try:
                Path(temp_path).unlink()
                print(f"🧹 Cleaned up temporary file")
            except Exception as e:
                print(f"⚠️  Failed to cleanup temporary file: {str(e)}") 