from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from config.settings import get_settings, Settings
from utils.constants import ResponseMessage, StatusCode
from services.document import DocumentService
from services.retrieval import RetrievalService
from models.schemas.chat import ChatRequest, ChatResponse
from core.langsmith import init_langsmith
import logging

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize services
document_service = DocumentService()
retrieval_service = RetrievalService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting up...")
    init_langsmith()  # Initialize LangSmith
    yield
    # Shutdown
    print("Shutting down...")


def create_application() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        root_path="/api" if not settings.DEBUG else ""  # Add root_path for production
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=settings.ALLOW_CREDENTIALS,
        allow_methods=settings.ALLOWED_METHODS,
        allow_headers=settings.ALLOWED_HEADERS,
    )

    return app


app = create_application()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": ResponseMessage.SUCCESS,
        "code": StatusCode.HTTP_200_OK,
        "message": "API is healthy"
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Query(None, description="Optional session ID")
):
    """Upload a PDF file and cache it for lazy retriever creation"""
    if not file.filename.endswith('.pdf'):
        print(f"Invalid file type: {file.filename}")
        return {
            "status": ResponseMessage.VALIDATION_ERROR,
            "code": StatusCode.HTTP_400_BAD_REQUEST,
            "message": "Only PDF files are supported"
        }
    
    try:
        print(f"Processing PDF file: {file.filename}")
        result = await document_service.process_pdf(
            file.file,
            file.filename,
            session_id
        )
        print(f"Successfully processed PDF file: {file.filename}")
        return {
            "status": ResponseMessage.SUCCESS,
            "code": StatusCode.HTTP_200_OK,
            "data": result
        }
    except Exception as e:
        print(f"Error processing PDF file: {file.filename}. Error: {str(e)}")
        return {
            "status": ResponseMessage.INTERNAL_ERROR,
            "code": StatusCode.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": str(e)
        }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with session-based RAG support and lazy retriever loading"""
    try:
        print(f"Processing chat request for session {request.session_id} with message: {request.message}")
        result = await retrieval_service.get_response(
            request.openai_api_key,
            request.message,
            request.retrieval_strategies,
            request.session_id
        )
        print("Successfully processed chat request")
        return {
            "status": ResponseMessage.SUCCESS,
            "code": StatusCode.HTTP_200_OK,
            "data": result
        }
    except ValueError as e:
        print(f"Validation error in chat request: {str(e)}")
        return {
            "status": ResponseMessage.VALIDATION_ERROR,
            "code": StatusCode.HTTP_400_BAD_REQUEST,
            "message": str(e)
        }
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return {
            "status": ResponseMessage.INTERNAL_ERROR,
            "code": StatusCode.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": str(e)
        }


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session"""
    try:
        result = await retrieval_service.get_session_info(session_id)
        return {
            "status": ResponseMessage.SUCCESS,
            "code": StatusCode.HTTP_200_OK,
            "data": result
        }
    except ValueError as e:
        return {
            "status": ResponseMessage.VALIDATION_ERROR,
            "code": StatusCode.HTTP_400_BAD_REQUEST,
            "message": str(e)
        }
    except Exception as e:
        return {
            "status": ResponseMessage.INTERNAL_ERROR,
            "code": StatusCode.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": str(e)
        }

@app.get("/strategies")
async def list_strategies():
    """Get list of available retrieval strategies"""
    strategies = await retrieval_service.list_available_strategies()
    return {
        "status": ResponseMessage.SUCCESS,
        "code": StatusCode.HTTP_200_OK,
        "data": {"strategies": strategies}
    }

# Handler for Vercel serverless
handler = Mangum(app)
