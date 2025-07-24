"""LangSmith initialization and configuration"""
from langsmith import Client
from config.settings import get_settings

settings = get_settings()

def init_langsmith():
    """Initialize LangSmith client and configure tracing"""
    if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
        try:
            client = Client()
            print(f"Successfully initialized LangSmith client for project: {settings.LANGSMITH_PROJECT}")
            return client
        except Exception as e:
            print(f"Error initializing LangSmith client: {e}")
            return None
    return None 