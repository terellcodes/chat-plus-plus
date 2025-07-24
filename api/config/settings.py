from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "Chat++ API"
    APP_DESCRIPTION: str = "A FastAPI backend for the Chat++ project"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # Local frontend
        "http://localhost:3001",  # Additional local frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://*.vercel.app",   # Vercel preview deployments
        "https://chat-plus-plus.vercel.app"  # Production frontend
    ]
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    ALLOW_CREDENTIALS: bool = True

    # Environment
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache to prevent multiple reads of environment variables
    """
    return Settings() 