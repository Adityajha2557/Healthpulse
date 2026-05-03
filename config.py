"""Application Configuration"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # MongoDB
    MONGODB_URI: str
    
    # Reddit (now optional)
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    
    # News API
    NEWSAPI_KEY: str = ""
    
    # Application
    ENVIRONMENT: str = "development"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
