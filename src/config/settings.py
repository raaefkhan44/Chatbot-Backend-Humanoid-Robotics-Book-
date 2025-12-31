from pydantic_settings import BaseSettings
from typing import Optional
from cohere import Client as CohereClient
from google.generativeai import configure as configure_genai
import google.generativeai as genai


class Settings(BaseSettings):
    # Cohere settings for embeddings
    COHERE_API_KEY: str

    # Gemini settings for conversation
    GEMINI_API_KEY: str

    # Qdrant settings
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None

    # Neon Postgres settings
    NEON_DATABASE_URL: str

    # Context7 MCP Server settings
    CONTEXT7_MCP_SERVER_URL: Optional[str] = None

    # Optional API key for authentication
    API_KEY: Optional[str] = None

    # Application settings
    app_name: str = "RAG Chatbot API"
    debug: bool = False
    environment: str = "development"

    # Model configuration
    EMBEDDING_MODEL: str = "embed-multilingual-v3.0"
    CHAT_MODEL: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def cohere_client(self) -> CohereClient:
        """Create and return a Cohere client instance"""
        return CohereClient(api_key=self.COHERE_API_KEY)

    @property
    def gemini_client(self):
        """Configure and return Gemini client"""
        configure_genai(api_key=self.GEMINI_API_KEY)
        return genai


settings = Settings()