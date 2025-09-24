import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="rag_system", env="POSTGRES_DB")
    postgres_user: str = Field(default="rag_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="rag_password", env="POSTGRES_PASSWORD")

    # Vector Database
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection: str = Field(default="pdf_documents", env="QDRANT_COLLECTION")

    # LLM
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")

    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # Application
    chunk_size: int = Field(default=400, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    top_k_default: int = Field(default=5, env="TOP_K_DEFAULT")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    polling_interval: int = Field(default=2, env="POLLING_INTERVAL")

    @property
    def database_url(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()