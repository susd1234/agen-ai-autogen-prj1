import os
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API Keys Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    serper_api_key: Optional[str] = Field(None, env="SERPER_API_KEY")

    # Application Configuration
    app_name: str = "Agentic RAG Application"
    debug: bool = Field(default=False, env="DEBUG")

    # Storage Configuration
    base_dir: str = Field(default=os.path.dirname(os.path.dirname(__file__)))

    @property
    def storage_dir(self) -> str:
        return os.path.join(self.base_dir, "storage")

    @property
    def documents_dir(self) -> str:
        return os.path.join(self.storage_dir, "documents")

    @property
    def vector_db_dir(self) -> str:
        return os.path.join(self.storage_dir, "vector_db")

    # Agent Configuration
    agent_temperature: float = 0.7
    agent_max_tokens: int = 1000
    default_model: str = "llama3.2"
    available_models: List[str] = ["llama3.2", "gpt-4o-mini"]

    # RAG Configuration
    chunk_size: int = 1000  # ~140-150 words per chunk (better context)
    chunk_overlap: int = 200  # ~30 words overlap (maintains context between chunks)

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields in the settings


# Initialize settings
settings = Settings()

# Create necessary directories
os.makedirs(settings.documents_dir, exist_ok=True)
os.makedirs(settings.vector_db_dir, exist_ok=True)

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llama3.2": {"base_url": "http://localhost:11434", "api_key": None},
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_key": settings.openai_api_key,
    },
}

# Export settings for use in other modules
APP_NAME = settings.app_name
DEBUG = settings.debug
STORAGE_DIR = settings.storage_dir
DOCUMENTS_DIR = settings.documents_dir
VECTOR_DB_DIR = settings.vector_db_dir
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
AGENT_CONFIG = {
    "temperature": settings.agent_temperature,
    "max_tokens": settings.agent_max_tokens,
    "model": "llama3.2",
    "available_models": ["llama3.2", "gpt-4o-mini"],
    "model_configs": MODEL_CONFIGS,
}
