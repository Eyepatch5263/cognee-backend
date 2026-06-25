import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Settings:
    PROJECT_NAME: str = "CogniVerdict Backend"
    VERSION: str = "1.0.0"
    
    # Cognee Cloud configuration
    COGNEE_API_KEY: str = os.getenv("COGNEE_API_KEY", "")
    COGNEE_API_URL: str = os.getenv("COGNEE_API_URL", "https://api.cognee.ai").rstrip("/")
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # Ollama LLM configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    # NVIDIA API configuration
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY") or ""
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen3-next-80b-a3b-instruct")

    @property
    def has_cognee_credentials(self) -> bool:
        return bool(self.COGNEE_API_KEY)

settings = Settings()
