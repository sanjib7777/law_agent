from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI Settings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"

    # Gemini Settings
    gemini_api_key: str = ""

    # Qdrant Settings
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "legal_docs"

    class Config:
        env_file = ".env"


settings = Settings()