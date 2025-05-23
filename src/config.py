from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    # LLM Provider settings
    PRIMARY_LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str
    DEEPSEEK_API_KEY: str
    
    # Model settings
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    DEEPSEEK_CHAT_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    
    # Embedding settings - separate from LLM choice
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 3072  # OpenAI text-embedding-3-large dimensions
    
    # Elasticsearch settings
    ES_URL: str = "http://localhost:9200"
    ES_INDEX: str = "tractian_docs"
    
    # Apitally settings
    APITALLY_CLIENT_ID: str
    APITALLY_ENVIRONMENT: str = "dev"
    APITALLY_APP_NAME: str = "tractian-rag"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a global settings object
settings = Settings()