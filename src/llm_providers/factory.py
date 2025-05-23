from .base import LLMProvider
from .fallback import FallbackLLMProvider

def create_llm_provider() -> LLMProvider:
    """Factory function to create the appropriate LLM provider based on settings."""
    return FallbackLLMProvider() 