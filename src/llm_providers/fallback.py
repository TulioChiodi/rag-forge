from .base import LLMProvider
from .providers import OpenAIProvider, DeepSeekProvider
from src.config import settings
from src.logging_conf import logger

class FallbackLLMProvider(LLMProvider):
    def __init__(self):
        self.primary_provider = OpenAIProvider(settings.OPENAI_API_KEY)
        self.fallback_provider = DeepSeekProvider(settings.DEEPSEEK_API_KEY)
        self.primary_name = "OpenAI"
        self.fallback_name = "DeepSeek"

        # Optionally flip order based on configuration
        if settings.PRIMARY_LLM_PROVIDER == "deepseek":
            self.primary_provider, self.fallback_provider = self.fallback_provider, self.primary_provider
            self.primary_name, self.fallback_name = self.fallback_name, self.primary_name

    async def generate_completion(self, prompt: str, system_message: str) -> str:
        try:
            logger.info(f"Attempting to generate completion with {self.primary_name}")
            result = await self.primary_provider.generate_completion(prompt, system_message)
            logger.info(f"Successfully generated completion using {self.primary_name}")
            return result
        except Exception as e:
            logger.warning(f"{self.primary_name} failed with error: {str(e)}. Falling back to {self.fallback_name}.")
            try:
                result = await self.fallback_provider.generate_completion(prompt, system_message)
                logger.info(f"Successfully generated completion using fallback provider {self.fallback_name}")
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback provider {self.fallback_name} also failed: {str(fallback_error)}")
                raise Exception(f"Both LLM providers failed. Primary error: {str(e)}, Fallback error: {str(fallback_error)}")
