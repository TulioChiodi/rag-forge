from openai import AsyncOpenAI
from .base import LLMProvider
from src.config import settings
from src.logging_conf import logger

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = settings.OPENAI_CHAT_MODEL
    
    async def generate_completion(self, prompt: str, system_message: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=settings.DEEPSEEK_BASE_URL)
        self.model = settings.DEEPSEEK_CHAT_MODEL
    
    async def generate_completion(self, prompt: str, system_message: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise