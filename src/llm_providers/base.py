from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def generate_completion(self, prompt: str, system_message: str) -> str:
        """Generate a completion for the given prompt."""
        pass
