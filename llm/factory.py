from app.config import Settings
from llm.base import DisabledLLMProvider
from llm.openai_compatible import OpenAICompatibleLLMProvider


def create_llm_provider(settings: Settings):
    if not settings.llm_enabled:
        return DisabledLLMProvider()
    return OpenAICompatibleLLMProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
