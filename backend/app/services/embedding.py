from openai import OpenAI

from app.config.settings import AppSettings
from app.services.errors import ServiceConfigurationError


def create_openai_embedding_model(settings: AppSettings):
    client = None

    def embed(text: str):
        nonlocal client

        if not settings.openai_api_key:
            raise ServiceConfigurationError(
                "OPENAI_API_KEY is required for embedding generation."
            )

        if client is None:
            client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
            )

        response = client.embeddings.create(
            model=settings.embedding_model_name,
            input=text,
        )

        return response.data[0].embedding

    return embed


def get_embedding(text: str):
    settings = AppSettings.from_env()
    embed = create_openai_embedding_model(settings)
    return embed(text)
