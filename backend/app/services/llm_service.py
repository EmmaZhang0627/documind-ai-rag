from openai import OpenAI

from app.config.settings import AppSettings
from app.services.errors import ServiceConfigurationError
from app.services.rag_types import LLMModel


def create_openai_llm(settings: AppSettings) -> LLMModel:
    client = None

    def generate(query: str, context: str) -> str:
        nonlocal client

        if not settings.openai_api_key:
            raise ServiceConfigurationError(
                "OPENAI_API_KEY is required for answer generation."
            )

        if client is None:
            client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
            )

        response = client.responses.create(
            model=settings.chat_model_name,
            input=f"""
You are a helpful assistant for an enterprise document knowledge base.

Answer the user's question using only the context below.
If the answer cannot be found in the context, say:
"I cannot find the answer in the uploaded documents."

Context:
{context}

Question:
{query}
"""
        )

        return response.output_text

    return generate


def generate_llm_answer(query: str, context: str) -> str:
    settings = AppSettings.from_env()
    llm = create_openai_llm(settings)
    return llm(query, context)


class LLMService:
    def __init__(self, llm: LLMModel) -> None:
        self.llm = llm

    def generate(self, query: str, context: str) -> str:
        return self.llm(query, context)
