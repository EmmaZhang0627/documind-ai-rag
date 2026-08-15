import re
from collections.abc import Callable

from openai import OpenAI

from app.config.settings import AppSettings


QueryRewriteModel = Callable[[str], str]
VAGUE_REFERENCE_PATTERN = re.compile(
    r"\b(?:it|its|that|this|they|them|those|these|the former|the latter|what about|how about)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
CAPITALIZED_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9-]*\b")
GENERIC_CAPITALIZED_TOKENS = {"What", "Which", "Who", "When", "Where", "Why", "How", "Can", "Could", "Does", "Do", "Is", "Are"}
ANSWER_LIKE_PREFIXES = (
    "the answer is",
    "answer:",
    "based on the",
)


def create_openai_query_rewriter(settings: AppSettings) -> QueryRewriteModel:
    client = None

    def rewrite(query: str) -> str:
        nonlocal client
        if not settings.openai_api_key:
            return query
        if client is None:
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.openai_timeout_seconds,
            )
        response = client.responses.create(
            model=settings.chat_model_name,
            input=f"""
Rewrite the user text as one concise standalone search query for document retrieval.

Rules:
- Preserve the user's exact intent and constraints.
- Do not answer the question.
- Do not add facts, dates, numbers, names, or entities not present in the input.
- Do not add synonyms merely to make the query longer.
- If a reference cannot be resolved from this input alone, keep it unchanged.
- If the input is already a clear retrieval query, return it unchanged.
- Return only the query, on one line, with no label or explanation.

User text:
{query}
"""
        )
        return response.output_text

    return rewrite


class QueryRewriteService:
    def __init__(
        self,
        rewriter: QueryRewriteModel,
        enabled: bool = True,
        minimum_natural_language_words: int = 9,
    ) -> None:
        self.rewriter = rewriter
        self.enabled = enabled
        self.minimum_natural_language_words = minimum_natural_language_words

    def should_rewrite(self, query: str) -> bool:
        if not self.enabled:
            return False
        words = re.findall(r"\b[\w'-]+\b", query)
        return bool(VAGUE_REFERENCE_PATTERN.search(query)) or (
            len(words) >= self.minimum_natural_language_words
            and query.rstrip().endswith("?")
        )

    def rewrite(self, query: str) -> str:
        original = " ".join(query.split())
        if not self.should_rewrite(original):
            return original
        try:
            rewritten = " ".join(self.rewriter(original).split()).strip(" `\"'")
        except Exception:
            return original
        if not self._is_valid(original, rewritten):
            return original
        return rewritten

    def _is_valid(self, original: str, rewritten: str) -> bool:
        if not rewritten or len(rewritten) > 500:
            return False
        if len(rewritten) > max(120, len(original) * 3):
            return False
        if rewritten.casefold().startswith(ANSWER_LIKE_PREFIXES):
            return False
        original_numbers = set(NUMBER_PATTERN.findall(original))
        rewritten_numbers = set(NUMBER_PATTERN.findall(rewritten))
        if not rewritten_numbers <= original_numbers:
            return False
        original_entities = set(CAPITALIZED_TOKEN_PATTERN.findall(original))
        rewritten_entities = (
            set(CAPITALIZED_TOKEN_PATTERN.findall(rewritten))
            - GENERIC_CAPITALIZED_TOKENS
        )
        return rewritten_entities <= original_entities
