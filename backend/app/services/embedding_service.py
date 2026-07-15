from app.services.rag_types import EmbeddingModel


class EmbeddingService:
    def __init__(self, embed_model: EmbeddingModel) -> None:
        self.model = embed_model

    def embed(self, text: str) -> list[float]:
        return self.model(text)
