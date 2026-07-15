from app.services.rag_types import Candidate, RerankerModel


class RerankService:
    def __init__(self, reranker: RerankerModel) -> None:
        self.reranker = reranker

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        return self.reranker(query, candidates)
