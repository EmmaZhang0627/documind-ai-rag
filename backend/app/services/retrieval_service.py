from app.services.rag_types import Candidate, Chunk, VectorStore
from app.services.vector_db import (
    add_chunks_to_db,
    clear_vector_store,
    retrieve_candidates,
    vector_store,
)


class InMemoryVectorStore:
    def add(self, chunks: list[Chunk]) -> None:
        add_chunks_to_db(chunks)

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        return retrieve_candidates(query_embedding, query_text)[:top_k]

    def count(self) -> int:
        return len(vector_store)

    def clear(self) -> None:
        clear_vector_store()


class RetrievalService:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def add(self, chunks: list[Chunk]) -> None:
        self.vector_store.add(chunks)

    def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> list[Candidate]:
        return self.vector_store.search(query_embedding, query_text, top_k)

    def count(self) -> int:
        return self.vector_store.count()

    def clear(self) -> None:
        self.vector_store.clear()
