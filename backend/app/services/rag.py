from openai import OpenAI

from app.services.embedding import get_embedding
from app.services.vector_db import search

client = OpenAI()


def ask_question(question: str):
    query_embedding = get_embedding(question)

    results = search(query_embedding, top_k=3)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(chunks)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"""
You are a helpful assistant for an enterprise document knowledge base.

Answer the user's question using only the context below.
If the answer cannot be found in the context, say:
"I cannot find the answer in the uploaded documents."

Context:
{context}

Question:
{question}
"""
    )

    return {
        "answer": response.output_text,
        "sources": metadatas,
    }
