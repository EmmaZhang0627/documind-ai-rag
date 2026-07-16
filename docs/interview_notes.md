# DocuMind Interview Notes

## 60-Second Project Introduction

DocuMind is a FastAPI-based RAG document question-answering system for PDFs. It supports PDF upload, page-aware parsing, chunking, OpenAI embeddings, hybrid retrieval with embedding similarity and BM25, optional reranking with fallback behavior, confidence gating, LLM answer generation, and source citations. I refactored the system into thin routers, centralized configuration, dependency providers, service-layer components, structured trace logging, and a lightweight evaluation runner so the project is easier to debug, test, and extend.

## System Design Explanation

The system is split into clear layers:

- API layer: handles HTTP requests and responses
- config layer: centralizes runtime settings
- dependency provider: creates and caches `RAGService`
- service layer: implements RAG capabilities
- `RAGService`: orchestrates the pipeline
- observability layer: writes trace logs
- eval layer: runs local regression cases

This separation keeps the project maintainable. Routers do not know how embedding or retrieval works. `RAGService` does not know how HTTP requests are parsed. Individual services can be replaced independently.

## Why Hybrid Retrieval Was Used

Hybrid retrieval combines semantic search and keyword search:

```text
embedding similarity + BM25 keyword score
```

Embedding similarity helps with paraphrases and semantic meaning. BM25 helps with exact terms, names, policy numbers, and keyword-heavy questions. Weighted fusion gives the system a better first-stage retrieval signal than either method alone.

## Why Confidence Gating Was Added

The confidence gate prevents the LLM from answering when retrieved evidence is weak.

If the top score is below the threshold, the system returns a low-confidence refusal instead of forcing an answer. This reduces hallucination risk and avoids unnecessary LLM calls.

Trade-off:

- threshold too low can allow unsupported answers
- threshold too high can block valid answers
- the threshold should be tuned with traces and evaluation cases

## Why Routers Were Kept Thin

Routers should represent the API boundary, not the RAG workflow.

In DocuMind:

- `chat.py` validates the chat request, calls `RAGService.ask()`, and returns the response.
- `documents.py` validates PDF upload, parses the file, chunks text, and delegates ingestion.
- RAG logic stays in the service layer.

This makes the API easier to read and the business logic easier to test.

## Why The Service Layer Was Introduced

The service layer separates capabilities:

- `EmbeddingService`: embedding generation
- `RetrievalService`: candidate retrieval
- `RerankService`: reranking or fallback ranking
- `LLMService`: answer generation
- `RAGService`: orchestration

This makes the system easier to maintain and replace. For example, replacing the in-memory vector store with Chroma or Qdrant should mostly affect the retrieval layer, not every router or the whole pipeline.

## Why Trace Logging Matters

RAG failures are not always LLM failures. A wrong answer may come from parsing, chunking, retrieval, reranking, thresholding, context building, or generation.

Each chat request gets a `trace_id`. The system logs structured JSONL traces with:

- retrieved candidate count
- source metadata
- embedding score
- BM25 score
- final score
- rerank score
- confidence decision
- `llm_called`
- final status

This allows engineers to debug by trace ID instead of guessing.

## Why The Evaluation Dataset Matters

The evaluation dataset defines expected RAG behavior:

- grounded questions should be answered with sources
- unrelated questions should be refused
- expected keywords should appear
- citations should come from expected files or pages when specified

This protects against regressions when changing retrieval weights, rerankers, thresholds, prompts, or models.

## Current Limitations

- The vector store is in-memory and not production-ready.
- Indexed documents are lost on process restart.
- Chroma, Qdrant, or pgvector should replace the in-memory store in a production version.
- Cross-encoder reranking is optional and may fall back locally because of PyTorch / Windows dependency issues.
- JSONL trace logging is an MVP observability solution.
- The eval dataset is lightweight and should be expanded with real domain-reviewed cases.
- The eval runner does not yet use RAGAS, LLM-as-judge, or CI.
- Authentication, authorization, persistence, and deployment hardening are not implemented in this phase.

## Future Improvements

- Replace in-memory storage with a persistent vector database.
- Persist document, chunk, embedding, and metadata records.
- Add authentication and document-level access control.
- Add production observability with metrics, latency, token usage, and dashboards.
- Expand evaluation with real user questions and bug-derived regression cases.
- Add CI-based eval checks for critical cases.
- Improve prompt templates and answer citation formatting.
- Make reranker deployment more reliable outside local Windows development.

## Chinese Interview Answer

```text
DocuMind 是一个基于 FastAPI 的 PDF 文档问答 RAG 系统。它支持 PDF 上传、按页解析、文本切块、OpenAI embedding、混合检索、rerank fallback、confidence gate、LLM 生成答案和 source citation。

我在 Phase 4 里重点做了工程化重构：把 FastAPI router 保持成 thin controller，把配置集中到 settings 层，把 RAGService 通过 dependency provider 创建，并且把 embedding、retrieval、rerank、LLM generation 拆成独立 service。这样每一层职责更清楚，后续替换模型、向量数据库或 reranker 时，不需要重写整个系统。

检索层使用 hybrid retrieval，把 embedding similarity 和 BM25 keyword score 做加权融合。这样既能处理语义相似问题，也能处理关键词比较强的问题。之后系统会尝试 rerank，如果本地 cross-encoder 或 PyTorch 环境不可用，就 fallback 到 retrieval score，保证 API 不会因为 reranker 失败而不可用。

我还加入了 confidence gate。当 top score 低于阈值时，系统会拒答，而不是强行调用 LLM。这可以减少幻觉，也能减少不必要的 LLM 调用。

为了方便 debug，我给每个 chat request 加了 trace_id，并把 retrieval、rerank、confidence decision、llm_called、status 等信息写入 JSONL trace。这样如果用户说答案不对，可以直接用 trace_id 查完整链路。

最后，我加了轻量 eval dataset 和 run_eval.py。它会跑固定问题，检查是否 answered、是否有 sources、关键词是否存在、无关问题是否 low_confidence。这个 eval 层的作用是防止后续调整 retrieval、rerank、threshold、prompt 或模型时，把原本正常的行为改坏。
```

## English Interview Answer

```text
DocuMind is a FastAPI-based RAG system for PDF document question answering. It supports PDF upload, page-aware parsing, chunking, OpenAI embeddings, hybrid retrieval, reranking with fallback behavior, confidence gating, LLM answer generation, and source citations.

In Phase 4, I focused on engineering the system into maintainable layers. FastAPI routers are thin controllers, runtime configuration is centralized in a settings layer, RAGService is created through a dependency provider, and embedding, retrieval, reranking, and LLM generation are separated into dedicated services. This makes the system easier to debug, test, and replace component by component.

For retrieval, I used hybrid search by combining embedding similarity and BM25 keyword score. Embeddings help with semantic matching, while BM25 helps with exact terms and keyword-heavy questions. The scores are fused with configurable weights.

Reranking is supported, but it has fallback behavior because cross-encoder and PyTorch dependencies can fail in a local Windows environment. If reranking is unavailable, the system falls back to retrieval-score ordering so the API remains usable.

I added a confidence gate to reduce hallucinations. If the top retrieved score is below the threshold, the system returns a low-confidence refusal instead of calling the LLM.

I also added request-level observability. Each chat request receives a trace_id, and the system writes structured JSONL traces with retrieval scores, rerank information, confidence decisions, source metadata, whether the LLM was called, and final status. This makes debugging much more systematic.

Finally, I added a lightweight evaluation dataset and local regression runner. It checks whether grounded questions are answered with sources and whether unrelated questions are refused. This helps protect the RAG pipeline from regressions when changing retrieval weights, rerankers, thresholds, prompts, or models.
```
