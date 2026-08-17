# Metadata and Permission Filtering

## Purpose and architecture position

Enterprise RAG must prevent unauthorized evidence from reaching reranking, traces, context construction, or the LLM:

```text
query + verified access context
-> ACTIVE and permission-eligible chunks
-> Vector/BM25/Hybrid
-> CrossEncoder
-> confidence
-> generation
```

Prompt instructions are not a security boundary. If restricted text is placed in the prompt, the model has already received it and may quote, summarize, infer from, or accidentally expose it.

## Concepts

- **Metadata filtering** selects chunks using stored attributes such as tenant, department, status, date, or document type.
- **RBAC** maps authenticated users and roles to permissions. This MVP does not implement RBAC.
- **Lifecycle filtering** removes non-current material such as `ARCHIVED` versions.
- **Tenant isolation** prevents one customer or organization from retrieving another tenant's data, even when their documents are semantically similar.

## MVP metadata schema

Permission-aware chunks carry all three fields together:

| Field | Meaning |
|---|---|
| `tenant_id` | Exact tenant boundary |
| `department` | Owning department; `all` is shared inside the tenant |
| `access_level` | `public`, `internal`, or `confidential` |

`AccessContext` contains a normalized `tenant_id`, one or more departments, and the user's maximum access level. Access levels are ordered:

```text
public < internal < confidential
```

A chunk is eligible only when all rules pass:

```text
status == ACTIVE
AND chunk.tenant_id == context.tenant_id
AND (chunk.department == all OR chunk.department in context.departments)
AND chunk.access_level <= context.access_level
```

Unknown access levels, empty context fields, and partial permission metadata are rejected. When an access context is supplied, legacy chunks without permission metadata fail closed.

## Current implementation

`RAGService.ask()` and `_retrieve_and_rank()` accept an optional evaluation/demo `AccessContext`. It flows through `RetrievalService` to the vector-store adapter.

- The in-memory store builds Vector and BM25 candidates only from eligible ACTIVE records.
- The Chroma store applies a metadata `where` predicate before vector candidates are returned and builds query-time BM25 scores only from the same eligible corpus.
- CrossEncoder receives only filtered candidates.
- Existing API callers that do not provide an access context retain legacy behavior for compatibility.

The compatibility mode is not a production authorization design. A deployment must derive `AccessContext` from authenticated, server-controlled identity claims; clients must not be allowed to grant themselves tenant or clearance values.

## Important Python code concepts

`@dataclass` automatically generates initialization, representation, and equality behavior for `AccessContext`. `frozen=True` prevents ordinary mutation after creation, reducing the risk that query permissions are accidentally changed during retrieval.

`__post_init__()` runs immediately after the dataclass-generated `__init__()`. It normalizes case and whitespace, removes empty departments, validates required values, and writes the normalized values back with `object.__setattr__()` because the dataclass is frozen.

`frozenset` is an immutable, unordered collection of unique values. It suits departments because order and duplicates do not matter and the access context should remain unchanged. The normalization expression is a two-stage pipeline rather than two nested corpus loops:

```text
each department -> normalize -> remove empty values -> deduplicate -> freeze
```

`@property` exposes a calculated method result like an attribute. `allowed_access_levels` therefore uses `context.allowed_access_levels`, not a method call. Its return type `tuple[str, ...]` means an immutable, ordered tuple containing any number of strings.

The ordered access-level slice implements inherited clearance:

```text
public       -> (public)
internal     -> (public, internal)
confidential -> (public, internal, confidential)
```

Python slices exclude the end index, so `maximum + 1` is required to include the current level.

Chroma permission predicates use:

- `$and`: every condition must pass;
- `$eq`: the metadata value must equal one value;
- `$in`: the metadata value must be one of the allowed values.

This is analogous to SQL `WHERE ... AND ...`, `=`, and `IN (...)`.

## Security boundary and failure handling

The enforceable boundary is the retrieval adapter, before scoring and reranking. Filtering only after TopK would allow restricted chunks to displace accessible evidence and would let restricted text participate in downstream processing.

Important failure modes include:

- filtering after reranking or context construction;
- trusting prompt instructions to conceal evidence;
- checking department but not tenant;
- treating tenant and lifecycle checks as OR conditions;
- assigning permissive defaults to missing metadata;
- letting request payloads choose their own access level;
- normalizing Vector/BM25 over a corpus different from the eligible corpus.

`department="all"` means shared across departments inside the same tenant. It never means shared across tenants. Tenant, department, access level, and lifecycle are independent checks combined with AND.

Permission-aware retrieval is fail closed: incomplete or unknown permission metadata is rejected instead of treated as public. Such documents need metadata migration or explicit review before they can participate in secured retrieval.

## Verification

The focused tests cover allowed metadata, unauthorized exclusion, tenant isolation, department access, ordered access levels, ACTIVE lifecycle composition, partial metadata rejection, and confirmation that restricted chunks never reach the reranker or LLM context.

The deterministic permission benchmark covers six scenarios, including a restricted document with the strongest semantic match and combined ARCHIVED/unauthorized filtering. Its required leakage target is zero.

## Evolution and limitations

This is a local enterprise-aware MVP, not authentication:

1. Prototype: manually supplied access context and centralized rules.
2. Reliable MVP: tested vector-store enforcement and evaluation benchmark (current stage).
3. Deployment-ready: authenticated identity claims, server-derived context, audit logs, deny-by-default API behavior, migration of legacy metadata.
4. Enterprise: SSO/OIDC, centralized RBAC or ABAC policy, policy versioning, revocation, security review, and adversarial authorization testing.

The current MVP does not implement OAuth, JWT verification, SSO, a user database, role inheritance, document-level ACL lists, or a policy engine.

A portfolio-level authenticated extension would normally take roughly 4–10 focused development days: learn JWT/OIDC basics, verify tokens in FastAPI, derive `AccessContext` from server-verified claims, add simple role data, audit authorization decisions, and connect frontend login. A production enterprise authorization platform usually takes weeks or months and requires security and platform review.

## Transferable skills

- **MUST MASTER:** pre-retrieval authorization, fail-closed behavior, tenant isolation, lifecycle-plus-permission composition.
- **PROJECT-LEVEL FAMILIARITY:** metadata predicates and permission-aware lexical scoring.
- **AWARENESS ONLY:** enterprise identity providers, policy engines, complex RBAC/ABAC graphs.

The same boundary applies to AI agents and multimodal systems: unauthorized tools, records, images, or OCR text must be excluded before they are supplied to the model.

## Understanding review

The completed review established these distinctions:

- Metadata filtering executes attribute-based candidate selection.
- RBAC maps users and roles to permissions; it supplies authorization decisions rather than performing semantic retrieval.
- Lifecycle filtering prevents obsolete or archived versions from acting as current evidence.
- Filtering after reranking is too late because restricted text may already reach an external model, logs, traces, caches, and ranking logic.
- A confidential finance chunk from `tenant-a` is accessible only to a context that matches `tenant-a`, includes finance, has confidential clearance, and retrieves ACTIVE documents.

Keep four layers separate in explanations:

```text
authentication: who is the user?
authorization: what may the user access?
metadata enforcement: which chunks are eligible?
RAG quality: which eligible chunks best answer the question?
```

## Interview explanation

**中文：** 我在 RAG 检索层实现了基于 tenant、department 和 access level 的权限过滤，并与 ACTIVE 生命周期条件组合。过滤发生在 Vector、BM25 候选和 CrossEncoder 之前，所以受限文本不会进入 reranker 或 LLM context。我保留了旧 API 的兼容模式，同时明确它不是认证系统；生产环境应从经过验证的身份声明生成访问上下文。

**English:** I implemented retrieval-layer permission filtering using tenant, department, and ordered access-level metadata, composed with the existing ACTIVE lifecycle rule. Filtering happens before Vector/BM25 candidates and CrossEncoder reranking, so restricted text never reaches model context. The MVP preserves legacy API compatibility but intentionally leaves authenticated identity and enterprise RBAC integration as future work.

## Truthful resume wording

Implemented and evaluated fail-closed metadata permission filtering for a multi-document RAG pipeline, enforcing tenant, department, access-level, and lifecycle constraints before reranking and LLM context construction, with zero-leakage security tests on a deterministic benchmark.
