# Frontend Citation and PDF Evidence Verification

## Purpose and user flow

Enterprise RAG users need to verify generated claims against source evidence:

```text
AI answer
-> citation card
-> source PDF
-> cited page
-> visible supporting snippet
```

Source attribution identifies the document, page citation narrows the location,
and evidence highlighting helps locate the supporting words. A citation improves
traceability but does not prove that the generated answer is correct.

## Source schema

The frontend consumes only sources returned by `/api/chat`:

- `document_id` and `version`: stable PDF identity;
- `source_file` or `file_name`: user-facing label;
- `page_number`: one-based PDF page target;
- `chunk_index`: retained traceability;
- `source_snippet`: short supporting preview.

`source_snippet` is intentionally truncated presentation data. It is not the
full retrieved chunk, complete PDF, or evaluation ground truth. Internal
full-chunk evidence remains outside the public response.

## Current MVP

Successful parsed uploads persist a deterministic evidence PDF copy under a
hash derived from `document_id + version`. The raw identity is never used as a
filesystem path. The backend exposes it inline through:

```text
GET /api/documents/evidence-pdf?document_id=...&version=...
```

The Vue chat UI displays the AI answer separately from numbered citation cards.
Each card shows the file, page, and a short snippet. Clicking it selects an
evidence viewer, opens the backend PDF URL with `#page=<page_number>`, and keeps
the snippet visible in a highlighted evidence panel.

Exact duplicate sources are removed using document ID, version, page, and
normalized snippet. Distinct snippets on the same page remain visible because
they may support different claims.

## Highlight strategy

The repository had no PDF.js dependency or viewer. This focused MVP therefore
uses the browser's native PDF viewer for page navigation and a highlighted
snippet panel for evidence location. It does not claim coordinate-perfect
in-page highlighting.

PDF.js text-layer matching remains a future option when measured user needs
justify worker configuration and more robust matching. It would still require
fallback behavior because OCR text, ligatures, line breaks, truncation, and PDF
reading order can prevent exact snippet matches.

## Failure and fallback behavior

- Missing `document_id`: do not build a PDF URL; retain the snippet.
- PDF unavailable or viewer failure: the answer and supporting panel remain.
- Missing page: default navigation to page 1.
- Missing snippet: instruct the user to inspect the cited page manually.
- Native viewer cannot highlight text: page jump plus visible snippet is the
  deliberate fallback.
- Multiple sources: render all distinct evidence while removing exact repeats.

The chat answer never depends on viewer success.

## Security boundary

The frontend does not infer or filter permissions. It renders only the sources
returned by the backend's retrieval flow. The current PDF serving endpoint is a
local MVP without authenticated identity enforcement. A deployment-ready
version must authenticate the request and re-check tenant/document permission
on the server before returning PDF bytes; hiding a link in the UI is not access
control.

## Important code paths

- `documents._evidence_pdf_path()`: deterministic path-safe PDF identity.
- `documents.get_evidence_pdf()`: inline PDF response and unavailable fallback.
- `deduplicateSources()`: exact evidence-card deduplication.
- `evidencePdfUrl()`: query-safe PDF URL and one-based page fragment.
- `CitationList.vue`: source rendering and selection event.
- `PdfEvidenceViewer.vue`: page viewer, evidence panel, and graceful fallback.
- `App.vue`: upload, chat, citation selection, and evidence layout.

## Verification scope

Focused tests cover citation rendering, exact-source deduplication, distinct
same-page evidence preservation, citation selection, page-fragment navigation,
missing snippet fallback, missing document identity, deterministic safe backend
paths, successful PDF serving, and missing-file 404 behavior.

## Limitations and evolution

The current stage is a local verification MVP. It does not include PDF.js text
layers, fuzzy text matching, bounding-box highlights, authenticated PDF access,
range requests, signed URLs, or storage lifecycle cleanup. A production version
would add server-side authorization, durable object storage, audit logging,
content-disposition hardening, and measured PDF.js or coordinate-based
highlighting where text-layer quality supports it.

## Transferable skills

- **MUST MASTER:** public source schema, answer/source/evidence separation,
  server-side authorization boundary, and graceful viewer fallback.
- **PROJECT-LEVEL FAMILIARITY:** citation deduplication, PDF page fragments,
  text-layer matching constraints, and viewer component testing.
- **AWARENESS ONLY:** signed object-storage URLs, PDF coordinate highlights,
  range streaming, and enterprise document-viewer audit controls.
