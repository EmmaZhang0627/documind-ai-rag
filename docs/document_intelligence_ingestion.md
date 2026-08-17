# Document Intelligence Ingestion MVP

## Purpose and system position

Normal PDF text extraction reads a PDF text layer. Scanned pages often contain only pixels or many image tiles, so retrieval cannot recover missing words later. OCR therefore belongs to ingestion:

```text
PDF page
-> native text extraction
-> OCR decision and optional local OCR
-> optional table extraction
-> page-aware chunks
-> existing embedding and storage
```

Fixed-character chunking also damages tables when it separates a header from its rows or a value from its column. The table-aware path adds deterministic Markdown-like chunks that repeat the table caption and header whenever rows must be split.

## Feature gates

Both capabilities are disabled by default:

```text
OCR_FALLBACK_ENABLED=false
TABLE_AWARE_INGESTION_ENABLED=false
```

When both are disabled, `/api/documents/parse-pdf` retains its original `page.get_text()` path so the existing seven-document Corpus V2 benchmark and chunk boundaries remain unchanged.

## Page-level OCR fallback

OCR is attempted only when native text is below 200 non-whitespace characters and the page is image-heavy. Image-heavy means either estimated image coverage is at least 0.5 or the page contains at least three image objects. The second condition handles documents constructed from many small image tiles.

PyMuPDF calls local Tesseract using the bundled English `tessdata_fast` language data at 150 DPI. OCR output replaces native text only when it is materially larger. If OCR raises an error or produces no useful gain, ingestion retains native text and records the failure instead of failing the entire document.

Page and chunk metadata preserve:

- `page_number`;
- `source_file` and `document_id`;
- `extraction_method`: `text` or `ocr`;
- existing lifecycle, version, parent-child, and permission metadata.

### PyMuPDF objects and geometry

The important object relationship is:

```text
fitz.Document  -> complete PDF
fitz.Page      -> one PDF page
fitz.Rect      -> one rectangular area on a page
```

`page.get_image_info()` and `page.get_textpage_ocr()` belong to `fitz.Page`;
`page.rect` is the page boundary represented by `fitz.Rect`. Image metadata
normally contains `bbox=(x0, y0, x1, y1)`. The code normalizes and clips it:

```python
bbox = fitz.Rect(image.get("bbox") or page.rect)
clipped = bbox & page.rect
```

`fitz.Rect(...)` converts a tuple, list, or existing rectangle into an object
that supports area and intersection operations. `&` is geometric intersection,
not union, so an image extending beyond the page is clipped before its area is
counted. Missing `bbox` data conservatively falls back to the complete page.

The current coverage estimate adds separate image areas and may double-count
overlapping rectangles. A practical future improvement is maximum single-image
coverage; exact rectangle-union geometry is only needed if failures justify it.

### OCR call and decision rule

`page.get_textpage_ocr(...)` performs OCR and returns a PyMuPDF `TextPage`.
The string is then read through `page.get_text(textpage=text_page)`. The trigger
is:

```text
native characters < 200
AND
(image coverage >= 0.5 OR image count >= 3)
```

A page with 80 native characters, 0.2 coverage, and 20 image objects therefore
does trigger OCR. Triggering does not guarantee replacement: OCR must produce
materially more usable text, otherwise native text remains the safe fallback.

## Table-aware path

PyMuPDF `page.find_tables()` detects practical ruled tables. Each useful table is represented as:

```text
Table: <caption or deterministic page/index label>
| header A | header B |
| --- | --- |
| row A1 | row B1 |
```

Rows are greedily grouped under a character limit. Every group repeats the caption and headers so a row never loses its column interpretation solely because of chunk splitting.

Table chunks add:

- `content_type=table`;
- `table_index`;
- `table_caption`;
- `page_number`;
- `extraction_method`.

Normal text chunks remain available alongside table chunks. This improves structure availability but creates duplicate evidence and additional ranking competition.

## Relationship with Parent-Child Retrieval

The current chunk-selection branch is mutually exclusive:

```python
if parent_child_enabled:
    chunks = parent_child_chunks
elif table_aware_enabled:
    chunks = table_aware_chunks
else:
    chunks = flat_chunks
```

If both flags are true, Parent-Child wins and structured table chunks are not
created. This is an explicit MVP limitation, not a claim that parents cannot
contain tables.

A future combined design should first unify the data model instead of merely
appending the outputs of both branches:

```text
page extraction
-> text/table/OCR regions
-> deterministic parents
-> typed text or table children
-> retrieve and rerank children
-> resolve deduplicated parent/table context
```

Children would retain `parent_id`, `content_type`, `page_number`, and
`child_index`. Blindly appending both chunk sets could duplicate evidence,
conflict in identity, fill TopK with copies, and repeat context for the LLM.

## Verification results

For `Reflective Practice Transcript.pdf`:

- 4 pages;
- raw native PyMuPDF extraction: 542 characters (491 after normalization);
- local OCR extraction: 11,321 characters;
- all 4 pages improved;
- chunks increased from 4 native fragments to 18 OCR chunks;
- three representative visible-text queries changed from evidence-not-found to OCR ranks 1, 3, and 1.

For the two table papers:

| Document | Useful detected tables | Ordinary chunks | Table-aware chunks | Table chunks |
|---|---:|---:|---:|---:|
| Asleep at the Keyboard | 5 | 136 | 145 | 9 |
| Do Users Write More Insecure Code | 15 | 119 | 134 | 15 |

All six focused table cases retained their required header/row evidence in a structured table chunk. Ordinary text extraction also contained all six facts. With evaluation-only BM25 ranking, one case improved, four were unchanged, and one nearby-row case moved from rank 4 to rank 5. This result supports structure preservation but does not prove general retrieval improvement.

## Important code paths

- `extract_document_pages()`: native extraction, OCR trigger, safe fallback, and optional table detection.
- `should_use_ocr()`: deterministic text-plus-image heuristic.
- `extract_page_tables()`: filters unusable detections and preserves headers and rows.
- `serialize_table_chunks()`: repeats caption/header while grouping rows.
- `split_pages_into_table_aware_chunks()`: combines existing text chunks with typed table chunks.
- `/api/documents/parse-pdf`: selects legacy, OCR, table-aware, or existing parent-child ingestion paths using feature flags.

## Trade-offs and failure modes

- OCR is slower than native extraction and may introduce spelling, punctuation, or reading-order errors.
- Image-heavy title pages can trigger OCR, although output replaces native text only when it adds material text.
- English tessdata does not cover every language.
- Table detection can miss borderless, nested, graphical, rotated, or cross-page tables.
- Merged cells and visual charts inside cells may be flattened imperfectly.
- Caption association is a nearby-text heuristic.
- Added table chunks can improve evidence structure while worsening rank through candidate competition.
- The benchmark uses BM25 as a deterministic retrieval proxy and does not claim production reranker improvement.

## How the current limitations normally evolve

| Limitation | Practical next adjustment | When to do it |
|---|---|---|
| Parent-Child and table modes are mutually exclusive | Introduce typed children and a shared deterministic parent model | After a measured combined-use failure |
| Image coverage may double-count overlap | Prefer maximum single-image coverage; use geometric union only if necessary | Small diagnostic improvement |
| OCR supports English only | Configure OCR languages per document or manifest | When multilingual scans enter the corpus |
| OCR contains noise | Preserve raw text and normalize control characters, line breaks, and repeated headers without guessing facts | Based on observed OCR errors |
| Complex tables lose merged/multi-level headers | Preserve structured cells and generate self-contained row text with column names | Later table benchmark work |
| Cross-page tables are split | Join only when geometry, caption, and page-boundary signals agree; retain row page numbers | Advanced document intelligence |
| Table chunks compete with flat text | Remove duplicated table regions or deduplicate candidates by table identity | Diagnose retrieval failures first |
| Parent context is too large | Reconstruct a window around the matched child and enforce a context budget | Context-composition phase |

The adjustment rule is evidence-driven: first classify a failure as extraction,
chunking, retrieval, ranking, or context composition, then improve that stage.
Changing OCR, tables, Parent-Child, TopK, and retrieval weights together would
make the source of regressions difficult to identify.

## Evolution

The current stage is a reliable local MVP. A deployment-ready version would add OCR language selection, resource limits, background ingestion, richer error telemetry, table-quality review, and explicit asset packaging. Future document intelligence may use layout-aware models or a VLM for handwriting, diagrams, charts, and complex tables, but those are not justified by this MVP benchmark yet.

## Transferable skills

- **MUST MASTER:** ingestion-versus-retrieval failure classification, selective OCR, page traceability, header/row preservation.
- **PROJECT-LEVEL FAMILIARITY:** OCR heuristics, table serialization, duplicate evidence trade-offs.
- **AWARENESS ONLY:** layout transformers, VLM extraction, table reasoning engines, human annotation pipelines.

## Interview explanation

**中文：** 我为 RAG ingestion 增加了默认关闭的页面级 OCR fallback 和 table-aware chunks。系统只在 native text 明显不足且页面 image-heavy 时调用本地 Tesseract，并在失败时安全回退。表格通过 PyMuPDF 检测后序列化为重复表头的结构化文本，保留 page/source metadata。真实 OCR 文档从原始 542 个字符提高到 11,321 个字符；表格测试证明结构证据被完整保留，但排序收益有限，因此没有宣称或实施 retrieval tuning。

**English:** I implemented feature-gated page-level OCR fallback and table-aware ingestion for a RAG pipeline. Local OCR runs only when native text is insufficient and the page is image-heavy, with a safe native-text fallback. Detected tables are serialized into header-preserving chunks with page and source traceability. A real scanned PDF improved from 542 to 11,321 extracted characters, while the table benchmark preserved structured evidence but showed limited ranking gains, so retrieval settings remained unchanged.

## Truthful resume wording

Implemented and evaluated selective local OCR fallback and header-preserving table ingestion for a multi-document RAG system, improving a real image-based PDF from 542 to 11,321 extracted characters while preserving page/source metadata and maintaining backward-compatible text ingestion.
