from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
MANIFEST_PATH = EVAL_DIR / "corpus_v2_manifest.json"
RESULTS_PATH = EVAL_DIR / "corpus_v2_ingestion_audit_latest.json"
PERSIST_DIRECTORY = EVAL_DIR / ".chroma_corpus_v2_audit"
COLLECTION_NAME = "documind_corpus_v2_ingestion_audit"
ALLOWED_GROUPS = {"text", "table", "ocr_multimodal"}
REQUIRED_DOCUMENT_FIELDS = {
    "document_id",
    "file_path",
    "file_name",
    "display_name",
    "version",
    "status",
    "benchmark_group",
    "include_in_text_retrieval",
}


def ensure_backend_import_path() -> None:
    backend_path = PROJECT_ROOT / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Corpus V2 manifest must contain a non-empty documents list.")

    document_ids: set[str] = set()
    file_paths: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"Manifest document at index {index} must be an object.")
        missing = REQUIRED_DOCUMENT_FIELDS - document.keys()
        if missing:
            raise ValueError(
                f"Manifest document at index {index} is missing: {sorted(missing)}"
            )
        document_id = document["document_id"]
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"Invalid document_id at manifest index {index}.")
        if document_id in document_ids:
            raise ValueError(f"Duplicate manifest document_id: {document_id}")
        document_ids.add(document_id)

        file_path = document["file_path"]
        if file_path in file_paths:
            raise ValueError(f"Duplicate manifest file_path: {file_path}")
        file_paths.add(file_path)

        group = document["benchmark_group"]
        if group not in ALLOWED_GROUPS:
            raise ValueError(f"Unsupported benchmark_group: {group!r}")
        included = document["include_in_text_retrieval"]
        if not isinstance(included, bool):
            raise ValueError(
                f"include_in_text_retrieval must be boolean for {document_id}."
            )
        if included != (group == "text"):
            raise ValueError(
                f"Text inclusion and benchmark_group disagree for {document_id}."
            )
        if str(document["status"]).upper() not in {"ACTIVE", "ARCHIVED"}:
            raise ValueError(f"Unsupported document status for {document_id}.")

        resolved_path = (PROJECT_ROOT / file_path).resolve()
        try:
            resolved_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"Manifest path escapes the project root: {file_path}"
            ) from error
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Manifest PDF does not exist: {resolved_path}")
        if resolved_path.suffix.casefold() != ".pdf":
            raise ValueError(f"Manifest source must be a PDF: {file_path}")
        if resolved_path.name != document["file_name"]:
            raise ValueError(
                f"file_name does not match file_path for {document_id}: "
                f"{document['file_name']!r} != {resolved_path.name!r}"
            )

    return manifest


def extract_pdf_pages(pdf_path: Path) -> tuple[int, list[dict[str, Any]], int]:
    import fitz

    document = fitz.open(pdf_path)
    try:
        pages: list[dict[str, Any]] = []
        extracted_text_length = 0
        for page_index, page in enumerate(document):
            text = page.get_text()
            extracted_text_length += len(text)
            if text.strip():
                pages.append({"page_number": page_index + 1, "text": text})
        return len(document), pages, extracted_text_length
    finally:
        document.close()


def build_audit_rag_service():
    ensure_backend_import_path()
    from app.config.settings import AppSettings
    from app.dependencies.rag_dependencies import build_rag_service

    if COLLECTION_NAME in {"documind_chunks", "documind_eval_chunks"}:
        raise RuntimeError("Corpus V2 audit must not use a production or V1 collection.")
    if PERSIST_DIRECTORY.resolve() in {
        (PROJECT_ROOT / "backend" / "data" / "chroma").resolve(),
        (EVAL_DIR / ".chroma").resolve(),
    }:
        raise RuntimeError("Corpus V2 audit persist directory is not isolated.")

    settings = replace(
        AppSettings.from_env(),
        vector_store_backend="chroma",
        chroma_persist_directory=str(PERSIST_DIRECTORY),
        chroma_collection_name=COLLECTION_NAME,
    )
    return build_rag_service(settings)


def stop_chroma_store(rag_service: Any) -> None:
    from chromadb.api.shared_system_client import SharedSystemClient

    store = rag_service.retriever.vector_store
    store.client._system.stop()
    SharedSystemClient.clear_system_cache()


def get_collection_records(rag_service: Any) -> dict[str, Any]:
    store = rag_service.retriever.vector_store
    return store.collection.get(include=["documents", "metadatas"])


def validate_records(
    records: dict[str, Any],
    included_documents: list[dict[str, Any]],
    expected_chunk_counts: dict[str, int],
) -> dict[str, Any]:
    ids = records.get("ids") or []
    metadatas = records.get("metadatas") or []
    documents = records.get("documents") or []
    errors: list[str] = []
    counts = Counter(
        str((metadata or {}).get("document_id")) for metadata in metadatas
    )
    expected_by_id = {
        document["document_id"]: document for document in included_documents
    }
    required_metadata = {
        "document_id",
        "version",
        "status",
        "file_name",
        "source_file",
        "page_number",
        "chunk_index",
    }

    if not (len(ids) == len(metadatas) == len(documents)):
        errors.append("Chroma record IDs, documents, and metadatas differ in length.")

    for record_id, document_text, metadata in zip(ids, documents, metadatas):
        stored = metadata or {}
        document_id = str(stored.get("document_id"))
        expected = expected_by_id.get(document_id)
        if expected is None:
            errors.append(f"Unexpected document_id in record {record_id}: {document_id}")
            continue
        missing = required_metadata - stored.keys()
        if missing:
            errors.append(f"Record {record_id} missing metadata: {sorted(missing)}")
        if str(stored.get("version")) != str(expected["version"]):
            errors.append(f"Record {record_id} has incorrect version.")
        if str(stored.get("status")).upper() != str(expected["status"]).upper():
            errors.append(f"Record {record_id} has incorrect status.")
        if stored.get("source_file") != expected["file_name"]:
            errors.append(f"Record {record_id} has incorrect source_file.")
        if stored.get("file_name") != expected["file_name"]:
            errors.append(f"Record {record_id} has incorrect file_name.")
        if not isinstance(stored.get("page_number"), int):
            errors.append(f"Record {record_id} has invalid page_number.")
        if not isinstance(stored.get("chunk_index"), int):
            errors.append(f"Record {record_id} has invalid chunk_index.")
        if not isinstance(document_text, str) or not document_text.strip():
            errors.append(f"Record {record_id} has empty stored text.")
        expected_prefix = f"{document_id}:{expected['version']}:"
        if not str(record_id).startswith(expected_prefix):
            errors.append(f"Record {record_id} has inconsistent Chroma ID.")
    expected_ids = set(expected_by_id)
    actual_ids = set(counts)
    if actual_ids != expected_ids:
        errors.append(
            "Stored document IDs differ from manifest: "
            f"expected={sorted(expected_ids)}, actual={sorted(actual_ids)}"
        )
    for document_id, expected_count in expected_chunk_counts.items():
        if counts[document_id] != expected_count:
            errors.append(
                f"Stored chunk count mismatch for {document_id}: "
                f"expected={expected_count}, actual={counts[document_id]}"
            )

    samples = []
    for document_id in sorted(expected_ids):
        sample_index = next(
            (
                index
                for index, metadata in enumerate(metadatas)
                if (metadata or {}).get("document_id") == document_id
            ),
            None,
        )
        if sample_index is not None:
            samples.append({
                "record_id": ids[sample_index],
                "metadata": metadatas[sample_index],
            })

    return {
        "passed": not errors,
        "errors": errors,
        "stored_document_ids": sorted(actual_ids),
        "stored_chunks_by_document": dict(sorted(counts.items())),
        "sampled_chunks": samples,
    }


def write_results(output: dict[str, Any]) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")


def run_audit() -> int:
    manifest = load_manifest()
    documents = manifest["documents"]
    included = [doc for doc in documents if doc["include_in_text_retrieval"]]
    excluded = [doc for doc in documents if not doc["include_in_text_retrieval"]]
    rag_service = build_audit_rag_service()
    rag_service.retriever.clear()

    per_document: list[dict[str, Any]] = []
    expected_chunk_counts: dict[str, int] = {}
    try:
        ensure_backend_import_path()
        from app.services.chunker import split_pages_into_chunks

        for entry in included:
            pdf_path = (PROJECT_ROOT / entry["file_path"]).resolve()
            page_count, pages, text_length = extract_pdf_pages(pdf_path)
            file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            chunks = split_pages_into_chunks(
                pages=pages,
                document_id=entry["document_id"],
                source_file=entry["file_name"],
                version=str(entry["version"]),
                status=str(entry["status"]).upper(),
                file_hash=file_hash,
            )
            count_before = rag_service.retriever.count()
            rag_service.ingest_document(chunks)
            count_after = rag_service.retriever.count()
            stored_chunk_count = count_after - count_before
            expected_chunk_counts[entry["document_id"]] = len(chunks)
            per_document.append({
                "document_id": entry["document_id"],
                "display_name": entry["display_name"],
                "file_path": entry["file_path"],
                "file_name": entry["file_name"],
                "version": str(entry["version"]),
                "status": str(entry["status"]).upper(),
                "page_count": page_count,
                "pages_with_extracted_text": len(pages),
                "extracted_text_length": text_length,
                "generated_chunk_count": len(chunks),
                "stored_chunk_count": stored_chunk_count,
            })

        initial_count = rag_service.retriever.count()
        initial_validation = validate_records(
            get_collection_records(rag_service), included, expected_chunk_counts
        )
    finally:
        stop_chroma_store(rag_service)

    reopened_service = build_audit_rag_service()
    try:
        reopened_count = reopened_service.retriever.count()
        persistence_validation = validate_records(
            get_collection_records(reopened_service), included, expected_chunk_counts
        )
    finally:
        stop_chroma_store(reopened_service)

    total_generated = sum(item["generated_chunk_count"] for item in per_document)
    all_passed = (
        len(included) == 7
        and initial_count == total_generated
        and reopened_count == total_generated
        and initial_validation["passed"]
        and persistence_validation["passed"]
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(MANIFEST_PATH),
        "corpus_id": manifest.get("corpus_id"),
        "isolation": {
            "persist_directory": str(PERSIST_DIRECTORY),
            "collection_name": COLLECTION_NAME,
        },
        "summary": {
            "manifest_document_count": len(documents),
            "included_text_document_count": len(included),
            "excluded_document_count": len(excluded),
            "generated_chunk_count": total_generated,
            "initial_stored_chunk_count": initial_count,
            "reopened_stored_chunk_count": reopened_count,
            "passed": all_passed,
        },
        "documents": per_document,
        "excluded_documents": [
            {
                "document_id": entry["document_id"],
                "file_name": entry["file_name"],
                "benchmark_group": entry["benchmark_group"],
            }
            for entry in excluded
        ],
        "metadata_validation": initial_validation,
        "persistence_validation": {
            **persistence_validation,
            "chunk_count_before_restart": initial_count,
            "chunk_count_after_restart": reopened_count,
        },
    }
    write_results(output)

    print("Corpus V2 ingestion audit complete")
    print(f"Text documents: {len(included)}")
    print(f"Generated chunks: {total_generated}")
    print(f"Stored chunks after reopen: {reopened_count}")
    print(f"Metadata validation: {initial_validation['passed']}")
    print(f"Persistence validation: {persistence_validation['passed']}")
    print(f"Results written to: {RESULTS_PATH}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(run_audit())
