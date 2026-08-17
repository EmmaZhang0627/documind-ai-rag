from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from rank_bm25 import BM25Okapi


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
RESULTS_PATH = EVAL_DIR / "document_intelligence_ingestion_latest.json"
TESSDATA_DIRECTORY = BACKEND_ROOT / "app" / "assets" / "tessdata"
OCR_PATH = (
    EVAL_DIR / "fixtures" / "corpus_v2" / "ocr_multimodal_benchmark"
    / "Reflective Practice Transcript.pdf"
)
TABLE_PATHS = (
    EVAL_DIR / "fixtures" / "corpus_v2" / "table_benchmark"
    / "Asleep_at_the_Keyboard_Assessing_the_Security_of_GitHub_Copilots_Code_Contributions.pdf",
    EVAL_DIR / "fixtures" / "corpus_v2" / "table_benchmark"
    / "Do users write more insecure code.pdf",
)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.chunker import (
    split_pages_into_chunks,
    split_pages_into_table_aware_chunks,
)
from app.services.document_extraction import extract_document_pages


def normalized(value: str) -> str:
    return " ".join(re.sub(r"[^\w.%#-]+", " ", value.casefold()).split())


def tokenized(value: str) -> list[str]:
    return normalized(value).split()


def first_evidence_rank(
    chunks: list[dict[str, Any]],
    query: str,
    required_phrases: list[str],
) -> int | None:
    corpus = [chunk["content"] for chunk in chunks]
    if not corpus:
        return None
    model = BM25Okapi([tokenized(value) for value in corpus])
    scores = model.get_scores(tokenized(query))
    ranked = sorted(
        zip(scores, chunks), key=lambda item: float(item[0]), reverse=True
    )
    required = [normalized(value) for value in required_phrases]
    for rank, (_, chunk) in enumerate(ranked, start=1):
        text = normalized(chunk["content"])
        if all(value in text for value in required):
            return rank
    return None


def chunk_pages(pages: list[dict[str, Any]], document_id: str) -> list[dict[str, Any]]:
    return split_pages_into_chunks(
        pages,
        document_id=document_id,
        source_file=f"{document_id}.pdf",
    )


def ocr_benchmark() -> dict[str, Any]:
    with fitz.open(OCR_PATH) as document:
        raw_native_characters = sum(len(page.get_text()) for page in document)
    with fitz.open(OCR_PATH) as document:
        native_pages = extract_document_pages(document, ocr_enabled=False)
    with fitz.open(OCR_PATH) as document:
        ocr_pages = extract_document_pages(
            document,
            ocr_enabled=True,
            tessdata_directory=TESSDATA_DIRECTORY,
        )
    native_chunks = chunk_pages(native_pages, "reflective-native")
    ocr_chunks = chunk_pages(ocr_pages, "reflective-ocr")
    cases = [
        {
            "query": "Who proposed the two reflective practice frameworks?",
            "required_phrases": ["Graham Gibbs", "Donald A. Schon"],
        },
        {
            "query": "What questions should be considered in reflective practice?",
            "required_phrases": ["What are you learning about", "Why is it important"],
        },
        {
            "query": "What is reflective practice not simply about?",
            "required_phrases": ["not simply", "recording what you did"],
        },
    ]
    case_results = []
    for case in cases:
        case_results.append({
            **case,
            "native_evidence_rank": first_evidence_rank(
                native_chunks, case["query"], case["required_phrases"]
            ),
            "ocr_evidence_rank": first_evidence_rank(
                ocr_chunks, case["query"], case["required_phrases"]
            ),
        })
    return {
        "file_name": OCR_PATH.name,
        "page_count": len(ocr_pages),
        "normal_pymupdf_extracted_characters": raw_native_characters,
        "normalized_native_characters": sum(
            len(page["native_text"]) for page in native_pages
        ),
        "ocr_extracted_characters": sum(len(page["text"]) for page in ocr_pages),
        "pages_improved": [
            page["page_number"]
            for page in ocr_pages
            if page["extraction_method"] == "ocr"
        ],
        "native_generated_chunks": len(native_chunks),
        "ocr_generated_chunks": len(ocr_chunks),
        "representative_retrieval_cases": case_results,
        "representative_visible_text_retrievable": all(
            case["ocr_evidence_rank"] is not None for case in case_results
        ),
    }


TABLE_CASES = {
    "Do users write more insecure code.pdf": [
        {
            "case_id": "single_cell_q1_correctness",
            "query": "What are Q1 correctness and security values?",
            "required_phrases": ["Q1", "0.757", "0.813"],
        },
        {
            "case_id": "row_column_specification_prompts",
            "query": "For Specification, what proportion of prompts and users is reported?",
            "required_phrases": ["Specification", "42.1%", "63.8%"],
        },
        {
            "case_id": "comparison_temperature_adjustment_q3",
            "query": "Compare Q3 insecure answers for adjusted and unadjusted temperature.",
            "required_phrases": ["Did Adjust Temp", "50%", "Did Not Adjust Temp", "81%"],
        },
        {
            "case_id": "nearby_row_model_close",
            "query": "What proportion of prompts is Model Close?",
            "required_phrases": ["Model Close", "33.5%"],
        },
    ],
    "Asleep_at_the_Keyboard_Assessing_the_Security_of_GitHub_Copilots_Code_Contributions.pdf": [
        {
            "case_id": "scenario_787_0_values",
            "query": "For CWE scenario 787-0, how many valid and vulnerable programs are shown?",
            "required_phrases": ["787-0", "19", "9"],
        },
        {
            "case_id": "nearby_scenario_787_1",
            "query": "For scenario 787-1, what values distinguish it from 787-0?",
            "required_phrases": ["787-1", "17", "2"],
        },
    ],
}


def table_benchmark() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in TABLE_PATHS:
        with fitz.open(path) as document:
            pages = extract_document_pages(document, table_enabled=True)
        ordinary_chunks = chunk_pages(pages, path.stem + "-ordinary")
        table_chunks = split_pages_into_table_aware_chunks(
            pages,
            document_id=path.stem + "-aware",
            source_file=path.name,
        )
        cases = []
        for case in TABLE_CASES[path.name]:
            ordinary_rank = first_evidence_rank(
                ordinary_chunks, case["query"], case["required_phrases"]
            )
            aware_rank = first_evidence_rank(
                table_chunks, case["query"], case["required_phrases"]
            )
            table_representation_contains_evidence = any(
                all(
                    normalized(phrase) in normalized(chunk["content"])
                    for phrase in case["required_phrases"]
                )
                for chunk in table_chunks
                if chunk.get("content_type") == "table"
            )
            cases.append({
                **case,
                "ordinary_evidence_rank": ordinary_rank,
                "table_aware_evidence_rank": aware_rank,
                "table_representation_contains_evidence": (
                    table_representation_contains_evidence
                ),
                "table_aware_improved": (
                    aware_rank is not None
                    and (ordinary_rank is None or aware_rank < ordinary_rank)
                ),
            })
        documents.append({
            "file_name": path.name,
            "page_count": len(pages),
            "detected_table_count": sum(len(page["tables"]) for page in pages),
            "ordinary_chunk_count": len(ordinary_chunks),
            "table_aware_chunk_count": len(table_chunks),
            "generated_table_chunk_count": sum(
                chunk.get("content_type") == "table" for chunk in table_chunks
            ),
            "cases": cases,
        })
    return documents


def run() -> int:
    ocr = ocr_benchmark()
    tables = table_benchmark()
    table_cases = [case for document in tables for case in document["cases"]]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "OCR fallback and table-aware ingestion MVP",
        "configuration": {
            "ocr_feature_default_enabled": False,
            "table_feature_default_enabled": False,
            "ocr_language": "eng",
            "ocr_dpi": 150,
            "ocr_minimum_text_characters": 200,
            "ocr_minimum_image_coverage": 0.5,
            "ocr_minimum_image_count": 3,
        },
        "ocr": ocr,
        "tables": tables,
        "table_summary": {
            "case_count": len(table_cases),
            "ordinary_evidence_found": sum(
                case["ordinary_evidence_rank"] is not None for case in table_cases
            ),
            "table_aware_evidence_found": sum(
                case["table_aware_evidence_rank"] is not None for case in table_cases
            ),
            "table_aware_improved_cases": sum(
                case["table_aware_improved"] for case in table_cases
            ),
            "structured_table_evidence_preserved": sum(
                case["table_representation_contains_evidence"]
                for case in table_cases
            ),
        },
        "limitations": [
            "BM25 rank is used only as a deterministic retrieval proxy; production embeddings and CrossEncoder are unchanged.",
            "PyMuPDF table detection can miss borderless, nested, graphical, or cross-page tables.",
            "OCR quality varies with scan resolution, rotation, handwriting, and language data.",
        ],
    }
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    passed = (
        ocr["representative_visible_text_retrievable"]
        and output["table_summary"]["table_aware_evidence_found"]
        >= output["table_summary"]["ordinary_evidence_found"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
