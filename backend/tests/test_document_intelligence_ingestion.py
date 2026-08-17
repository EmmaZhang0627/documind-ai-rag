from __future__ import annotations

import sys
import unittest
from pathlib import Path

import fitz


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.settings import DEFAULT_TESSDATA_DIRECTORY
from app.services.chunker import (
    serialize_table_chunks,
    split_pages_into_table_aware_chunks,
)
from app.services.document_extraction import (
    extract_document_pages,
    should_use_ocr,
)


class FailingOCRPage:
    number = 0
    rect = fitz.Rect(0, 0, 100, 100)

    def get_text(self, *_args, **_kwargs):
        return "native fallback"

    def get_image_info(self):
        return [{"bbox": (0, 0, 100, 100)}]

    def get_textpage_ocr(self, **_kwargs):
        raise RuntimeError("synthetic OCR failure")


class DocumentIntelligenceIngestionTest(unittest.TestCase):
    def test_ocr_fallback_requires_insufficient_text_and_image_heavy_page(self) -> None:
        self.assertTrue(should_use_ocr("short", 0.9))
        self.assertTrue(should_use_ocr("short", 0.1, image_count=20))
        self.assertFalse(should_use_ocr("short", 0.1, image_count=1))
        self.assertFalse(should_use_ocr("x" * 250, 0.9))

    def test_ocr_failure_preserves_native_text_and_page_metadata(self) -> None:
        pages = extract_document_pages(
            [FailingOCRPage()],
            ocr_enabled=True,
            minimum_text_characters=200,
            minimum_image_coverage=0.5,
        )
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["text"], "native fallback")
        self.assertEqual(pages[0]["extraction_method"], "text")
        self.assertEqual(pages[0]["ocr_error"], "RuntimeError")

    def test_reserved_ocr_pdf_produces_retrievable_text_and_ocr_metadata(self) -> None:
        path = (
            PROJECT_ROOT / "eval" / "fixtures" / "corpus_v2"
            / "ocr_multimodal_benchmark" / "Reflective Practice Transcript.pdf"
        )
        with fitz.open(path) as document:
            pages = extract_document_pages(
                document,
                ocr_enabled=True,
                tessdata_directory=DEFAULT_TESSDATA_DIRECTORY,
            )
        self.assertEqual(len(pages), 4)
        self.assertTrue(all(page["extraction_method"] == "ocr" for page in pages))
        self.assertIn("Reflective Practice", pages[0]["text"])
        self.assertEqual([page["page_number"] for page in pages], [1, 2, 3, 4])

    def test_existing_text_pdf_stays_on_native_text_path(self) -> None:
        path = PROJECT_ROOT / "eval" / "fixtures" / "Study Plan - MSc Computer Science.pdf"
        with fitz.open(path) as document:
            pages = extract_document_pages(
                document,
                ocr_enabled=True,
                tessdata_directory=DEFAULT_TESSDATA_DIRECTORY,
            )
        self.assertTrue(pages)
        self.assertTrue(all(page["extraction_method"] == "text" for page in pages))
        self.assertFalse(any(page["ocr_attempted"] for page in pages))

    def test_table_serialization_repeats_header_when_rows_split(self) -> None:
        table = {
            "caption": "Security outcomes",
            "headers": ["Group", "Secure", "Insecure"],
            "rows": [[f"group-{index}", "10%", "90%"] for index in range(8)],
        }
        serialized = serialize_table_chunks(table, maximum_characters=150)
        self.assertGreater(len(serialized), 1)
        for value in serialized:
            self.assertIn("| Group | Secure | Insecure |", value)

    def test_table_chunks_preserve_page_source_and_table_metadata(self) -> None:
        pages = [{
            "page_number": 5,
            "text": "Nearby narrative.",
            "extraction_method": "text",
            "tables": [{
                "table_index": 2,
                "caption": "Table 3: Results",
                "headers": ["Question", "Value"],
                "rows": [["Q1", "21%"], ["Q2", "43%"]],
            }],
        }]
        chunks = split_pages_into_table_aware_chunks(
            pages,
            document_id="paper",
            source_file="paper.pdf",
        )
        table_chunk = next(chunk for chunk in chunks if chunk["content_type"] == "table")
        self.assertEqual(table_chunk["page_number"], 5)
        self.assertEqual(table_chunk["source_file"], "paper.pdf")
        self.assertEqual(table_chunk["table_index"], 2)
        self.assertEqual(table_chunk["table_caption"], "Table 3: Results")
        self.assertIn("| Question | Value |", table_chunk["content"])


if __name__ == "__main__":
    unittest.main()
