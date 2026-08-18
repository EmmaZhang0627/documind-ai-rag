import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.api import documents


class PdfEvidenceServingTests(unittest.IsolatedAsyncioTestCase):
    def test_evidence_path_is_deterministic_and_path_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(documents, "EVIDENCE_PDF_DIR", Path(directory)):
                first = documents._evidence_pdf_path("../../tenant/document", "v/1")
                second = documents._evidence_pdf_path("../../tenant/document", "v/1")

        self.assertEqual(first, second)
        self.assertEqual(first.parent, Path(directory))
        self.assertEqual(first.suffix, ".pdf")
        self.assertNotIn("tenant", first.name)

    async def test_stored_pdf_can_be_served_inline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(documents, "EVIDENCE_PDF_DIR", Path(directory)):
                documents._store_evidence_pdf("document-1", "2", b"%PDF-test")
                response = await documents.get_evidence_pdf("document-1", "2")

                self.assertEqual(response.media_type, "application/pdf")
                self.assertTrue(Path(response.path).is_file())
                self.assertEqual(Path(response.path).read_bytes(), b"%PDF-test")

    async def test_missing_pdf_returns_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(documents, "EVIDENCE_PDF_DIR", Path(directory)):
                with self.assertRaises(HTTPException) as raised:
                    await documents.get_evidence_pdf("missing", "1")

        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
