from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import fitz


logger = logging.getLogger(__name__)


def normalize_extracted_text(text: str) -> str:
    lines = [" ".join(line.replace("\x00", "").split()) for line in text.splitlines()]
    normalized: list[str] = []
    for line in lines:
        if line or (normalized and normalized[-1]):
            normalized.append(line)
    return "\n".join(normalized).strip()


def page_image_coverage(page: fitz.Page) -> float:
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    image_area = 0.0
    for image in page.get_image_info():
        bbox = fitz.Rect(image.get("bbox") or page.rect)
        clipped = bbox & page.rect
        if not clipped.is_empty:
            image_area += float(clipped.width * clipped.height)
    return min(image_area / page_area, 1.0)


def should_use_ocr(
    native_text: str,
    image_coverage: float,
    *,
    image_count: int = 0,
    minimum_text_characters: int = 200,
    minimum_image_coverage: float = 0.5,
    minimum_image_count: int = 3,
) -> bool:
    visible_characters = len(re.sub(r"\s+", "", native_text))
    return (
        visible_characters < minimum_text_characters
        and (
            image_coverage >= minimum_image_coverage
            or image_count >= minimum_image_count
        )
    )


def _nearby_table_caption(page: fitz.Page, bbox: fitz.Rect, table_index: int) -> str:
    candidates: list[tuple[float, str]] = []
    for block in page.get_text("blocks"):
        block_bbox = fitz.Rect(block[:4])
        text = normalize_extracted_text(str(block[4]))
        if not text or block_bbox.y1 > bbox.y0 + 2:
            continue
        distance = bbox.y0 - block_bbox.y1
        if distance <= 90:
            candidates.append((distance, text.splitlines()[-1]))
    if candidates:
        caption = min(candidates, key=lambda item: item[0])[1]
        if caption.casefold().startswith(("table", "tab.")):
            return caption[:240]
    return f"Table on page {page.number + 1}, index {table_index}"


def extract_page_tables(page: fitz.Page) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    try:
        detected = page.find_tables().tables
    except Exception as error:
        logger.warning("table_detection_failed page=%s error=%s", page.number + 1, error)
        return tables

    for table_index, table in enumerate(detected, start=1):
        extracted = table.extract()
        rows = [
            [normalize_extracted_text(str(cell or "")) for cell in row]
            for row in extracted
            if row and any(normalize_extracted_text(str(cell or "")) for cell in row)
        ]
        if len(rows) < 2 or max(len(row) for row in rows) < 2:
            continue
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        headers = [value or f"column_{index + 1}" for index, value in enumerate(padded[0])]
        tables.append({
            "table_index": table_index,
            "caption": _nearby_table_caption(page, fitz.Rect(table.bbox), table_index),
            "headers": headers,
            "rows": padded[1:],
        })
    return tables


def extract_document_pages(
    document: fitz.Document,
    *,
    ocr_enabled: bool = False,
    table_enabled: bool = False,
    tessdata_directory: str | Path | None = None,
    ocr_language: str = "eng",
    ocr_dpi: int = 150,
    minimum_text_characters: int = 200,
    minimum_image_coverage: float = 0.5,
    minimum_image_count: int = 3,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    tessdata = str(Path(tessdata_directory).resolve()) if tessdata_directory else None

    for page_index, page in enumerate(document, start=1):
        native_text = normalize_extracted_text(page.get_text())
        image_coverage = page_image_coverage(page)
        image_count = len(page.get_image_info())
        extraction_method = "text"
        ocr_attempted = False
        ocr_error: str | None = None
        text = native_text

        if ocr_enabled and should_use_ocr(
            native_text,
            image_coverage,
            image_count=image_count,
            minimum_text_characters=minimum_text_characters,
            minimum_image_coverage=minimum_image_coverage,
            minimum_image_count=minimum_image_count,
        ):
            ocr_attempted = True
            try:
                text_page = page.get_textpage_ocr(
                    language=ocr_language,
                    dpi=ocr_dpi,
                    full=True,
                    tessdata=tessdata,
                )
                ocr_text = normalize_extracted_text(page.get_text(textpage=text_page))
                if len(ocr_text) > max(len(native_text) + 50, int(len(native_text) * 1.5)):
                    text = ocr_text
                    extraction_method = "ocr"
            except Exception as error:
                ocr_error = error.__class__.__name__
                logger.warning("page_ocr_failed page=%s error=%s", page_index, error)

        pages.append({
            "page_number": page_index,
            "text": text,
            "native_text": native_text,
            "extraction_method": extraction_method,
            "ocr_attempted": ocr_attempted,
            "ocr_error": ocr_error,
            "image_coverage": image_coverage,
            "image_count": image_count,
            "tables": extract_page_tables(page) if table_enabled else [],
        })
    return pages
