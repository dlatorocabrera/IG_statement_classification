#!/usr/bin/env python3
"""
02_extract_pdf.py

Sub-task 1.2: Text Extraction.

This script extracts text page-by-page from a searchable PDF using pdfplumber,
creates:
  1. a raw TXT file with explicit <PAGE number="X"> markers,
  2. a page-level JSON file with extraction metadata,
  3. an extraction_events table/record in the SQLite database.

Expected project structure for this simple version:

IG_protocol_project/
├── codes/
│   ├── config.py
│   └── 02_extract_pdf.py
├── data/
│   ├── database/
│   │   └── corpus_inventory.sqlite
│   └── raw_text/
├── documents/
│   └── source_pdfs/
└── tables/

Run from the project root:

python codes/task_1_text_preparation/02_extract_pdf.py \
  --document-id CL_MMA_DEC_000001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber

# ---------------------------------------------------------------------
# Import project paths from codes/config.py
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from config import DATABASE_PATH, RAW_TEXT_DIR, SOURCE_PDF_DIR
except ImportError as error:
    raise SystemExit(
        "Could not import config.py. Make sure this file is stored as:\n"
        "  codes/task_1_text_preparation/02_extract_pdf.py\n"
        "and config.py is stored as:\n"
        "  codes/config.py"
    ) from error


EXTRACTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS extraction_events (
    extraction_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    extraction_tool TEXT NOT NULL,
    tool_version TEXT,
    extraction_date TEXT NOT NULL,
    ocr_language TEXT,
    page_count INTEGER,
    extracted_page_count INTEGER,
    layout_preserved TEXT,
    raw_text_path TEXT,
    page_json_path TEXT,
    extraction_status TEXT NOT NULL,
    extraction_notes TEXT,
    source_file_hash_sha256 TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);
"""


def utc_now_iso() -> str:
    """Return current UTC time as an ISO string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 file hash."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def choose_database_path() -> Path:
    """Return the canonical inventory database declared in config.py."""
    database_path = Path(DATABASE_PATH)
    if database_path.exists():
        return database_path
    raise FileNotFoundError(
        "Could not find corpus_inventory.sqlite.\n"
        f"Expected canonical path: {database_path}\n"
        "Run 01_build_document_inventory.py first or restore the canonical database."
    )


def project_relative(path: Path) -> str:
    """Store relative paths when possible, for portability."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Connect to SQLite and enforce foreign-key relationships."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def get_document_record(connection: sqlite3.Connection, document_id: str) -> sqlite3.Row:
    """Read the document inventory record for the selected document_id."""
    record = connection.execute(
        """
        SELECT
            document_id,
            document_title,
            file_name_original,
            page_count,
            file_hash_sha256,
            original_file_format,
            language,
            document_status
        FROM document_inventory
        WHERE document_id = ?
        """,
        (document_id,),
    ).fetchone()

    if record is None:
        raise ValueError(
            f"Document_id {document_id!r} was not found in document_inventory."
        )

    return record


def resolve_pdf_path(
    document_record: sqlite3.Row,
    explicit_pdf_path: Path | None,
) -> Path:
    """
    Use --pdf-path when provided.
    Otherwise, use documents/source_pdfs/<file_name_original>.
    """
    if explicit_pdf_path is not None:
        pdf_path = explicit_pdf_path.expanduser()
    else:
        file_name = document_record["file_name_original"]
        pdf_path = Path(SOURCE_PDF_DIR) / file_name

    pdf_path = pdf_path.resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Check that the original PDF is inside documents/source_pdfs/, "
            "or pass --pdf-path explicitly."
        )

    return pdf_path


def next_extraction_id(connection: sqlite3.Connection) -> str:
    """Generate extraction IDs such as EXT_000001."""
    rows = connection.execute("SELECT extraction_id FROM extraction_events").fetchall()

    highest = 0
    for row in rows:
        extraction_id = row["extraction_id"]
        if extraction_id and extraction_id.startswith("EXT_"):
            suffix = extraction_id.replace("EXT_", "")
            if suffix.isdigit():
                highest = max(highest, int(suffix))

    return f"EXT_{highest + 1:06d}"


def extract_pdf_pages(pdf_path: Path) -> tuple[list[dict[str, Any]], int]:
    """
    Extract raw text page by page.

    This is intentionally conservative:
    - no cleaning
    - no normalization
    - no correction
    - no legal interpretation
    """
    pages: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(
                x_tolerance=3,
                y_tolerance=3,
                layout=False,
            )

            pages.append(
                {
                    "page_number": page_number,
                    "raw_text": raw_text,
                    "extraction_method": "PDF text extraction",
                }
            )

    return pages, total_pages


def write_raw_text(output_path: Path, document_id: str, pages: list[dict[str, Any]]) -> None:
    """Write one raw TXT file preserving page boundaries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(f'<DOCUMENT document_id="{document_id}">\n')

        for page in pages:
            file.write(f'\n<PAGE number="{page["page_number"]}">\n')
            if page["raw_text"] is not None:
                file.write(page["raw_text"])
            file.write("\n")


def write_page_json(
    output_path: Path,
    *,
    extraction_id: str,
    document_record: sqlite3.Row,
    pdf_path: Path,
    pages: list[dict[str, Any]],
    source_page_count: int,
    source_hash: str,
    raw_text_path: Path,
) -> None:
    """Write page-level JSON with extraction metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = utc_now_iso()

    payload = {
        "document_id": document_record["document_id"],
        "document_metadata": {
            "document_title": document_record["document_title"],
            "language": document_record["language"],
            "original_file_format": document_record["original_file_format"],
            "file_name_original": document_record["file_name_original"],
        },
        "extraction_metadata": {
            "extraction_id": extraction_id,
            "extraction_method": "PDF text extraction",
            "extraction_tool": "pdfplumber",
            "tool_version": pdfplumber.__version__,
            "extraction_date": timestamp,
            "ocr_language": None,
            "source_pdf_path": project_relative(pdf_path),
            "source_file_hash_sha256": source_hash,
            "source_page_count": source_page_count,
            "layout_preserved": "Partially",
            "raw_text_path": project_relative(raw_text_path),
        },
        "pages": pages,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def insert_extraction_event(connection: sqlite3.Connection, event: dict[str, Any]) -> None:
    """Insert one row into extraction_events."""
    connection.execute(
        """
        INSERT INTO extraction_events (
            extraction_id,
            document_id,
            extraction_method,
            extraction_tool,
            tool_version,
            extraction_date,
            ocr_language,
            page_count,
            extracted_page_count,
            layout_preserved,
            raw_text_path,
            page_json_path,
            extraction_status,
            extraction_notes,
            source_file_hash_sha256,
            created_at
        )
        VALUES (
            :extraction_id,
            :document_id,
            :extraction_method,
            :extraction_tool,
            :tool_version,
            :extraction_date,
            :ocr_language,
            :page_count,
            :extracted_page_count,
            :layout_preserved,
            :raw_text_path,
            :page_json_path,
            :extraction_status,
            :extraction_notes,
            :source_file_hash_sha256,
            :created_at
        )
        """,
        event,
    )


def run_extraction(document_id: str, pdf_path_argument: Path | None = None) -> dict[str, Any]:
    """Run the full Step 1.2 extraction workflow."""
    database_path = choose_database_path()
    raw_text_base = Path(RAW_TEXT_DIR)
    raw_text_base.mkdir(parents=True, exist_ok=True)

    connection = connect_database(database_path)

    try:
        connection.execute(EXTRACTION_TABLE_SQL)

        document_record = get_document_record(connection, document_id)
        pdf_path = resolve_pdf_path(document_record, pdf_path_argument)

        extraction_id = next_extraction_id(connection)
        source_hash = sha256_file(pdf_path)
        pages, source_page_count = extract_pdf_pages(pdf_path)

        extracted_page_count = sum(
            1
            for page in pages
            if page["raw_text"] is not None and page["raw_text"].strip()
        )

        document_output_dir = raw_text_base / document_id
        raw_text_path = document_output_dir / f"{extraction_id}_raw.txt"
        page_json_path = document_output_dir / f"{extraction_id}_pages.json"

        write_raw_text(raw_text_path, document_id, pages)
        write_page_json(
            page_json_path,
            extraction_id=extraction_id,
            document_record=document_record,
            pdf_path=pdf_path,
            pages=pages,
            source_page_count=source_page_count,
            source_hash=source_hash,
            raw_text_path=raw_text_path,
        )

        warnings: list[str] = []

        inventory_hash = document_record["file_hash_sha256"]
        if inventory_hash and source_hash != inventory_hash:
            warnings.append("Source PDF hash differs from document_inventory hash.")

        inventory_page_count = document_record["page_count"]
        if inventory_page_count is not None and int(inventory_page_count) != source_page_count:
            warnings.append("Source PDF page count differs from document_inventory page_count.")

        if extracted_page_count < source_page_count:
            warnings.append(
                f"Only {extracted_page_count} of {source_page_count} pages returned non-empty text."
            )

        extraction_status = "Completed with warnings" if warnings else "Completed"
        extraction_notes = " ".join(warnings) if warnings else None
        timestamp = utc_now_iso()

        event = {
            "extraction_id": extraction_id,
            "document_id": document_id,
            "extraction_method": "PDF text extraction",
            "extraction_tool": "pdfplumber",
            "tool_version": pdfplumber.__version__,
            "extraction_date": timestamp[:10],
            "ocr_language": "Not applicable",
            "page_count": source_page_count,
            "extracted_page_count": extracted_page_count,
            "layout_preserved": "Partially",
            "raw_text_path": project_relative(raw_text_path),
            "page_json_path": project_relative(page_json_path),
            "extraction_status": extraction_status,
            "extraction_notes": extraction_notes,
            "source_file_hash_sha256": source_hash,
            "created_at": timestamp,
        }

        insert_extraction_event(connection, event)
        connection.commit()

        print("\nExtraction completed")
        print("--------------------")
        for key, value in event.items():
            print(f"{key}: {value}")

        print("\nGenerated files")
        print("---------------")
        print(project_relative(raw_text_path))
        print(project_relative(page_json_path))

        print("\nDatabase used")
        print("-------------")
        print(project_relative(database_path))

        return event

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sub-task 1.2 PDF text extraction.")
    parser.add_argument(
        "--document-id",
        required=True,
        help="document_id already stored in document_inventory.",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=None,
        help="Optional explicit path to the source PDF.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        run_extraction(args.document_id, args.pdf_path)
    except Exception as error:
        raise SystemExit(f"\nExtraction failed: {error}") from error
