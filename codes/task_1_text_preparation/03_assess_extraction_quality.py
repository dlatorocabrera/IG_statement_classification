#!/usr/bin/env python3
"""
03_assess_extraction_quality.py

Sub-task 1.3: Extraction-Quality Assessment.

Version 2 adds explicit manual-review variables:

- requires_manual_review
- review_reason_code
- review_reason_note

This script reads the page-level JSON created in Sub-task 1.2 and creates:

1. a quality-checked JSON file,
2. a page-level CSV quality report,
3. SQLite records for the assessment event and page-level flags.

Run from the project root:

python codes/task_1_text_preparation/03_assess_extraction_quality.py \
  --document-id CL_MMA_DEC_000001

or for a specific extraction event:

python codes/task_1_text_preparation/03_assess_extraction_quality.py \
  --extraction-id EXT_000001
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


# ---------------------------------------------------------------------
# Import project paths from codes/config.py
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from config import DATABASE_PATH, QUALITY_CHECKED_DIR, QUALITY_REPORT_DIR
except ImportError as error:
    raise SystemExit(
        "Could not import config.py. Make sure this file is stored as:\n"
        "  codes/task_1_text_preparation/03_assess_extraction_quality.py\n"
        "and config.py is stored as:\n"
        "  codes/config.py"
    ) from error


QUALITY_ASSESSMENT_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS quality_assessment_events (
    assessment_id TEXT PRIMARY KEY,
    extraction_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    assessment_date TEXT NOT NULL,
    quality_tool TEXT NOT NULL,
    near_blank_threshold INTEGER NOT NULL,
    expected_page_count INTEGER,
    observed_page_count INTEGER NOT NULL,
    page_count_match INTEGER NOT NULL,
    blank_page_count INTEGER NOT NULL,
    near_blank_page_count INTEGER NOT NULL,
    replacement_char_page_count INTEGER NOT NULL,
    manual_review_page_count INTEGER NOT NULL,
    review_reason_summary TEXT,
    min_nonspace_chars INTEGER,
    max_nonspace_chars INTEGER,
    mean_nonspace_chars REAL,
    quality_status TEXT NOT NULL,
    quality_checked_json_path TEXT NOT NULL,
    page_quality_csv_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (extraction_id)
        REFERENCES extraction_events(extraction_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);
"""


QUALITY_PAGE_FLAGS_SQL = """
CREATE TABLE IF NOT EXISTS quality_page_flags (
    assessment_id TEXT NOT NULL,
    extraction_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    total_char_count INTEGER NOT NULL,
    nonspace_char_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    blank_extraction INTEGER NOT NULL,
    near_blank_extraction INTEGER NOT NULL,
    replacement_char_count INTEGER NOT NULL,
    requires_manual_review INTEGER NOT NULL,
    review_reason_code TEXT NOT NULL,
    review_reason_note TEXT,
    quality_status TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (assessment_id, page_number),
    FOREIGN KEY (assessment_id)
        REFERENCES quality_assessment_events(assessment_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (extraction_id)
        REFERENCES extraction_events(extraction_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);
"""


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO 8601."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def choose_database_path() -> Path:
    """Return the canonical inventory database declared in config.py."""
    database_path = Path(DATABASE_PATH)
    if database_path.exists():
        return database_path
    raise FileNotFoundError(
        "Could not find corpus_inventory.sqlite.\n"
        f"Expected canonical path: {database_path}"
    )


def project_relative(path: Path) -> str:
    """Return project-relative paths where possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(path_value: str | None) -> Path:
    """
    Resolve a path stored in SQLite.

    Relative paths are interpreted from the project root.
    Absolute paths are used directly.
    """
    if not path_value:
        raise ValueError("Missing path value.")

    path = Path(path_value)
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Connect to SQLite and enforce foreign-key relationships."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """
    Add a column to an existing SQLite table if it is missing.

    This is useful when you already ran an older version of this script and
    the table exists without the new review-reason fields.
    """
    columns = connection.execute(f"PRAGMA table_info({table_name});").fetchall()
    existing_column_names = {row["name"] for row in columns}

    if column_name not in existing_column_names:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition};"
        )


def ensure_quality_tables(connection: sqlite3.Connection) -> None:
    """
    Create quality tables and migrate older versions by adding missing columns.
    """
    connection.execute(QUALITY_ASSESSMENT_EVENTS_SQL)
    connection.execute(QUALITY_PAGE_FLAGS_SQL)

    ensure_column(
        connection,
        "quality_assessment_events",
        "review_reason_summary",
        "TEXT",
    )
    ensure_column(
        connection,
        "quality_page_flags",
        "review_reason_code",
        "TEXT NOT NULL DEFAULT 'none'",
    )
    ensure_column(
        connection,
        "quality_page_flags",
        "review_reason_note",
        "TEXT",
    )


def get_latest_extraction_for_document(
    connection: sqlite3.Connection,
    document_id: str,
) -> sqlite3.Row:
    """Return the most recent extraction event for a document."""
    record = connection.execute(
        """
        SELECT *
        FROM extraction_events
        WHERE document_id = ?
        ORDER BY created_at DESC, extraction_id DESC
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()

    if record is None:
        raise ValueError(
            f"No extraction event found for document_id {document_id!r}. "
            "Run Sub-task 1.2 first."
        )

    return record


def get_extraction_by_id(
    connection: sqlite3.Connection,
    extraction_id: str,
) -> sqlite3.Row:
    """Return one extraction event by extraction_id."""
    record = connection.execute(
        """
        SELECT *
        FROM extraction_events
        WHERE extraction_id = ?
        """,
        (extraction_id,),
    ).fetchone()

    if record is None:
        raise ValueError(
            f"No extraction event found for extraction_id {extraction_id!r}."
        )

    return record


def next_assessment_id(connection: sqlite3.Connection) -> str:
    """Generate assessment identifiers such as QA_000001."""
    rows = connection.execute(
        "SELECT assessment_id FROM quality_assessment_events"
    ).fetchall()

    highest = 0
    for row in rows:
        value = row["assessment_id"]
        if value and value.startswith("QA_"):
            suffix = value.replace("QA_", "")
            if suffix.isdigit():
                highest = max(highest, int(suffix))

    return f"QA_{highest + 1:06d}"


def count_words(text: str) -> int:
    """Count simple word-like tokens, including accented Spanish characters."""
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def detect_review_reason(
    *,
    raw_text: Any,
    nonspace_char_count: int,
    replacement_char_count: int,
    near_blank_threshold: int,
) -> tuple[bool, str, str | None, str]:
    """
    Assign manual-review flags and reason codes.

    Returns:
        requires_manual_review
        review_reason_code
        review_reason_note
        quality_status

    Reason-code hierarchy:
    1. blank_extraction
    2. near_blank_extraction
    3. replacement_character
    4. none

    If more than one issue is possible, the earliest one in this hierarchy
    is used as the primary review reason.
    """
    text = "" if raw_text is None else str(raw_text)
    stripped = text.strip()

    blank_extraction = raw_text is None or stripped == ""
    near_blank_extraction = (
        not blank_extraction
        and nonspace_char_count < near_blank_threshold
    )

    if blank_extraction:
        return (
            True,
            "blank_extraction",
            "No text was extracted from this page. The page may be empty, scanned, image-based, or affected by extraction failure.",
            "blank_extraction",
        )

    if near_blank_extraction:
        return (
            True,
            "near_blank_extraction",
            (
                "The extracted text is below the near-blank threshold "
                f"({nonspace_char_count} < {near_blank_threshold} non-space characters). "
                "The page may contain a table, figure, signature, stamp, scanned content, or another complex layout."
            ),
            "near_blank_extraction",
        )

    if replacement_char_count > 0:
        return (
            True,
            "replacement_character",
            (
                "The replacement character � was detected. This may indicate an encoding problem "
                "that should not be automatically corrected without inspection."
            ),
            "requires_review",
        )

    return (
        False,
        "none",
        None,
        "text_extracted",
    )


def assess_page(
    page: dict[str, Any],
    near_blank_threshold: int,
) -> dict[str, Any]:
    """Create page-level quality flags."""
    raw_text = page.get("raw_text")
    text = "" if raw_text is None else str(raw_text)

    total_char_count = len(text)
    nonspace_char_count = len(re.sub(r"\s+", "", text))
    word_count = count_words(text)
    line_count = len(text.splitlines()) if text else 0
    replacement_char_count = text.count("\ufffd") + text.count("�")

    blank_extraction = raw_text is None or text.strip() == ""
    near_blank_extraction = (
        not blank_extraction
        and nonspace_char_count < near_blank_threshold
    )

    (
        requires_manual_review,
        review_reason_code,
        review_reason_note,
        quality_status,
    ) = detect_review_reason(
        raw_text=raw_text,
        nonspace_char_count=nonspace_char_count,
        replacement_char_count=replacement_char_count,
        near_blank_threshold=near_blank_threshold,
    )

    return {
        "page_number": int(page["page_number"]),
        "total_char_count": total_char_count,
        "nonspace_char_count": nonspace_char_count,
        "word_count": word_count,
        "line_count": line_count,
        "blank_extraction": blank_extraction,
        "near_blank_extraction": near_blank_extraction,
        "replacement_char_count": replacement_char_count,
        "requires_manual_review": requires_manual_review,
        "review_reason_code": review_reason_code,
        "review_reason_note": review_reason_note,
        "quality_status": quality_status,
        # notes is kept for backward compatibility and readability.
        "notes": review_reason_note,
    }


def summarize_review_reasons(
    page_assessments: list[dict[str, Any]],
) -> str | None:
    """Create a compact document-level summary of review reason codes."""
    counts: dict[str, int] = {}

    for page in page_assessments:
        code = page["review_reason_code"]
        if code != "none":
            counts[code] = counts.get(code, 0) + 1

    if not counts:
        return None

    return "; ".join(f"{code}: {count}" for code, count in sorted(counts.items()))


def build_quality_summary(
    *,
    extraction_event: sqlite3.Row,
    page_assessments: list[dict[str, Any]],
    near_blank_threshold: int,
) -> dict[str, Any]:
    """Build document-level quality summary."""
    expected_page_count = extraction_event["page_count"]
    observed_page_count = len(page_assessments)
    page_count_match = (
        expected_page_count is not None
        and int(expected_page_count) == observed_page_count
    )

    nonspace_counts = [p["nonspace_char_count"] for p in page_assessments]

    blank_page_count = sum(1 for p in page_assessments if p["blank_extraction"])
    near_blank_page_count = sum(1 for p in page_assessments if p["near_blank_extraction"])
    replacement_char_page_count = sum(
        1 for p in page_assessments if p["replacement_char_count"] > 0
    )
    manual_review_page_count = sum(
        1 for p in page_assessments if p["requires_manual_review"]
    )

    review_reason_summary = summarize_review_reasons(page_assessments)

    notes: list[str] = []
    if not page_count_match:
        notes.append("Observed page count does not match extraction event page count.")
    if blank_page_count > 0:
        notes.append("At least one page has blank extraction.")
    if near_blank_page_count > 0:
        notes.append("At least one page has near-blank extraction.")
    if replacement_char_page_count > 0:
        notes.append("At least one page contains replacement characters.")

    quality_status = "Completed with warnings" if notes else "Completed"

    return {
        "expected_page_count": expected_page_count,
        "observed_page_count": observed_page_count,
        "page_count_match": page_count_match,
        "blank_page_count": blank_page_count,
        "near_blank_page_count": near_blank_page_count,
        "replacement_char_page_count": replacement_char_page_count,
        "manual_review_page_count": manual_review_page_count,
        "review_reason_summary": review_reason_summary,
        "min_nonspace_chars": min(nonspace_counts) if nonspace_counts else None,
        "max_nonspace_chars": max(nonspace_counts) if nonspace_counts else None,
        "mean_nonspace_chars": round(mean(nonspace_counts), 2) if nonspace_counts else None,
        "near_blank_threshold": near_blank_threshold,
        "quality_status": quality_status,
        "notes": " ".join(notes) if notes else None,
    }


def write_quality_checked_json(
    output_path: Path,
    *,
    input_payload: dict[str, Any],
    assessment_id: str,
    extraction_event: sqlite3.Row,
    summary: dict[str, Any],
    page_assessments: list[dict[str, Any]],
) -> None:
    """Write quality-checked JSON, retaining raw text and adding quality flags."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_page = {p["page_number"]: p for p in page_assessments}

    output_payload = dict(input_payload)
    output_payload["quality_assessment_metadata"] = {
        "assessment_id": assessment_id,
        "extraction_id": extraction_event["extraction_id"],
        "document_id": extraction_event["document_id"],
        "assessment_date": utc_now_iso(),
        "quality_tool": "03_assess_extraction_quality.py",
        "near_blank_threshold": summary["near_blank_threshold"],
    }
    output_payload["document_level_quality_summary"] = summary

    quality_checked_pages = []
    for page in input_payload.get("pages", []):
        page_copy = dict(page)
        page_number = int(page_copy["page_number"])
        page_copy["quality_assessment"] = by_page[page_number]
        quality_checked_pages.append(page_copy)

    output_payload["pages"] = quality_checked_pages

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)


def write_page_quality_csv(
    output_path: Path,
    page_assessments: list[dict[str, Any]],
) -> None:
    """Write a page-level quality CSV report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "page_number",
        "total_char_count",
        "nonspace_char_count",
        "word_count",
        "line_count",
        "blank_extraction",
        "near_blank_extraction",
        "replacement_char_count",
        "requires_manual_review",
        "review_reason_code",
        "review_reason_note",
        "quality_status",
        "notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(page_assessments)


def insert_quality_assessment_event(
    connection: sqlite3.Connection,
    event: dict[str, Any],
) -> None:
    """Insert the document-level quality assessment event."""
    connection.execute(
        """
        INSERT INTO quality_assessment_events (
            assessment_id,
            extraction_id,
            document_id,
            assessment_date,
            quality_tool,
            near_blank_threshold,
            expected_page_count,
            observed_page_count,
            page_count_match,
            blank_page_count,
            near_blank_page_count,
            replacement_char_page_count,
            manual_review_page_count,
            review_reason_summary,
            min_nonspace_chars,
            max_nonspace_chars,
            mean_nonspace_chars,
            quality_status,
            quality_checked_json_path,
            page_quality_csv_path,
            created_at,
            notes
        )
        VALUES (
            :assessment_id,
            :extraction_id,
            :document_id,
            :assessment_date,
            :quality_tool,
            :near_blank_threshold,
            :expected_page_count,
            :observed_page_count,
            :page_count_match,
            :blank_page_count,
            :near_blank_page_count,
            :replacement_char_page_count,
            :manual_review_page_count,
            :review_reason_summary,
            :min_nonspace_chars,
            :max_nonspace_chars,
            :mean_nonspace_chars,
            :quality_status,
            :quality_checked_json_path,
            :page_quality_csv_path,
            :created_at,
            :notes
        )
        """,
        event,
    )


def insert_quality_page_flags(
    connection: sqlite3.Connection,
    *,
    assessment_id: str,
    extraction_event: sqlite3.Row,
    page_assessments: list[dict[str, Any]],
) -> None:
    """Insert page-level quality flags."""
    rows = []

    for page in page_assessments:
        rows.append(
            {
                "assessment_id": assessment_id,
                "extraction_id": extraction_event["extraction_id"],
                "document_id": extraction_event["document_id"],
                "page_number": page["page_number"],
                "total_char_count": page["total_char_count"],
                "nonspace_char_count": page["nonspace_char_count"],
                "word_count": page["word_count"],
                "line_count": page["line_count"],
                "blank_extraction": int(page["blank_extraction"]),
                "near_blank_extraction": int(page["near_blank_extraction"]),
                "replacement_char_count": page["replacement_char_count"],
                "requires_manual_review": int(page["requires_manual_review"]),
                "review_reason_code": page["review_reason_code"],
                "review_reason_note": page["review_reason_note"],
                "quality_status": page["quality_status"],
                "notes": page["notes"],
            }
        )

    connection.executemany(
        """
        INSERT INTO quality_page_flags (
            assessment_id,
            extraction_id,
            document_id,
            page_number,
            total_char_count,
            nonspace_char_count,
            word_count,
            line_count,
            blank_extraction,
            near_blank_extraction,
            replacement_char_count,
            requires_manual_review,
            review_reason_code,
            review_reason_note,
            quality_status,
            notes
        )
        VALUES (
            :assessment_id,
            :extraction_id,
            :document_id,
            :page_number,
            :total_char_count,
            :nonspace_char_count,
            :word_count,
            :line_count,
            :blank_extraction,
            :near_blank_extraction,
            :replacement_char_count,
            :requires_manual_review,
            :review_reason_code,
            :review_reason_note,
            :quality_status,
            :notes
        )
        """,
        rows,
    )


def run_quality_assessment(
    *,
    document_id: str | None,
    extraction_id: str | None,
    near_blank_threshold: int,
) -> dict[str, Any]:
    """Run Sub-task 1.3."""
    if not document_id and not extraction_id:
        raise ValueError("Provide either --document-id or --extraction-id.")

    database_path = choose_database_path()
    connection = connect_database(database_path)

    try:
        ensure_quality_tables(connection)

        if extraction_id:
            extraction_event = get_extraction_by_id(connection, extraction_id)
        else:
            extraction_event = get_latest_extraction_for_document(connection, document_id)

        assessment_id = next_assessment_id(connection)

        page_json_path = resolve_project_path(extraction_event["page_json_path"])
        if not page_json_path.exists():
            raise FileNotFoundError(
                f"Page-level JSON not found: {page_json_path}\n"
                "Check the page_json_path stored in extraction_events."
            )

        with page_json_path.open("r", encoding="utf-8") as file:
            input_payload = json.load(file)

        pages = input_payload.get("pages", [])
        page_assessments = [
            assess_page(page, near_blank_threshold=near_blank_threshold)
            for page in pages
        ]

        summary = build_quality_summary(
            extraction_event=extraction_event,
            page_assessments=page_assessments,
            near_blank_threshold=near_blank_threshold,
        )

        document_output_dir = QUALITY_CHECKED_DIR / extraction_event["document_id"]
        report_output_dir = QUALITY_REPORT_DIR / extraction_event["document_id"]

        quality_checked_json_path = (
            document_output_dir
            / f'{extraction_event["extraction_id"]}_quality_checked.json'
        )
        page_quality_csv_path = (
            report_output_dir
            / f'{extraction_event["extraction_id"]}_page_quality.csv'
        )

        write_quality_checked_json(
            quality_checked_json_path,
            input_payload=input_payload,
            assessment_id=assessment_id,
            extraction_event=extraction_event,
            summary=summary,
            page_assessments=page_assessments,
        )

        write_page_quality_csv(page_quality_csv_path, page_assessments)

        timestamp = utc_now_iso()
        assessment_event = {
            "assessment_id": assessment_id,
            "extraction_id": extraction_event["extraction_id"],
            "document_id": extraction_event["document_id"],
            "assessment_date": timestamp[:10],
            "quality_tool": "03_assess_extraction_quality.py",
            "near_blank_threshold": near_blank_threshold,
            "expected_page_count": summary["expected_page_count"],
            "observed_page_count": summary["observed_page_count"],
            "page_count_match": int(summary["page_count_match"]),
            "blank_page_count": summary["blank_page_count"],
            "near_blank_page_count": summary["near_blank_page_count"],
            "replacement_char_page_count": summary["replacement_char_page_count"],
            "manual_review_page_count": summary["manual_review_page_count"],
            "review_reason_summary": summary["review_reason_summary"],
            "min_nonspace_chars": summary["min_nonspace_chars"],
            "max_nonspace_chars": summary["max_nonspace_chars"],
            "mean_nonspace_chars": summary["mean_nonspace_chars"],
            "quality_status": summary["quality_status"],
            "quality_checked_json_path": project_relative(quality_checked_json_path),
            "page_quality_csv_path": project_relative(page_quality_csv_path),
            "created_at": timestamp,
            "notes": summary["notes"],
        }

        insert_quality_assessment_event(connection, assessment_event)
        insert_quality_page_flags(
            connection,
            assessment_id=assessment_id,
            extraction_event=extraction_event,
            page_assessments=page_assessments,
        )

        connection.commit()

        print("\nQuality assessment completed")
        print("----------------------------")
        for key, value in assessment_event.items():
            print(f"{key}: {value}")

        print("\nGenerated files")
        print("---------------")
        print(project_relative(quality_checked_json_path))
        print(project_relative(page_quality_csv_path))

        print("\nDatabase used")
        print("-------------")
        print(project_relative(database_path))

        return assessment_event

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sub-task 1.3 extraction-quality assessment."
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help=(
            "Document ID. If --extraction-id is not supplied, the latest "
            "extraction event for this document will be assessed."
        ),
    )
    parser.add_argument(
        "--extraction-id",
        default=None,
        help="Specific extraction event to assess, for example EXT_000001.",
    )
    parser.add_argument(
        "--near-blank-threshold",
        type=int,
        default=50,
        help="Minimum number of non-space characters required before a page is not near-blank.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        run_quality_assessment(
            document_id=args.document_id,
            extraction_id=args.extraction_id,
            near_blank_threshold=args.near_blank_threshold,
        )
    except Exception as error:
        raise SystemExit(f"\nQuality assessment failed: {error}") from error
