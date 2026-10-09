#!/usr/bin/env python3
"""
06_segment_candidate_statements.py

Task 2, first implementation: Spanish sentence segmentation and
candidate-statement storage.

The script reads substantive text-bearing elements produced by Sub-task 1.5,
segments each element independently with a configurable Spanish spaCy pipeline,
and applies conservative post-processing for Spanish legal abbreviations.

It creates:
1. a hierarchical-metadata JSON file with candidate statements,
2. a one-row-per-statement CSV file,
3. segmentation_events and candidate_statements records in SQLite.

Character offsets use Python slice semantics: source_char_start is inclusive
and source_char_end is exclusive relative to structure_elements.element_text.

Run from the project root:

python codes/task_2_statement_segmentation/06_segment_candidate_statements.py \
  --document-id CL_MMA_DEC_000001

python codes/task_2_statement_segmentation/06_segment_candidate_statements.py \
  --structure-id STRUCT_000009
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from config import (
        DATABASE_PATH,
        STATEMENT_SEGMENTATION_DIR,
        STATEMENT_SEGMENTATION_REPORT_DIR,
    )
except ImportError as error:
    raise SystemExit(
        "Could not import codes/config.py. Store this file as:\n"
        "  codes/task_2_statement_segmentation/06_segment_candidate_statements.py"
    ) from error


# ---------------------------------------------------------------------
# Workflow constants
# ---------------------------------------------------------------------
SEGMENTATION_VERSION = "0.2"
SEGMENTATION_TOOL = "06_segment_candidate_statements.py"
DEFAULT_SPACY_MODEL = "es_core_news_md"
SEGMENTATION_METHOD = "spacy_sentence_boundaries_with_legal_postprocessing"

INCLUDED_ELEMENT_TYPES = (
    "paragraph",
    "numbered_item",
    "lettered_item",
    "compound_item",
    "roman_item",
    "bullet_item",
)

COMPLETED_STRUCTURE_STATUSES = (
    "Completed",
    "Completed with warnings",
)

PROTECTED_TOKEN_CASES = (
    "Art.",
    "art.",
    "Arts.",
    "arts.",
    "N°",
    "n°",
    "Nº",
    "nº",
    "D.S.",
    "d.s.",
    "D.L.",
    "d.l.",
    "D.F.L.",
    "d.f.l.",
    "Sr.",
    "sr.",
    "Sra.",
    "sra.",
    "Núm.",
    "núm.",
    "No.",
    "no.",
    "etc.",
    "Etc.",
)

LETTER_PATTERN = r"A-Za-zÁÉÍÓÚÜÑáéíóúüñ"
ARTICLE_ABBREVIATION_RE = re.compile(r"\bArts?\.$", re.IGNORECASE)
LEGAL_DECREE_ABBREVIATION_RE = re.compile(
    r"\b(?:D\.S\.|D\.L\.|D\.F\.L\.)$",
    re.IGNORECASE,
)
PERSON_ABBREVIATION_RE = re.compile(r"\b(?:Sr|Sra)\.$", re.IGNORECASE)
NUMBER_ABBREVIATION_RE = re.compile(r"\b(?:Núm|No)\.$", re.IGNORECASE)
ETC_ABBREVIATION_RE = re.compile(r"\betc\.$", re.IGNORECASE)
INITIAL_RE = re.compile(rf"(?:^|\s)[{LETTER_PATTERN}]\.$")
MULTI_PERIOD_ABBREVIATION_RE = re.compile(
    rf"(?:^|\s)(?:[{LETTER_PATTERN}]\.){{2,}}$"
)
# A numbered marker is protected only when the entire text to the left of a
# proposed boundary is the marker itself. A suffix-only pattern would also
# match ordinary sentence endings such as "Gráfico I 3." and incorrectly join
# the following grammatical sentence.
NUMBERED_MARKER_RE = re.compile(
    r"^\s*\(?\d{1,3}(?:\.\d{1,3}){0,3}[.)]?\s*$"
)
RIGHT_LEGAL_REFERENCE_RE = re.compile(
    r"^(?:N[°º]|Núm\.?|No\.?|\d|del?\b|de\b|que\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# SQLite schemas
# ---------------------------------------------------------------------
SEGMENTATION_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS segmentation_events (
    segmentation_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    segmentation_date TEXT NOT NULL,
    segmentation_tool TEXT NOT NULL,
    spacy_model TEXT NOT NULL,
    spacy_version TEXT NOT NULL,
    segmentation_version TEXT NOT NULL,
    source_elements_count INTEGER NOT NULL,
    candidate_statements_count INTEGER NOT NULL,
    manual_review_count INTEGER NOT NULL,
    candidate_json_path TEXT NOT NULL,
    candidate_csv_path TEXT NOT NULL,
    segmentation_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (structure_id)
        REFERENCES structure_events(structure_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);
"""

CANDIDATE_STATEMENTS_SQL = """
CREATE TABLE IF NOT EXISTS candidate_statements (
    statement_id TEXT NOT NULL,
    segmentation_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    structure_id TEXT NOT NULL,
    source_element_id TEXT NOT NULL,
    source_parent_id TEXT,
    source_element_type TEXT NOT NULL,
    source_element_order INTEGER NOT NULL,
    sentence_order_within_element INTEGER NOT NULL,
    statement_order INTEGER NOT NULL,
    statement_text TEXT NOT NULL,
    source_char_start INTEGER NOT NULL,
    source_char_end INTEGER NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    page_assignment_method TEXT NOT NULL,
    segmentation_method TEXT NOT NULL,
    segmentation_status TEXT NOT NULL,
    requires_manual_review INTEGER NOT NULL,
    review_reason_code TEXT,
    review_reason_note TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (segmentation_id, statement_id),
    UNIQUE (segmentation_id, statement_order),
    UNIQUE (
        segmentation_id,
        source_element_id,
        sentence_order_within_element
    ),
    FOREIGN KEY (segmentation_id)
        REFERENCES segmentation_events(segmentation_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (structure_id, source_element_id)
        REFERENCES structure_elements(structure_id, element_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);
"""

SEGMENTATION_INDEXES_SQL = (
    """
    CREATE INDEX IF NOT EXISTS idx_segmentation_events_document_created
        ON segmentation_events(document_id, created_at);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_segmentation_events_structure
        ON segmentation_events(structure_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_candidate_statements_document_order
        ON candidate_statements(document_id, segmentation_id, statement_order);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_candidate_statements_source
        ON candidate_statements(structure_id, source_element_id);
    """,
)


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def utc_now_iso() -> str:
    """Return the current UTC timestamp without microseconds."""
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
    """Store project-relative paths where possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(path_value: str) -> Path:
    """Resolve an absolute path or a path stored relative to the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Connect to SQLite and enforce foreign keys."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def ensure_segmentation_tables(connection: sqlite3.Connection) -> None:
    """Create Task 2 tables and indexes if they do not exist."""
    connection.execute(SEGMENTATION_EVENTS_SQL)
    connection.execute(CANDIDATE_STATEMENTS_SQL)
    for statement in SEGMENTATION_INDEXES_SQL:
        connection.execute(statement)


def next_segmentation_id(connection: sqlite3.Connection) -> str:
    """Generate process-event identifiers such as SEG_000001."""
    rows = connection.execute(
        "SELECT segmentation_id FROM segmentation_events"
    ).fetchall()

    highest = 0
    for row in rows:
        value = row["segmentation_id"]
        if value and value.startswith("SEG_"):
            suffix = value.replace("SEG_", "")
            if suffix.isdigit():
                highest = max(highest, int(suffix))

    return f"SEG_{highest + 1:06d}"


def get_structure_event(
    connection: sqlite3.Connection,
    *,
    document_id: str | None,
    structure_id: str | None,
) -> sqlite3.Row:
    """Resolve an explicit structure event or the latest completed event."""
    if structure_id:
        record = connection.execute(
            """
            SELECT *
            FROM structure_events
            WHERE structure_id = ?
            """,
            (structure_id,),
        ).fetchone()

        if record is None:
            raise ValueError(
                f"No structure event found for structure_id {structure_id!r}."
            )

        if record["structure_status"] not in COMPLETED_STRUCTURE_STATUSES:
            raise ValueError(
                f"Structure event {structure_id!r} is not completed "
                f"(status: {record['structure_status']!r})."
            )

        return record

    record = connection.execute(
        """
        SELECT *
        FROM structure_events
        WHERE document_id = ?
          AND structure_status IN (?, ?)
        ORDER BY created_at DESC, structure_id DESC
        LIMIT 1
        """,
        (
            document_id,
            COMPLETED_STRUCTURE_STATUSES[0],
            COMPLETED_STRUCTURE_STATUSES[1],
        ),
    ).fetchone()

    if record is None:
        raise ValueError(
            f"No completed structure event found for document_id {document_id!r}. "
            "Run Sub-task 1.5 first."
        )

    return record


def get_source_elements(
    connection: sqlite3.Connection,
    structure_id: str,
) -> list[sqlite3.Row]:
    """Return eligible structural elements in original document order."""
    placeholders = ", ".join("?" for _ in INCLUDED_ELEMENT_TYPES)
    return connection.execute(
        f"""
        SELECT
            structure_id,
            element_id,
            parent_id,
            document_id,
            element_order,
            element_type,
            element_text,
            page_start,
            page_end,
            requires_manual_review,
            review_reason_code,
            review_reason_note
        FROM structure_elements
        WHERE structure_id = ?
          AND element_type IN ({placeholders})
        ORDER BY element_order
        """,
        (structure_id, *INCLUDED_ELEMENT_TYPES),
    ).fetchall()


def get_document_record(
    connection: sqlite3.Connection,
    document_id: str,
) -> sqlite3.Row | None:
    """Return the corpus inventory row for document metadata."""
    return connection.execute(
        """
        SELECT *
        FROM document_inventory
        WHERE document_id = ?
        """,
        (document_id,),
    ).fetchone()


def load_structure_payload(structure_event: sqlite3.Row) -> dict[str, Any]:
    """Load the hierarchical Step 1.5 JSON for metadata provenance."""
    path = resolve_project_path(structure_event["structure_json_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Structure JSON not found: {path}\n"
            "Check structure_json_path in structure_events."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def document_metadata(
    structure_payload: dict[str, Any],
    inventory_record: sqlite3.Row | None,
) -> dict[str, Any]:
    """Prefer Step 1.5 metadata and fill only missing values from inventory."""
    metadata = dict(structure_payload.get("document_metadata", {}) or {})

    if inventory_record is not None:
        fallback_fields = (
            "document_title",
            "document_type",
            "source_country",
            "source_region",
            "issuing_institution",
            "publication_date",
            "promulgation_date",
            "effective_date",
            "language",
            "original_file_format",
        )
        for field in fallback_fields:
            if metadata.get(field) is None and field in inventory_record.keys():
                metadata[field] = inventory_record[field]

    return metadata


# ---------------------------------------------------------------------
# spaCy loading and legal-boundary post-processing
# ---------------------------------------------------------------------
def load_spacy_pipeline(model_name: str) -> tuple[Any, str]:
    """Load the requested Spanish model or raise an actionable error."""
    try:
        spacy = importlib.import_module("spacy")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "spaCy is not installed in the existing project environment.\n"
            "Install the pinned dependency with:\n"
            "  ig_env/bin/python -m pip install -r requirements.txt"
        ) from error

    try:
        nlp = spacy.load(
            model_name,
            disable=("ner", "lemmatizer", "attribute_ruler", "morphologizer"),
        )
    except OSError as error:
        raise RuntimeError(
            f"Spanish spaCy model {model_name!r} is not installed or cannot be loaded.\n"
            "Install it in the existing environment with:\n"
            f"  ig_env/bin/python -m spacy download {model_name}\n"
            "Do not substitute an English pipeline."
        ) from error

    if nlp.lang != "es":
        raise RuntimeError(
            f"spaCy model {model_name!r} reports language {nlp.lang!r}, not 'es'. "
            "Use a Spanish pipeline such as es_core_news_md."
        )

    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        raise RuntimeError(
            f"spaCy model {model_name!r} has no enabled parser or sentence recognizer. "
            "Sentence boundaries cannot be generated."
        )

    orth = spacy.symbols.ORTH
    for token_text in PROTECTED_TOKEN_CASES:
        nlp.tokenizer.add_special_case(token_text, [{orth: token_text}])

    return nlp, spacy.__version__


def strip_closing_punctuation(text: str) -> str:
    """Remove closing quotes/brackets when checking terminal punctuation."""
    return re.sub(r"""["'»”’)\]]+$""", "", text.rstrip()).rstrip()


def first_alpha_character(text: str) -> str | None:
    """Return the first Spanish alphabetic character in a string."""
    match = re.search(rf"[{LETTER_PATTERN}]", text)
    return match.group(0) if match else None


def starts_with_lowercase(text: str) -> bool:
    """Return True when the first alphabetic character is lowercase."""
    first = first_alpha_character(text)
    if first is None:
        return False
    return first == first.lower() and first != first.upper()


def starts_with_uppercase(text: str) -> bool:
    """Return True when the first alphabetic character is uppercase."""
    first = first_alpha_character(text)
    if first is None:
        return False
    return first == first.upper() and first != first.lower()


def protected_boundary_reason(
    source_text: str,
    left_end: int,
    right_start: int,
) -> str | None:
    """
    Return the reason a spaCy boundary must be merged.

    The inputs are offsets into one source element. The function never examines
    another structure element, so post-processing cannot join across elements.
    """
    left = source_text[:left_end].rstrip()
    right = source_text[right_start:].lstrip()

    if not left or not right:
        return None

    left_terminal = strip_closing_punctuation(left)
    right_clean = right.lstrip("\"'«“‘([{")

    if left_terminal.endswith((";", ":")):
        return "semicolon_or_colon"

    if not left_terminal.endswith((".", "?", "!")):
        return "nonterminal_or_linebreak"

    if left_terminal.endswith("...") and starts_with_lowercase(right_clean):
        return "ellipsis_continuation"

    if ARTICLE_ABBREVIATION_RE.search(left_terminal):
        if re.match(r"^(?:\d|N[°º]|Núm\.?|No\.?)", right_clean, re.IGNORECASE):
            return "article_abbreviation"

    if LEGAL_DECREE_ABBREVIATION_RE.search(left_terminal):
        if RIGHT_LEGAL_REFERENCE_RE.match(right_clean):
            return "legal_decree_abbreviation"

    if NUMBER_ABBREVIATION_RE.search(left_terminal):
        if re.match(r"^\d", right_clean):
            return "number_abbreviation"

    if PERSON_ABBREVIATION_RE.search(left_terminal):
        if starts_with_uppercase(right_clean):
            return "person_title_abbreviation"

    if ETC_ABBREVIATION_RE.search(left_terminal):
        if starts_with_lowercase(right_clean) or re.match(r"^\d", right_clean):
            return "etc_abbreviation"

    if MULTI_PERIOD_ABBREVIATION_RE.search(left_terminal):
        if RIGHT_LEGAL_REFERENCE_RE.match(right_clean) or starts_with_uppercase(right_clean):
            return "multi_period_abbreviation"

    if INITIAL_RE.search(left_terminal) and starts_with_uppercase(right_clean):
        return "initial"

    if NUMBERED_MARKER_RE.search(left_terminal):
        if first_alpha_character(right_clean) is not None:
            return "numbered_legal_marker"

    # Defensive protection if spaCy ever places a boundary inside an unspaced
    # decimal number, dotted date, abbreviation, or ellipsis.
    boundary_window = source_text[max(0, left_end - 8): min(len(source_text), right_start + 8)]
    if re.search(r"\d\.\d", boundary_window):
        return "numeric_expression"
    if re.search(r"\.{2,}", boundary_window) and left_end < right_start + 2:
        return "ellipsis_internal"

    return None


def trim_span(source_text: str, start: int, end: int) -> tuple[int, int]:
    """Trim only leading/trailing whitespace and return corrected offsets."""
    while start < end and source_text[start].isspace():
        start += 1
    while end > start and source_text[end - 1].isspace():
        end -= 1
    return start, end


def legal_sentence_spans(
    source_text: str,
    doc: Any,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """
    Merge unsafe spaCy boundaries and return exact source-text spans.

    Each output tuple is (inclusive_start, exclusive_end).
    """
    raw_spans = list(doc.sents)
    if not raw_spans:
        start, end = trim_span(source_text, 0, len(source_text))
        return ([(start, end)] if start < end else []), []

    merged_spans: list[tuple[int, int]] = []
    merged_boundaries: list[dict[str, Any]] = []
    current_start = int(raw_spans[0].start_char)
    current_end = int(raw_spans[0].end_char)

    for next_span in raw_spans[1:]:
        next_start = int(next_span.start_char)
        next_end = int(next_span.end_char)
        reason = protected_boundary_reason(source_text, current_end, next_start)

        if reason is not None:
            merged_boundaries.append(
                {
                    "left_end": current_end,
                    "right_start": next_start,
                    "reason": reason,
                }
            )
            current_end = next_end
            continue

        start, end = trim_span(source_text, current_start, current_end)
        if start < end:
            merged_spans.append((start, end))
        current_start = next_start
        current_end = next_end

    start, end = trim_span(source_text, current_start, current_end)
    if start < end:
        merged_spans.append((start, end))

    return merged_spans, merged_boundaries


def boundary_looks_like_abbreviation_split(left_text: str, right_text: str) -> bool:
    """Flag an accepted boundary that still follows a known abbreviation."""
    left = strip_closing_punctuation(left_text)
    if not right_text.strip():
        return False
    return bool(
        ARTICLE_ABBREVIATION_RE.search(left)
        or LEGAL_DECREE_ABBREVIATION_RE.search(left)
        or PERSON_ABBREVIATION_RE.search(left)
        or NUMBER_ABBREVIATION_RE.search(left)
        or ETC_ABBREVIATION_RE.search(left)
        or MULTI_PERIOD_ABBREVIATION_RE.search(left)
        or INITIAL_RE.search(left)
    )


def has_terminal_punctuation(text: str) -> bool:
    """Check period, question mark, or exclamation mark after closing marks."""
    clean = strip_closing_punctuation(text)
    return clean.endswith((".", "?", "!"))


def append_warning(
    warnings: list[tuple[str, str]],
    code: str,
    note: str,
) -> None:
    """Append a warning once while preserving deterministic order."""
    if code not in {existing_code for existing_code, _ in warnings}:
        warnings.append((code, note))


def statement_warnings(
    *,
    statement_text: str,
    next_statement_text: str | None,
    source_element: sqlite3.Row,
) -> list[tuple[str, str]]:
    """Return non-destructive review flags for one candidate statement."""
    warnings: list[tuple[str, str]] = []
    words = re.findall(rf"\b[{LETTER_PATTERN}0-9]+\b", statement_text)

    if len(statement_text) < 20 or len(words) < 3:
        append_warning(
            warnings,
            "very_short_sentence",
            "Candidate has fewer than 20 characters or fewer than three word-like tokens.",
        )

    if starts_with_lowercase(statement_text):
        append_warning(
            warnings,
            "sentence_starts_lowercase",
            "The first alphabetic character is lowercase; inspect for an over-split continuation.",
        )

    if next_statement_text and boundary_looks_like_abbreviation_split(
        statement_text,
        next_statement_text,
    ):
        append_warning(
            warnings,
            "possible_abbreviation_split",
            "An accepted boundary follows a known abbreviation and should be reviewed.",
        )

    if not has_terminal_punctuation(statement_text):
        append_warning(
            warnings,
            "missing_terminal_punctuation",
            "Candidate lacks final period, question mark, or exclamation mark; list items may be legitimate.",
        )

    if bool(source_element["requires_manual_review"]):
        inherited_code = source_element["review_reason_code"] or "unspecified"
        inherited_note = source_element["review_reason_note"] or (
            "The source structure element inherited a manual-review flag."
        )
        append_warning(
            warnings,
            f"source_element_review:{inherited_code}",
            inherited_note,
        )

    return warnings


# ---------------------------------------------------------------------
# Candidate creation and validation
# ---------------------------------------------------------------------
def build_candidate_statements(
    *,
    nlp: Any,
    segmentation_id: str,
    structure_event: sqlite3.Row,
    source_elements: list[sqlite3.Row],
    created_at: str,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    """Segment source elements and create ordered candidate records."""
    text_elements = [
        element
        for element in source_elements
        if element["element_text"] is not None
        and str(element["element_text"]).strip()
    ]
    source_texts = [str(element["element_text"]) for element in text_elements]

    candidates: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    merged_boundaries: list[dict[str, Any]] = []
    statement_order = 0

    docs: Iterable[Any] = nlp.pipe(source_texts, batch_size=64)
    for source_element, source_text, doc in zip(
        text_elements,
        source_texts,
        docs,
        strict=True,
    ):
        spans, element_merged_boundaries = legal_sentence_spans(source_text, doc)

        for boundary in element_merged_boundaries:
            merged_boundaries.append(
                {
                    "source_element_id": source_element["element_id"],
                    **boundary,
                }
            )

        span_texts = [source_text[start:end] for start, end in spans]
        for sentence_index, (start, end) in enumerate(spans, start=1):
            statement_order += 1
            statement_text = source_text[start:end]
            next_text = (
                span_texts[sentence_index]
                if sentence_index < len(span_texts)
                else None
            )
            warnings = statement_warnings(
                statement_text=statement_text,
                next_statement_text=next_text,
                source_element=source_element,
            )
            for code, _ in warnings:
                warning_counts[code] += 1

            candidate = {
                "statement_id": (
                    f"{structure_event['document_id']}_STMT_{statement_order:06d}"
                ),
                "segmentation_id": segmentation_id,
                "document_id": structure_event["document_id"],
                "structure_id": structure_event["structure_id"],
                "source_element_id": source_element["element_id"],
                "source_parent_id": source_element["parent_id"],
                "source_element_type": source_element["element_type"],
                "source_element_order": int(source_element["element_order"]),
                "sentence_order_within_element": sentence_index,
                "statement_order": statement_order,
                "statement_text": statement_text,
                "source_char_start": start,
                "source_char_end": end,
                "page_start": int(source_element["page_start"]),
                "page_end": int(source_element["page_end"]),
                "page_assignment_method": "element_inherited",
                "segmentation_method": SEGMENTATION_METHOD,
                "segmentation_status": (
                    "completed_with_warnings" if warnings else "completed"
                ),
                "requires_manual_review": int(bool(warnings)),
                "review_reason_code": (
                    ";".join(code for code, _ in warnings) if warnings else None
                ),
                "review_reason_note": (
                    " ".join(note for _, note in warnings) if warnings else None
                ),
                "created_at": created_at,
            }
            candidates.append(candidate)

    return candidates, warning_counts, merged_boundaries


def validate_candidates(
    *,
    candidates: list[dict[str, Any]],
    source_elements: list[sqlite3.Row],
) -> dict[str, Any]:
    """Run required in-memory traceability and boundary checks."""
    source_by_id = {
        element["element_id"]: element
        for element in source_elements
    }
    nonempty_element_ids = {
        element["element_id"]
        for element in source_elements
        if element["element_text"] is not None
        and str(element["element_text"]).strip()
    }
    produced_element_ids = {
        candidate["source_element_id"]
        for candidate in candidates
    }

    no_empty_statements = all(
        bool(candidate["statement_text"].strip())
        for candidate in candidates
    )
    sequential_order = [
        candidate["statement_order"]
        for candidate in candidates
    ] == list(range(1, len(candidates) + 1))
    valid_source_ids = all(
        candidate["source_element_id"] in source_by_id
        for candidate in candidates
    )
    all_nonempty_elements_produced = (
        nonempty_element_ids <= produced_element_ids
    )

    offsets_reproduce_text = True
    protected_boundary_violations: list[dict[str, Any]] = []
    candidates_by_element: dict[str, list[dict[str, Any]]] = {}

    for candidate in candidates:
        element = source_by_id[candidate["source_element_id"]]
        source_text = str(element["element_text"])
        start = int(candidate["source_char_start"])
        end = int(candidate["source_char_end"])
        if source_text[start:end] != candidate["statement_text"]:
            offsets_reproduce_text = False
        candidates_by_element.setdefault(
            candidate["source_element_id"],
            [],
        ).append(candidate)

    for element_id, element_candidates in candidates_by_element.items():
        source_text = str(source_by_id[element_id]["element_text"])
        for left, right in zip(
            element_candidates,
            element_candidates[1:],
        ):
            reason = protected_boundary_reason(
                source_text,
                int(left["source_char_end"]),
                int(right["source_char_start"]),
            )
            if reason is not None:
                protected_boundary_violations.append(
                    {
                        "source_element_id": element_id,
                        "left_statement_id": left["statement_id"],
                        "right_statement_id": right["statement_id"],
                        "reason": reason,
                    }
                )

    checks = {
        "no_empty_candidate_statements": no_empty_statements,
        "candidate_statement_order_is_sequential": sequential_order,
        "every_statement_has_valid_source_element_id": valid_source_ids,
        "every_nonempty_processed_element_produces_statement": (
            all_nonempty_elements_produced
        ),
        "character_offsets_reproduce_statement_text": offsets_reproduce_text,
        "protected_legal_abbreviation_boundary_violations": len(
            protected_boundary_violations
        ),
        "protected_boundary_violation_details": protected_boundary_violations,
    }

    failed = [
        name
        for name, result in checks.items()
        if name != "protected_boundary_violation_details"
        and (
            result is False
            or (
                name == "protected_legal_abbreviation_boundary_violations"
                and result != 0
            )
        )
    ]
    if failed:
        raise ValueError(
            "Candidate validation failed: " + ", ".join(failed)
        )

    return checks


# ---------------------------------------------------------------------
# Output and database persistence
# ---------------------------------------------------------------------
CANDIDATE_FIELDNAMES = [
    "statement_id",
    "segmentation_id",
    "document_id",
    "structure_id",
    "source_element_id",
    "source_parent_id",
    "source_element_type",
    "source_element_order",
    "sentence_order_within_element",
    "statement_order",
    "statement_text",
    "source_char_start",
    "source_char_end",
    "page_start",
    "page_end",
    "page_assignment_method",
    "segmentation_method",
    "segmentation_status",
    "requires_manual_review",
    "review_reason_code",
    "review_reason_note",
    "created_at",
]


def write_candidate_json(
    path: Path,
    *,
    document_metadata_payload: dict[str, Any],
    structure_event: sqlite3.Row,
    segmentation_metadata: dict[str, Any],
    quality_checks: dict[str, Any],
    warning_summary: dict[str, int],
    merged_boundaries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    """Write Task 2 JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": structure_event["document_id"],
        "document_metadata": document_metadata_payload,
        "structure_event_metadata": dict(structure_event),
        "segmentation_metadata": segmentation_metadata,
        "quality_checks": quality_checks,
        "warning_summary": warning_summary,
        "merged_protected_boundaries": merged_boundaries,
        "candidate_statements": candidates,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_candidate_csv(
    path: Path,
    candidates: list[dict[str, Any]],
) -> None:
    """Write one human-readable CSV row per candidate statement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CANDIDATE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(candidates)


def insert_segmentation_event(
    connection: sqlite3.Connection,
    event: dict[str, Any],
) -> None:
    """Insert one process-level Task 2 event."""
    connection.execute(
        """
        INSERT INTO segmentation_events (
            segmentation_id,
            structure_id,
            document_id,
            segmentation_date,
            segmentation_tool,
            spacy_model,
            spacy_version,
            segmentation_version,
            source_elements_count,
            candidate_statements_count,
            manual_review_count,
            candidate_json_path,
            candidate_csv_path,
            segmentation_status,
            created_at,
            notes
        )
        VALUES (
            :segmentation_id,
            :structure_id,
            :document_id,
            :segmentation_date,
            :segmentation_tool,
            :spacy_model,
            :spacy_version,
            :segmentation_version,
            :source_elements_count,
            :candidate_statements_count,
            :manual_review_count,
            :candidate_json_path,
            :candidate_csv_path,
            :segmentation_status,
            :created_at,
            :notes
        )
        """,
        event,
    )


def insert_candidate_statements(
    connection: sqlite3.Connection,
    candidates: list[dict[str, Any]],
) -> None:
    """Insert observation-level candidate statements."""
    connection.executemany(
        """
        INSERT INTO candidate_statements (
            statement_id,
            segmentation_id,
            document_id,
            structure_id,
            source_element_id,
            source_parent_id,
            source_element_type,
            source_element_order,
            sentence_order_within_element,
            statement_order,
            statement_text,
            source_char_start,
            source_char_end,
            page_start,
            page_end,
            page_assignment_method,
            segmentation_method,
            segmentation_status,
            requires_manual_review,
            review_reason_code,
            review_reason_note,
            created_at
        )
        VALUES (
            :statement_id,
            :segmentation_id,
            :document_id,
            :structure_id,
            :source_element_id,
            :source_parent_id,
            :source_element_type,
            :source_element_order,
            :sentence_order_within_element,
            :statement_order,
            :statement_text,
            :source_char_start,
            :source_char_end,
            :page_start,
            :page_end,
            :page_assignment_method,
            :segmentation_method,
            :segmentation_status,
            :requires_manual_review,
            :review_reason_code,
            :review_reason_note,
            :created_at
        )
        """,
        candidates,
    )


def verify_persisted_outputs(
    *,
    connection: sqlite3.Connection,
    segmentation_id: str,
    json_path: Path,
    csv_path: Path,
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    """Verify SQLite, JSON, and CSV counts and exact statement text."""
    with json_path.open("r", encoding="utf-8") as file:
        json_candidates = json.load(file)["candidate_statements"]

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        csv_candidates = list(csv.DictReader(file))

    database_rows = connection.execute(
        """
        SELECT statement_id, statement_order, statement_text
        FROM candidate_statements
        WHERE segmentation_id = ?
        ORDER BY statement_order
        """,
        (segmentation_id,),
    ).fetchall()

    expected = [
        (
            candidate["statement_id"],
            int(candidate["statement_order"]),
            candidate["statement_text"],
        )
        for candidate in candidates
    ]
    from_json = [
        (
            candidate["statement_id"],
            int(candidate["statement_order"]),
            candidate["statement_text"],
        )
        for candidate in json_candidates
    ]
    from_csv = [
        (
            candidate["statement_id"],
            int(candidate["statement_order"]),
            candidate["statement_text"],
        )
        for candidate in csv_candidates
    ]
    from_database = [
        (
            row["statement_id"],
            int(row["statement_order"]),
            row["statement_text"],
        )
        for row in database_rows
    ]

    if expected != from_json:
        raise ValueError("JSON candidate statements differ from in-memory output.")
    if expected != from_csv:
        raise ValueError("CSV candidate statements differ from in-memory output.")
    if expected != from_database:
        raise ValueError("SQLite candidate statements differ from in-memory output.")

    return {
        "expected_rows": len(expected),
        "sqlite_rows": len(from_database),
        "json_rows": len(from_json),
        "csv_rows": len(from_csv),
    }


def run_segmentation(
    *,
    document_id: str | None,
    structure_id: str | None,
    spacy_model: str,
) -> dict[str, Any]:
    """Run Task 2 sentence segmentation and persistence."""
    database_path = choose_database_path()
    connection = connect_database(database_path)

    try:
        ensure_segmentation_tables(connection)
        structure_event = get_structure_event(
            connection,
            document_id=document_id,
            structure_id=structure_id,
        )
        source_elements = get_source_elements(
            connection,
            structure_event["structure_id"],
        )
        inventory_record = get_document_record(
            connection,
            structure_event["document_id"],
        )
        structure_payload = load_structure_payload(structure_event)

        nlp, spacy_version = load_spacy_pipeline(spacy_model)
        segmentation_id = next_segmentation_id(connection)
        timestamp = utc_now_iso()

        candidates, warning_counts, merged_boundaries = build_candidate_statements(
            nlp=nlp,
            segmentation_id=segmentation_id,
            structure_event=structure_event,
            source_elements=source_elements,
            created_at=timestamp,
        )
        quality_checks = validate_candidates(
            candidates=candidates,
            source_elements=source_elements,
        )

        manual_review_count = sum(
            int(candidate["requires_manual_review"])
            for candidate in candidates
        )
        segmentation_status = (
            "Completed with warnings"
            if manual_review_count > 0
            else "Completed"
        )
        notes = (
            "; ".join(
                f"{code}: {count}"
                for code, count in sorted(warning_counts.items())
            )
            if warning_counts
            else None
        )

        output_dir = (
            STATEMENT_SEGMENTATION_DIR
            / structure_event["document_id"]
        )
        report_dir = (
            STATEMENT_SEGMENTATION_REPORT_DIR
            / structure_event["document_id"]
        )
        candidate_json_path = output_dir / (
            f"{structure_event['structure_id']}_{segmentation_id}"
            "_candidate_statements.json"
        )
        candidate_csv_path = report_dir / (
            f"{structure_event['structure_id']}_{segmentation_id}"
            "_candidate_statements.csv"
        )

        event = {
            "segmentation_id": segmentation_id,
            "structure_id": structure_event["structure_id"],
            "document_id": structure_event["document_id"],
            "segmentation_date": timestamp[:10],
            "segmentation_tool": SEGMENTATION_TOOL,
            "spacy_model": spacy_model,
            "spacy_version": spacy_version,
            "segmentation_version": SEGMENTATION_VERSION,
            "source_elements_count": len(source_elements),
            "candidate_statements_count": len(candidates),
            "manual_review_count": manual_review_count,
            "candidate_json_path": project_relative(candidate_json_path),
            "candidate_csv_path": project_relative(candidate_csv_path),
            "segmentation_status": segmentation_status,
            "created_at": timestamp,
            "notes": notes,
        }

        segmentation_metadata = {
            **event,
            "included_element_types": list(INCLUDED_ELEMENT_TYPES),
            "page_assignment_method": "element_inherited",
            "segmentation_method": SEGMENTATION_METHOD,
            "source_char_end_semantics": "exclusive",
            "merged_protected_boundaries_count": len(merged_boundaries),
        }

        insert_segmentation_event(connection, event)
        insert_candidate_statements(connection, candidates)
        write_candidate_json(
            candidate_json_path,
            document_metadata_payload=document_metadata(
                structure_payload,
                inventory_record,
            ),
            structure_event=structure_event,
            segmentation_metadata=segmentation_metadata,
            quality_checks=quality_checks,
            warning_summary=dict(sorted(warning_counts.items())),
            merged_boundaries=merged_boundaries,
            candidates=candidates,
        )
        write_candidate_csv(candidate_csv_path, candidates)

        output_counts = verify_persisted_outputs(
            connection=connection,
            segmentation_id=segmentation_id,
            json_path=candidate_json_path,
            csv_path=candidate_csv_path,
            candidates=candidates,
        )

        if len(set(output_counts.values())) != 1:
            raise ValueError(
                "SQLite, JSON, and CSV candidate counts do not match."
            )

        connection.commit()

        print("\nCandidate-statement segmentation completed")
        print("------------------------------------------")
        for key, value in event.items():
            print(f"{key}: {value}")

        print("\nQuality checks")
        print("--------------")
        for key, value in quality_checks.items():
            if key != "protected_boundary_violation_details":
                print(f"{key}: {value}")
        print(f"merged_protected_boundaries: {len(merged_boundaries)}")
        for key, value in output_counts.items():
            print(f"{key}: {value}")

        print("\nGenerated files")
        print("---------------")
        print(project_relative(candidate_json_path))
        print(project_relative(candidate_csv_path))

        print("\nDatabase used")
        print("-------------")
        print(project_relative(database_path))

        return {
            **event,
            "quality_checks": quality_checks,
            "warning_summary": dict(sorted(warning_counts.items())),
            "merged_protected_boundaries_count": len(merged_boundaries),
            "output_counts": output_counts,
        }

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Task 2 Spanish sentence segmentation and candidate storage."
        )
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--document-id",
        default=None,
        help=(
            "Document ID. Selects the latest completed structure event for "
            "the document."
        ),
    )
    selector.add_argument(
        "--structure-id",
        default=None,
        help="Explicit completed structure event, for example STRUCT_000009.",
    )
    parser.add_argument(
        "--spacy-model",
        default=DEFAULT_SPACY_MODEL,
        help=(
            "Spanish spaCy pipeline name. Default: es_core_news_md. "
            "The script does not fall back to another language."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        run_segmentation(
            document_id=args.document_id,
            structure_id=args.structure_id,
            spacy_model=args.spacy_model,
        )
    except Exception as error:
        raise SystemExit(f"\nSentence segmentation failed: {error}") from error
