#!/usr/bin/env python3
"""
04_normalize_text.py

Sub-task 1.4: Text Normalization.

This script reads the quality-checked JSON created in Sub-task 1.3 and creates:

1. a normalized page-level JSON file,
2. a page-level CSV normalization report,
3. SQLite records for the normalization event and page-level normalization stats.

Important design choices:
- raw_text is never overwritten.
- normalized_text is stored as a separate field.
- Spanish accents are preserved in normalized_text.
- An auxiliary matching_text field is created for accent-insensitive matching.
- Technical repository headers/footers are removed only when they match explicit patterns.
- Legal structure markers such as CAPÍTULO, Artículo, numbered items, lettered items,
  tables, and figures are preserved.
- Repository metadata key-value lines such as Tipo Norma, Fecha Publicación,
  Organismo, Título, Id Norma, and URL are preserved as separate lines.

Run from the project root:

python codes/task_1_text_preparation/04_normalize_text.py \
  --document-id CL_MMA_DEC_000001

or for a specific quality assessment:

python codes/task_1_text_preparation/04_normalize_text.py \
  --assessment-id QA_000001
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
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
    from config import DATABASE_PATH, NORMALIZED_DIR, NORMALIZATION_REPORT_DIR
except ImportError as error:
    raise SystemExit(
        "Could not import config.py. Make sure this file is stored as:\n"
        "  codes/task_1_text_preparation/04_normalize_text.py\n"
        "and config.py is stored as:\n"
        "  codes/config.py"
    ) from error


NORMALIZATION_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS normalization_events (
    normalization_id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL,
    extraction_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    normalization_date TEXT NOT NULL,
    normalization_tool TEXT NOT NULL,
    normalization_rules_json TEXT NOT NULL,
    pages_normalized_count INTEGER NOT NULL,
    pages_with_warnings_count INTEGER NOT NULL,
    normalized_json_path TEXT NOT NULL,
    page_normalization_csv_path TEXT NOT NULL,
    normalization_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (assessment_id)
        REFERENCES quality_assessment_events(assessment_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
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


NORMALIZATION_PAGE_STATS_SQL = """
CREATE TABLE IF NOT EXISTS normalization_page_stats (
    normalization_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    raw_char_count INTEGER NOT NULL,
    normalized_char_count INTEGER NOT NULL,
    matching_text_char_count INTEGER NOT NULL,
    removed_technical_lines_count INTEGER NOT NULL,
    replacement_char_count INTEGER NOT NULL,
    quality_requires_manual_review INTEGER NOT NULL,
    normalization_status TEXT NOT NULL,
    normalization_warning TEXT,
    PRIMARY KEY (normalization_id, page_number),
    FOREIGN KEY (normalization_id)
        REFERENCES normalization_events(normalization_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);
"""


NORMALIZATION_RULES = {
    "unicode_normalization": "NFC composition normalization for main normalized_text",
    "targeted_compatibility_replacements": "typographic ligatures and selected symbols only; legal N°/Nº markers are preserved",
    "quotation_mark_standardization": "typographic quotes converted to plain quotes",
    "dash_standardization": "en/em/minus dashes converted to hyphen-minus",
    "space_normalization": "non-breaking and irregular spaces converted to ordinary spaces",
    "soft_hyphen_removal": "soft hyphen and line-break hyphenation removed conservatively",
    "zero_width_character_removal": "zero-width characters removed",
    "technical_content_removal": "explicit repository headers/footers and HTML tags removed from normalized_text",
    "legal_structure_preservation": "legal structure markers are preserved as block-start markers; title/caption/source continuations may be joined only when the marker block lacks clear sentence-ending punctuation and the following line does not start a new marker",
    "metadata_line_preservation": "repository metadata key-value lines are preserved as separate lines",
    "matching_text": "accent-insensitive lowercase auxiliary field generated separately from normalized_text",
}


TECHNICAL_LINE_PATTERNS = [
    re.compile(
        r"^\s*Biblioteca del Congreso Nacional de Chile\s*-\s*www\.leychile\.cl\s*-\s*documento generado el\b.*$",
        flags=re.IGNORECASE,
    ),
    re.compile(r"^\s*Página\s+\d+\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*Page\s+\d+\s*$", flags=re.IGNORECASE),
]

STRUCTURAL_MARKER_PATTERN = re.compile(
    r"""^\s*(?:
        (?:CAP[IÍ]TULO|Cap[íi]tulo)\b|
        (?:T[IÍ]TULO|T[íi]tulo)\b|
        (?:SECCI[OÓ]N|Secci[oó]n)\b|
        (?:SUBSECCI[OÓ]N|Subsecci[oó]n)\b|
        (?:P[ÁA]RRAFO|P[áa]rrafo)\b|

        # Article heading: requires punctuation or end of line
        (?:ART[IÍ]CULO|Art[íi]culo)
        \s+\d+[A-Za-z°º]?
        \s*(?=[:.\-–—]|$)|

        # Roman-decimal headings: I.1, I.2.1, VI.6.2
        [IVXLCDM]+\.\d+(?:\.\d+)*\b|

        # List-item markers
        [0-9]+[.)]\s+|
        [a-zA-Z]\)\s+|
        [ivxlcdm]+\)\s+|

        # Visual and supplementary elements
        (?:Tabla|TABLA)\b|
        (?:Gr[áa]fico|GR[ÁA]FICO)\b|
        (?:Figura|FIGURA)\b|
        (?:Anexo|ANEXO)\b|
        (?:Fuente|FUENTE):
    )""",
    flags=re.VERBOSE,
)

# LeyChile-style metadata lines are not ordinary wrapped sentences.
# They should remain as separate lines during normalization.
METADATA_LINE_PATTERN = re.compile(
    r"""^\s*(
        Tipo\s+Norma|
        Fecha\s+Publicaci[oó]n|
        Fecha\s+Promulgaci[oó]n|
        Organismo|
        T[ií]tulo|
        Tipo\s+Versi[oó]n|
        Inicio\s+Vigencia|
        Id\s+Norma|
        URL
    )\s*:""",
    flags=re.IGNORECASE | re.VERBOSE,
)

TERMINAL_PUNCTUATION = tuple(".;:?!)]}\"'")


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
    """Resolve paths stored in SQLite."""
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


def ensure_normalization_tables(connection: sqlite3.Connection) -> None:
    """Create normalization tables."""
    connection.execute(NORMALIZATION_EVENTS_SQL)
    connection.execute(NORMALIZATION_PAGE_STATS_SQL)


def get_latest_assessment_for_document(
    connection: sqlite3.Connection,
    document_id: str,
) -> sqlite3.Row:
    """Return the most recent quality assessment for a document."""
    record = connection.execute(
        """
        SELECT *
        FROM quality_assessment_events
        WHERE document_id = ?
        ORDER BY created_at DESC, assessment_id DESC
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()

    if record is None:
        raise ValueError(
            f"No quality assessment found for document_id {document_id!r}. "
            "Run Sub-task 1.3 first."
        )

    return record


def get_assessment_by_id(
    connection: sqlite3.Connection,
    assessment_id: str,
) -> sqlite3.Row:
    """Return one quality assessment event by assessment_id."""
    record = connection.execute(
        """
        SELECT *
        FROM quality_assessment_events
        WHERE assessment_id = ?
        """,
        (assessment_id,),
    ).fetchone()

    if record is None:
        raise ValueError(
            f"No quality assessment found for assessment_id {assessment_id!r}."
        )

    return record


def next_normalization_id(connection: sqlite3.Connection) -> str:
    """Generate normalization identifiers such as NORM_000001."""
    rows = connection.execute(
        "SELECT normalization_id FROM normalization_events"
    ).fetchall()

    highest = 0
    for row in rows:
        value = row["normalization_id"]
        if value and value.startswith("NORM_"):
            suffix = value.replace("NORM_", "")
            if suffix.isdigit():
                highest = max(highest, int(suffix))

    return f"NORM_{highest + 1:06d}"


def remove_accents(text: str) -> str:
    """
    Create an accent-insensitive version for matching_text.

    This is not used for normalized_text.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in decomposed
        if unicodedata.category(char) != "Mn"
    )
    return unicodedata.normalize("NFC", without_marks)


def standardize_characters(text: str) -> str:
    """
    Normalize characters conservatively.

    We use NFC instead of global NFKC because NFKC may alter legally meaningful
    symbols such as ordinal indicators in legal numbering. Selected compatibility
    replacements are applied explicitly.
    """
    text = "" if text is None else str(text)

    # Decode common HTML entities if present.
    text = html.unescape(text)

    # Unicode composition normalization. This preserves accents.
    text = unicodedata.normalize("NFC", text)

    # Targeted compatibility replacements.
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u00ad": "",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "—": "-",
        "–": "-",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "―": "-",
        "…": "...",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def is_technical_line(line: str) -> bool:
    """Return True when a line is clear repository-generated noise."""
    return any(pattern.match(line) for pattern in TECHNICAL_LINE_PATTERNS)


def remove_html_tags(text: str) -> str:
    """Remove simple HTML/XML tags if present."""
    return re.sub(r"<[^>\n]+>", "", text)


def remove_technical_lines(text: str) -> tuple[str, int]:
    """
    Remove technical repository-generated lines.

    This is intentionally narrow to avoid deleting legal titles, article headings,
    tables, figures, or definitions.
    """
    kept_lines: list[str] = []
    removed_count = 0

    for line in text.splitlines():
        if is_technical_line(line):
            removed_count += 1
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines), removed_count


def repair_linebreak_hyphenation(text: str) -> str:
    """
    Join words broken by line-break hyphenation.

    Example:
        regula-
        ción
    becomes:
        regulación
    """
    return re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)


def is_structural_marker(line: str) -> bool:
    """Detect lines that should keep their own line break."""
    return bool(STRUCTURAL_MARKER_PATTERN.match(line))


def is_metadata_line(line: str) -> bool:
    """Detect repository metadata key-value lines that should keep line breaks."""
    return bool(METADATA_LINE_PATTERN.match(line))


def should_preserve_line_break(line: str) -> bool:
    """Return True when the line is a structural or metadata boundary."""
    return is_structural_marker(line) or is_metadata_line(line)


CONTINUABLE_MARKER_PATTERN = re.compile(
    r"""^\s*(
        CAP[IÍ]TULO\b|
        T[IÍ]TULO\b|
        SECCI[OÓ]N\b|
        SUBSECCI[OÓ]N\b|
        P[ÁA]RRAFO\b|
        [IVXLCDM]+\.\d+(\.\d+)*\b|
        Tabla\b|
        Gr[áa]fico\b|
        Figura\b|
        Anexo\b|
        Fuente:
    )""",
    flags=re.IGNORECASE | re.VERBOSE,
)



def is_continuable_marker(line: str) -> bool:
    """Markers whose title/caption/source note may continue in the next line."""
    return bool(CONTINUABLE_MARKER_PATTERN.match(line))


def has_clear_sentence_end(line: str) -> bool:
    """
    Return True when a line appears to end a complete sentence or heading.

    This is stricter than TERMINAL_PUNCTUATION. A closing parenthesis alone
    does not count as a clear sentence end, but a period, question mark, or
    exclamation mark followed by an optional closing parenthesis/bracket does.
    """
    clean = line.strip()

    if not clean:
        return False

    # Remove closing quotes that may appear after punctuation.
    clean = re.sub(r"[\"']+$", "", clean).strip()

    # Clear endings:
    #   texto.
    #   texto.)
    #   texto].
    #   texto?
    #   texto!
    return bool(re.search(r"(\.|\?|\!)(\)|\])?$", clean))


def should_join_marker_continuation(previous: str, current: str) -> bool:
    """
    Join a continuation line after a structural marker only when the marker
    block does not already have a clear sentence/heading ending.
    """
    if not is_continuable_marker(previous):
        return False

    if has_clear_sentence_end(previous):
        return False

    if should_preserve_line_break(current):
        return False

    return True


def is_incomplete_article_marker(previous: str, current: str) -> bool:
    """
    Detect article markers that were split from their opening text.
    Example:
        Artículo 57
        : Se entenderá...
    """
    previous_clean = previous.strip()
    current_clean = current.strip()

    article_only = re.match(
        r"^Art[íi]culo\s+\d+\s*$",
        previous_clean,
        flags=re.IGNORECASE,
    )

    current_starts_continuation = current_clean.startswith((":", "-", ".-"))

    return bool(article_only and current_starts_continuation)



def clean_line_spacing(text: str) -> str:
    """
    Normalize spaces and line breaks while preserving legal structure markers.

    The procedure joins obvious line-wrapped text and keeps new lines before
    explicit legal structure markers. Some marker blocks (chapter headings,
    numbered headings, visual captions, annexes, and source notes) may continue
    across following lines, but only while the marker block has no clear
    sentence-ending punctuation and the following line is not a new marker.
    """
    # Standardize line breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Reduce spaces/tabs inside each line.
    raw_lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_lines]

    # Remove repeated blank lines.
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and previous_blank:
            continue
        compact_lines.append(line)
        previous_blank = is_blank

    # Join wrapped lines conservatively.
    output_lines: list[str] = []

    for line in compact_lines:
        if line == "":
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue

        if not output_lines or output_lines[-1] == "":
            output_lines.append(line)
            continue

        previous = output_lines[-1]

        current_is_structure = should_preserve_line_break(line)

        if current_is_structure:
            output_lines.append(line)

        elif should_join_marker_continuation(previous, line):
            output_lines[-1] = previous + " " + line

        elif is_incomplete_article_marker(previous, line):
            output_lines[-1] = previous + " " + line

        elif previous.endswith(TERMINAL_PUNCTUATION):
            output_lines.append(line)

        else:
            output_lines[-1] = previous + " " + line

    normalized = "\n".join(output_lines)

    # Normalize multiple spaces again.
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"\s+([:;,.])", r"\1", normalized)

    return normalized.strip()


def normalize_page_text(raw_text: str | None) -> tuple[str, str, dict[str, Any]]:
    """
    Normalize one page and return:
    - normalized_text
    - matching_text
    - stats
    """
    text = "" if raw_text is None else str(raw_text)
    replacement_char_count = text.count("\ufffd") + text.count("�")

    text = standardize_characters(text)
    text = remove_html_tags(text)
    text, removed_technical_lines_count = remove_technical_lines(text)
    text = repair_linebreak_hyphenation(text)
    normalized_text = clean_line_spacing(text)

    # Create auxiliary matching_text. This intentionally removes accents only here.
    matching_text = remove_accents(normalized_text).lower()
    matching_text = re.sub(r"\s+", " ", matching_text).strip()

    warnings: list[str] = []
    if replacement_char_count > 0:
        warnings.append(
            "Replacement character detected in raw_text; it was flagged but not automatically corrected."
        )

    stats = {
        "raw_char_count": len("" if raw_text is None else str(raw_text)),
        "normalized_char_count": len(normalized_text),
        "matching_text_char_count": len(matching_text),
        "removed_technical_lines_count": removed_technical_lines_count,
        "replacement_char_count": replacement_char_count,
        "normalization_status": "completed_with_warnings" if warnings else "completed",
        "normalization_warning": " ".join(warnings) if warnings else None,
    }

    return normalized_text, matching_text, stats


def write_normalized_json(
    output_path: Path,
    *,
    input_payload: dict[str, Any],
    normalization_id: str,
    assessment_event: sqlite3.Row,
    normalized_pages: list[dict[str, Any]],
    normalization_summary: dict[str, Any],
) -> None:
    """Write normalized page-level JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = dict(input_payload)
    output_payload["normalization_metadata"] = {
        "normalization_id": normalization_id,
        "assessment_id": assessment_event["assessment_id"],
        "extraction_id": assessment_event["extraction_id"],
        "document_id": assessment_event["document_id"],
        "normalization_date": utc_now_iso(),
        "normalization_tool": "04_normalize_text.py",
        "normalization_rules": NORMALIZATION_RULES,
    }
    output_payload["document_level_normalization_summary"] = normalization_summary
    output_payload["pages"] = normalized_pages

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)


def write_page_normalization_csv(
    output_path: Path,
    page_stats: list[dict[str, Any]],
) -> None:
    """Write page-level normalization report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "page_number",
        "raw_char_count",
        "normalized_char_count",
        "matching_text_char_count",
        "removed_technical_lines_count",
        "replacement_char_count",
        "quality_requires_manual_review",
        "normalization_status",
        "normalization_warning",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(page_stats)


def insert_normalization_event(
    connection: sqlite3.Connection,
    event: dict[str, Any],
) -> None:
    """Insert one normalization event."""
    connection.execute(
        """
        INSERT INTO normalization_events (
            normalization_id,
            assessment_id,
            extraction_id,
            document_id,
            normalization_date,
            normalization_tool,
            normalization_rules_json,
            pages_normalized_count,
            pages_with_warnings_count,
            normalized_json_path,
            page_normalization_csv_path,
            normalization_status,
            created_at,
            notes
        )
        VALUES (
            :normalization_id,
            :assessment_id,
            :extraction_id,
            :document_id,
            :normalization_date,
            :normalization_tool,
            :normalization_rules_json,
            :pages_normalized_count,
            :pages_with_warnings_count,
            :normalized_json_path,
            :page_normalization_csv_path,
            :normalization_status,
            :created_at,
            :notes
        )
        """,
        event,
    )


def insert_normalization_page_stats(
    connection: sqlite3.Connection,
    *,
    normalization_id: str,
    document_id: str,
    page_stats: list[dict[str, Any]],
) -> None:
    """Insert page-level normalization stats."""
    rows = []

    for page in page_stats:
        rows.append(
            {
                "normalization_id": normalization_id,
                "document_id": document_id,
                "page_number": page["page_number"],
                "raw_char_count": page["raw_char_count"],
                "normalized_char_count": page["normalized_char_count"],
                "matching_text_char_count": page["matching_text_char_count"],
                "removed_technical_lines_count": page["removed_technical_lines_count"],
                "replacement_char_count": page["replacement_char_count"],
                "quality_requires_manual_review": int(page["quality_requires_manual_review"]),
                "normalization_status": page["normalization_status"],
                "normalization_warning": page["normalization_warning"],
            }
        )

    connection.executemany(
        """
        INSERT INTO normalization_page_stats (
            normalization_id,
            document_id,
            page_number,
            raw_char_count,
            normalized_char_count,
            matching_text_char_count,
            removed_technical_lines_count,
            replacement_char_count,
            quality_requires_manual_review,
            normalization_status,
            normalization_warning
        )
        VALUES (
            :normalization_id,
            :document_id,
            :page_number,
            :raw_char_count,
            :normalized_char_count,
            :matching_text_char_count,
            :removed_technical_lines_count,
            :replacement_char_count,
            :quality_requires_manual_review,
            :normalization_status,
            :normalization_warning
        )
        """,
        rows,
    )


def build_normalization_outputs(
    input_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add normalization fields to each page and build page stats."""
    normalized_pages: list[dict[str, Any]] = []
    page_stats: list[dict[str, Any]] = []

    for page in input_payload.get("pages", []):
        raw_text = page.get("raw_text")
        normalized_text, matching_text, stats = normalize_page_text(raw_text)

        quality_assessment = page.get("quality_assessment", {})
        quality_requires_manual_review = bool(
            quality_assessment.get("requires_manual_review", False)
        )

        page_number = int(page["page_number"])

        page_copy = dict(page)
        page_copy["normalization"] = {
            "normalized_text": normalized_text,
            "matching_text": matching_text,
            "normalization_steps": list(NORMALIZATION_RULES.keys()),
            "normalization_status": stats["normalization_status"],
            "normalization_warning": stats["normalization_warning"],
        }

        normalized_pages.append(page_copy)

        stats_row = {
            "page_number": page_number,
            "quality_requires_manual_review": quality_requires_manual_review,
            **stats,
        }
        page_stats.append(stats_row)

    return normalized_pages, page_stats


def run_normalization(
    *,
    document_id: str | None,
    assessment_id: str | None,
) -> dict[str, Any]:
    """Run Sub-task 1.4."""
    if not document_id and not assessment_id:
        raise ValueError("Provide either --document-id or --assessment-id.")

    database_path = choose_database_path()
    connection = connect_database(database_path)

    try:
        ensure_normalization_tables(connection)

        if assessment_id:
            assessment_event = get_assessment_by_id(connection, assessment_id)
        else:
            assessment_event = get_latest_assessment_for_document(connection, document_id)

        normalization_id = next_normalization_id(connection)

        quality_checked_json_path = resolve_project_path(
            assessment_event["quality_checked_json_path"]
        )
        if not quality_checked_json_path.exists():
            raise FileNotFoundError(
                f"Quality-checked JSON not found: {quality_checked_json_path}\n"
                "Check the quality_checked_json_path stored in quality_assessment_events."
            )

        with quality_checked_json_path.open("r", encoding="utf-8") as file:
            input_payload = json.load(file)

        normalized_pages, page_stats = build_normalization_outputs(input_payload)

        pages_normalized_count = len(page_stats)
        pages_with_warnings_count = sum(
            1 for page in page_stats
            if page["normalization_status"] != "completed"
        )

        document_output_dir = NORMALIZED_DIR / assessment_event["document_id"]
        report_output_dir = NORMALIZATION_REPORT_DIR / assessment_event["document_id"]

        normalized_json_path = (
            document_output_dir
            / f'{assessment_event["assessment_id"]}_{normalization_id}_normalized.json'
        )
        page_normalization_csv_path = (
            report_output_dir
            / f'{assessment_event["assessment_id"]}_{normalization_id}_page_normalization.csv'
        )

        normalization_status = (
            "Completed with warnings" if pages_with_warnings_count > 0 else "Completed"
        )
        notes = (
            f"{pages_with_warnings_count} page(s) contain normalization warnings."
            if pages_with_warnings_count > 0 else None
        )

        normalization_summary = {
            "normalization_id": normalization_id,
            "assessment_id": assessment_event["assessment_id"],
            "extraction_id": assessment_event["extraction_id"],
            "document_id": assessment_event["document_id"],
            "pages_normalized_count": pages_normalized_count,
            "pages_with_warnings_count": pages_with_warnings_count,
            "normalization_status": normalization_status,
            "notes": notes,
        }

        write_normalized_json(
            normalized_json_path,
            input_payload=input_payload,
            normalization_id=normalization_id,
            assessment_event=assessment_event,
            normalized_pages=normalized_pages,
            normalization_summary=normalization_summary,
        )

        write_page_normalization_csv(page_normalization_csv_path, page_stats)

        timestamp = utc_now_iso()
        normalization_event = {
            "normalization_id": normalization_id,
            "assessment_id": assessment_event["assessment_id"],
            "extraction_id": assessment_event["extraction_id"],
            "document_id": assessment_event["document_id"],
            "normalization_date": timestamp[:10],
            "normalization_tool": "04_normalize_text.py",
            "normalization_rules_json": json.dumps(NORMALIZATION_RULES, ensure_ascii=False),
            "pages_normalized_count": pages_normalized_count,
            "pages_with_warnings_count": pages_with_warnings_count,
            "normalized_json_path": project_relative(normalized_json_path),
            "page_normalization_csv_path": project_relative(page_normalization_csv_path),
            "normalization_status": normalization_status,
            "created_at": timestamp,
            "notes": notes,
        }

        insert_normalization_event(connection, normalization_event)
        insert_normalization_page_stats(
            connection,
            normalization_id=normalization_id,
            document_id=assessment_event["document_id"],
            page_stats=page_stats,
        )

        connection.commit()

        print("\nText normalization completed")
        print("----------------------------")
        for key, value in normalization_event.items():
            print(f"{key}: {value}")

        print("\nGenerated files")
        print("---------------")
        print(project_relative(normalized_json_path))
        print(project_relative(page_normalization_csv_path))

        print("\nDatabase used")
        print("-------------")
        print(project_relative(database_path))

        return normalization_event

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sub-task 1.4 text normalization."
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help=(
            "Document ID. If --assessment-id is not supplied, the latest "
            "quality assessment for this document will be normalized."
        ),
    )
    parser.add_argument(
        "--assessment-id",
        default=None,
        help="Specific quality assessment to normalize, for example QA_000001.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        run_normalization(
            document_id=args.document_id,
            assessment_id=args.assessment_id,
        )
    except Exception as error:
        raise SystemExit(f"\nText normalization failed: {error}") from error
