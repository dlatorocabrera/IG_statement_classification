#!/usr/bin/env python3
"""
05_identify_legal_structure.py

Sub-task 1.5: Legal-Structure Identification and Hierarchical Representation.

Rule-based first implementation for normalized LeyChile-style legal documents.

Input:
    data/normalized/<document_id>/..._normalized.json

Outputs:
    data/legal_structure/<document_id>/<normalization_id>_<structure_id>_structure.json
    tables/legal_structure/<document_id>/<normalization_id>_<structure_id>_structure_elements.csv

SQL tables:
    structure_events
    structure_elements

Run from project root:
    python codes/task_1_text_preparation/05_identify_legal_structure.py \
        --document-id CL_MMA_DEC_000001
    python codes/task_1_text_preparation/05_identify_legal_structure.py \
        --normalization-id NORM_000001
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
from typing import Any


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from config import DATABASE_PATH, LEGAL_STRUCTURE_DIR, LEGAL_STRUCTURE_REPORT_DIR
except ImportError as error:
    raise SystemExit(
        "Could not import config.py. Store this file as "
        "codes/task_1_text_preparation/05_identify_legal_structure.py"
    ) from error


# ---------------------------------------------------------------------
# SQL schemas
# ---------------------------------------------------------------------
STRUCTURE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS structure_events (
    structure_id TEXT PRIMARY KEY,
    normalization_id TEXT NOT NULL,
    assessment_id TEXT NOT NULL,
    extraction_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    structure_date TEXT NOT NULL,
    structure_tool TEXT NOT NULL,
    parser_method TEXT NOT NULL,
    structure_version TEXT NOT NULL,
    source_text_field TEXT NOT NULL,
    elements_count INTEGER NOT NULL,
    uncertain_elements_count INTEGER NOT NULL,
    manual_review_elements_count INTEGER NOT NULL,
    structure_json_path TEXT NOT NULL,
    structure_csv_path TEXT NOT NULL,
    structure_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (normalization_id)
        REFERENCES normalization_events(normalization_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);
"""

STRUCTURE_ELEMENTS_SQL = """
CREATE TABLE IF NOT EXISTS structure_elements (
    structure_id TEXT NOT NULL,
    element_id TEXT NOT NULL,
    parent_id TEXT,
    document_id TEXT NOT NULL,
    element_order INTEGER NOT NULL,
    nesting_level INTEGER NOT NULL,
    element_type TEXT NOT NULL,
    label TEXT,
    title TEXT,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    element_text TEXT,
    text_source TEXT,
    content_status TEXT,
    structure_confidence TEXT NOT NULL,
    requires_manual_review INTEGER NOT NULL,
    review_reason_code TEXT,
    review_reason_note TEXT,
    PRIMARY KEY (structure_id, element_id),
    FOREIGN KEY (structure_id)
        REFERENCES structure_events(structure_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);
"""


# ---------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------
METADATA_FIELD_PATTERN = re.compile(
    r"^\s*(?P<label>Tipo\s+Norma|Fecha\s+Publicaci[oó]n|Fecha\s+Promulgaci[oó]n|Organismo|T[ií]tulo|Tipo\s+Versi[oó]n|Inicio\s+Vigencia|Id\s+Norma|URL)\s*:\s*(?P<value>.*)$",
    re.IGNORECASE,
)
TITLE_LIKE_PATTERN = re.compile(r"^[A-ZÁÉÍÓÚÜÑ0-9 ,.;:\-()°º/]+$")
DECREE_HEADER_PATTERN = re.compile(r"^\s*(N[úu]m\.?|N[°º]\.?)\s*\d+[\.\-]?\s+.*$", re.IGNORECASE)
PREAMBLE_PATTERN = re.compile(r"^\s*(Visto|Considerando|Decreto|Resuelvo)\s*:\s*$", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"^\s*(?P<label>CAP[IÍ]TULO\s+[IVXLCDM0-9]+)\s*(?P<title>.*)$", re.IGNORECASE)
TITLE_SECTION_PATTERN = re.compile(r"^\s*(?P<label>T[IÍ]TULO\s+[IVXLCDM0-9]+)\s*(?P<title>.*)$", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^\s*(?P<label>SECCI[OÓ]N\s+[IVXLCDM0-9]+)\s*(?P<title>.*)$", re.IGNORECASE)
SUBSECTION_PATTERN = re.compile(r"^\s*(?P<label>(?:SUBSECCI[OÓ]N|P[ÁA]RRAFO)\s+[IVXLCDM0-9]+)\s*(?P<title>.*)$", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"^\s*(?P<label>Art[íi]culo\s+(?P<number>[0-9]+[A-Za-z°º]?))\s*(?:(?:[:.\-–—]+)\s*(?P<rest>.*)|$)", re.IGNORECASE)
NUMBERED_ITEM_PATTERN = re.compile(r"^\s*(?P<label>[0-9]+)[\.\)]\s+(?P<text>.+)$")
COMPOUND_ITEM_PATTERN = re.compile(r"^\s*(?P<label>[a-zA-Z]\.\d+)\)\s+(?P<text>.+)$")
ROMAN_ITEM_PATTERN = re.compile(r"^\s*(?P<label>i{1,3}|iv|v|vi{0,3}|ix|x)[\.\)]\s+(?P<text>.+)$")
LETTERED_ITEM_PATTERN = re.compile(r"^\s*(?P<label>[a-zA-Z])[\.\)]\s+(?P<text>.+)$")
BULLET_ITEM_PATTERN = re.compile(r"^\s*[-–—]\s+(?P<text>.+)$")
# Visual-object captions are intentionally restricted to a compact label
# followed by ':' or '.'. This avoids false positives such as prose that
# merely begins with "Tabla III-9 o la Tabla III-10 ...".
VISUAL_NUMBER_LABEL = r"(?:[IVXLCDM]+(?:[-\s]\d+(?:[-.]\d+)*)?|\d+(?:[-.]\d+)*)"
TABLE_PATTERN = re.compile(
    rf"^\s*(?P<label>(?:Tabla|Cuadro)\s+{VISUAL_NUMBER_LABEL})\s*[:.]\s*(?P<title>.*)$",
    re.IGNORECASE,
)
GRAPH_PATTERN = re.compile(
    rf"^\s*(?P<label>Gr[áa]fico\s+{VISUAL_NUMBER_LABEL})\s*[:.]\s*(?P<title>.*)$",
    re.IGNORECASE,
)
FIGURE_PATTERN = re.compile(
    rf"^\s*(?P<label>Figura\s+{VISUAL_NUMBER_LABEL})\s*[:.]\s*(?P<title>.*)$",
    re.IGNORECASE,
)
ANNEX_PATTERN = re.compile(
    r"^\s*(?P<label>Anexo(?:\s+[A-Za-z0-9IVXLCDM]+)?)\s*[:.]\s*(?P<title>.*)$",
    re.IGNORECASE,
)
SOURCE_LINE_PATTERN = re.compile(r"^\s*Fuente\s*:\s*(?P<text>.*)$", re.IGNORECASE)
OBJECT_NOTE_LINE_PATTERN = re.compile(
    r"^\s*(?:PI\s*=|CI\s*=|\(\*+\)|\*+\)|Nota\s*:|Observaci[oó]n\b).+",
    re.IGNORECASE,
)
SIGNATURE_START_PATTERN = re.compile(r"^\s*(An[oó]tese|T[oó]mese raz[oó]n|Publ[ií]quese|Comun[ií]quese|Reg[ií]strese)\b.*", re.IGNORECASE)
TRANSITORY_ARTICLE_PATTERN = re.compile(r"^\s*(?P<label>Art[íi]culo\s+(?:[úu]nico\s+)?Transitorio|Art[íi]culo\s+(?:primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo)\s+transitorio)\s*[:.\-–—]?\s*(?P<rest>.*)$", re.IGNORECASE)
NUMBERED_HEADING_PATTERN = re.compile(r"^\s*(?P<label>[IVXLCDM]+\.\d+(?:\.\d+)*)\s+(?P<title>.+)$", re.IGNORECASE)
UPPERCASE_LETTER_HEADING_PATTERN = re.compile(r"^\s*(?P<label>[A-Z])\.\s+(?P<title>.+)$")

# ---------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def choose_database_path() -> Path:
    """Return the canonical inventory database declared in config.py."""
    database_path = Path(DATABASE_PATH)
    if database_path.exists():
        return database_path
    raise FileNotFoundError(f"Could not find database at {database_path}")


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def connect_database(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def get_latest_normalization_for_document(connection: sqlite3.Connection, document_id: str) -> sqlite3.Row:
    record = connection.execute(
        """
        SELECT *
        FROM normalization_events
        WHERE document_id = ?
        ORDER BY created_at DESC, normalization_id DESC
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()
    if record is None:
        raise ValueError(f"No normalization event found for document_id {document_id!r}. Run Step 1.4 first.")
    return record


def get_normalization_by_id(connection: sqlite3.Connection, normalization_id: str) -> sqlite3.Row:
    record = connection.execute(
        "SELECT * FROM normalization_events WHERE normalization_id = ?",
        (normalization_id,),
    ).fetchone()
    if record is None:
        raise ValueError(f"No normalization event found for normalization_id {normalization_id!r}.")
    return record


def get_document_inventory_record(connection: sqlite3.Connection, document_id: str) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM document_inventory WHERE document_id = ?", (document_id,)).fetchone()


def next_structure_id(connection: sqlite3.Connection) -> str:
    rows = connection.execute("SELECT structure_id FROM structure_events").fetchall()
    highest = 0
    for row in rows:
        value = row["structure_id"]
        if value and value.startswith("STRUCT_"):
            suffix = value.replace("STRUCT_", "")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return f"STRUCT_{highest + 1:06d}"


def numbered_heading_depth(label: str | None) -> int:
    """
    Return the hierarchy depth of a roman-decimal heading.

    Examples:
        I.5 -> 2
        VI.6.1 -> 3
    """
    if not label:
        return 0

    return len(label.split("."))

# ---------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------

def clean_small_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    return value or None

def clean_title_separator(value: str | None) -> str | None:
    """
    Remove leading separators from titles.

    Example:
        ': CONTROL DE EMISIONES'
    becomes:
        'CONTROL DE EMISIONES'
    """
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^\s*[:.\-–—]+\s*", "", value)
    return value.strip() or None


def normalize_marker_label(value: str | None) -> str | None:
    """
    Normalize internal spaces in structural labels.

    Example:
        'CAPÍTULO   V'
    becomes:
        'CAPÍTULO V'
    """
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value).strip()
    return value or None


CAPITALIZED_SPANISH_FUNCTION_STARTERS = [
    # Determiners / articles / demonstratives
    r"El\b",
    r"La\b",
    r"Los\b",
    r"Las\b",
    r"Un\b",
    r"Una\b",
    r"Unos\b",
    r"Unas\b",
    r"Este\b",
    r"Esta\b",
    r"Estos\b",
    r"Estas\b",
    r"Dicho\b",
    r"Dicha\b",
    r"Dichos\b",
    r"Dichas\b",

    # Contractions
    r"Al\b",
    r"Del\b",

    # Simple prepositions. These are intentionally capitalized only.
    r"A\b",
    r"Ante\b",
    r"Bajo\b",
    r"Cabe\b",
    r"Con\b",
    r"Contra\b",
    r"De\b",
    r"Desde\b",
    r"Durante\b",
    r"En\b",
    r"Entre\b",
    r"Hacia\b",
    r"Hasta\b",
    r"Mediante\b",
    r"Para\b",
    r"Por\b",
    r"Según\b",
    r"Sin\b",
    r"So\b",
    r"Sobre\b",
    r"Tras\b",
    r"Vía\b",

    # Conjunctions / subordinators / sentence connectors
    r"Que\b",
    r"Si\b",
    r"Aunque\b",
    r"Cuando\b",
    r"Mientras\b",
    r"Como\b",
    r"Donde\b",
    r"Porque\b",
    r"Pues\b",
    r"Asimismo\b",
    r"Además\b",
    r"No\s+obstante\b",
    r"Sin\s+embargo\b",
    r"Tal\s+como\b",
    r"Cabe\s+señalar\b",

    # Reflexive/legal sentence openings
    r"Se\b",
]


def split_title_from_remainder(
    value: str | None,
    *,
    min_title_chars: int = 10,
    starters: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """
    Split a structural title/caption from body text accidentally joined
    to the same normalized line.

    The split uses capitalized Spanish function-word starters.
    Capitalization is intentionally preserved as part of the signal.
    """

    value = clean_title_separator(value)

    if not value:
        return None, None

    starter_patterns = starters or CAPITALIZED_SPANISH_FUNCTION_STARTERS

    # No re.IGNORECASE here: capitalization is the boundary signal.
    starter_regex = re.compile(
        r"\s+(" + "|".join(starter_patterns) + r")"
    )

    for match in starter_regex.finditer(value):
        split_at = match.start()
        title = value[:split_at].strip()
        remainder = value[split_at:].strip()

        if len(title) >= min_title_chars and remainder:
            return title, remainder

    return value, None


def split_numbered_heading_title(value: str | None) -> tuple[str | None, str | None]:
    """
    Split numbered-heading title from accidentally attached body text.
    """

    return split_title_from_remainder(
        value,
        min_title_chars=10,
    )


def split_visual_title(value: str | None) -> tuple[str | None, str | None]:
    """
    Split table/graph/figure caption title from notes or body text.
    """

    value = clean_title_separator(value)

    if not value:
        return None, None

    # First split explicit visual-object notes/legends.
    note_marker = re.search(
        r"\s+(?=(?:PI\s*=|CI\s*=|\(\*+\)|Nota\s*:|Observaci[oó]n\b|Fuente\s*:))",
        value,
        flags=re.IGNORECASE,
    )

    if note_marker:
        title = value[:note_marker.start()].strip()
        remainder = value[note_marker.start():].strip()

        if len(title) >= 8 and remainder:
            return title, remainder

    # Then split ordinary body text accidentally joined after the caption.
    return split_title_from_remainder(
        value,
        min_title_chars=12,
    )


def is_visual_note_like(value: str | None) -> bool:
    """
    Detect whether a visual-object remainder is likely a table/graph note
    rather than an ordinary paragraph.
    """

    if not value:
        return False

    return bool(
        re.match(
            r"^\s*(?:PI\s*=|CI\s*=|\(\*+\)|Nota\s*:|Observaci[oó]n\b|Fuente\s*:)",
            value,
            flags=re.IGNORECASE,
        )
    )




def starts_with_lowercase_letter(value: str | None) -> bool:
    """Return True when the first alphabetic character is lowercase."""

    if not value:
        return False

    match = re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", value.strip())
    if not match:
        return False

    first = match.group(0)
    return first == first.lower() and first != first.upper()


def is_appendable_text_element(element: dict[str, Any] | None) -> bool:
    """Elements that can safely receive a page-break continuation."""

    if not element:
        return False

    return element.get("element_type") in {
        "paragraph",
        "numbered_item",
        "lettered_item",
        "roman_item",
        "compound_item",
        "bullet_item",
    }


def should_append_page_continuation(
    previous_element: dict[str, Any] | None,
    current_text: str,
    current_page: int,
) -> bool:
    """
    Merge a paragraph/list continuation split only by a page break.

    The conservative rule used here is: append only when the new text is on
    a later page and starts with a lowercase letter. Structural markers are
    already handled before this function is reached.
    """

    if not is_appendable_text_element(previous_element):
        return False

    if int(current_page) <= int(previous_element.get("page_end", previous_element.get("page_start", current_page))):
        return False

    return starts_with_lowercase_letter(current_text)


def split_source_note_text(text: str) -> tuple[str, str | None]:
    """
    Split a Fuente line when body text was accidentally joined to it.

    The split reuses the same capitalized Spanish function-word starters used
    for title-boundary detection. The source-note label itself is preserved.
    """

    source_match = SOURCE_LINE_PATTERN.match(text)
    if not source_match:
        return text, None

    source_body = source_match.group("text").strip()
    source_text, remainder = split_title_from_remainder(
        source_body,
        min_title_chars=15,
    )

    if not source_text:
        return text, remainder

    return f"Fuente: {source_text}", remainder


def visual_parent_for_remainder(builder: "StructureBuilder") -> dict[str, Any] | None:
    """Parent for body text split from a source note or visual caption."""

    return builder.current_parent()

def safe_id_part(value: str | None, fallback: str) -> str:
    value = value or fallback
    value = re.sub(r"\W+", "_", value, flags=re.UNICODE).strip("_")
    return value or fallback


def is_title_like(line: str) -> bool:
    line = line.strip()

    if len(line) < 20:
        return False

    if not TITLE_LIKE_PATTERN.match(line):
        return False

    # Avoid classifying explicit structural markers as generic titles.
    if (
        ARTICLE_PATTERN.match(line)
        or TRANSITORY_ARTICLE_PATTERN.match(line)
        or CHAPTER_PATTERN.match(line)
        or TITLE_SECTION_PATTERN.match(line)
        or SECTION_PATTERN.match(line)
        or SUBSECTION_PATTERN.match(line)
        or NUMBERED_HEADING_PATTERN.match(line)
        or UPPERCASE_LETTER_HEADING_PATTERN.match(line)
        or TABLE_PATTERN.match(line)
        or GRAPH_PATTERN.match(line)
        or FIGURE_PATTERN.match(line)
        or ANNEX_PATTERN.match(line)
    ):
        return False

    return True


def page_review_flags(page: dict[str, Any]) -> dict[str, Any]:
    qa = page.get("quality_assessment", {}) or {}
    return {
        "requires_manual_review": bool(qa.get("requires_manual_review", False)),
        "review_reason_code": qa.get("review_reason_code"),
        "review_reason_note": qa.get("review_reason_note"),
    }


EMBEDDED_PREAMBLE_MARKER_PATTERN = re.compile(
    r"\b(Visto|Considerando|Decreto|Resuelvo)\s*:",
    re.IGNORECASE,
)


def is_valid_embedded_marker_start(text: str, start: int) -> bool:
    """
    Decide whether a preamble marker inside a longer line should be split.

    We allow splitting when the marker appears:
    - at the beginning of the line
    - after punctuation
    - after a legal transition like '; y '
    """

    if start == 0:
        return True

    prefix = text[:start]

    return bool(
        re.search(
            r"([.;:]\s*|;\s+y\s*)$",
            prefix,
            flags=re.IGNORECASE,
        )
    )


def split_embedded_structural_markers(text: str) -> list[str]:
    """
    Split embedded preamble markers into separate logical lines.

    Example:
    '... Contraloría General de la República; y Considerando:'
    becomes:
    '... Contraloría General de la República; y'
    'Considerando:'

    This does not classify every colon as a preamble.
    It only splits known legal preamble markers.
    """

    text = text.strip()
    if not text:
        return []

    output_parts: list[str] = []
    cursor = 0

    for match in EMBEDDED_PREAMBLE_MARKER_PATTERN.finditer(text):
        start, end = match.span()

        if not is_valid_embedded_marker_start(text, start):
            continue

        before = text[cursor:start].strip()
        marker = text[start:end].strip()

        if before:
            output_parts.append(before)

        output_parts.append(marker)
        cursor = end

    after = text[cursor:].strip()
    if after:
        output_parts.append(after)

    return output_parts if output_parts else [text]



def flatten_normalized_lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []

    for page in payload.get("pages", []):
        page_number = int(page["page_number"])
        text = (page.get("normalization", {}) or {}).get("normalized_text", "")
        flags = page_review_flags(page)

        for raw_line in str(text).splitlines():
            line = raw_line.strip()
            if not line:
                continue

            logical_lines = split_embedded_structural_markers(line)

            for logical_line in logical_lines:
                lines.append(
                    {
                        "page_number": page_number,
                        "text": logical_line,
                        **flags,
                    }
                )

    return lines


class StructureBuilder:
    def __init__(self, document_id: str, structure_id: str):
        self.document_id = document_id
        self.structure_id = structure_id
        self.roots: list[dict[str, Any]] = []
        self.flat: list[dict[str, Any]] = []
        self.order = 0
        self.counts: dict[str, int] = {}
        self.used_element_ids: set[str] = set()
        self.by_id: dict[str, dict[str, Any]] = {}
        self.current_metadata_block: dict[str, Any] | None = None
        self.current_metadata_field: dict[str, Any] | None = None
        self.in_metadata_block = True
        self.current_title: dict[str, Any] | None = None
        self.current_preamble: dict[str, Any] | None = None
        self.current_chapter: dict[str, Any] | None = None
        self.current_section: dict[str, Any] | None = None
        self.current_subsection: dict[str, Any] | None = None
        self.current_numbered_headings: list[dict[str, Any]] = []
        self.current_lettered_heading: dict[str, Any] | None = None
        self.current_article: dict[str, Any] | None = None
        self.last_visual_object: dict[str, Any] | None = None
        self.current_signature_block: dict[str, Any] | None = None
        

    def count(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def make_unique_id(self, candidate_id: str) -> str:
        """Ensure element IDs are unique within one structure event."""
        if candidate_id not in self.used_element_ids:
            self.used_element_ids.add(candidate_id)
            return candidate_id

        suffix = 2
        while f"{candidate_id}_{suffix:03d}" in self.used_element_ids:
            suffix += 1

        unique_id = f"{candidate_id}_{suffix:03d}"
        self.used_element_ids.add(unique_id)
        return unique_id

    def make_id(self, element_type: str, label: str | None, parent: dict[str, Any] | None) -> str:
        doc = self.document_id
        if element_type == "metadata_block":
            return f"{doc}_META_001"
        if element_type == "metadata_field":
            return f"{doc}_META_001_FIELD_{self.count('metadata_field'):03d}"
        if element_type == "title":
            return f"{doc}_TITLE_{self.count('title'):03d}"
        if element_type == "decree_header":
            return f"{doc}_DECREE_HEADER_{self.count('decree_header'):03d}"
        if element_type == "preamble":
            return f"{doc}_PREAMBLE_{safe_id_part(label, str(self.count('preamble'))).upper()}"
        if element_type == "chapter":
            return f"{doc}_CH_{safe_id_part(label, str(self.count('chapter')))}"
        if element_type == "section":
            return f"{doc}_SEC_{safe_id_part(label, str(self.count('section')))}"
        if element_type == "subsection":
            return f"{doc}_SUBSEC_{safe_id_part(label, str(self.count('subsection')))}"
        if element_type == "article":
            return f"{doc}_ART_{safe_id_part(label, str(self.count('article')))}"
        if element_type == "paragraph" and parent:
            return f"{parent['element_id']}_P{self.count(parent['element_id'] + '_paragraph'):03d}"
        if element_type in {"numbered_item", "lettered_item", "roman_item", "compound_item"} and parent:
            return f"{parent['element_id']}_ITEM_{safe_id_part(label, str(self.count(element_type)))}"
        if element_type == "bullet_item" and parent:
            return f"{parent['element_id']}_BULLET_{self.count(parent['element_id'] + '_bullet'):03d}"
        if element_type == "table":
            return f"{doc}_TABLE_{safe_id_part(label, str(self.count('table')))}"
        if element_type == "graph":
            return f"{doc}_GRAPH_{safe_id_part(label, str(self.count('graph')))}"
        if element_type == "figure":
            return f"{doc}_FIGURE_{safe_id_part(label, str(self.count('figure')))}"
        if element_type == "annex":
            return f"{doc}_ANNEX_{safe_id_part(label, str(self.count('annex')))}"
        if element_type == "signature_block":
            return f"{doc}_SIGNATURE_{self.count('signature_block'):03d}"
        if element_type == "transitory_article":
            return f"{doc}_TRANSITORY_ART_{safe_id_part(label, str(self.count('transitory_article')))}"
        if element_type == "numbered_heading":
            return f"{doc}_HEAD_{safe_id_part(label, str(self.count('numbered_heading')))}"
        if element_type == "lettered_heading":
            return f"{doc}_LHEAD_{safe_id_part(label, str(self.count('lettered_heading')))}"
        if element_type == "object_note" and parent:
            return f"{parent['element_id']}_NOTE_{self.count(parent['element_id'] + '_note'):03d}"
        return f"{doc}_OTHER_{self.count('other'):03d}"

    def add(
        self,
        *,
        element_type: str,
        page_number: int,
        label: str | None = None,
        title: str | None = None,
        element_text: str | None = None,
        parent: dict[str, Any] | None = None,
        content_status: str | None = None,
        structure_confidence: str = "high",
        requires_manual_review: bool = False,
        review_reason_code: str | None = None,
        review_reason_note: str | None = None,
        text_source: str | None = "normalization.normalized_text",
    ) -> dict[str, Any]:
        self.order += 1
        parent_id = parent["element_id"] if parent else None
        nesting_level = int(parent.get("_nesting_level", 0)) + 1 if parent else 0
        element_id = self.make_unique_id(self.make_id(element_type, label, parent))

        element = {
            "element_id": element_id,
            "parent_id": parent_id,
            "element_type": element_type,
            "label": clean_small_text(label),
            "title": clean_small_text(title),
            "page_start": int(page_number),
            "page_end": int(page_number),
            "structure_confidence": structure_confidence,
            "requires_manual_review": bool(requires_manual_review),
            "review_reason_code": review_reason_code,
            "review_reason_note": review_reason_note,
            "children": [],
            "_order": self.order,
            "_nesting_level": nesting_level,
        }

        if element_text is not None:
            element["element_text"] = element_text
            element["text_source"] = text_source
            element["content_status"] = content_status or "text_extracted"

        if parent:
            parent["children"].append(element)
            parent["page_end"] = max(parent["page_end"], int(page_number))
            if requires_manual_review:
                parent["requires_manual_review"] = True
                parent["review_reason_code"] = parent.get("review_reason_code") or review_reason_code
                parent["review_reason_note"] = parent.get("review_reason_note") or review_reason_note
        else:
            self.roots.append(element)

        self.flat.append(element)
        self.by_id[element_id] = element
        return element

    def current_parent(self) -> dict[str, Any] | None:
        return (
            self.current_article
            or self.current_lettered_heading
            or self.current_numbered_parent()
            or self.current_subsection
            or self.current_section
            or self.current_chapter
            or self.current_preamble
        )

    def append_to_leaf(self, element: dict[str, Any], text: str, page_number: int) -> None:
        element["element_text"] = (element.get("element_text", "").rstrip() + " " + text).strip()
        element["page_end"] = max(element["page_end"], int(page_number))

        # Keep ancestor page ranges consistent when a leaf receives a
        # continuation from a later page.
        parent_id = element.get("parent_id")
        while parent_id:
            parent = self.by_id.get(parent_id)
            if parent is None:
                break
            parent["page_end"] = max(parent["page_end"], int(page_number))
            parent_id = parent.get("parent_id")

    def strip_internal(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.strip_internal(x) for x in value]
        if isinstance(value, dict):
            return {k: self.strip_internal(v) for k, v in value.items() if not k.startswith("_")}
        return value
    
    def current_numbered_parent(self) -> dict[str, Any] | None:
        if self.current_numbered_headings:
            return self.current_numbered_headings[-1]
        return None

    def hierarchy(self) -> list[dict[str, Any]]:
        return self.strip_internal(self.roots)

    def flat_rows(self) -> list[dict[str, Any]]:
        rows = []
        for e in self.flat:
            rows.append({
                "structure_id": self.structure_id,
                "element_id": e["element_id"],
                "parent_id": e.get("parent_id"),
                "document_id": self.document_id,
                "element_order": e["_order"],
                "nesting_level": e["_nesting_level"],
                "element_type": e["element_type"],
                "label": e.get("label"),
                "title": e.get("title"),
                "page_start": e["page_start"],
                "page_end": e["page_end"],
                "element_text": e.get("element_text"),
                "text_source": e.get("text_source"),
                "content_status": e.get("content_status"),
                "structure_confidence": e["structure_confidence"],
                "requires_manual_review": int(bool(e.get("requires_manual_review", False))),
                "review_reason_code": e.get("review_reason_code"),
                "review_reason_note": e.get("review_reason_note"),
            })
        return rows


def parse_structure(document_id: str, structure_id: str, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    builder = StructureBuilder(document_id, structure_id)
    last_leaf: dict[str, Any] | None = None

    for line in flatten_normalized_lines(payload):
        text = line["text"]
        page = line["page_number"]
        flags = {
            "requires_manual_review": line["requires_manual_review"],
            "review_reason_code": line["review_reason_code"],
            "review_reason_note": line["review_reason_note"],
        }

        # Metadata fields at beginning.
        meta = METADATA_FIELD_PATTERN.match(text)
        if meta and builder.in_metadata_block:
            if builder.current_metadata_block is None:
                builder.current_metadata_block = builder.add(
                    element_type="metadata_block", page_number=page, text_source=None, **flags
                )
            label = meta.group("label")
            element_text = f"{label} :{meta.group('value')}".strip()
            builder.current_metadata_field = builder.add(
                element_type="metadata_field",
                page_number=page,
                label=label,
                element_text=element_text,
                parent=builder.current_metadata_block,
                content_status="text_extracted",
                **flags,
            )
            last_leaf = builder.current_metadata_field
            continue

        # Continuation of long metadata title.
        if (
            builder.in_metadata_block
            and builder.current_metadata_field is not None
            and builder.current_metadata_field.get("label", "").lower() in {"título", "titulo"}
            and not meta
            and not DECREE_HEADER_PATTERN.match(text)
            and not PREAMBLE_PATTERN.match(text)
        ):
            builder.append_to_leaf(builder.current_metadata_field, text, page)
            continue

        if builder.current_metadata_block is not None and not meta:
            builder.in_metadata_block = False

        if is_title_like(text) and builder.current_title is None and builder.current_article is None:
            builder.current_title = builder.add(
                element_type="title",
                page_number=page,
                title=text,
                element_text=text,
                content_status="text_extracted",
                structure_confidence="medium",
                **flags,
            )
            last_leaf = builder.current_title
            continue

        if DECREE_HEADER_PATTERN.match(text):
            last_leaf = builder.add(
                element_type="decree_header",
                page_number=page,
                element_text=text,
                content_status="text_extracted",
                **flags,
            )
            continue

        preamble = PREAMBLE_PATTERN.match(text)
        if preamble:
            builder.current_preamble = builder.add(
                element_type="preamble",
                page_number=page,
                label=preamble.group(1),
                text_source=None,
                **flags,
            )
            builder.current_article = None
            builder.last_visual_object = None
            last_leaf = None
            continue

        chapter = CHAPTER_PATTERN.match(text) or TITLE_SECTION_PATTERN.match(text)
        if chapter:
            builder.current_chapter = builder.add(
                element_type="chapter",
                page_number=page,
                label=normalize_marker_label(chapter.group("label")),
                title=clean_title_separator(chapter.group("title")),
                text_source=None,
                **flags,
            )
            builder.current_numbered_headings = []
            builder.current_lettered_heading = None
            builder.current_section = None
            builder.current_subsection = None
            builder.current_article = None
            builder.current_lettered_heading = None
            builder.current_preamble = None
            builder.last_visual_object = None
            last_leaf = None
            continue

        section = SECTION_PATTERN.match(text)
        if section:
            builder.current_section = builder.add(
                element_type="section",
                page_number=page,
                label=normalize_marker_label(section.group("label")),
                title=clean_title_separator(section.group("title")),
                parent=builder.current_chapter,
                text_source=None,
                **flags,
            )
            builder.current_numbered_headings = []
            builder.current_lettered_heading = None
            builder.current_subsection = None
            builder.current_article = None
            builder.current_lettered_heading = None
            builder.current_preamble = None
            builder.last_visual_object = None
            last_leaf = None
            continue

        subsection = SUBSECTION_PATTERN.match(text)
        if subsection:
            builder.current_subsection = builder.add(
                element_type="subsection",
                page_number=page,
                label=normalize_marker_label(subsection.group("label")),
                title=clean_title_separator(subsection.group("title")),
                parent=builder.current_section or builder.current_chapter,
                text_source=None,
                **flags,
            )
            builder.current_numbered_headings = []
            builder.current_lettered_heading = None
            builder.current_article = None
            builder.current_lettered_heading = None
            builder.current_preamble = None
            builder.last_visual_object = None
            last_leaf = None
            continue

        numbered_heading = NUMBERED_HEADING_PATTERN.match(text)
        if numbered_heading:
            label = normalize_marker_label(numbered_heading.group("label"))
            title, heading_remainder = split_numbered_heading_title(numbered_heading.group("title"))

            depth = numbered_heading_depth(label)

            # Remove headings at the same or deeper level.
            while (
                builder.current_numbered_headings
                and builder.current_numbered_headings[-1].get("_numbered_depth", 0) >= depth
            ):
                builder.current_numbered_headings.pop()

            parent = (
                builder.current_lettered_heading
                or builder.current_numbered_parent()
                or builder.current_subsection
                or builder.current_section
                or builder.current_chapter
            )

            heading_element = builder.add(
                element_type="numbered_heading",
                page_number=page,
                label=label,
                title=title,
                parent=parent,
                text_source=None,
                structure_confidence="medium",
                **flags,
            )

            heading_element["_numbered_depth"] = depth

            builder.current_numbered_headings.append(heading_element)
            builder.current_article = None
            builder.current_lettered_heading = None
            builder.current_preamble = None
            builder.last_visual_object = None
            last_leaf = None

            if heading_remainder:
                last_leaf = builder.add(
                    element_type="paragraph",
                    page_number=page,
                    element_text=heading_remainder,
                    parent=heading_element,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )

            continue

        uppercase_letter_heading = UPPERCASE_LETTER_HEADING_PATTERN.match(text)
        if uppercase_letter_heading:
            label = normalize_marker_label(uppercase_letter_heading.group("label"))
            title, heading_remainder = split_numbered_heading_title(uppercase_letter_heading.group("title"))

            parent = (
                builder.current_numbered_parent()
                or builder.current_subsection
                or builder.current_section
                or builder.current_chapter
            )

            builder.current_lettered_heading = builder.add(
                element_type="lettered_heading",
                page_number=page,
                label=label,
                title=title,
                parent=parent,
                text_source=None,
                structure_confidence="medium",
                **flags,
            )

            builder.current_article = None
            builder.current_preamble = None
            builder.last_visual_object = None
            last_leaf = None

            if heading_remainder:
                last_leaf = builder.add(
                    element_type="paragraph",
                    page_number=page,
                    element_text=heading_remainder,
                    parent=builder.current_lettered_heading,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )

            continue

        transitory_article = TRANSITORY_ARTICLE_PATTERN.match(text)
        if transitory_article:
            label = normalize_marker_label(transitory_article.group("label"))
            rest = (transitory_article.group("rest") or "").strip()

            parent = (
                builder.current_lettered_heading
                or builder.current_numbered_parent()
                or builder.current_subsection
                or builder.current_section
                or builder.current_chapter
            )

            builder.current_article = builder.add(
                element_type="transitory_article",
                page_number=page,
                label=label,
                parent=parent,
                text_source=None,
                **flags,
            )

            builder.current_preamble = None
            builder.last_visual_object = None
            last_leaf = None

            if rest:
                last_leaf = builder.add(
                    element_type="paragraph",
                    page_number=page,
                    element_text=clean_title_separator(rest),
                    parent=builder.current_article,
                    content_status="text_extracted",
                    **flags,
                )

            continue

        article = ARTICLE_PATTERN.match(text)
        if article:
            article_label = normalize_marker_label(article.group("label"))
            rest = clean_title_separator(article.group("rest") or "")

            parent = (
                builder.current_lettered_heading
                or builder.current_numbered_parent()
                or builder.current_subsection
                or builder.current_section
                or builder.current_chapter
            )

            builder.current_article = builder.add(
                element_type="article",
                page_number=page,
                label=article_label,
                parent=parent,
                text_source=None,
                **flags,
            )
            builder.current_preamble = None
            builder.last_visual_object = None
            last_leaf = None
            if rest:
                last_leaf = builder.add(
                    element_type="paragraph",
                    page_number=page,
                    element_text=rest,
                    parent=builder.current_article,
                    content_status="text_extracted",
                    **flags,
                )
            continue

        for kind, pattern, element_type, content_status in [
            ("table", TABLE_PATTERN, "table", "title_or_caption_extracted"),
            ("graph", GRAPH_PATTERN, "graph", "title_or_caption_extracted"),
            ("figure", FIGURE_PATTERN, "figure", "title_or_caption_extracted"),
            ("annex", ANNEX_PATTERN, "annex", "text_extracted"),
        ]:
            m = pattern.match(text)
            if m:
                visual_label = normalize_marker_label(m.group("label"))
                visual_title, visual_remainder = split_visual_title(m.group("title"))

                parent = builder.current_parent()
                if element_type == "annex":
                    parent = builder.current_chapter

                visual_element_text = (
                    f"{visual_label}: {visual_title}"
                    if visual_label and visual_title
                    else text
                )

                visual_element = builder.add(
                    element_type=element_type,
                    page_number=page,
                    label=visual_label,
                    title=visual_title,
                    element_text=visual_element_text,
                    parent=parent,
                    content_status=content_status,
                    structure_confidence="medium",
                    **flags,
                )

                last_leaf = visual_element
                if element_type in {"table", "graph", "figure"}:
                    builder.last_visual_object = visual_element

                if visual_remainder:
                    if is_visual_note_like(visual_remainder):
                        last_leaf = builder.add(
                            element_type="object_note",
                            page_number=page,
                            element_text=visual_remainder,
                            parent=visual_element,
                            content_status="text_extracted",
                            structure_confidence="medium",
                            **flags,
                        )
                    else:
                        last_leaf = builder.add(
                            element_type="paragraph",
                            page_number=page,
                            element_text=visual_remainder,
                            parent=parent,
                            content_status="text_extracted",
                            structure_confidence="medium",
                            **flags,
                        )

                break
        else:
            m_source = SOURCE_LINE_PATTERN.match(text)
            if m_source:
                source_text, source_remainder = split_source_note_text(text)
                parent = builder.last_visual_object or builder.current_parent()
                last_leaf = builder.add(
                    element_type="source_note",
                    page_number=page,
                    label="Fuente",
                    element_text=source_text,
                    parent=parent,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )

                if source_remainder:
                    last_leaf = builder.add(
                        element_type="paragraph",
                        page_number=page,
                        element_text=source_remainder,
                        parent=visual_parent_for_remainder(builder),
                        content_status="text_extracted",
                        structure_confidence="medium",
                        **flags,
                    )

                continue

            object_note = OBJECT_NOTE_LINE_PATTERN.match(text)
            if object_note and builder.last_visual_object is not None:
                last_leaf = builder.add(
                    element_type="object_note",
                    page_number=page,
                    element_text=text,
                    parent=builder.last_visual_object,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )
                continue

            compound = COMPOUND_ITEM_PATTERN.match(text)
            numbered = NUMBERED_ITEM_PATTERN.match(text)
            if numbered and re.fullmatch(r"(?:19|20)\d{2}", numbered.group("label")):
                numbered = None

            roman = ROMAN_ITEM_PATTERN.match(text)
            lettered = LETTERED_ITEM_PATTERN.match(text)
            bullet = BULLET_ITEM_PATTERN.match(text)

            if compound:
                parent = builder.current_article or builder.current_parent()
                last_leaf = builder.add(
                    element_type="compound_item",
                    page_number=page,
                    label=compound.group("label"),
                    element_text=compound.group("text"),
                    parent=parent,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )
                continue

            if numbered:
                parent = builder.current_article or builder.current_parent()
                last_leaf = builder.add(
                    element_type="numbered_item",
                    page_number=page,
                    label=numbered.group("label"),
                    element_text=numbered.group("text"),
                    parent=parent,
                    content_status="text_extracted",
                    **flags,
                )
                continue

            if roman:
                parent = builder.current_article or builder.current_parent()
                last_leaf = builder.add(
                    element_type="roman_item",
                    page_number=page,
                    label=roman.group("label"),
                    element_text=roman.group("text"),
                    parent=parent,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )
                continue

            if lettered:
                parent = builder.current_article or builder.current_parent()
                last_leaf = builder.add(
                    element_type="lettered_item",
                    page_number=page,
                    label=lettered.group("label"),
                    element_text=lettered.group("text"),
                    parent=parent,
                    content_status="text_extracted",
                    **flags,
                )
                continue

            if bullet:
                parent = builder.current_article or builder.current_parent()
                last_leaf = builder.add(
                    element_type="bullet_item",
                    page_number=page,
                    label="-",
                    element_text=bullet.group("text"),
                    parent=parent,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )
                continue

            if SIGNATURE_START_PATTERN.match(text):
                builder.current_signature_block = builder.add(
                    element_type="signature_block",
                    page_number=page,
                    element_text=text,
                    content_status="text_extracted",
                    structure_confidence="medium",
                    **flags,
                )
                last_leaf = builder.current_signature_block
                continue

            if builder.current_signature_block is not None and page >= builder.current_signature_block["page_start"]:
                builder.append_to_leaf(builder.current_signature_block, text, page)
                last_leaf = builder.current_signature_block
                continue

            if should_append_page_continuation(last_leaf, text, page):
                builder.append_to_leaf(last_leaf, text, page)
                continue

            parent = builder.current_parent()
            if parent:
                confidence = "high" if parent["element_type"] in {"preamble", "article"} else "medium"
                last_leaf = builder.add(
                    element_type="paragraph",
                    page_number=page,
                    element_text=text,
                    parent=parent,
                    content_status="text_extracted",
                    structure_confidence=confidence,
                    **flags,
                )
            else:
                last_leaf = builder.add(
                    element_type="other",
                    page_number=page,
                    element_text=text,
                    content_status="text_extracted",
                    structure_confidence="uncertain",
                    **flags,
                )

    return builder.hierarchy(), builder.flat_rows()


# ---------------------------------------------------------------------
# Output and run
# ---------------------------------------------------------------------
def build_document_metadata(inventory_record: sqlite3.Row | None, payload: dict[str, Any]) -> dict[str, Any]:
    payload_metadata = payload.get("document_metadata", {}) or {}

    def get_value(*names: str) -> Any:
        for name in names:
            if inventory_record is not None and name in inventory_record.keys() and inventory_record[name] is not None:
                return inventory_record[name]
            if name in payload_metadata and payload_metadata[name] is not None:
                return payload_metadata[name]
        return None

    return {
        "document_title": get_value("document_title"),
        "document_type": get_value("document_type"),
        "source_country": get_value("source_country"),
        "source_region": get_value("source_region"),
        "issuing_institution": get_value("issuing_institution", "organism"),
        "publication_date": get_value("publication_date"),
        "promulgation_date": get_value("promulgation_date"),
        "effective_date": get_value("effective_date"),
        "source_url": get_value("source_url_repository", "source_url"),
        "language": get_value("language"),
        "original_file_format": get_value("original_file_format"),
    }


def write_structure_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_structure_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "structure_id", "element_id", "parent_id", "document_id", "element_order",
        "nesting_level", "element_type", "label", "title", "page_start", "page_end",
        "element_text", "text_source", "content_status", "structure_confidence",
        "requires_manual_review", "review_reason_code", "review_reason_note"
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def insert_structure_event(connection: sqlite3.Connection, event: dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO structure_events (
            structure_id, normalization_id, assessment_id, extraction_id, document_id,
            structure_date, structure_tool, parser_method, structure_version,
            source_text_field, elements_count, uncertain_elements_count,
            manual_review_elements_count, structure_json_path, structure_csv_path,
            structure_status, created_at, notes
        )
        VALUES (
            :structure_id, :normalization_id, :assessment_id, :extraction_id, :document_id,
            :structure_date, :structure_tool, :parser_method, :structure_version,
            :source_text_field, :elements_count, :uncertain_elements_count,
            :manual_review_elements_count, :structure_json_path, :structure_csv_path,
            :structure_status, :created_at, :notes
        )
        """,
        event,
    )


def insert_structure_elements(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    connection.executemany(
        """
        INSERT INTO structure_elements (
            structure_id, element_id, parent_id, document_id, element_order, nesting_level,
            element_type, label, title, page_start, page_end, element_text, text_source,
            content_status, structure_confidence, requires_manual_review,
            review_reason_code, review_reason_note
        )
        VALUES (
            :structure_id, :element_id, :parent_id, :document_id, :element_order, :nesting_level,
            :element_type, :label, :title, :page_start, :page_end, :element_text, :text_source,
            :content_status, :structure_confidence, :requires_manual_review,
            :review_reason_code, :review_reason_note
        )
        """,
        rows,
    )


def run_structure_identification(document_id: str | None, normalization_id: str | None) -> dict[str, Any]:
    if not document_id and not normalization_id:
        raise ValueError("Provide either --document-id or --normalization-id.")

    database_path = choose_database_path()
    connection = connect_database(database_path)

    try:
        connection.execute(STRUCTURE_EVENTS_SQL)
        connection.execute(STRUCTURE_ELEMENTS_SQL)

        if normalization_id:
            normalization = get_normalization_by_id(connection, normalization_id)
        else:
            normalization = get_latest_normalization_for_document(connection, document_id)

        document_id_value = normalization["document_id"]
        inventory = get_document_inventory_record(connection, document_id_value)
        structure_id = next_structure_id(connection)

        normalized_json_path = resolve_project_path(normalization["normalized_json_path"])
        if not normalized_json_path.exists():
            raise FileNotFoundError(f"Normalized JSON not found: {normalized_json_path}")

        with normalized_json_path.open("r", encoding="utf-8") as file:
            input_payload = json.load(file)

        hierarchy, rows = parse_structure(document_id_value, structure_id, input_payload)

        page_numbers = [int(page["page_number"]) for page in input_payload.get("pages", [])]
        source_page_range = {
            "start": min(page_numbers) if page_numbers else 0,
            "end": max(page_numbers) if page_numbers else 0,
        }

        uncertain_count = sum(1 for row in rows if row["structure_confidence"] == "uncertain")
        review_count = sum(1 for row in rows if row["requires_manual_review"] == 1)
        notes_parts = []
        if uncertain_count:
            notes_parts.append(f"{uncertain_count} element(s) have uncertain structure.")
        if review_count:
            notes_parts.append(f"{review_count} element(s) inherit manual-review flags.")
        notes = " ".join(notes_parts) if notes_parts else None
        status = "Completed with warnings" if notes else "Completed"

        output_dir = LEGAL_STRUCTURE_DIR / document_id_value
        report_dir = LEGAL_STRUCTURE_REPORT_DIR / document_id_value
        structure_json_path = output_dir / f'{normalization["normalization_id"]}_{structure_id}_structure.json'
        structure_csv_path = report_dir / f'{normalization["normalization_id"]}_{structure_id}_structure_elements.csv'

        output_payload = {
            "document_id": document_id_value,
            "document_metadata": build_document_metadata(inventory, input_payload),
            "workflow_metadata": {
                "structure_id": structure_id,
                "normalization_id": normalization["normalization_id"],
                "assessment_id": normalization["assessment_id"],
                "extraction_id": normalization["extraction_id"],
                "structure_created_by": "rule_based_parser",
                "structure_tool": "05_identify_legal_structure.py",
                "parser_method": "regular_expression_state_parser",
                "structure_version": "0.6",
                "source_text_field": "normalization.normalized_text",
            },
            "source_page_range": source_page_range,
            "structure_summary": {
                "elements_count": len(rows),
                "uncertain_elements_count": uncertain_count,
                "manual_review_elements_count": review_count,
                "structure_status": status,
                "notes": notes,
            },
            "structure": hierarchy,
        }

        write_structure_json(structure_json_path, output_payload)
        write_structure_csv(structure_csv_path, rows)

        timestamp = utc_now_iso()
        event = {
            "structure_id": structure_id,
            "normalization_id": normalization["normalization_id"],
            "assessment_id": normalization["assessment_id"],
            "extraction_id": normalization["extraction_id"],
            "document_id": document_id_value,
            "structure_date": timestamp[:10],
            "structure_tool": "05_identify_legal_structure.py",
            "parser_method": "regular_expression_state_parser",
            "structure_version": "0.6",
            "source_text_field": "normalization.normalized_text",
            "elements_count": len(rows),
            "uncertain_elements_count": uncertain_count,
            "manual_review_elements_count": review_count,
            "structure_json_path": project_relative(structure_json_path),
            "structure_csv_path": project_relative(structure_csv_path),
            "structure_status": status,
            "created_at": timestamp,
            "notes": notes,
        }

        insert_structure_event(connection, event)
        insert_structure_elements(connection, rows)
        connection.commit()

        print("\nLegal-structure identification completed")
        print("----------------------------------------")
        for key, value in event.items():
            print(f"{key}: {value}")

        print("\nGenerated files")
        print("---------------")
        print(project_relative(structure_json_path))
        print(project_relative(structure_csv_path))

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
    parser = argparse.ArgumentParser(description="Run Sub-task 1.5 legal-structure identification.")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--normalization-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        run_structure_identification(
            document_id=args.document_id,
            normalization_id=args.normalization_id,
        )
    except Exception as error:
        raise SystemExit(f"\nLegal-structure identification failed: {error}") from error
