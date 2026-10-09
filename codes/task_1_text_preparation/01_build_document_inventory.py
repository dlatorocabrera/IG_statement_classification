#!/usr/bin/env python3
"""
01_build_document_inventory.py

First workflow step for the IG automation protocol:
Document Intake and Corpus Inventory.

What it does
------------
1. Reads one or more PDF files.
2. Extracts document-level metadata from the first page when the document follows
   the LeyChile header format.
3. Computes technical provenance fields: page count, file size, SHA-256 hash,
   and whether the PDF appears to have a usable text layer.
4. Exports the inventory as CSV, JSON, and SQLite.

Dependencies
------------
pip install pdfplumber pypdf

Design choice
-------------
The SQLite database is the recommended prototype storage format because it keeps
the inventory relational, queryable, and migration-ready. CSV/JSON are provided
as review and interchange exports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pdfplumber
from pypdf import PdfReader


SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from config import DATABASE_PATH, INVENTORY_TABLE_DIR
except ImportError as error:
    raise SystemExit("Could not import codes/config.py.") from error


LEYCHILE_FIELD_PATTERNS = {
    "document_type_raw": r"Tipo Norma\s*:\s*(.+)",
    "publication_date_raw": r"Fecha Publicación\s*:\s*(.+)",
    "promulgation_date_raw": r"Fecha Promulgación\s*:\s*(.+)",
    "issuing_institution": r"Organismo\s*:\s*(.+)",
    "document_title_raw": r"Título\s*:\s*(.+?)(?=\n\s*Tipo Versión\s*:|\n\s*Tipo Version\s*:)",
    "version_type_raw": r"Tipo Versión\s*:\s*(.+)",
    "effective_date_raw": r"Inicio Vigencia\s*:\s*(.+)",
    "source_external_id": r"Id Norma\s*:\s*(.+)",
    "source_url_repository": r"URL\s*:\s*(.+)",
}

SPANISH_MONTHS = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "setiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}


@dataclass
class DocumentInventoryRecord:
    document_id: str
    document_title: str
    document_type: str
    document_number: Optional[str]
    source_country: str
    source_region: Optional[str]
    jurisdiction_scope: Optional[str]
    issuing_institution: Optional[str]
    publication_date: Optional[str]
    promulgation_date: Optional[str]
    effective_date: Optional[str]
    version_type: Optional[str]
    source_url_repository: Optional[str]
    source_database: Optional[str]
    source_external_id: Optional[str]
    language: str
    original_file_format: str
    file_name_original: str
    file_size_bytes: int
    page_count: int
    file_hash_sha256: str
    repository_generated_date: Optional[str]
    access_date: str
    ingestion_date: str
    inventory_created_at: str
    document_status: str
    notes: str


def sha256_file(path: Path) -> str:
    """Compute SHA-256 without loading the whole file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_dd_mm_yyyy(value: Optional[str]) -> Optional[str]:
    """Parse LeyChile dates such as 24-11-2017 into ISO format."""
    if not value:
        return None
    value = normalize_whitespace(value)
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", value)
    if not match:
        return None
    dd, mm, yyyy = match.groups()
    return f"{yyyy}-{mm}-{dd}"


def parse_repository_generated_date(first_page_text: str) -> Optional[str]:
    """Parse 'documento generado el 22-May-2018' as ISO date."""
    match = re.search(
        r"documento generado el\s+(\d{1,2})-([A-Za-zÁÉÍÓÚáéíóúñÑ]+)-(\d{4})",
        first_page_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    day, mon_raw, year = match.groups()
    mon_key = mon_raw.lower()
    # LeyChile often uses English-style 'May'; normalize both common cases.
    month = SPANISH_MONTHS.get(mon_key) or {"may": "05"}.get(mon_key[:3])
    if not month:
        return None
    return f"{year}-{month}-{int(day):02d}"


def extract_leychile_metadata(first_page_text: str) -> dict:
    """Extract metadata from a LeyChile first-page header."""
    out = {}
    for key, pattern in LEYCHILE_FIELD_PATTERNS.items():
        m = re.search(pattern, first_page_text, flags=re.IGNORECASE | re.DOTALL)
        out[key] = normalize_whitespace(m.group(1)) if m else None

    # Clean fields
    raw_type = out.get("document_type_raw")
    document_number = None
    document_type = None
    if raw_type:
        m = re.match(r"([A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s+(.+)", raw_type)
        if m:
            document_type, document_number = m.group(1), m.group(2)
        else:
            document_type = raw_type

    title = out.get("document_title_raw")
    if title:
        title = normalize_whitespace(title).title()
        # Keep Spanish capitalization readable but avoid forcing all acronyms.
        title = title.replace("Atmosférica Para", "Atmosférica para")
        title = title.replace("La Región", "la Región")

    version_type = out.get("version_type_raw")
    if version_type:
        version_type = version_type.split(" De :")[0].strip()

    return {
        "document_type": document_type,
        "document_number": document_number,
        "document_title": title,
        "publication_date": parse_dd_mm_yyyy(out.get("publication_date_raw")),
        "promulgation_date": parse_dd_mm_yyyy(out.get("promulgation_date_raw")),
        "effective_date": parse_dd_mm_yyyy(out.get("effective_date_raw")),
        "issuing_institution": out.get("issuing_institution"),
        "version_type": version_type,
        "source_external_id": out.get("source_external_id"),
        "source_url_repository": out.get("source_url_repository"),
        "repository_generated_date": parse_repository_generated_date(first_page_text),
    }


def inspect_pdf(path: Path, near_blank_threshold: int = 50) -> dict:
    """Inspect page count and text-layer availability."""
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        texts = [(p.extract_text() or "") for p in pdf.pages]
        char_counts = [len(t.strip()) for t in texts]

    extracted_pages = sum(c >= near_blank_threshold for c in char_counts)
    if extracted_pages == page_count:
        original_file_format = "Searchable PDF"
        status = "Ready for processing"
    elif extracted_pages == 0:
        original_file_format = "Scanned PDF"
        status = "Requires manual review"
    else:
        original_file_format = "Mixed PDF"
        status = "Requires manual review"

    return {
        "page_count": page_count,
        "first_page_text": texts[0] if texts else "",
        "char_counts": char_counts,
        "extracted_pages": extracted_pages,
        "original_file_format": original_file_format,
        "document_status": status,
    }


def build_document_id(country_code: str, institution_code: str, document_type_code: str, sequence: int) -> str:
    """Stable corpus-internal ID. Do not use the filename as the document ID."""
    return f"{country_code}_{institution_code}_{document_type_code}_{sequence:06d}"


def build_record(path: Path, sequence: int, access_date: str) -> DocumentInventoryRecord:
    pdf_info = inspect_pdf(path)
    meta = extract_leychile_metadata(pdf_info["first_page_text"])

    # Prefer a stable corpus-internal identifier. Store LeyChile Id Norma separately.
    document_id = build_document_id("CL", "MMA", "DEC", sequence)

    notes = (
        f"Initial inspection found text on {pdf_info['extracted_pages']} of "
        f"{pdf_info['page_count']} pages. Tables/figures should be reviewed in later "
        "extraction and quality-control tasks."
    )

    return DocumentInventoryRecord(
        document_id=document_id,
        document_title=meta.get("document_title") or path.stem,
        document_type="Decree" if (meta.get("document_type") or "").lower().startswith("decreto") else (meta.get("document_type") or "Other"),
        document_number=meta.get("document_number"),
        source_country="Chile",
        source_region="Región Metropolitana de Santiago",
        jurisdiction_scope="Regional plan issued by national ministry; applies to Región Metropolitana de Santiago",
        issuing_institution=meta.get("issuing_institution"),
        publication_date=meta.get("publication_date"),
        promulgation_date=meta.get("promulgation_date"),
        effective_date=meta.get("effective_date"),
        version_type=meta.get("version_type"),
        source_url_repository=meta.get("source_url_repository"),
        source_database="LeyChile / Biblioteca del Congreso Nacional de Chile",
        source_external_id=meta.get("source_external_id"),
        language="Spanish (es)",
        original_file_format=pdf_info["original_file_format"],
        file_name_original=path.name,
        file_size_bytes=path.stat().st_size,
        page_count=pdf_info["page_count"],
        file_hash_sha256=sha256_file(path),
        repository_generated_date=meta.get("repository_generated_date"),
        access_date=access_date,
        ingestion_date=access_date,
        inventory_created_at=date.today().isoformat(),
        document_status=pdf_info["document_status"],
        notes=notes,
    )


def write_csv(records: list[DocumentInventoryRecord], path: Path) -> None:
    rows = [asdict(r) for r in records]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(records: list[DocumentInventoryRecord], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8")


def write_sqlite(records: list[DocumentInventoryRecord], path: Path) -> None:
    if path.exists():
        path.unlink()
    rows = [asdict(r) for r in records]
    columns = list(rows[0].keys())

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    col_defs = []
    for col in columns:
        if col == "document_id":
            col_defs.append("document_id TEXT PRIMARY KEY")
        elif isinstance(rows[0][col], int):
            col_defs.append(f"{col} INTEGER")
        else:
            col_defs.append(f"{col} TEXT")
    cur.execute(f"CREATE TABLE document_inventory ({', '.join(col_defs)});")

    placeholders = ", ".join(["?"] * len(columns))
    cur.executemany(
        f"INSERT INTO document_inventory ({', '.join(columns)}) VALUES ({placeholders})",
        [[row[col] for col in columns] for row in rows],
    )
    conn.commit()
    conn.close()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, retaining canonical output paths as defaults."""
    parser = argparse.ArgumentParser(
        description="Build the document inventory and canonical SQLite database."
    )
    parser.add_argument("--pdf", nargs="+", required=True, help="One or more PDF paths.")
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default=str(INVENTORY_TABLE_DIR),
        help="CSV/JSON inventory directory. Default: tables/inventory.",
    )
    parser.add_argument(
        "--database",
        default=str(DATABASE_PATH),
        help="SQLite inventory database. Default: data/database/corpus_inventory.sqlite.",
    )
    parser.add_argument(
        "--access-date",
        "--access_date",
        dest="access_date",
        default=date.today().isoformat(),
        help="ISO date for retrieval/ingestion.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()

    out_dir = Path(args.out_dir).expanduser()
    database_path = Path(args.database).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    records = [build_record(Path(p), sequence=i + 1, access_date=args.access_date) for i, p in enumerate(args.pdf)]

    write_csv(records, out_dir / "document_inventory.csv")
    write_json(records, out_dir / "document_inventory.json")
    write_sqlite(records, database_path)

    print(f"Created CSV/JSON inventory outputs in {out_dir}")
    print(f"Created SQLite inventory database at {database_path}")


if __name__ == "__main__":
    main()
