#!/usr/bin/env python3
"""Create reproducible Task 3 statement-level training and test datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from config import DATABASE_PATH, PACKAGES_DIR, REPORTS_DIR
except ImportError as error:
    raise SystemExit(
        "Could not import codes/config.py. Store this file as:\n"
        "  codes/task_3_statement_classification/07_create_train_test_split.py"
    ) from error


# ---------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------
TASK_ID = "task_3_statement_classification"
SCRIPT_ID = "07"
SCRIPT_NAME = "07_create_train_test_split.py"
SCRIPT_VERSION = "1.3"
DEFAULT_REVIEW_FILENAME = "IG_statement_alignment_review.xlsx"
DEFAULT_SUMMARY_FILENAME = "IG_statement_alignment_summary.xlsx"
DEFAULT_SHEET = "Alignment Review"
DEFAULT_HEADER_ROW = 5
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.90

TRAINING_FILENAME = "IG_statement_training_set.csv"
TEST_FILENAME = "IG_statement_test_set.csv"
WORKBOOK_FILENAME = "IG_statement_train_test_split.xlsx"
MANIFEST_FILENAME = "IG_statement_split_manifest.json"
HISTORY_DIRNAME = "history"

VALID_LABELS = ("regulative", "constitutive", "non_institutional")
EXPECTED_SOURCE_COUNTS = {
    "total_review_records": 768,
    "include_in_final_set": {"yes": 703, "no": 43, "hold": 22},
    "included_reviewer_gold_label": {
        "regulative": 338,
        "constitutive": 156,
        "non_institutional": 209,
    },
}
EXPECTED_ELIGIBLE_COUNTS = {
    "eligible_records": 702,
    "reviewer_gold_label": {
        "regulative": 339,
        "constitutive": 154,
        "non_institutional": 209,
    },
}
SOURCE_REVIEW_ADJUSTMENT_NOTE = (
    "REV_0752 was corrected from constitutive to regulative so its reviewed label "
    "matches exact-text duplicate REV_0283. This shifts the included source-label "
    "counts from the original task specification by +1 regulative and -1 constitutive."
)
ELIGIBILITY_POLICY_NOTE = (
    "REV_0768 is a workbook-only gap rather than a valid candidate statement; "
    "it is excluded because candidate_id and candidate_text_original are blank. "
    "This user-authorized policy yields 702 eligible statements."
)

REQUIRED_SOURCE_COLUMNS = (
    "review_record_id",
    "candidate_id",
    "alignment_group_id",
    "article",
    "scope",
    "source_parent_id",
    "match_type",
    "candidate_text_original",
    "reviewer_gold_label",
    "include_in_final_set",
)

SUMMARY_COMPARISON_COLUMNS = (
    "review_record_id",
    "candidate_id",
    "article",
    "scope",
    "match_type",
    "reviewer_gold_label",
    "include_in_final_set",
)

OUTPUT_COLUMNS = (
    "review_record_id",
    "candidate_id",
    "alignment_group_id",
    "article",
    "scope",
    "source_parent_id",
    "match_type",
    "candidate_text_original",
    "reviewer_gold_label",
    "split",
    "duplicate_group_id",
    "random_seed",
)

DUPLICATE_AUDIT_COLUMNS = (
    "audit_type",
    "duplicate_group_id_a",
    "duplicate_group_id_b",
    "duplicate_group_size_a",
    "duplicate_group_size_b",
    "review_record_ids_a",
    "review_record_ids_b",
    "candidate_ids_a",
    "candidate_ids_b",
    "reviewer_gold_labels_a",
    "reviewer_gold_labels_b",
    "similarity_ratio",
    "candidate_text_original_a",
    "candidate_text_original_b",
    "cross_label",
    "action",
)


SPLIT_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS split_events (
    split_id TEXT PRIMARY KEY,
    matching_id TEXT NOT NULL,
    segmentation_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    split_date TEXT NOT NULL,
    split_tool TEXT NOT NULL,
    split_tool_version TEXT NOT NULL,
    source_review_path TEXT NOT NULL,
    source_review_hash_sha256 TEXT NOT NULL,
    source_summary_path TEXT NOT NULL,
    source_summary_hash_sha256 TEXT NOT NULL,
    eligibility_rules_json TEXT NOT NULL,
    valid_labels_json TEXT NOT NULL,
    eligible_records_count INTEGER NOT NULL,
    excluded_records_count INTEGER NOT NULL,
    training_records_count INTEGER NOT NULL,
    test_records_count INTEGER NOT NULL,
    class_counts_json TEXT NOT NULL,
    requested_test_proportion REAL NOT NULL,
    observed_test_proportion REAL NOT NULL,
    random_seed INTEGER NOT NULL,
    stratification_field TEXT NOT NULL,
    grouping_field TEXT NOT NULL,
    split_method TEXT NOT NULL,
    duplicate_group_normalization TEXT NOT NULL,
    near_duplicate_threshold REAL NOT NULL,
    duplicate_audit_json TEXT NOT NULL,
    quality_checks_json TEXT NOT NULL,
    training_csv_path TEXT NOT NULL,
    training_csv_hash_sha256 TEXT NOT NULL,
    test_csv_path TEXT NOT NULL,
    test_csv_hash_sha256 TEXT NOT NULL,
    split_workbook_path TEXT NOT NULL,
    split_workbook_hash_sha256 TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_hash_sha256 TEXT NOT NULL,
    split_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    CHECK (length(split_date) = 10),
    CHECK (eligible_records_count >= 0),
    CHECK (excluded_records_count >= 0),
    CHECK (training_records_count >= 0),
    CHECK (test_records_count >= 0),
    CHECK (training_records_count + test_records_count = eligible_records_count),
    CHECK (requested_test_proportion > 0 AND requested_test_proportion < 1),
    CHECK (observed_test_proportion > 0 AND observed_test_proportion < 1),
    CHECK (near_duplicate_threshold > 0 AND near_duplicate_threshold <= 1),
    CHECK (json_valid(eligibility_rules_json)),
    CHECK (json_valid(valid_labels_json)),
    CHECK (json_valid(class_counts_json)),
    CHECK (json_valid(duplicate_audit_json)),
    CHECK (json_valid(quality_checks_json)),
    FOREIGN KEY (matching_id)
        REFERENCES matching_events(matching_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (segmentation_id)
        REFERENCES segmentation_events(segmentation_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS split_assignments (
    split_id TEXT NOT NULL,
    segmentation_id TEXT NOT NULL,
    review_record_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    partition TEXT NOT NULL,
    gold_label TEXT NOT NULL,
    duplicate_group_id TEXT NOT NULL,
    duplicate_group_size INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (split_id, review_record_id),
    UNIQUE (split_id, candidate_id),
    CHECK (partition IN ('train', 'test')),
    CHECK (gold_label IN ('regulative', 'constitutive', 'non_institutional')),
    CHECK (duplicate_group_size >= 1),
    FOREIGN KEY (split_id)
        REFERENCES split_events(split_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (segmentation_id, candidate_id)
        REFERENCES candidate_statements(segmentation_id, statement_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_split_events_matching_date
    ON split_events(matching_id, split_date, created_at);

CREATE INDEX IF NOT EXISTS idx_split_assignments_partition_label
    ON split_assignments(split_id, partition, gold_label);

CREATE INDEX IF NOT EXISTS idx_split_assignments_candidate
    ON split_assignments(segmentation_id, candidate_id);
"""


class DuplicateLabelConflictError(ValueError):
    """Raised when identical normalized texts have different reviewed labels."""


def utc_now_iso() -> str:
    """Return the current UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_relative(path: Path) -> str:
    """Store project-relative paths when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 hash without loading the full file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_hashes(
    paths: dict[str, Path],
    expected_hashes: dict[str, str],
    stage: str,
) -> None:
    """Fail if a source workbook changes while a run is in progress."""
    changed = []
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != expected_hashes[name]:
            changed.append(
                f"{name}: expected {expected_hashes[name]}, observed {observed}"
            )
    if changed:
        raise ValueError(
            f"Source workbook changed {stage}; restart with a stable reviewed pair. "
            + " | ".join(changed)
        )


def is_blank(value: Any) -> bool:
    """Return True for null values or strings containing only whitespace."""
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def normalize_category(value: Any) -> str:
    """Normalize a short categorical value for comparisons only."""
    if is_blank(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def normalize_for_duplicate_check(text: str) -> str:
    """Normalize text only for exact and near-duplicate detection."""
    if not isinstance(text, str):
        raise TypeError("candidate_text_original must be a string for duplicate checks.")
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _comparison_value(value: Any) -> str:
    """Create a stable scalar representation for workbook cross-checks."""
    if is_blank(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        # Excel stores numeric cells at finite precision. Twelve significant
        # digits distinguish audit values while absorbing harmless binary
        # round-trip noise (for example, partition proportions).
        return format(value, ".12g")
    return str(value)


def _require_columns(data: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required columns: {', '.join(missing)}"
        )


def load_review_workbook(
    input_path: Path,
    sheet: str = DEFAULT_SHEET,
    header_row: int = DEFAULT_HEADER_ROW,
) -> pd.DataFrame:
    """Load the reviewed alignment table without changing reviewed text."""
    if not input_path.exists():
        raise FileNotFoundError(f"Review workbook not found: {input_path}")
    if header_row < 1:
        raise ValueError("header_row must be at least 1.")

    try:
        data = pd.read_excel(
            input_path,
            sheet_name=sheet,
            header=header_row - 1,
            dtype=object,
            keep_default_na=False,
            engine="openpyxl",
        )
    except ValueError as error:
        raise ValueError(
            f"Could not read worksheet {sheet!r} from {input_path.name}: {error}"
        ) from error

    data.columns = [str(column).strip() for column in data.columns]
    _require_columns(data, REQUIRED_SOURCE_COLUMNS, "Alignment Review")
    data = data.copy()
    data["_source_excel_row"] = range(header_row + 1, header_row + 1 + len(data))
    return data


def load_summary_workbook(summary_path: Path) -> pd.DataFrame:
    """Load the summary workbook's source-data lineage table."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary workbook not found: {summary_path}")
    try:
        data = pd.read_excel(
            summary_path,
            sheet_name="Source Data",
            header=0,
            dtype=object,
            keep_default_na=False,
            engine="openpyxl",
        )
    except ValueError as error:
        raise ValueError(
            f"Could not read 'Source Data' from {summary_path.name}: {error}"
        ) from error
    data.columns = [str(column).strip() for column in data.columns]
    _require_columns(data, SUMMARY_COMPARISON_COLUMNS, "Summary Source Data")
    return data


def validate_summary_pair(
    review_data: pd.DataFrame,
    summary_data: pd.DataFrame,
) -> dict[str, Any]:
    """Verify that the review and summary workbooks describe the same rows."""
    if len(review_data) != len(summary_data):
        raise ValueError(
            "Review/summary workbook mismatch: "
            f"{len(review_data)} review rows versus {len(summary_data)} summary rows."
        )

    for name, data in (("review", review_data), ("summary", summary_data)):
        identifiers = data["review_record_id"].map(_comparison_value)
        duplicates = sorted(
            identifier
            for identifier, count in Counter(identifiers).items()
            if identifier and count > 1
        )
        if duplicates:
            raise ValueError(
                f"{name.title()} workbook has duplicate review_record_id values: "
                + ", ".join(duplicates[:10])
            )

    review_indexed = review_data.set_index(
        review_data["review_record_id"].map(_comparison_value),
        drop=False,
    )
    summary_indexed = summary_data.set_index(
        summary_data["review_record_id"].map(_comparison_value),
        drop=False,
    )
    missing_from_summary = sorted(set(review_indexed.index) - set(summary_indexed.index))
    extra_in_summary = sorted(set(summary_indexed.index) - set(review_indexed.index))
    if missing_from_summary or extra_in_summary:
        raise ValueError(
            "Review/summary workbook record identifiers differ. "
            f"Missing from summary: {missing_from_summary[:10]}; "
            f"extra in summary: {extra_in_summary[:10]}."
        )

    differences: list[str] = []
    for record_id in review_indexed.index:
        for column in SUMMARY_COMPARISON_COLUMNS:
            review_value = _comparison_value(review_indexed.at[record_id, column])
            summary_value = _comparison_value(summary_indexed.at[record_id, column])
            if review_value != summary_value:
                differences.append(
                    f"{record_id}.{column}: review={review_value!r}, "
                    f"summary={summary_value!r}"
                )
                if len(differences) >= 10:
                    break
        if len(differences) >= 10:
            break

    if differences:
        raise ValueError(
            "Review/summary workbook values differ; use a matched reviewed pair. "
            + " | ".join(differences)
        )

    return {
        "status": "passed",
        "compared_records": len(review_data),
        "compared_columns": list(SUMMARY_COMPARISON_COLUMNS),
        "differing_cells": 0,
    }


def validate_source_counts(
    data: pd.DataFrame,
    allow_count_mismatch: bool = False,
) -> dict[str, Any]:
    """Validate the expected reviewed workbook totals."""
    _require_columns(
        data,
        ("include_in_final_set", "reviewer_gold_label"),
        "Review data",
    )
    include_values = data["include_in_final_set"].map(normalize_category)
    label_values = data["reviewer_gold_label"].map(normalize_category)
    include_counts = Counter(include_values)
    included_label_counts = Counter(label_values[include_values.eq("yes")])

    observed = {
        "total_review_records": int(len(data)),
        "include_in_final_set": {
            key: int(include_counts.get(key, 0))
            for key in ("yes", "no", "hold")
        },
        "included_reviewer_gold_label": {
            key: int(included_label_counts.get(key, 0))
            for key in VALID_LABELS
        },
        "unexpected_include_values": {
            key or "<blank>": int(value)
            for key, value in sorted(include_counts.items())
            if key not in {"yes", "no", "hold"}
        },
        "unexpected_included_labels": {
            key or "<blank>": int(value)
            for key, value in sorted(included_label_counts.items())
            if key not in set(VALID_LABELS)
        },
    }

    mismatches: list[str] = []
    if observed["total_review_records"] != EXPECTED_SOURCE_COUNTS["total_review_records"]:
        mismatches.append(
            "total_review_records: expected "
            f"{EXPECTED_SOURCE_COUNTS['total_review_records']}, observed "
            f"{observed['total_review_records']}"
        )
    for section in ("include_in_final_set", "included_reviewer_gold_label"):
        for key, expected_value in EXPECTED_SOURCE_COUNTS[section].items():
            observed_value = observed[section].get(key, 0)
            if observed_value != expected_value:
                mismatches.append(
                    f"{section}.{key}: expected {expected_value}, "
                    f"observed {observed_value}"
                )
    if observed["unexpected_include_values"]:
        mismatches.append(
            "unexpected include_in_final_set values: "
            + json.dumps(observed["unexpected_include_values"], ensure_ascii=False)
        )
    if observed["unexpected_included_labels"]:
        mismatches.append(
            "unexpected labels among included records: "
            + json.dumps(observed["unexpected_included_labels"], ensure_ascii=False)
        )

    if mismatches and not allow_count_mismatch:
        raise ValueError(
            "Source count validation failed. This may be an unexpected workbook "
            "version. Use --allow-count-mismatch only after reviewing the differences. "
            + " | ".join(mismatches)
        )
    if mismatches:
        print("WARNING: Source count mismatches were explicitly allowed:")
        for mismatch in mismatches:
            print(f"- {mismatch}")

    return {
        "status": "overridden" if mismatches else "passed",
        "expected": EXPECTED_SOURCE_COUNTS,
        "observed": observed,
        "mismatches": mismatches,
    }


def verify_candidate_id_uniqueness(data: pd.DataFrame) -> dict[str, int]:
    """Verify uniqueness of all nonblank source candidate identifiers."""
    _require_columns(data, ("candidate_id",), "Review data")
    identifiers = [
        str(value).strip()
        for value in data["candidate_id"]
        if not is_blank(value)
    ]
    duplicate_ids = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "candidate_id must be unique; duplicate values found: "
            + ", ".join(duplicate_ids[:20])
        )
    return {
        "source_records": int(len(data)),
        "nonblank_candidate_ids": len(identifiers),
        "unique_nonblank_candidate_ids": len(set(identifiers)),
        "blank_candidate_ids": int(len(data) - len(identifiers)),
    }


def _exclusion_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    inclusion = normalize_category(row["include_in_final_set"])
    label = normalize_category(row["reviewer_gold_label"])
    if inclusion != "yes":
        reasons.append(f"include_in_final_set_{inclusion or 'blank'}")
    if label not in VALID_LABELS:
        reasons.append("invalid_reviewer_gold_label")
    if is_blank(row["candidate_text_original"]):
        reasons.append("blank_candidate_text_original")
    if is_blank(row["candidate_id"]):
        reasons.append("blank_candidate_id")
    return reasons


def select_eligible_records(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply every eligibility condition and retain exclusion reasons."""
    _require_columns(data, REQUIRED_SOURCE_COLUMNS, "Review data")
    reasons = data.apply(_exclusion_reasons, axis=1)
    eligible_mask = reasons.map(lambda values: not values)

    eligible = data.loc[eligible_mask].copy()
    excluded = data.loc[~eligible_mask].copy()
    excluded["exclusion_reasons"] = reasons.loc[~eligible_mask].map("; ".join)

    non_string_text_rows = eligible.loc[
        ~eligible["candidate_text_original"].map(lambda value: isinstance(value, str))
    ]
    if not non_string_text_rows.empty:
        ids = non_string_text_rows["review_record_id"].map(_comparison_value).tolist()
        raise ValueError(
            "Eligible candidate_text_original values must be strings; affected "
            f"review records: {ids[:20]}"
        )

    eligible["reviewer_gold_label"] = eligible["reviewer_gold_label"].map(
        normalize_category
    )

    reason_counts: Counter[str] = Counter()
    for value in excluded.get("exclusion_reasons", pd.Series(dtype=object)):
        reason_counts.update(reason.strip() for reason in str(value).split(";") if reason.strip())
    combination_counts = Counter(
        excluded.get("exclusion_reasons", pd.Series(dtype=object)).tolist()
    )

    summary = {
        "eligible_records": int(len(eligible)),
        "excluded_records": int(len(excluded)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "reason_combination_counts": {
            key: int(value) for key, value in sorted(combination_counts.items())
        },
    }
    return eligible, excluded, summary


def assess_eligibility_counts(data: pd.DataFrame) -> dict[str, Any]:
    """Compare eligible statement counts with the authorized Task 3 policy."""
    observed_by_class = _counts_by_class(data)
    mismatches: list[str] = []
    expected_total = EXPECTED_ELIGIBLE_COUNTS["eligible_records"]
    if len(data) != expected_total:
        mismatches.append(
            f"eligible_records: expected {expected_total}, observed {len(data)}"
        )
    for label, expected_count in EXPECTED_ELIGIBLE_COUNTS[
        "reviewer_gold_label"
    ].items():
        observed_count = observed_by_class[label]
        if observed_count != expected_count:
            mismatches.append(
                f"reviewer_gold_label.{label}: expected {expected_count}, "
                f"observed {observed_count}"
            )
    return {
        "status": "mismatch" if mismatches else "passed",
        "expected": EXPECTED_ELIGIBLE_COUNTS,
        "observed": {
            "eligible_records": int(len(data)),
            "reviewer_gold_label": observed_by_class,
        },
        "mismatches": mismatches,
        "policy_note": ELIGIBILITY_POLICY_NOTE,
    }


def validate_authorized_review_adjustments(data: pd.DataFrame) -> dict[str, Any]:
    """Bind the authorized source deviations to their exact reviewed records."""
    _require_columns(
        data,
        (
            "review_record_id",
            "candidate_id",
            "candidate_text_original",
            "reviewer_gold_label",
            "include_in_final_set",
        ),
        "Review data",
    )

    records: dict[str, pd.Series] = {}
    for review_record_id in ("REV_0283", "REV_0752", "REV_0768"):
        matches = data.loc[data["review_record_id"].eq(review_record_id)]
        if len(matches) != 1:
            raise ValueError(
                "Authorized review-adjustment validation requires exactly one "
                f"{review_record_id} row; observed {len(matches)}."
            )
        records[review_record_id] = matches.iloc[0]

    checks = {
        "REV_0283_label_regulative": normalize_category(
            records["REV_0283"]["reviewer_gold_label"]
        )
        == "regulative",
        "REV_0752_label_regulative": normalize_category(
            records["REV_0752"]["reviewer_gold_label"]
        )
        == "regulative",
        "REV_0752_matches_REV_0283_normalized_text": normalize_for_duplicate_check(
            records["REV_0752"]["candidate_text_original"]
        )
        == normalize_for_duplicate_check(
            records["REV_0283"]["candidate_text_original"]
        ),
        "REV_0768_included_yes": normalize_category(
            records["REV_0768"]["include_in_final_set"]
        )
        == "yes",
        "REV_0768_blank_candidate_id": is_blank(
            records["REV_0768"]["candidate_id"]
        ),
        "REV_0768_blank_candidate_text_original": is_blank(
            records["REV_0768"]["candidate_text_original"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Authorized review-adjustment validation failed: " + ", ".join(failed)
        )
    return {
        "status": "passed",
        "checks": checks,
        "source_review_adjustment_note": SOURCE_REVIEW_ADJUSTMENT_NOTE,
        "eligibility_policy_note": ELIGIBILITY_POLICY_NOTE,
    }


def _duplicate_group_id(normalized_text: str) -> str:
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16]
    return f"DUP_{digest.upper()}"


def assign_duplicate_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Assign stable text-hash groups and reject label conflicts."""
    _require_columns(
        data,
        ("candidate_text_original", "reviewer_gold_label", "candidate_id"),
        "Eligible data",
    )
    grouped = data.copy()
    grouped["_normalized_text"] = grouped["candidate_text_original"].map(
        normalize_for_duplicate_check
    )
    if grouped["_normalized_text"].eq("").any():
        raise ValueError("Eligible candidate text became blank after normalization.")
    grouped["duplicate_group_id"] = grouped["_normalized_text"].map(
        _duplicate_group_id
    )

    collision_check = grouped.groupby("duplicate_group_id")["_normalized_text"].nunique()
    if collision_check.gt(1).any():
        raise ValueError("A duplicate_group_id hash collision was detected.")

    grouped["_duplicate_group_size"] = grouped.groupby("duplicate_group_id")[
        "duplicate_group_id"
    ].transform("size")
    conflict_ids = grouped.groupby("duplicate_group_id")[
        "reviewer_gold_label"
    ].nunique()
    conflict_ids = conflict_ids[conflict_ids.gt(1)].index.tolist()
    if conflict_ids:
        details: list[str] = []
        for group_id in conflict_ids:
            rows = grouped.loc[grouped["duplicate_group_id"].eq(group_id)]
            members = ", ".join(
                f"{_comparison_value(row.review_record_id)} / "
                f"{_comparison_value(row.candidate_id)}={row.reviewer_gold_label}"
                for row in rows.itertuples()
            )
            text = rows.iloc[0]["candidate_text_original"]
            details.append(f"{group_id} [{members}] text={text!r}")
        raise DuplicateLabelConflictError(
            "Identical normalized texts have conflicting reviewer_gold_label "
            "values. Correct the alignment review before splitting: "
            + " | ".join(details)
        )
    return grouped


def duplicate_statistics(data: pd.DataFrame) -> dict[str, int]:
    """Summarize exact normalized-text groups."""
    sizes = data.groupby("duplicate_group_id").size()
    multi = sizes[sizes.gt(1)]
    return {
        "total_duplicate_groups_including_singletons": int(len(sizes)),
        "multi_record_duplicate_groups": int(len(multi)),
        "records_in_multi_record_duplicate_groups": int(multi.sum()),
        "surplus_duplicate_records": int((multi - 1).sum()),
        "largest_duplicate_group": int(multi.max()) if not multi.empty else 1,
        "conflicting_label_groups": 0,
    }


def build_duplicate_audit(
    data: pd.DataFrame,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build exact-group and report-only near-duplicate audit rows."""
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1].")

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for group_id, members in data.groupby("duplicate_group_id", sort=True):
        members = members.sort_values("candidate_id")
        group = {
            "group_id": group_id,
            "size": int(len(members)),
            "normalized_text": members.iloc[0]["_normalized_text"],
            "text": members.iloc[0]["candidate_text_original"],
            "review_record_ids": " | ".join(
                members["review_record_id"].map(_comparison_value)
            ),
            "candidate_ids": " | ".join(
                members["candidate_id"].map(_comparison_value)
            ),
            "labels": " | ".join(
                sorted(set(members["reviewer_gold_label"].map(str)))
            ),
        }
        groups.append(group)
        if group["size"] > 1:
            rows.append(
                {
                    "audit_type": "exact_normalized_duplicate",
                    "duplicate_group_id_a": group_id,
                    "duplicate_group_id_b": "",
                    "duplicate_group_size_a": group["size"],
                    "duplicate_group_size_b": "",
                    "review_record_ids_a": group["review_record_ids"],
                    "review_record_ids_b": "",
                    "candidate_ids_a": group["candidate_ids"],
                    "candidate_ids_b": "",
                    "reviewer_gold_labels_a": group["labels"],
                    "reviewer_gold_labels_b": "",
                    "similarity_ratio": 1.0,
                    "candidate_text_original_a": group["text"],
                    "candidate_text_original_b": "",
                    "cross_label": False,
                    "action": "kept_together_in_one_partition",
                }
            )

    near_pair_count = 0
    cross_label_near_pair_count = 0
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            left_text = left["normalized_text"]
            right_text = right["normalized_text"]
            if left_text == right_text:
                continue
            maximum_ratio = (
                2 * min(len(left_text), len(right_text))
                / (len(left_text) + len(right_text))
            )
            if maximum_ratio < near_duplicate_threshold:
                continue
            ratio = SequenceMatcher(
                None,
                left_text,
                right_text,
                autojunk=False,
            ).ratio()
            if ratio < near_duplicate_threshold:
                continue
            cross_label = left["labels"] != right["labels"]
            near_pair_count += 1
            cross_label_near_pair_count += int(cross_label)
            rows.append(
                {
                    "audit_type": "possible_near_duplicate",
                    "duplicate_group_id_a": left["group_id"],
                    "duplicate_group_id_b": right["group_id"],
                    "duplicate_group_size_a": left["size"],
                    "duplicate_group_size_b": right["size"],
                    "review_record_ids_a": left["review_record_ids"],
                    "review_record_ids_b": right["review_record_ids"],
                    "candidate_ids_a": left["candidate_ids"],
                    "candidate_ids_b": right["candidate_ids"],
                    "reviewer_gold_labels_a": left["labels"],
                    "reviewer_gold_labels_b": right["labels"],
                    "similarity_ratio": round(ratio, 6),
                    "candidate_text_original_a": left["text"],
                    "candidate_text_original_b": right["text"],
                    "cross_label": cross_label,
                    "action": "report_only_no_automatic_exclusion_or_grouping",
                }
            )

    if not rows:
        rows.append(
            {
                **{column: "" for column in DUPLICATE_AUDIT_COLUMNS},
                "audit_type": "none",
                "action": "no_exact_or_near_duplicates_detected",
            }
        )

    audit = pd.DataFrame(rows, columns=DUPLICATE_AUDIT_COLUMNS)
    return audit, {
        "near_duplicate_text_group_pairs": near_pair_count,
        "cross_label_near_duplicate_text_group_pairs": cross_label_near_pair_count,
    }


def _seeded_group_order(group_ids: Iterable[str], random_state: int) -> list[str]:
    return sorted(
        group_ids,
        key=lambda group_id: hashlib.sha256(
            f"{random_state}\0{group_id}".encode("utf-8")
        ).digest(),
    )


def _reachable_group_subsets(
    group_sizes: dict[str, int],
    random_state: int,
) -> dict[int, tuple[str, ...]]:
    """Return one seeded deterministic group subset per reachable row count."""
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id in _seeded_group_order(group_sizes, random_state):
        size = int(group_sizes[group_id])
        snapshot = list(reachable.items())
        for current_count, selected in snapshot:
            new_count = current_count + size
            if new_count not in reachable:
                reachable[new_count] = (*selected, group_id)
    return reachable


def _create_group_aware_split(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Choose the closest attainable stratified group allocation."""
    group_label_counts = data.groupby("duplicate_group_id")[
        "reviewer_gold_label"
    ].nunique()
    if group_label_counts.gt(1).any():
        raise DuplicateLabelConflictError(
            "Duplicate groups must contain exactly one reviewed label."
        )

    class_options: list[tuple[str, int, dict[int, tuple[str, ...]]]] = []
    for label in VALID_LABELS:
        label_data = data.loc[data["reviewer_gold_label"].eq(label)]
        group_sizes = {
            str(group_id): int(size)
            for group_id, size in label_data.groupby("duplicate_group_id").size().items()
        }
        if len(group_sizes) < 2:
            raise ValueError(
                f"Class {label!r} needs at least two distinct duplicate groups "
                "so it can appear in training and test sets."
            )
        total = int(len(label_data))
        reachable = _reachable_group_subsets(group_sizes, random_state)
        reachable = {
            count: selected
            for count, selected in reachable.items()
            if 0 < count < total
        }
        if not reachable:
            raise ValueError(
                f"No nonempty train/test group allocation is possible for {label!r}."
            )
        class_options.append((label, total, reachable))

    # For every attainable total, retain the allocation closest to per-class
    # targets. Insertion order supplies the seeded deterministic tie-break.
    states: dict[int, tuple[float, tuple[tuple[str, ...], ...]]] = {0: (0.0, ())}
    class_targets: dict[str, float] = {}
    for label, class_total, reachable in class_options:
        target = class_total * test_size
        class_targets[label] = target
        next_states: dict[int, tuple[float, tuple[tuple[str, ...], ...]]] = {}
        for previous_total, (previous_deviation, previous_selections) in states.items():
            for test_count, selected_groups in reachable.items():
                combined_total = previous_total + test_count
                candidate = (
                    previous_deviation + abs(test_count - target),
                    (*previous_selections, selected_groups),
                )
                existing = next_states.get(combined_total)
                if existing is None or candidate[0] < existing[0]:
                    next_states[combined_total] = candidate
        states = next_states

    target_total = math.ceil(len(data) * test_size)
    achieved_total, (class_deviation, selections) = min(
        states.items(),
        key=lambda item: (abs(item[0] - target_total), item[1][0]),
    )
    test_group_ids = {
        group_id for selected_groups in selections for group_id in selected_groups
    }
    test_data = data.loc[data["duplicate_group_id"].isin(test_group_ids)].copy()
    training_data = data.loc[~data["duplicate_group_id"].isin(test_group_ids)].copy()

    achieved_by_class = {
        label: int(test_data["reviewer_gold_label"].eq(label).sum())
        for label in VALID_LABELS
    }
    metadata = {
        "method": "deterministic_weighted_duplicate_group_allocation",
        "target_test_records": target_total,
        "achieved_test_records": achieved_total,
        "class_target_test_records": class_targets,
        "class_achieved_test_records": achieved_by_class,
        "summed_absolute_class_count_deviation": class_deviation,
    }
    return training_data, test_data, metadata


def create_stratified_split(
    data: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create a reproducible stratified split, respecting duplicate groups."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")
    _require_columns(
        data,
        ("candidate_id", "reviewer_gold_label", "duplicate_group_id"),
        "Eligible grouped data",
    )

    group_sizes = data.groupby("duplicate_group_id").size()
    if group_sizes.le(1).all():
        try:
            training_data, test_data = train_test_split(
                data,
                test_size=test_size,
                stratify=data["reviewer_gold_label"],
                random_state=random_state,
            )
        except ValueError as error:
            raise ValueError(f"Stratified train/test split failed: {error}") from error
        split_metadata = {
            "method": "sklearn_train_test_split",
            "target_test_records": math.ceil(len(data) * test_size),
            "achieved_test_records": int(len(test_data)),
            "class_target_test_records": {
                label: int(data["reviewer_gold_label"].eq(label).sum()) * test_size
                for label in VALID_LABELS
            },
            "class_achieved_test_records": {
                label: int(test_data["reviewer_gold_label"].eq(label).sum())
                for label in VALID_LABELS
            },
        }
    else:
        training_data, test_data, split_metadata = _create_group_aware_split(
            data,
            test_size,
            random_state,
        )

    training_data = training_data.sort_index().copy()
    test_data = test_data.sort_index().copy()
    training_data["split"] = "training"
    test_data["split"] = "test"
    training_data["random_seed"] = int(random_state)
    test_data["random_seed"] = int(random_state)
    return training_data, test_data, split_metadata


def validate_split(
    eligible: pd.DataFrame,
    training: pd.DataFrame,
    test: pd.DataFrame,
    test_size: float,
) -> dict[str, Any]:
    """Validate membership, labels, text, groups, and class coverage."""
    training_ids = set(training["candidate_id"].map(_comparison_value))
    test_ids = set(test["candidate_id"].map(_comparison_value))
    eligible_ids = set(eligible["candidate_id"].map(_comparison_value))
    combined_ids = training_ids | test_ids
    duplicate_groups_crossing = sorted(
        set(training["duplicate_group_id"]) & set(test["duplicate_group_id"])
    )
    class_presence = {
        label: {
            "training": int(training["reviewer_gold_label"].eq(label).sum()),
            "test": int(test["reviewer_gold_label"].eq(label).sum()),
        }
        for label in VALID_LABELS
    }

    combined = pd.concat([training, test], axis=0).sort_index()
    original_text = {
        _comparison_value(row.candidate_id): row.candidate_text_original
        for row in eligible.itertuples()
    }
    exported_text = {
        _comparison_value(row.candidate_id): row.candidate_text_original
        for row in combined.itertuples()
    }

    checks = {
        "training_plus_test_equals_eligible": len(training) + len(test) == len(eligible),
        "candidate_ids_do_not_overlap": not bool(training_ids & test_ids),
        "all_eligible_candidate_ids_assigned_once": (
            combined_ids == eligible_ids and len(combined_ids) == len(combined)
        ),
        "duplicate_groups_do_not_cross_partitions": not duplicate_groups_crossing,
        "no_blank_candidate_id": not combined["candidate_id"].map(is_blank).any(),
        "no_blank_candidate_text": not combined["candidate_text_original"].map(is_blank).any(),
        "no_blank_reviewed_label": not combined["reviewer_gold_label"].map(is_blank).any(),
        "only_approved_labels": set(combined["reviewer_gold_label"]) == set(VALID_LABELS),
        "every_class_appears_in_both_sets": all(
            values["training"] > 0 and values["test"] > 0
            for values in class_presence.values()
        ),
        "candidate_text_original_unchanged": original_text == exported_text,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Split validation failed: " + ", ".join(failed))

    overall_proportions = eligible["reviewer_gold_label"].value_counts(normalize=True)
    test_proportions = test["reviewer_gold_label"].value_counts(normalize=True)
    proportion_differences = {
        label: abs(
            float(test_proportions.get(label, 0.0))
            - float(overall_proportions.get(label, 0.0))
        )
        for label in VALID_LABELS
    }
    return {
        "checks": checks,
        "class_presence": class_presence,
        "requested_test_proportion": test_size,
        "observed_test_proportion": len(test) / len(eligible),
        "test_vs_overall_absolute_class_proportion_difference": proportion_differences,
        "duplicate_groups_crossing_partitions": duplicate_groups_crossing,
        "design_constraints": {
            "split_grouping_field": "duplicate_group_id",
            "alignment_group_id_used_for_grouping": False,
            "modeling_components_fitted": [],
        },
    }


def _counts_by_class(data: pd.DataFrame) -> dict[str, int]:
    return {
        label: int(data["reviewer_gold_label"].eq(label).sum())
        for label in VALID_LABELS
    }


def build_split_summary(
    eligible: pd.DataFrame,
    training: pd.DataFrame,
    test: pd.DataFrame,
    excluded: pd.DataFrame,
    duplicate_summary: dict[str, int],
    near_summary: dict[str, int],
) -> pd.DataFrame:
    """Build a compact human-readable split summary sheet."""
    rows: list[dict[str, Any]] = []
    for partition, frame in (
        ("Eligible", eligible),
        ("Training", training),
        ("Test", test),
    ):
        total = len(frame)
        rows.append(
            {
                "section": "partition_counts",
                "partition": partition,
                "metric": "total",
                "value": total,
                "proportion": 1.0,
            }
        )
        for label in VALID_LABELS:
            count = int(frame["reviewer_gold_label"].eq(label).sum())
            rows.append(
                {
                    "section": "partition_counts",
                    "partition": partition,
                    "metric": label,
                    "value": count,
                    "proportion": count / total if total else 0.0,
                }
            )
    rows.append(
        {
            "section": "exclusions",
            "partition": "Excluded",
            "metric": "total",
            "value": len(excluded),
            "proportion": "",
        }
    )
    for metric, value in {**duplicate_summary, **near_summary}.items():
        rows.append(
            {
                "section": "duplicate_audit",
                "partition": "Eligible",
                "metric": metric.replace("_", " "),
                "value": value,
                "proportion": "",
            }
        )
    return pd.DataFrame(rows)


def build_readme_sheet(
    *,
    review_path: Path,
    summary_path: Path,
    run_id: str,
    created_at: str,
    test_size: float,
    random_state: int,
    split_method: str,
    near_duplicate_threshold: float,
) -> pd.DataFrame:
    """Build the workbook README sheet."""
    rows = [
        ("Task", "Task 3: institutional-statement classification dataset split"),
        ("Script", SCRIPT_NAME),
        ("Run ID", run_id),
        ("Split date", created_at[:10]),
        ("Created at", created_at),
        ("Review source", project_relative(review_path)),
        ("Summary source", project_relative(summary_path)),
        ("Input text", "candidate_text_original; preserved exactly as reviewed"),
        ("Target", "reviewer_gold_label"),
        ("Approved labels", ", ".join(VALID_LABELS)),
        ("Eligibility", "include_in_final_set=yes; approved label; nonblank text and candidate_id"),
        ("Expected eligible statements", EXPECTED_ELIGIBLE_COUNTS["eligible_records"]),
        ("Reviewed-source adjustment", SOURCE_REVIEW_ADJUSTMENT_NOTE),
        ("Eligibility policy", ELIGIBILITY_POLICY_NOTE),
        ("Test proportion", test_size),
        ("Random seed", random_state),
        ("Split method", split_method),
        ("Duplicate grouping", "NFKC Unicode normalization, lowercase, whitespace collapse, trim"),
        ("Near-duplicate threshold", near_duplicate_threshold),
        ("Near duplicates", "Report only; they are not excluded or grouped automatically"),
        ("Audit metadata", "Available for auditing only; do not use as model features"),
        ("Modeling", "No vectorizer or classifier was fitted by this script"),
        ("Training sheet", "Raw training statements plus required audit metadata"),
        ("Test sheet", "Raw held-out test statements plus required audit metadata"),
        ("Excluded sheet", "Source rows failing one or more eligibility conditions"),
        ("Duplicate Check", "Exact normalized groups and report-only near-duplicate pairs"),
    ]
    return pd.DataFrame(rows, columns=("item", "details"))


def next_split_run_id(
    output_dir: Path,
    database_path: Path | None = None,
) -> str:
    """Increment the highest filesystem or database SPLIT_###### identifier."""
    highest = 0
    manifest_path = output_dir / MANIFEST_FILENAME
    data_paths = [
        output_dir / filename
        for filename in (TRAINING_FILENAME, TEST_FILENAME, WORKBOOK_FILENAME)
    ]
    existing_data_paths = [path for path in data_paths if path.exists()]
    if existing_data_paths and not manifest_path.exists():
        raise ValueError(
            "Output directory contains Task 3 data files without a manifest; "
            "refusing to reuse a run ID or overwrite ambiguous prior state: "
            + ", ".join(path.name for path in existing_data_paths)
        )
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Existing Task 3 manifest is unreadable or invalid: {manifest_path}"
            ) from error
        prior = str(payload.get("run_id", ""))
        match = re.fullmatch(r"SPLIT_(\d{6})", prior)
        if not match:
            raise ValueError(
                f"Existing Task 3 manifest has invalid run_id {prior!r}."
            )
        missing_outputs = [path.name for path in data_paths if not path.exists()]
        if missing_outputs:
            raise ValueError(
                "Existing Task 3 package is incomplete; refusing to overwrite it. "
                "Missing: " + ", ".join(missing_outputs)
            )
        highest = int(match.group(1))

    if database_path is not None and database_path.exists():
        with sqlite3.connect(database_path) as connection:
            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='split_events'"
            ).fetchone()
            if table_exists:
                database_ids = [
                    str(row[0])
                    for row in connection.execute("SELECT split_id FROM split_events")
                ]
                invalid_ids = [
                    split_id
                    for split_id in database_ids
                    if not re.fullmatch(r"SPLIT_\d{6}", split_id)
                ]
                if invalid_ids:
                    raise ValueError(
                        "The split registry contains invalid split_id values: "
                        + ", ".join(invalid_ids[:10])
                    )
                highest = max(
                    [highest]
                    + [int(split_id.removeprefix("SPLIT_")) for split_id in database_ids]
                )
    return f"SPLIT_{highest + 1:06d}"


def resolve_matching_event(
    database_path: Path,
    *,
    review_hash_sha256: str,
    summary_hash_sha256: str,
) -> dict[str, str]:
    """Resolve the single registered matching event for the exact source pair."""
    if not database_path.exists():
        raise FileNotFoundError(f"Corpus inventory database not found: {database_path}")
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='matching_events'"
        ).fetchone()
        if not table_exists:
            raise ValueError(
                "The corpus inventory has no matching_events table. Register the "
                "alignment process before creating a split."
            )
        rows = connection.execute(
            """
            SELECT matching_id, segmentation_id, document_id
            FROM matching_events
            WHERE alignment_review_hash_sha256 = ?
              AND alignment_summary_hash_sha256 = ?
            """,
            (review_hash_sha256, summary_hash_sha256),
        ).fetchall()
    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one matching event for the review/summary hashes; "
            f"observed {len(rows)}. Register the exact alignment artifacts first."
        )
    return {key: str(rows[0][key]) for key in rows[0].keys()}


def _safe_table_name(sheet_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", sheet_name.replace(" ", "")) + "Table"


def _style_workbook(workbook: Any) -> None:
    """Apply consistent, audit-friendly workbook formatting."""
    dark_blue = "1F4E78"
    light_blue = "D9EAF7"
    white = "FFFFFF"
    light_border = Side(style="thin", color="D9E2F3")
    header_fill = PatternFill("solid", fgColor=dark_blue)
    header_font = Font(name="Aptos", bold=True, color=white)
    body_font = Font(name="Aptos", size=10)

    width_overrides = {
        "review_record_id": 18,
        "candidate_id": 34,
        "alignment_group_id": 24,
        "article": 16,
        "scope": 22,
        "source_parent_id": 32,
        "match_type": 20,
        "candidate_text_original": 90,
        "reviewer_gold_label": 22,
        "split": 12,
        "duplicate_group_id": 25,
        "random_seed": 13,
        "exclusion_reasons": 38,
        "candidate_text_original_a": 65,
        "candidate_text_original_b": 65,
        "details": 80,
        "item": 28,
        "section": 18,
        "partition": 11,
        "metric": 25,
        "value": 10,
        "proportion": 14,
    }

    for worksheet in workbook.worksheets:
        worksheet.sheet_view.showGridLines = False
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.sheet_properties.tabColor = dark_blue

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            cell.border = Border(bottom=Side(style="medium", color=dark_blue))
        worksheet.row_dimensions[1].height = 30

        headers = {cell.column: str(cell.value) for cell in worksheet[1]}
        for column_index, header in headers.items():
            letter = get_column_letter(column_index)
            width = width_overrides.get(header)
            if width is None:
                sample_values = [header]
                for row_index in range(2, min(worksheet.max_row, 80) + 1):
                    value = worksheet.cell(row=row_index, column=column_index).value
                    sample_values.append("" if value is None else str(value))
                width = min(max(max(map(len, sample_values)) + 2, 12), 36)
            worksheet.column_dimensions[letter].width = width

        wrap_headers = {
            "candidate_text_original",
            "candidate_text_original_a",
            "candidate_text_original_b",
            "details",
            "section",
            "metric",
            "proportion",
            "action",
            "exclusion_reasons",
        }
        wrap_columns = {
            column_index
            for column_index, header in headers.items()
            if header in wrap_headers
        }
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = body_font
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=cell.column in wrap_columns,
                )
                cell.border = Border(bottom=light_border)
            if wrap_columns:
                estimated_lines = 1
                for column_index in wrap_columns:
                    cell = worksheet.cell(row=row[0].row, column=column_index)
                    text = "" if cell.value is None else str(cell.value)
                    letter = get_column_letter(column_index)
                    width = worksheet.column_dimensions[letter].width or 12
                    line_capacity = max(int(width * 0.92), 8)
                    wrapped_lines = sum(
                        max(1, math.ceil(len(fragment) / line_capacity))
                        for fragment in (text.splitlines() or [""])
                    )
                    estimated_lines = max(estimated_lines, wrapped_lines)
                worksheet.row_dimensions[row[0].row].height = min(
                    409, max(20, 15 * estimated_lines + 4)
                )

        if "similarity_ratio" in headers.values():
            index = next(key for key, value in headers.items() if value == "similarity_ratio")
            for row_index in range(2, worksheet.max_row + 1):
                worksheet.cell(row=row_index, column=index).number_format = "0.000000"
        if "proportion" in headers.values():
            index = next(
                key
                for key, value in headers.items()
                if value == "proportion"
            )
            for row_index in range(2, worksheet.max_row + 1):
                worksheet.cell(row=row_index, column=index).number_format = "0.0%"

        if worksheet.max_row >= 2 and worksheet.max_column >= 1:
            table = Table(
                displayName=_safe_table_name(worksheet.title),
                ref=f"A1:{get_column_letter(worksheet.max_column)}{worksheet.max_row}",
            )
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            worksheet.add_table(table)

        # A subtle first-column cue improves scanning without boxing every cell.
        for row_index in range(2, worksheet.max_row + 1):
            worksheet.cell(row=row_index, column=1).fill = PatternFill(
                "solid", fgColor=light_blue
            )

    workbook.properties.creator = "IG protocol pipeline"
    workbook.properties.title = "IG statement train/test split"
    workbook.properties.subject = "Task 3 raw statement classification datasets"


def _canonical_frame_rows(data: pd.DataFrame, columns: Iterable[str]) -> list[tuple[str, ...]]:
    return [
        tuple(_comparison_value(value) for value in row)
        for row in data.loc[:, list(columns)].itertuples(index=False, name=None)
    ]


def verify_exported_outputs(
    *,
    training_csv_path: Path,
    test_csv_path: Path,
    workbook_path: Path,
    training: pd.DataFrame,
    test: pd.DataFrame,
    excluded: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    split_summary: pd.DataFrame,
    readme: pd.DataFrame,
) -> dict[str, Any]:
    """Reread CSV/XLSX outputs and verify exact rows, text, and sheet names."""
    exported_training = pd.read_csv(
        training_csv_path, dtype=object, keep_default_na=False, encoding="utf-8"
    )
    exported_test = pd.read_csv(
        test_csv_path, dtype=object, keep_default_na=False, encoding="utf-8"
    )
    expected_training = training.loc[:, OUTPUT_COLUMNS]
    expected_test = test.loc[:, OUTPUT_COLUMNS]
    if _canonical_frame_rows(exported_training, OUTPUT_COLUMNS) != _canonical_frame_rows(
        expected_training, OUTPUT_COLUMNS
    ):
        raise ValueError("Training CSV differs from the validated in-memory data.")
    if _canonical_frame_rows(exported_test, OUTPUT_COLUMNS) != _canonical_frame_rows(
        expected_test, OUTPUT_COLUMNS
    ):
        raise ValueError("Test CSV differs from the validated in-memory data.")

    expected_sheets = [
        "Training",
        "Test",
        "Excluded",
        "Duplicate Check",
        "Split Summary",
        "README",
    ]
    with pd.ExcelFile(workbook_path, engine="openpyxl") as excel_file:
        observed_sheets = excel_file.sheet_names
    if observed_sheets != expected_sheets:
        raise ValueError(
            f"Workbook sheets differ: expected {expected_sheets}, "
            f"observed {observed_sheets}."
        )

    expected_frames = {
        "Training": expected_training,
        "Test": expected_test,
        "Excluded": excluded,
        "Duplicate Check": duplicate_audit,
        "Split Summary": split_summary,
        "README": readme,
    }
    for sheet_name, expected_frame in expected_frames.items():
        observed_frame = pd.read_excel(
            workbook_path,
            sheet_name=sheet_name,
            dtype=object,
            keep_default_na=False,
            engine="openpyxl",
        )
        if list(observed_frame.columns) != list(expected_frame.columns):
            raise ValueError(f"{sheet_name} worksheet columns differ after export.")
        if _canonical_frame_rows(observed_frame, expected_frame.columns) != _canonical_frame_rows(
            expected_frame, expected_frame.columns
        ):
            raise ValueError(f"{sheet_name} worksheet rows differ after export.")

    return {
        "training_csv_rows": int(len(exported_training)),
        "test_csv_rows": int(len(exported_test)),
        "workbook_sheets": expected_sheets,
        "candidate_text_round_trip_exact": True,
    }


def validate_manifest_payload(
    payload: dict[str, Any],
    *,
    training: pd.DataFrame,
    test: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    """Validate required manifest fields, membership, counts, and file hashes."""
    required_keys = {
        "task_id",
        "script_id",
        "script_name",
        "run_id",
        "matching_id",
        "split_date",
        "created_at",
        "source",
        "eligibility_rules",
        "eligibility_policy",
        "exclusions",
        "split",
        "counts",
        "duplicate_audit",
        "training_review_record_ids",
        "training_candidate_ids",
        "test_review_record_ids",
        "test_candidate_ids",
        "quality_checks",
        "modeling_steps_performed",
        "execution_status",
        "output_files",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise ValueError("Manifest is missing required keys: " + ", ".join(missing))
    if not re.fullmatch(r"SPLIT_\d{6}", str(payload["run_id"])):
        raise ValueError(f"Manifest run_id is invalid: {payload['run_id']!r}")
    if payload["execution_status"] not in {"Completed", "Completed with warnings"}:
        raise ValueError("Manifest execution_status is not a completed status.")

    eligibility_validation = payload["source"].get("eligibility_count_validation")
    if not isinstance(eligibility_validation, dict) or eligibility_validation.get(
        "status"
    ) not in {"passed", "overridden"}:
        raise ValueError(
            "Manifest eligibility_count_validation is missing or not completed."
        )
    review_adjustment_validation = payload["source"].get(
        "review_adjustment_validation"
    )
    if not isinstance(
        review_adjustment_validation, dict
    ) or review_adjustment_validation.get("status") != "passed":
        raise ValueError(
            "Manifest review_adjustment_validation is missing or not passed."
        )
    eligibility_policy = payload["eligibility_policy"]
    if not isinstance(eligibility_policy, dict) or not eligibility_policy.get(
        "expected_eligible_counts"
    ):
        raise ValueError("Manifest eligibility_policy metadata is missing.")

    expected_counts = {
        "eligible_total": int(len(training) + len(test)),
        "training_total": int(len(training)),
        "test_total": int(len(test)),
    }
    for key, expected in expected_counts.items():
        if int(payload["counts"].get(key, -1)) != expected:
            raise ValueError(
                f"Manifest {key} mismatch: expected {expected}, "
                f"observed {payload['counts'].get(key)!r}."
            )
    if int(payload["counts"]["eligible_total"]) != int(
        payload["counts"]["training_total"]
    ) + int(payload["counts"]["test_total"]):
        raise ValueError(
            "Manifest eligible_total must equal training_total plus test_total."
        )

    membership_checks = {
        "training_review_record_ids": training["review_record_id"].map(
            _comparison_value
        ).tolist(),
        "training_candidate_ids": training["candidate_id"].map(
            _comparison_value
        ).tolist(),
        "test_review_record_ids": test["review_record_id"].map(
            _comparison_value
        ).tolist(),
        "test_candidate_ids": test["candidate_id"].map(_comparison_value).tolist(),
    }
    for key, expected in membership_checks.items():
        if payload[key] != expected:
            raise ValueError(f"Manifest {key} differs from exported membership.")

    for filename, path in output_paths.items():
        file_entry = payload["output_files"].get(filename)
        if not isinstance(file_entry, dict):
            raise ValueError(f"Manifest output_files is missing {filename}.")
        observed_hash = sha256_file(path)
        if file_entry.get("sha256") != observed_hash:
            raise ValueError(
                f"Manifest hash mismatch for {filename}: expected {observed_hash}, "
                f"observed {file_entry.get('sha256')!r}."
            )


def export_outputs(
    *,
    output_dir: Path,
    training: pd.DataFrame,
    test: pd.DataFrame,
    excluded: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    split_summary: pd.DataFrame,
    readme: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Stage, verify, hash, and publish all required outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_columns = [
        column
        for column in (
            "_source_excel_row",
            *REQUIRED_SOURCE_COLUMNS,
            "proposed_gold_label",
            "review_status",
            "exclusion_reasons",
        )
        if column in excluded.columns
    ]
    excluded_export = excluded.loc[:, excluded_columns].copy()

    with tempfile.TemporaryDirectory(prefix=".split_build_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        temp_paths = {
            TRAINING_FILENAME: temp_dir / TRAINING_FILENAME,
            TEST_FILENAME: temp_dir / TEST_FILENAME,
            WORKBOOK_FILENAME: temp_dir / WORKBOOK_FILENAME,
            MANIFEST_FILENAME: temp_dir / MANIFEST_FILENAME,
        }
        training.loc[:, OUTPUT_COLUMNS].to_csv(
            temp_paths[TRAINING_FILENAME],
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        test.loc[:, OUTPUT_COLUMNS].to_csv(
            temp_paths[TEST_FILENAME],
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        with pd.ExcelWriter(
            temp_paths[WORKBOOK_FILENAME],
            engine="openpyxl",
        ) as writer:
            training.loc[:, OUTPUT_COLUMNS].to_excel(
                writer, sheet_name="Training", index=False
            )
            test.loc[:, OUTPUT_COLUMNS].to_excel(
                writer, sheet_name="Test", index=False
            )
            excluded_export.to_excel(writer, sheet_name="Excluded", index=False)
            duplicate_audit.to_excel(writer, sheet_name="Duplicate Check", index=False)
            split_summary.to_excel(writer, sheet_name="Split Summary", index=False)
            readme.to_excel(writer, sheet_name="README", index=False)
            _style_workbook(writer.book)

        verification = verify_exported_outputs(
            training_csv_path=temp_paths[TRAINING_FILENAME],
            test_csv_path=temp_paths[TEST_FILENAME],
            workbook_path=temp_paths[WORKBOOK_FILENAME],
            training=training,
            test=test,
            excluded=excluded_export,
            duplicate_audit=duplicate_audit,
            split_summary=split_summary,
            readme=readme,
        )

        output_files: dict[str, Any] = {}
        for filename in (TRAINING_FILENAME, TEST_FILENAME, WORKBOOK_FILENAME):
            path = temp_paths[filename]
            output_files[filename] = {
                "path": project_relative(output_dir / filename),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        output_files[TRAINING_FILENAME]["records"] = int(len(training))
        output_files[TEST_FILENAME]["records"] = int(len(test))
        output_files[WORKBOOK_FILENAME]["worksheets"] = verification[
            "workbook_sheets"
        ]

        manifest = {**manifest, "output_files": output_files}
        hashed_output_paths = {
            filename: temp_paths[filename]
            for filename in (TRAINING_FILENAME, TEST_FILENAME, WORKBOOK_FILENAME)
        }
        validate_manifest_payload(
            manifest,
            training=training,
            test=test,
            output_paths=hashed_output_paths,
        )
        temp_paths[MANIFEST_FILENAME].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        parsed_manifest = json.loads(
            temp_paths[MANIFEST_FILENAME].read_text(encoding="utf-8")
        )
        validate_manifest_payload(
            parsed_manifest,
            training=training,
            test=test,
            output_paths=hashed_output_paths,
        )

        final_paths: dict[str, Path] = {}
        publish_filenames = (
            TRAINING_FILENAME,
            TEST_FILENAME,
            WORKBOOK_FILENAME,
            MANIFEST_FILENAME,
        )
        backup_paths: dict[str, Path] = {}
        published: list[str] = []
        try:
            for filename in publish_filenames:
                final_path = output_dir / filename
                if final_path.exists():
                    backup_path = temp_dir / f"prior_{filename}"
                    shutil.copy2(final_path, backup_path)
                    backup_paths[filename] = backup_path
                temp_paths[filename].replace(final_path)
                final_paths[filename] = final_path
                published.append(filename)
        except Exception:
            for filename in reversed(published):
                final_path = output_dir / filename
                backup_path = backup_paths.get(filename)
                if backup_path is not None and backup_path.exists():
                    backup_path.replace(final_path)
                else:
                    final_path.unlink(missing_ok=True)
            raise

    return final_paths, verification


def _json_text(value: Any) -> str:
    """Serialize registry JSON deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def archive_published_run(
    *,
    output_dir: Path,
    run_id: str,
    published_paths: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """Copy a completed package to an immutable run-specific history folder."""
    archive_dir = output_dir / HISTORY_DIRNAME / run_id
    if archive_dir.exists():
        raise ValueError(
            f"Historical split package already exists and will not be overwritten: "
            f"{archive_dir}"
        )
    archive_dir.mkdir(parents=True)
    archive_paths: dict[str, Path] = {}
    try:
        for filename, source_path in published_paths.items():
            archive_path = archive_dir / filename
            shutil.copy2(source_path, archive_path)
            archive_paths[filename] = archive_path
        for filename in (TRAINING_FILENAME, TEST_FILENAME, WORKBOOK_FILENAME):
            expected_hash = str(manifest["output_files"][filename]["sha256"])
            observed_hash = sha256_file(archive_paths[filename])
            if observed_hash != expected_hash:
                raise ValueError(
                    f"Historical archive hash mismatch for {filename}: expected "
                    f"{expected_hash}, observed {observed_hash}."
                )
        if sha256_file(archive_paths[MANIFEST_FILENAME]) != sha256_file(
            published_paths[MANIFEST_FILENAME]
        ):
            raise ValueError("Historical manifest copy differs from the published file.")
    except Exception:
        for path in archive_paths.values():
            path.unlink(missing_ok=True)
        archive_dir.rmdir()
        raise
    return archive_paths


def register_split_run(
    *,
    database_path: Path,
    matching_event: dict[str, str],
    manifest: dict[str, Any],
    training: pd.DataFrame,
    test: pd.DataFrame,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    """Register one immutable split event and its lightweight membership rows."""
    if not database_path.exists():
        raise FileNotFoundError(f"Corpus inventory database not found: {database_path}")

    split_id = str(manifest["run_id"])
    created_at = str(manifest["created_at"])
    split_date = str(manifest["split_date"])
    matching_id = str(matching_event["matching_id"])
    segmentation_id = str(matching_event["segmentation_id"])
    document_id = str(matching_event["document_id"])
    source = manifest["source"]
    split_details = manifest["split"]
    counts = manifest["counts"]
    output_files = manifest["output_files"]

    required_output_paths = {
        TRAINING_FILENAME,
        TEST_FILENAME,
        WORKBOOK_FILENAME,
        MANIFEST_FILENAME,
    }
    missing_paths = sorted(required_output_paths - set(output_paths))
    if missing_paths:
        raise ValueError(
            "Split registration is missing published output paths: "
            + ", ".join(missing_paths)
        )

    combined = pd.concat(
        [
            training.assign(_registry_partition="train"),
            test.assign(_registry_partition="test"),
        ],
        ignore_index=True,
    )
    if len(combined) != int(counts["eligible_total"]):
        raise ValueError("Registry assignment count differs from manifest eligible_total.")
    if combined["candidate_id"].map(_comparison_value).duplicated().any():
        raise ValueError("Registry assignments contain duplicate candidate_id values.")
    if combined["review_record_id"].map(_comparison_value).duplicated().any():
        raise ValueError("Registry assignments contain duplicate review_record_id values.")
    duplicate_group_sizes = combined.groupby("duplicate_group_id").size().to_dict()

    training_entry = output_files[TRAINING_FILENAME]
    test_entry = output_files[TEST_FILENAME]
    workbook_entry = output_files[WORKBOOK_FILENAME]
    event_values = {
        "split_id": split_id,
        "matching_id": matching_id,
        "segmentation_id": segmentation_id,
        "document_id": document_id,
        "split_date": split_date,
        "split_tool": SCRIPT_NAME,
        "split_tool_version": str(manifest["script_version"]),
        "source_review_path": str(source["review_workbook"]["path"]),
        "source_review_hash_sha256": str(source["review_workbook"]["sha256"]),
        "source_summary_path": str(source["summary_workbook"]["path"]),
        "source_summary_hash_sha256": str(source["summary_workbook"]["sha256"]),
        "eligibility_rules_json": _json_text(manifest["eligibility_rules"]),
        "valid_labels_json": _json_text(list(VALID_LABELS)),
        "eligible_records_count": int(counts["eligible_total"]),
        "excluded_records_count": int(manifest["exclusions"]["excluded_records"]),
        "training_records_count": int(counts["training_total"]),
        "test_records_count": int(counts["test_total"]),
        "class_counts_json": _json_text(
            {
                "eligible": counts["eligible_by_class"],
                "training": counts["training_by_class"],
                "test": counts["test_by_class"],
            }
        ),
        "requested_test_proportion": float(split_details["test_proportion_requested"]),
        "observed_test_proportion": float(
            manifest["quality_checks"]["observed_test_proportion"]
        ),
        "random_seed": int(split_details["random_seed"]),
        "stratification_field": str(split_details["stratification_field"]),
        "grouping_field": str(split_details["grouping_field"]),
        "split_method": str(split_details["method"]),
        "duplicate_group_normalization": str(
            manifest["duplicate_audit"]["normalization"]
        ),
        "near_duplicate_threshold": float(
            manifest["duplicate_audit"]["near_duplicate_threshold"]
        ),
        "duplicate_audit_json": _json_text(manifest["duplicate_audit"]),
        "quality_checks_json": _json_text(manifest["quality_checks"]),
        "training_csv_path": project_relative(output_paths[TRAINING_FILENAME]),
        "training_csv_hash_sha256": str(training_entry["sha256"]),
        "test_csv_path": project_relative(output_paths[TEST_FILENAME]),
        "test_csv_hash_sha256": str(test_entry["sha256"]),
        "split_workbook_path": project_relative(output_paths[WORKBOOK_FILENAME]),
        "split_workbook_hash_sha256": str(workbook_entry["sha256"]),
        "manifest_path": project_relative(output_paths[MANIFEST_FILENAME]),
        "manifest_hash_sha256": sha256_file(output_paths[MANIFEST_FILENAME]),
        "split_status": str(manifest["execution_status"]),
        "created_at": created_at,
        "notes": (
            "Append-only dataset snapshot with immutable run-specific files. Full "
            "statement text and generated datasets remain outside SQLite; "
            "split_assignments stores identifiers, labels, partitions, and "
            "duplicate-group membership only."
        ),
    }

    assignment_values = [
        (
            split_id,
            segmentation_id,
            _comparison_value(row["review_record_id"]),
            _comparison_value(row["candidate_id"]),
            str(row["_registry_partition"]),
            normalize_category(row["reviewer_gold_label"]),
            str(row["duplicate_group_id"]),
            int(duplicate_group_sizes[row["duplicate_group_id"]]),
            created_at,
        )
        for row in combined.to_dict(orient="records")
    ]

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SPLIT_REGISTRY_SCHEMA)
        connection.row_factory = sqlite3.Row
        registered_match = connection.execute(
            """
            SELECT matching_id, segmentation_id, document_id,
                   alignment_review_hash_sha256, alignment_summary_hash_sha256
            FROM matching_events
            WHERE matching_id = ?
            """,
            (matching_id,),
        ).fetchone()
        if registered_match is None:
            raise ValueError(f"Matching event is not registered: {matching_id}")
        lineage_checks = {
            "segmentation_id": segmentation_id,
            "document_id": document_id,
            "alignment_review_hash_sha256": event_values[
                "source_review_hash_sha256"
            ],
            "alignment_summary_hash_sha256": event_values[
                "source_summary_hash_sha256"
            ],
        }
        mismatches = [
            f"{field}: matching_events={registered_match[field]!r}, split={expected!r}"
            for field, expected in lineage_checks.items()
            if str(registered_match[field]) != str(expected)
        ]
        if mismatches:
            raise ValueError(
                "Split-to-matching lineage validation failed: " + " | ".join(mismatches)
            )
        if connection.execute(
            "SELECT 1 FROM split_events WHERE split_id = ?", (split_id,)
        ).fetchone():
            raise ValueError(
                f"Split ID {split_id} is already registered; split history is append-only."
            )

        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO split_events (
                split_id, matching_id, segmentation_id, document_id, split_date,
                split_tool, split_tool_version, source_review_path,
                source_review_hash_sha256, source_summary_path,
                source_summary_hash_sha256, eligibility_rules_json,
                valid_labels_json, eligible_records_count, excluded_records_count,
                training_records_count, test_records_count, class_counts_json,
                requested_test_proportion, observed_test_proportion, random_seed,
                stratification_field, grouping_field, split_method,
                duplicate_group_normalization, near_duplicate_threshold,
                duplicate_audit_json, quality_checks_json, training_csv_path,
                training_csv_hash_sha256, test_csv_path, test_csv_hash_sha256,
                split_workbook_path, split_workbook_hash_sha256, manifest_path,
                manifest_hash_sha256, split_status, created_at, notes
            ) VALUES (
                :split_id, :matching_id, :segmentation_id, :document_id, :split_date,
                :split_tool, :split_tool_version, :source_review_path,
                :source_review_hash_sha256, :source_summary_path,
                :source_summary_hash_sha256, :eligibility_rules_json,
                :valid_labels_json, :eligible_records_count, :excluded_records_count,
                :training_records_count, :test_records_count, :class_counts_json,
                :requested_test_proportion, :observed_test_proportion, :random_seed,
                :stratification_field, :grouping_field, :split_method,
                :duplicate_group_normalization, :near_duplicate_threshold,
                :duplicate_audit_json, :quality_checks_json, :training_csv_path,
                :training_csv_hash_sha256, :test_csv_path, :test_csv_hash_sha256,
                :split_workbook_path, :split_workbook_hash_sha256, :manifest_path,
                :manifest_hash_sha256, :split_status, :created_at, :notes
            )
            """,
            event_values,
        )
        connection.executemany(
            """
            INSERT INTO split_assignments (
                split_id, segmentation_id, review_record_id, candidate_id,
                partition, gold_label, duplicate_group_id,
                duplicate_group_size, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            assignment_values,
        )

        observed_assignment_count = connection.execute(
            "SELECT COUNT(*) FROM split_assignments WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        observed_partition_counts = dict(
            connection.execute(
                """
                SELECT partition, COUNT(*)
                FROM split_assignments
                WHERE split_id = ?
                GROUP BY partition
                """,
                (split_id,),
            ).fetchall()
        )
        expected_partition_counts = {
            "train": int(counts["training_total"]),
            "test": int(counts["test_total"]),
        }
        if observed_assignment_count != int(counts["eligible_total"]):
            raise RuntimeError("Registered split assignment total is incorrect.")
        if observed_partition_counts != expected_partition_counts:
            raise RuntimeError(
                "Registered partition counts are incorrect: "
                f"expected {expected_partition_counts}, observed "
                f"{observed_partition_counts}."
            )
        foreign_key_violations = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_violations:
            raise RuntimeError(
                "Foreign-key violations detected while registering the split: "
                f"{foreign_key_violations[:5]}"
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return {
        "database_path": project_relative(database_path),
        "split_id": split_id,
        "matching_id": matching_id,
        "split_date": split_date,
        "assignment_records": len(assignment_values),
        "partition_counts": expected_partition_counts,
        "tables": ["split_events", "split_assignments"],
    }


def run_split_creation(
    *,
    input_path: Path,
    summary_path: Path,
    output_dir: Path,
    database_path: Path = DATABASE_PATH,
    sheet: str = DEFAULT_SHEET,
    header_row: int = DEFAULT_HEADER_ROW,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    allow_count_mismatch: bool = False,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    """Run source validation, splitting, QA, and export."""
    created_at = utc_now_iso()
    split_date = created_at[:10]
    source_paths = {"review_workbook": input_path, "summary_workbook": summary_path}
    source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    matching_event = resolve_matching_event(
        database_path,
        review_hash_sha256=source_hashes["review_workbook"],
        summary_hash_sha256=source_hashes["summary_workbook"],
    )
    review_data = load_review_workbook(input_path, sheet, header_row)
    summary_data = load_summary_workbook(summary_path)
    verify_source_hashes(source_paths, source_hashes, "while loading")
    summary_pair_validation = validate_summary_pair(review_data, summary_data)
    source_validation = validate_source_counts(review_data, allow_count_mismatch)
    review_adjustment_validation = validate_authorized_review_adjustments(review_data)
    candidate_id_validation = verify_candidate_id_uniqueness(review_data)
    eligible, excluded, exclusion_summary = select_eligible_records(review_data)
    eligibility_count_validation = assess_eligibility_counts(eligible)

    preflight_issues: list[str] = []
    if eligibility_count_validation["mismatches"]:
        if allow_count_mismatch:
            print("WARNING: Eligibility count mismatches were explicitly allowed:")
            for mismatch in eligibility_count_validation["mismatches"]:
                print(f"- {mismatch}")
            eligibility_count_validation["status"] = "overridden"
        else:
            preflight_issues.extend(eligibility_count_validation["mismatches"])

    grouped: pd.DataFrame | None = None
    try:
        grouped = assign_duplicate_groups(eligible)
    except DuplicateLabelConflictError as error:
        preflight_issues.append(str(error))

    if preflight_issues:
        raise ValueError(
            "Task 3 preflight validation failed:\n- "
            + "\n- ".join(preflight_issues)
        )
    assert grouped is not None

    duplicate_summary = duplicate_statistics(grouped)
    duplicate_audit, near_summary = build_duplicate_audit(
        grouped,
        near_duplicate_threshold,
    )
    training, test, split_metadata = create_stratified_split(
        grouped,
        test_size,
        random_state,
    )
    split_validation = validate_split(
        grouped,
        training,
        test,
        test_size,
    )

    run_id = next_split_run_id(output_dir, database_path)
    split_summary = build_split_summary(
        grouped,
        training,
        test,
        excluded,
        duplicate_summary,
        near_summary,
    )
    readme = build_readme_sheet(
        review_path=input_path,
        summary_path=summary_path,
        run_id=run_id,
        created_at=created_at,
        test_size=test_size,
        random_state=random_state,
        split_method=split_metadata["method"],
        near_duplicate_threshold=near_duplicate_threshold,
    )

    execution_status = (
        "Completed with warnings"
        if (
            source_validation["mismatches"]
            or eligibility_count_validation["status"] == "overridden"
        )
        else "Completed"
    )
    manifest = {
        "task_id": TASK_ID,
        "script_id": SCRIPT_ID,
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "matching_id": matching_event["matching_id"],
        "split_date": split_date,
        "created_at": created_at,
        "source": {
            "review_workbook": {
                "filename": input_path.name,
                "path": project_relative(input_path),
                "sha256": source_hashes["review_workbook"],
                "sheet": sheet,
                "header_row": header_row,
            },
            "summary_workbook": {
                "filename": summary_path.name,
                "path": project_relative(summary_path),
                "sha256": source_hashes["summary_workbook"],
                "source_data_sheet": "Source Data",
            },
            "summary_pair_validation": summary_pair_validation,
            "count_validation": source_validation,
            "eligibility_count_validation": eligibility_count_validation,
            "review_adjustment_validation": review_adjustment_validation,
            "candidate_id_validation": candidate_id_validation,
            "review_adjustment_note": SOURCE_REVIEW_ADJUSTMENT_NOTE,
        },
        "eligibility_rules": [
            "include_in_final_set normalized to yes",
            "reviewer_gold_label is regulative, constitutive, or non_institutional",
            "candidate_text_original is not blank",
            "candidate_id is not blank",
        ],
        "input_text_field": "candidate_text_original",
        "classification_target": "reviewer_gold_label",
        "metadata_fields_are_model_features": False,
        "eligibility_policy": {
            "expected_eligible_counts": EXPECTED_ELIGIBLE_COUNTS,
            "note": ELIGIBILITY_POLICY_NOTE,
        },
        "exclusions": exclusion_summary,
        "split": {
            "training_proportion_requested": 1.0 - test_size,
            "test_proportion_requested": test_size,
            "random_seed": random_state,
            "stratification_field": "reviewer_gold_label",
            "grouping_field": "duplicate_group_id",
            "alignment_group_id_used_for_grouping": False,
            **split_metadata,
        },
        "counts": {
            "eligible_total": int(len(grouped)),
            "training_total": int(len(training)),
            "test_total": int(len(test)),
            "eligible_by_class": _counts_by_class(grouped),
            "training_by_class": _counts_by_class(training),
            "test_by_class": _counts_by_class(test),
        },
        "duplicate_audit": {
            "normalization": "Unicode NFKC; lowercase; collapse whitespace; trim",
            **duplicate_summary,
            "near_duplicate_method": "difflib.SequenceMatcher on normalized text",
            "near_duplicate_threshold": near_duplicate_threshold,
            "near_duplicates_automatically_excluded": False,
            **near_summary,
        },
        "training_review_record_ids": training["review_record_id"].map(
            _comparison_value
        ).tolist(),
        "training_candidate_ids": training["candidate_id"].map(
            _comparison_value
        ).tolist(),
        "test_review_record_ids": test["review_record_id"].map(
            _comparison_value
        ).tolist(),
        "test_candidate_ids": test["candidate_id"].map(
            _comparison_value
        ).tolist(),
        "quality_checks": split_validation,
        "modeling_steps_performed": [],
        "execution_status": execution_status,
    }

    verify_source_hashes(source_paths, source_hashes, "before output publication")
    output_paths, export_verification = export_outputs(
        output_dir=output_dir,
        training=training,
        test=test,
        excluded=excluded,
        duplicate_audit=duplicate_audit,
        split_summary=split_summary,
        readme=readme,
        manifest=manifest,
    )
    published_manifest = json.loads(
        output_paths[MANIFEST_FILENAME].read_text(encoding="utf-8")
    )
    archive_paths = archive_published_run(
        output_dir=output_dir,
        run_id=run_id,
        published_paths=output_paths,
        manifest=published_manifest,
    )
    database_registration = register_split_run(
        database_path=database_path,
        matching_event=matching_event,
        manifest=published_manifest,
        training=training,
        test=test,
        output_paths=archive_paths,
    )

    print("\nTask 3 train/test split completed")
    print("---------------------------------")
    print(f"run_id: {run_id}")
    print(f"split_date: {split_date}")
    print(f"matching_id: {matching_event['matching_id']}")
    print(f"execution_status: {execution_status}")
    print(f"eligible_records: {len(grouped)}")
    print(f"training_records: {len(training)}")
    print(f"test_records: {len(test)}")
    print(f"split_method: {split_metadata['method']}")
    print("\nPer-class counts")
    print("----------------")
    for label in VALID_LABELS:
        print(
            f"{label}: training={int(training['reviewer_gold_label'].eq(label).sum())}, "
            f"test={int(test['reviewer_gold_label'].eq(label).sum())}"
        )
    print("\nDuplicate checks")
    print("----------------")
    for key, value in {**duplicate_summary, **near_summary}.items():
        print(f"{key}: {value}")
    print("\nGenerated files")
    print("---------------")
    for path in output_paths.values():
        print(project_relative(path))
    print("\nImmutable historical package")
    print("----------------------------")
    print(project_relative(output_dir / HISTORY_DIRNAME / run_id))
    print("\nDatabase registration")
    print("---------------------")
    print(f"database: {project_relative(database_path)}")
    print(f"split_events: {run_id}")
    print(f"split_assignments: {database_registration['assignment_records']}")

    return {
        "run_id": run_id,
        "split_date": split_date,
        "matching_id": matching_event["matching_id"],
        "execution_status": execution_status,
        "counts": manifest["counts"],
        "duplicate_audit": manifest["duplicate_audit"],
        "exclusions": exclusion_summary,
        "quality_checks": split_validation,
        "export_verification": export_verification,
        "database_registration": database_registration,
        "output_paths": {
            filename: project_relative(path)
            for filename, path in output_paths.items()
        },
        "archive_paths": {
            filename: project_relative(path)
            for filename, path in archive_paths.items()
        },
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible raw training/test datasets from the completed "
            "institutional-statement alignment review."
        )
    )
    parser.add_argument(
        "--input",
        default=str(REPORTS_DIR / DEFAULT_REVIEW_FILENAME),
        help="Reviewed alignment workbook (project-relative or absolute path).",
    )
    parser.add_argument(
        "--summary-input",
        default=None,
        help=(
            "Matched alignment summary workbook. Default: "
            "IG_statement_alignment_summary.xlsx beside --input."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGES_DIR / TASK_ID),
        help="Output package directory (project-relative or absolute path).",
    )
    parser.add_argument(
        "--database",
        default=str(DATABASE_PATH),
        help="Corpus inventory SQLite database (project-relative or absolute path).",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help=f"Review worksheet name. Default: {DEFAULT_SHEET!r}.",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=DEFAULT_HEADER_ROW,
        help="One-based Excel header row. Default: 5.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Requested test fraction. Default: 0.20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Reproducible random seed. Default: 42.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
        help="SequenceMatcher report-only threshold. Default: 0.90.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help=(
            "Explicitly continue after reviewed source/eligibility count differences. "
            "This never bypasses candidate-ID or duplicate-label conflicts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_arguments()
    input_path = resolve_project_path(args.input)
    summary_path = (
        resolve_project_path(args.summary_input)
        if args.summary_input
        else input_path.with_name(DEFAULT_SUMMARY_FILENAME)
    )
    output_dir = resolve_project_path(args.output_dir)
    database_path = resolve_project_path(args.database)
    try:
        run_split_creation(
            input_path=input_path,
            summary_path=summary_path,
            output_dir=output_dir,
            database_path=database_path,
            sheet=args.sheet,
            header_row=args.header_row,
            test_size=args.test_size,
            random_state=args.random_state,
            allow_count_mismatch=args.allow_count_mismatch,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
    except Exception as error:
        raise SystemExit(f"\nTrain/test split creation failed: {error}") from error


if __name__ == "__main__":
    main()
