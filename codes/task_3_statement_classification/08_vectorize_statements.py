#!/usr/bin/env python3
"""Create reusable TF-IDF representations from a registered train/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    from config import DATABASE_PATH
except ImportError as error:
    raise SystemExit("Could not import codes/config.py.") from error

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "task_3_statement_classification" / "vectorization"
)

SCRIPT_NAME = "08_vectorize_statements.py"
SCRIPT_VERSION = "1.0"
VALID_LABELS = ("regulative", "constitutive", "non_institutional")
SPANISH_SINGLE_CHARACTER_TOKENS = ("a", "e", "o", "u", "y")
TOKEN_PATTERN = r"(?u)\b(?:[aeouy]|\w{2,})\b"

TRAIN_MATRIX_FILENAME = "X_train_tfidf.npz"
TEST_MATRIX_FILENAME = "X_test_tfidf.npz"
TRAIN_INDEX_FILENAME = "train_vectorization_index.csv"
TEST_INDEX_FILENAME = "test_vectorization_index.csv"
FEATURES_FILENAME = "tfidf_features.csv"
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"

REQUIRED_INPUT_COLUMNS = (
    "review_record_id",
    "candidate_id",
    "candidate_text_original",
    "reviewer_gold_label",
    "split",
)
INDEX_COLUMNS = (
    "matrix_row_index",
    "candidate_id",
    "candidate_text_original",
    "reviewer_gold_label",
    "split",
)

TFIDF_PARAMETERS = {
    "analyzer": "word",
    "lowercase": True,
    "ngram_range": [1, 3],
    "token_pattern": TOKEN_PATTERN,
    "stop_words": None,
    "strip_accents": None,
    "use_idf": True,
    "smooth_idf": True,
    "sublinear_tf": False,
    "norm": "l2",
    "dtype": "numpy.float64",
}

ARTIFACT_DESCRIPTIONS = {
    "training_matrix": "Sparse TF-IDF matrix for training statements.",
    "test_matrix": "Sparse TF-IDF matrix for test statements using the training-fitted vocabulary and IDF values.",
    "training_index": "Row-order mapping between the training matrix, candidate identifiers, original text, and reviewed gold labels.",
    "test_index": "Row-order mapping between the test matrix, candidate identifiers, original text, and reviewed gold labels.",
    "features": "Feature-column index with n-gram text, n-gram length, and learned training IDF value.",
    "fitted_vectorizer": "Fitted scikit-learn TF-IDF vectorizer for reproducible transformation of future statements.",
}


VECTOR_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectorization_events (
    vectorization_id TEXT PRIMARY KEY,
    split_id TEXT NOT NULL,
    matching_id TEXT NOT NULL,
    segmentation_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    vectorization_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    script_name TEXT NOT NULL,
    script_version TEXT NOT NULL,
    vectorization_tool TEXT NOT NULL,
    vectorization_tool_version TEXT NOT NULL,
    input_training_path TEXT NOT NULL,
    input_training_hash_sha256 TEXT NOT NULL,
    input_test_path TEXT NOT NULL,
    input_test_hash_sha256 TEXT NOT NULL,
    input_training_rows INTEGER NOT NULL,
    input_test_rows INTEGER NOT NULL,
    class_distributions_json TEXT NOT NULL,
    tfidf_parameters_json TEXT NOT NULL,
    tokenization_method TEXT NOT NULL,
    spanish_single_character_tokens_json TEXT NOT NULL,
    fit_scope TEXT NOT NULL,
    vocabulary_size INTEGER NOT NULL,
    training_matrix_rows INTEGER NOT NULL,
    training_matrix_columns INTEGER NOT NULL,
    training_matrix_nonzero INTEGER NOT NULL,
    test_matrix_rows INTEGER NOT NULL,
    test_matrix_columns INTEGER NOT NULL,
    test_matrix_nonzero INTEGER NOT NULL,
    validation_results_json TEXT NOT NULL,
    output_artifacts_count INTEGER NOT NULL,
    vectorization_status TEXT NOT NULL,
    notes TEXT,
    CHECK (length(vectorization_date) = 10),
    CHECK (input_training_rows > 0),
    CHECK (input_test_rows > 0),
    CHECK (vocabulary_size > 0),
    CHECK (training_matrix_rows = input_training_rows),
    CHECK (test_matrix_rows = input_test_rows),
    CHECK (training_matrix_columns = vocabulary_size),
    CHECK (test_matrix_columns = vocabulary_size),
    CHECK (training_matrix_nonzero >= 0),
    CHECK (test_matrix_nonzero >= 0),
    CHECK (output_artifacts_count = 6),
    CHECK (json_valid(class_distributions_json)),
    CHECK (json_valid(tfidf_parameters_json)),
    CHECK (json_valid(spanish_single_character_tokens_json)),
    CHECK (json_valid(validation_results_json)),
    FOREIGN KEY (split_id)
        REFERENCES split_events(split_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
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

CREATE TABLE IF NOT EXISTS vectorization_artifacts (
    vectorization_id TEXT NOT NULL,
    artifact_role TEXT NOT NULL,
    file_name TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_hash_sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    artifact_description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (vectorization_id, artifact_role),
    UNIQUE (vectorization_id, file_name),
    CHECK (artifact_role IN (
        'training_matrix', 'test_matrix', 'training_index', 'test_index',
        'features', 'fitted_vectorizer'
    )),
    CHECK (size_bytes > 0),
    FOREIGN KEY (vectorization_id)
        REFERENCES vectorization_events(vectorization_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_vectorization_events_split_created
    ON vectorization_events(split_id, created_at);

CREATE INDEX IF NOT EXISTS idx_vectorization_artifacts_path
    ON vectorization_artifacts(artifact_path);
"""


def utc_now_iso() -> str:
    """Return a UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_relative(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve an absolute or project-relative path."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 hash."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    """Serialize JSON deterministically for SQLite."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def ensure_registry_schema(database_path: Path) -> None:
    """Create vectorization registry tables without changing prior records."""
    if not database_path.exists():
        raise FileNotFoundError(f"Corpus inventory database not found: {database_path}")
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(VECTOR_REGISTRY_SCHEMA)


def load_split_event(database_path: Path, split_id: str | None) -> dict[str, Any]:
    """Load one completed split event and verify its registered lineage."""
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        if split_id is None:
            row = connection.execute(
                """
                SELECT * FROM split_events
                WHERE split_status IN ('Completed', 'Completed with warnings')
                ORDER BY created_at DESC, split_id DESC
                LIMIT 1
                """
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM split_events WHERE split_id = ?", (split_id,)
            ).fetchone()
        if row is None:
            requested = split_id or "latest completed split"
            raise ValueError(f"Registered split event not found: {requested}")
        matching = connection.execute(
            """
            SELECT matching_id, segmentation_id, document_id
            FROM matching_events
            WHERE matching_id = ?
            """,
            (row["matching_id"],),
        ).fetchone()
        if matching is None:
            raise ValueError(f"Split {row['split_id']} has no matching event lineage.")
        for field in ("matching_id", "segmentation_id", "document_id"):
            if str(row[field]) != str(matching[field]):
                raise ValueError(
                    f"Split lineage mismatch for {field}: split={row[field]!r}, "
                    f"matching={matching[field]!r}."
                )
    if row["split_status"] not in {"Completed", "Completed with warnings"}:
        raise ValueError(f"Split {row['split_id']} is not completed.")
    return dict(row)


def next_vectorization_id(database_path: Path) -> str:
    """Return the next append-only VECT_###### identifier."""
    with sqlite3.connect(database_path) as connection:
        identifiers = [
            str(row[0])
            for row in connection.execute(
                "SELECT vectorization_id FROM vectorization_events"
            )
        ]
    invalid = [value for value in identifiers if not re.fullmatch(r"VECT_\d{6}", value)]
    if invalid:
        raise ValueError(
            "Invalid vectorization identifiers in the registry: "
            + ", ".join(invalid[:10])
        )
    highest = max(
        [0] + [int(value.removeprefix("VECT_")) for value in identifiers]
    )
    return f"VECT_{highest + 1:06d}"


def load_partition(path: Path) -> pd.DataFrame:
    """Load one split CSV without converting empty strings to missing values."""
    if not path.exists():
        raise FileNotFoundError(f"Registered split file not found: {path}")
    data = pd.read_csv(
        path,
        dtype=object,
        keep_default_na=False,
        encoding="utf-8",
    )
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {', '.join(missing)}"
        )
    return data


def class_counts(data: pd.DataFrame) -> dict[str, int]:
    """Count reviewed labels in a stable order."""
    counts = Counter(str(value) for value in data["reviewer_gold_label"])
    return {label: int(counts.get(label, 0)) for label in VALID_LABELS}


def validate_inputs(
    *,
    database_path: Path,
    split_event: dict[str, Any],
    training: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Validate files, rows, labels, membership, and registered split counts."""
    training_path = resolve_project_path(split_event["training_csv_path"])
    test_path = resolve_project_path(split_event["test_csv_path"])
    checks: dict[str, bool] = {
        "training_input_hash_matches_split_event": sha256_file(training_path)
        == split_event["training_csv_hash_sha256"],
        "test_input_hash_matches_split_event": sha256_file(test_path)
        == split_event["test_csv_hash_sha256"],
        "training_row_count_matches_split_event": len(training)
        == int(split_event["training_records_count"]),
        "test_row_count_matches_split_event": len(test)
        == int(split_event["test_records_count"]),
    }

    for name, data, expected_partition in (
        ("training", training, "training"),
        ("test", test, "test"),
    ):
        candidate_ids = data["candidate_id"].map(str)
        texts = data["candidate_text_original"]
        labels = data["reviewer_gold_label"].map(str)
        partitions = data["split"].map(str)
        checks[f"{name}_candidate_ids_nonblank"] = candidate_ids.str.strip().ne("").all()
        checks[f"{name}_candidate_ids_unique"] = not candidate_ids.duplicated().any()
        checks[f"{name}_texts_are_strings"] = texts.map(
            lambda value: isinstance(value, str)
        ).all()
        checks[f"{name}_texts_nonblank"] = texts.map(str).str.strip().ne("").all()
        checks[f"{name}_labels_approved"] = labels.isin(VALID_LABELS).all()
        checks[f"{name}_partition_preserved"] = partitions.eq(expected_partition).all()

    training_ids = set(training["candidate_id"].map(str))
    test_ids = set(test["candidate_id"].map(str))
    checks["candidate_ids_do_not_cross_partitions"] = not training_ids & test_ids

    observed_counts = {
        "training": class_counts(training),
        "test": class_counts(test),
    }
    observed_counts["eligible"] = {
        label: observed_counts["training"][label] + observed_counts["test"][label]
        for label in VALID_LABELS
    }
    registered_counts = json.loads(split_event["class_counts_json"])
    checks["class_distributions_match_split_event"] = (
        observed_counts == registered_counts
    )

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT review_record_id, candidate_id, partition, gold_label
            FROM split_assignments
            WHERE split_id = ?
            """,
            (split_event["split_id"],),
        ).fetchall()
    registered_assignments = {
        str(candidate_id): (str(review_record_id), str(partition), str(gold_label))
        for review_record_id, candidate_id, partition, gold_label in rows
    }
    file_assignments: dict[str, tuple[str, str, str]] = {}
    for data, partition in ((training, "train"), (test, "test")):
        for row in data.itertuples(index=False):
            file_assignments[str(row.candidate_id)] = (
                str(row.review_record_id),
                partition,
                str(row.reviewer_gold_label),
            )
    checks["assignment_count_matches_split_registry"] = (
        len(file_assignments) == len(registered_assignments)
    )
    checks["assignment_membership_matches_split_registry"] = (
        file_assignments == registered_assignments
    )

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Input validation failed: " + ", ".join(failed))
    return observed_counts, {"status": "passed", "checks": checks}


def build_vectorizer() -> TfidfVectorizer:
    """Create the protocol-specified TF-IDF vectorizer."""
    return TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, 3),
        token_pattern=TOKEN_PATTERN,
        stop_words=None,
        strip_accents=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        norm="l2",
        dtype=np.float64,
    )


def build_index(data: pd.DataFrame) -> pd.DataFrame:
    """Build a zero-based matrix-row index without changing source values."""
    index = data.loc[
        :, ["candidate_id", "candidate_text_original", "reviewer_gold_label", "split"]
    ].copy()
    index.insert(0, "matrix_row_index", np.arange(len(index), dtype=np.int64))
    return index.loc[:, list(INDEX_COLUMNS)]


def build_features(vectorizer: TfidfVectorizer) -> pd.DataFrame:
    """Build a zero-based feature index with learned IDF values."""
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names), dtype=np.int64),
            "ngram_text": feature_names,
            "ngram_length": [str(value).count(" ") + 1 for value in feature_names],
            "idf_value": vectorizer.idf_.astype(np.float64, copy=False),
        }
    )


def sparse_matrices_close(
    left: sparse.spmatrix,
    right: sparse.spmatrix,
    tolerance: float = 1e-12,
) -> bool:
    """Compare two sparse matrices without densifying them."""
    if left.shape != right.shape:
        return False
    difference = (left - right).tocsr()
    return difference.nnz == 0 or bool(np.max(np.abs(difference.data)) <= tolerance)


def write_outputs(
    *,
    output_dir: Path,
    training_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    training_index: pd.DataFrame,
    test_index: pd.DataFrame,
    features: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> dict[str, Path]:
    """Write all vectorization artifacts to a staging directory."""
    paths = {
        "training_matrix": output_dir / TRAIN_MATRIX_FILENAME,
        "test_matrix": output_dir / TEST_MATRIX_FILENAME,
        "training_index": output_dir / TRAIN_INDEX_FILENAME,
        "test_index": output_dir / TEST_INDEX_FILENAME,
        "features": output_dir / FEATURES_FILENAME,
        "fitted_vectorizer": output_dir / VECTORIZER_FILENAME,
    }
    sparse.save_npz(paths["training_matrix"], training_matrix, compressed=True)
    sparse.save_npz(paths["test_matrix"], test_matrix, compressed=True)
    training_index.to_csv(
        paths["training_index"], index=False, encoding="utf-8", lineterminator="\n"
    )
    test_index.to_csv(
        paths["test_index"], index=False, encoding="utf-8", lineterminator="\n"
    )
    features.to_csv(
        paths["features"], index=False, encoding="utf-8", lineterminator="\n"
    )
    joblib.dump(vectorizer, paths["fitted_vectorizer"], compress=3)
    return paths


def validate_outputs(
    *,
    paths: dict[str, Path],
    training: pd.DataFrame,
    test: pd.DataFrame,
    training_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    training_index: pd.DataFrame,
    test_index: pd.DataFrame,
    features: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> dict[str, Any]:
    """Read back and validate every vectorization artifact."""
    loaded_training_matrix = sparse.load_npz(paths["training_matrix"]).tocsr()
    loaded_test_matrix = sparse.load_npz(paths["test_matrix"]).tocsr()
    loaded_training_index = pd.read_csv(
        paths["training_index"], dtype=object, keep_default_na=False
    )
    loaded_test_index = pd.read_csv(
        paths["test_index"], dtype=object, keep_default_na=False
    )
    loaded_features = pd.read_csv(
        paths["features"], dtype=object, keep_default_na=False
    )
    loaded_vectorizer: TfidfVectorizer = joblib.load(paths["fitted_vectorizer"])

    feature_names = vectorizer.get_feature_names_out()
    analyzer = vectorizer.build_analyzer()
    probe_tokens = set(analyzer("a e o u y x ñ 1 casa árbol"))
    training_terms = {
        term
        for text in training["candidate_text_original"]
        for term in analyzer(str(text))
    }
    test_terms = {
        term
        for text in test["candidate_text_original"]
        for term in analyzer(str(text))
    }
    test_only_terms = test_terms - training_terms
    vocabulary = set(feature_names)

    def index_matches(index_data: pd.DataFrame, source_data: pd.DataFrame) -> bool:
        if list(index_data.columns) != list(INDEX_COLUMNS):
            return False
        if [int(value) for value in index_data["matrix_row_index"]] != list(
            range(len(source_data))
        ):
            return False
        for column in INDEX_COLUMNS[1:]:
            if index_data[column].map(str).tolist() != source_data[column].map(str).tolist():
                return False
        return True

    transformed_training = loaded_vectorizer.transform(
        training["candidate_text_original"].tolist()
    ).tocsr()
    transformed_test = loaded_vectorizer.transform(
        test["candidate_text_original"].tolist()
    ).tocsr()

    checks = {
        "training_matrix_rows_equal_training_index_rows": loaded_training_matrix.shape[0]
        == len(loaded_training_index),
        "test_matrix_rows_equal_test_index_rows": loaded_test_matrix.shape[0]
        == len(loaded_test_index),
        "training_and_test_use_identical_feature_count": loaded_training_matrix.shape[1]
        == loaded_test_matrix.shape[1],
        "feature_count_equals_matrix_columns": len(loaded_features)
        == loaded_training_matrix.shape[1],
        "idf_count_equals_matrix_columns": len(vectorizer.idf_)
        == loaded_training_matrix.shape[1],
        "training_row_order_and_values_preserved": index_matches(
            loaded_training_index, training
        ),
        "test_row_order_and_values_preserved": index_matches(loaded_test_index, test),
        "training_matrix_values_finite": np.isfinite(loaded_training_matrix.data).all(),
        "test_matrix_values_finite": np.isfinite(loaded_test_matrix.data).all(),
        "training_matrix_readback_matches_memory": sparse_matrices_close(
            loaded_training_matrix, training_matrix
        ),
        "test_matrix_readback_matches_memory": sparse_matrices_close(
            loaded_test_matrix, test_matrix
        ),
        "loaded_vectorizer_reproduces_training_matrix": sparse_matrices_close(
            transformed_training, loaded_training_matrix
        ),
        "loaded_vectorizer_reproduces_test_matrix": sparse_matrices_close(
            transformed_test, loaded_test_matrix
        ),
        "feature_order_preserved": loaded_features["ngram_text"].map(str).tolist()
        == feature_names.tolist(),
        "feature_indices_preserved": [int(value) for value in loaded_features["feature_index"]]
        == list(range(len(feature_names))),
        "ngram_lengths_preserved": [int(value) for value in loaded_features["ngram_length"]]
        == features["ngram_length"].astype(int).tolist(),
        "idf_values_preserved": np.allclose(
            loaded_features["idf_value"].astype(float).to_numpy(),
            vectorizer.idf_,
            rtol=0,
            atol=1e-12,
        ),
        "spanish_single_character_tokens_retained": set(
            SPANISH_SINGLE_CHARACTER_TOKENS
        ).issubset(probe_tokens),
        "other_probe_single_character_tokens_excluded": not {"x", "ñ", "1"}
        & probe_tokens,
        "test_only_terms_absent_from_vocabulary": not test_only_terms & vocabulary,
        "all_output_files_nonempty": all(
            path.exists() and path.stat().st_size > 0 for path in paths.values()
        ),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Output validation failed: " + ", ".join(failed))
    return {
        "status": "passed",
        "checks": checks,
        "test_only_ngram_count": len(test_only_terms),
        "test_only_ngrams_in_vocabulary": 0,
    }


def artifact_rows(
    *,
    vectorization_id: str,
    final_dir: Path,
    staged_paths: dict[str, Path],
    created_at: str,
) -> list[dict[str, Any]]:
    """Build artifact registry rows from validated staged files."""
    rows = []
    for role, staged_path in staged_paths.items():
        rows.append(
            {
                "vectorization_id": vectorization_id,
                "artifact_role": role,
                "file_name": staged_path.name,
                "artifact_path": project_relative(final_dir / staged_path.name),
                "artifact_hash_sha256": sha256_file(staged_path),
                "size_bytes": staged_path.stat().st_size,
                "artifact_description": ARTIFACT_DESCRIPTIONS[role],
                "created_at": created_at,
            }
        )
    return rows


def register_and_publish(
    *,
    database_path: Path,
    staged_dir: Path,
    final_dir: Path,
    event_values: dict[str, Any],
    artifacts: list[dict[str, Any]],
    validation_results: dict[str, Any],
) -> dict[str, Any]:
    """Publish files and register the event in one SQLite transaction."""
    if final_dir.exists():
        raise ValueError(f"Vectorization output already exists: {final_dir}")
    connection = sqlite3.connect(database_path)
    published = False
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute(
            "SELECT 1 FROM vectorization_events WHERE vectorization_id = ?",
            (event_values["vectorization_id"],),
        ).fetchone():
            raise ValueError(
                f"Vectorization ID is already registered: "
                f"{event_values['vectorization_id']}"
            )
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO vectorization_events (
                vectorization_id, split_id, matching_id, segmentation_id,
                document_id, vectorization_date, created_at, script_name,
                script_version, vectorization_tool, vectorization_tool_version,
                input_training_path, input_training_hash_sha256, input_test_path,
                input_test_hash_sha256, input_training_rows, input_test_rows,
                class_distributions_json, tfidf_parameters_json,
                tokenization_method, spanish_single_character_tokens_json,
                fit_scope, vocabulary_size, training_matrix_rows,
                training_matrix_columns, training_matrix_nonzero,
                test_matrix_rows, test_matrix_columns, test_matrix_nonzero,
                validation_results_json, output_artifacts_count,
                vectorization_status, notes
            ) VALUES (
                :vectorization_id, :split_id, :matching_id, :segmentation_id,
                :document_id, :vectorization_date, :created_at, :script_name,
                :script_version, :vectorization_tool, :vectorization_tool_version,
                :input_training_path, :input_training_hash_sha256, :input_test_path,
                :input_test_hash_sha256, :input_training_rows, :input_test_rows,
                :class_distributions_json, :tfidf_parameters_json,
                :tokenization_method, :spanish_single_character_tokens_json,
                :fit_scope, :vocabulary_size, :training_matrix_rows,
                :training_matrix_columns, :training_matrix_nonzero,
                :test_matrix_rows, :test_matrix_columns, :test_matrix_nonzero,
                :validation_results_json, :output_artifacts_count,
                :vectorization_status, :notes
            )
            """,
            event_values,
        )
        connection.executemany(
            """
            INSERT INTO vectorization_artifacts (
                vectorization_id, artifact_role, file_name, artifact_path,
                artifact_hash_sha256, size_bytes, artifact_description, created_at
            ) VALUES (
                :vectorization_id, :artifact_role, :file_name, :artifact_path,
                :artifact_hash_sha256, :size_bytes, :artifact_description, :created_at
            )
            """,
            artifacts,
        )

        staged_dir.replace(final_dir)
        published = True
        for artifact in artifacts:
            path = resolve_project_path(artifact["artifact_path"])
            if sha256_file(path) != artifact["artifact_hash_sha256"]:
                raise RuntimeError(
                    f"Published artifact hash mismatch: {artifact['file_name']}"
                )
        artifact_count = connection.execute(
            """
            SELECT COUNT(*) FROM vectorization_artifacts
            WHERE vectorization_id = ?
            """,
            (event_values["vectorization_id"],),
        ).fetchone()[0]
        if artifact_count != len(artifacts):
            raise RuntimeError("Vectorization artifact registry count is incorrect.")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key validation failed during registration.")

        final_validation = {
            **validation_results,
            "database_event_corresponds_to_exported_artifacts": True,
            "database_registration_transactional": True,
        }
        connection.execute(
            """
            UPDATE vectorization_events
            SET validation_results_json = ?
            WHERE vectorization_id = ?
            """,
            (json_text(final_validation), event_values["vectorization_id"]),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        if published and final_dir.exists() and not staged_dir.exists():
            final_dir.replace(staged_dir)
        raise
    finally:
        connection.close()
    return {
        "status": "registered",
        "vectorization_id": event_values["vectorization_id"],
        "artifact_records": len(artifacts),
        "validation_status": "passed",
    }


def run_vectorization(
    *,
    database_path: Path,
    output_base_dir: Path,
    split_id: str | None = None,
) -> dict[str, Any]:
    """Validate, vectorize, export, and register one split."""
    created_at = utc_now_iso()
    ensure_registry_schema(database_path)
    split_event = load_split_event(database_path, split_id)
    training_path = resolve_project_path(split_event["training_csv_path"])
    test_path = resolve_project_path(split_event["test_csv_path"])
    training = load_partition(training_path)
    test = load_partition(test_path)
    distributions, input_validation = validate_inputs(
        database_path=database_path,
        split_event=split_event,
        training=training,
        test=test,
    )

    vectorizer = build_vectorizer()
    training_matrix = vectorizer.fit_transform(
        training["candidate_text_original"].tolist()
    ).tocsr()
    test_matrix = vectorizer.transform(test["candidate_text_original"].tolist()).tocsr()
    training_index = build_index(training)
    test_index = build_index(test)
    features = build_features(vectorizer)

    vectorization_id = next_vectorization_id(database_path)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_base_dir / vectorization_id
    staged_dir = Path(
        tempfile.mkdtemp(prefix=f".{vectorization_id}_", dir=output_base_dir)
    )
    try:
        staged_paths = write_outputs(
            output_dir=staged_dir,
            training_matrix=training_matrix,
            test_matrix=test_matrix,
            training_index=training_index,
            test_index=test_index,
            features=features,
            vectorizer=vectorizer,
        )
        output_validation = validate_outputs(
            paths=staged_paths,
            training=training,
            test=test,
            training_matrix=training_matrix,
            test_matrix=test_matrix,
            training_index=training_index,
            test_index=test_index,
            features=features,
            vectorizer=vectorizer,
        )
        validation_results = {
            "status": "passed",
            "input_validation": input_validation,
            "output_validation": output_validation,
            "fit_used_training_text_only": True,
            "test_text_or_labels_used_for_fit": False,
            "supervised_feature_selection_performed": False,
            "classifier_training_performed": False,
        }
        artifacts = artifact_rows(
            vectorization_id=vectorization_id,
            final_dir=final_dir,
            staged_paths=staged_paths,
            created_at=created_at,
        )
        event_values = {
            "vectorization_id": vectorization_id,
            "split_id": split_event["split_id"],
            "matching_id": split_event["matching_id"],
            "segmentation_id": split_event["segmentation_id"],
            "document_id": split_event["document_id"],
            "vectorization_date": created_at[:10],
            "created_at": created_at,
            "script_name": SCRIPT_NAME,
            "script_version": SCRIPT_VERSION,
            "vectorization_tool": "scikit-learn TfidfVectorizer",
            "vectorization_tool_version": sklearn.__version__,
            "input_training_path": project_relative(training_path),
            "input_training_hash_sha256": sha256_file(training_path),
            "input_test_path": project_relative(test_path),
            "input_test_hash_sha256": sha256_file(test_path),
            "input_training_rows": len(training),
            "input_test_rows": len(test),
            "class_distributions_json": json_text(distributions),
            "tfidf_parameters_json": json_text(TFIDF_PARAMETERS),
            "tokenization_method": (
                "scikit-learn word analyzer with the protocol token_pattern; "
                "no stemming, lemmatization, accent removal, or stop-word removal"
            ),
            "spanish_single_character_tokens_json": json_text(
                list(SPANISH_SINGLE_CHARACTER_TOKENS)
            ),
            "fit_scope": "training candidate_text_original only",
            "vocabulary_size": training_matrix.shape[1],
            "training_matrix_rows": training_matrix.shape[0],
            "training_matrix_columns": training_matrix.shape[1],
            "training_matrix_nonzero": training_matrix.nnz,
            "test_matrix_rows": test_matrix.shape[0],
            "test_matrix_columns": test_matrix.shape[1],
            "test_matrix_nonzero": test_matrix.nnz,
            "validation_results_json": json_text(validation_results),
            "output_artifacts_count": len(artifacts),
            "vectorization_status": "Completed",
            "notes": (
                "The vectorizer was fitted on training text only. Test text and labels "
                "were not used to learn vocabulary or IDF values. No classifier, "
                "parameter tuning, supervised feature selection, stemming, "
                "lemmatization, or 70-feature restriction was applied."
            ),
        }
        registration = register_and_publish(
            database_path=database_path,
            staged_dir=staged_dir,
            final_dir=final_dir,
            event_values=event_values,
            artifacts=artifacts,
            validation_results=validation_results,
        )
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)

    print(f"linked_split_id: {split_event['split_id']}")
    print(f"training_records: {len(training)} {distributions['training']}")
    print(f"test_records: {len(test)} {distributions['test']}")
    print(f"vocabulary_size: {training_matrix.shape[1]}")
    print(f"training_matrix_shape: {training_matrix.shape}")
    print(f"test_matrix_shape: {test_matrix.shape}")
    print(f"output_directory: {project_relative(final_dir)}")
    print(f"registered_vectorization_id: {vectorization_id}")
    print("final_validation_status: passed")

    return {
        "vectorization_id": vectorization_id,
        "split_id": split_event["split_id"],
        "output_directory": project_relative(final_dir),
        "training_records": len(training),
        "test_records": len(test),
        "class_distributions": distributions,
        "vocabulary_size": training_matrix.shape[1],
        "training_matrix_shape": list(training_matrix.shape),
        "test_matrix_shape": list(test_matrix.shape),
        "training_matrix_nonzero": training_matrix.nnz,
        "test_matrix_nonzero": test_matrix.nnz,
        "registration": registration,
        "validation_status": "passed",
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create and register reusable TF-IDF statement representations."
    )
    parser.add_argument(
        "--database",
        default=str(DATABASE_PATH),
        help="Corpus inventory SQLite database.",
    )
    parser.add_argument(
        "--split-id",
        default=None,
        help="Registered split ID. Default: latest completed split.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base directory for append-only VECT_###### output folders.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_arguments()
    try:
        run_vectorization(
            database_path=resolve_project_path(args.database),
            output_base_dir=resolve_project_path(args.output_dir),
            split_id=args.split_id,
        )
    except Exception as error:
        raise SystemExit(f"Vectorization failed: {error}") from error


if __name__ == "__main__":
    main()
