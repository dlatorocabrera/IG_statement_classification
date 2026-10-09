#!/usr/bin/env python3
"""Train and register nested-RFECV Random Forest Model C."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as ExcelImage
from PIL import Image as PillowImage
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT_PATH = SCRIPT_DIR / "09_train_random_forest_model_a.py"
MODEL_B_SCRIPT_PATH = SCRIPT_DIR / "10_train_random_forest_model_b.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared model utilities: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module("task3_random_forest_model_a_base_for_c", BASE_SCRIPT_PATH)
MODEL_B = _load_module("task3_random_forest_model_b_base_for_c", MODEL_B_SCRIPT_PATH)

DATABASE_PATH = BASE.DATABASE_PATH
DEFAULT_OUTPUT_DIR = BASE.DEFAULT_OUTPUT_DIR
SCRIPT_NAME = "11_train_random_forest_model_c.py"
SCRIPT_VERSION = "1.1"
MODEL_NAME = "Model C: paired nested-RFECV Random Forests"
MODEL_DESIGNATION = "Model C"
MODEL_DESCRIPTION = (
    "Paired three-class Random Forest evaluation comparing unweighted C1 and "
    "class-balanced C2. Random-Forest recursive feature elimination with inner "
    "cross-validation is repeated inside every outer training fold; both "
    "complete-training configurations receive post-hoc held-out reporting."
)

RANDOM_STATE = BASE.RANDOM_STATE
CLASS_ORDER = BASE.CLASS_ORDER
OUTER_CV_SPLITS = BASE.CV_SPLITS
INNER_CV_SPLITS = BASE.CV_SPLITS
STORED_METRIC_DECIMALS = BASE.STORED_METRIC_DECIMALS
EXPECTED_VOCABULARY_SIZE = BASE.EXPECTED_CURRENT_VOCABULARY_SIZE
RANDOM_FOREST_PARAMETERS = BASE.RANDOM_FOREST_PARAMETERS

REFERENCE_VOCABULARY_SIZE = 3_770
REFERENCE_MINIMUM_FEATURE_COUNT = 20
REFERENCE_MINIMUM_PROPORTION = (
    REFERENCE_MINIMUM_FEATURE_COUNT / REFERENCE_VOCABULARY_SIZE
)
EXPECTED_MINIMUM_FEATURE_COUNT = round(
    EXPECTED_VOCABULARY_SIZE
    * REFERENCE_MINIMUM_FEATURE_COUNT
    / REFERENCE_VOCABULARY_SIZE
)
MODEL_A_FEATURE_REFERENCE = 436
DEFAULT_RFECV_STEP = 0.05

CONFIGURATIONS = {
    "C1": {"class_weight": None, "label": "nested-RFECV unweighted"},
    "C2": {"class_weight": "balanced", "label": "nested-RFECV balanced"},
}

MODEL_FILENAME = "model_c_rfecv_rf.joblib"
CV_RESULTS_FILENAME = "model_c_cv_results.csv"
SELECTED_FEATURES_FILENAME = "model_c_selected_features.csv"
TEST_PREDICTIONS_FILENAME = "model_c_test_predictions.csv"
CONFUSION_MATRIX_FILENAME = "model_c_confusion_matrix.csv"
PERFORMANCE_REPORT_FILENAME = "model_c_rfecv_performance.xlsx"
FEATURE_CURVE_FILENAME = "model_c_rfecv_feature_curve.csv"
FEATURE_CURVE_PLOT_FILENAME = "model_c_macro_f1_vs_features.png"

ARTIFACT_DESCRIPTIONS = {
    "fitted_model": (
        "Reusable fitted C1 and C2 complete-training pipelines, including each RFECV "
        "selector and final Random Forest classifier, with the outer-CV selection marked."
    ),
    "cross_validation_results": (
        "Fold-level and aggregate macro-F1 and secondary metrics comparing A1/A2, "
        "B1/B2, and the newly evaluated nested-RFECV C1/C2 configurations."
    ),
    "selected_features": (
        "Features retained by the selected configuration's complete-training RFECV "
        "fit, with source TF-IDF indices, IDF values, and final forest importances."
    ),
    "test_predictions": (
        "Candidate-aligned post-hoc held-out predictions for C1 and C2; neither result "
        "is used to change the outer-CV configuration selection."
    ),
    "confusion_matrix": (
        "Post-hoc held-out confusion matrices for C1 and C2 in the predefined protocol "
        "class order."
    ),
    "performance_report": (
        "Human-readable workbook with lineage, six-configuration CV comparison, "
        "post-hoc held-out comparison, selected features, and RFECV feature curves."
    ),
    "rfecv_feature_curve": (
        "All inner-CV macro-F1 feature-count curves from the five outer fits and the "
        "complete-training fit for C1 and C2, including fold-specific scores."
    ),
    "rfecv_feature_plot": (
        "Training-only complete-training RFECV macro-F1 curves for C1 and C2 with "
        "uncertainty bands, proportional-minimum and Model A references, and optima."
    ),
}


def utc_now_iso() -> str:
    """Return a UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def proportional_minimum_feature_count(
    current_vocabulary: int,
    reference_vocabulary: int = REFERENCE_VOCABULARY_SIZE,
    reference_minimum: int = REFERENCE_MINIMUM_FEATURE_COUNT,
) -> int:
    """Scale the released classifier's minimum feature count proportionally."""
    if min(current_vocabulary, reference_vocabulary, reference_minimum) <= 0:
        raise ValueError("Vocabulary and minimum counts must be positive.")
    return round(current_vocabulary * reference_minimum / reference_vocabulary)


def ensure_flexible_model_registry_schema(database_path: Path) -> dict[str, Any]:
    """Allow positive artifact counts and nonblank roles while preserving all rows."""
    MODEL_B.ensure_general_model_registry_schema(database_path)
    with sqlite3.connect(database_path) as connection:
        event_sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_events'"
        ).fetchone()
        artifact_sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_artifacts'"
        ).fetchone()
    if event_sql_row is None or artifact_sql_row is None:
        raise RuntimeError("Model registry tables were not created.")
    event_sql = str(event_sql_row[0])
    artifact_sql = str(artifact_sql_row[0])
    event_general = "output_artifacts_count > 0" in event_sql
    artifact_general = "length(trim(artifact_role)) > 0" in artifact_sql
    if event_general and artifact_general:
        return {"status": "already_flexible", "rows_preserved": True}
    if not event_general and "output_artifacts_count = 6" not in event_sql:
        raise RuntimeError("Unrecognized model_events artifact-count constraint.")
    if not artifact_general and "artifact_role IN" not in artifact_sql:
        raise RuntimeError("Unrecognized model_artifacts role constraint.")

    new_event_sql = re.sub(
        r'CREATE TABLE(?: IF NOT EXISTS)?\s+"?model_events"?',
        "CREATE TABLE model_events_new",
        event_sql,
        count=1,
        flags=re.IGNORECASE,
    ).replace(
        "CHECK (output_artifacts_count = 6)",
        "CHECK (output_artifacts_count > 0)",
    )
    new_artifact_sql = re.sub(
        r'CREATE TABLE(?: IF NOT EXISTS)?\s+"?model_artifacts"?',
        "CREATE TABLE model_artifacts_new",
        artifact_sql,
        count=1,
        flags=re.IGNORECASE,
    )
    new_artifact_sql = re.sub(
        r"CHECK\s*\(\s*artifact_role\s+IN\s*\([^)]*\)\s*\)",
        "CHECK (length(trim(artifact_role)) > 0)",
        new_artifact_sql,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        before_events = int(connection.execute("SELECT COUNT(*) FROM model_events").fetchone()[0])
        before_artifacts = int(
            connection.execute("SELECT COUNT(*) FROM model_artifacts").fetchone()[0]
        )
        event_columns = [
            str(row[1]) for row in connection.execute("PRAGMA table_info(model_events)")
        ]
        artifact_columns = [
            str(row[1]) for row in connection.execute("PRAGMA table_info(model_artifacts)")
        ]
        quoted_event_columns = ", ".join(f'"{column}"' for column in event_columns)
        quoted_artifact_columns = ", ".join(
            f'"{column}"' for column in artifact_columns
        )
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(new_event_sql)
        connection.execute(new_artifact_sql)
        connection.execute(
            f"INSERT INTO model_events_new ({quoted_event_columns}) "
            f"SELECT {quoted_event_columns} FROM model_events"
        )
        connection.execute(
            f"INSERT INTO model_artifacts_new ({quoted_artifact_columns}) "
            f"SELECT {quoted_artifact_columns} FROM model_artifacts"
        )
        copied_events = int(
            connection.execute("SELECT COUNT(*) FROM model_events_new").fetchone()[0]
        )
        copied_artifacts = int(
            connection.execute("SELECT COUNT(*) FROM model_artifacts_new").fetchone()[0]
        )
        if (copied_events, copied_artifacts) != (before_events, before_artifacts):
            raise RuntimeError("Registry migration did not preserve every model row.")
        connection.execute("DROP TABLE model_artifacts")
        connection.execute("DROP TABLE model_events")
        connection.execute("ALTER TABLE model_events_new RENAME TO model_events")
        connection.execute("ALTER TABLE model_artifacts_new RENAME TO model_artifacts")
        connection.execute(
            "CREATE INDEX idx_model_events_vectorization_created "
            "ON model_events(vectorization_id, created_at)"
        )
        connection.execute(
            "CREATE INDEX idx_model_artifacts_path ON model_artifacts(artifact_path)"
        )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed after registry migration.")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("SQLite integrity check failed after registry migration.")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {
        "status": "migrated",
        "previous_output_artifact_constraint": "exactly 6",
        "current_output_artifact_constraint": "positive count",
        "previous_artifact_role_constraint": "six enumerated roles",
        "current_artifact_role_constraint": "any nonblank role",
        "preserved_model_event_rows": before_events,
        "preserved_model_artifact_rows": before_artifacts,
        "rows_preserved": True,
    }


def build_pipeline(
    class_weight: str | None,
    *,
    minimum_features: int = EXPECTED_MINIMUM_FEATURE_COUNT,
    rfecv_step: int | float = DEFAULT_RFECV_STEP,
    inner_cv_splits: int = INNER_CV_SPLITS,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 100,
    estimator_n_jobs: int = -1,
    selector_n_jobs: int = -1,
) -> Pipeline:
    """Build the leakage-safe RFECV-plus-Random-Forest Model C pipeline."""
    if minimum_features <= 0:
        raise ValueError("minimum_features must be positive.")
    if not (isinstance(rfecv_step, int) and rfecv_step >= 1) and not (
        isinstance(rfecv_step, float) and 0 < rfecv_step < 1
    ):
        raise ValueError("rfecv_step must be a positive integer or a float in (0, 1).")
    inner_cv = StratifiedKFold(
        n_splits=inner_cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    forest_parameters = {
        **RANDOM_FOREST_PARAMETERS,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "n_jobs": estimator_n_jobs,
        "class_weight": class_weight,
    }
    return Pipeline(
        [
            (
                "feature_selection",
                RFECV(
                    estimator=RandomForestClassifier(**forest_parameters),
                    step=rfecv_step,
                    min_features_to_select=minimum_features,
                    scoring="f1_macro",
                    cv=inner_cv,
                    n_jobs=selector_n_jobs,
                ),
            ),
            ("classifier", RandomForestClassifier(**forest_parameters)),
        ]
    )


def extract_rfecv_curve(
    selector: RFECV,
    *,
    configuration_id: str,
    fit_scope: str,
    outer_fold: int | None,
    minimum_features: int,
    rfecv_step: int | float,
) -> pd.DataFrame:
    """Convert one fitted RFECV result dictionary into an auditable long table."""
    results = selector.cv_results_
    required = {"n_features", "mean_test_score", "std_test_score"}
    missing = required - set(results)
    if missing:
        raise RuntimeError(f"RFECV results are missing: {sorted(missing)}")
    split_keys = sorted(
        (key for key in results if re.fullmatch(r"split\d+_test_score", key)),
        key=lambda key: int(re.search(r"\d+", key).group()),
    )
    rows: list[dict[str, Any]] = []
    for index, feature_count in enumerate(results["n_features"]):
        row: dict[str, Any] = {
            "configuration_id": configuration_id,
            "configuration_label": CONFIGURATIONS[configuration_id]["label"],
            "class_weight": CONFIGURATIONS[configuration_id]["class_weight"] or "None",
            "fit_scope": fit_scope,
            "outer_fold": "" if outer_fold is None else int(outer_fold),
            "number_of_retained_features": int(feature_count),
            "mean_inner_cv_macro_f1": float(results["mean_test_score"][index]),
            "std_inner_cv_macro_f1": float(results["std_test_score"][index]),
            "minimum_features_to_select": int(minimum_features),
            "rfecv_step": rfecv_step,
            "selected_as_optimal": int(feature_count) == int(selector.n_features_),
        }
        for fold_number, key in enumerate(split_keys, start=1):
            row[f"inner_cv_macro_f1_fold_{fold_number}"] = float(results[key][index])
        rows.append(row)
    curve = pd.DataFrame(rows).sort_values("number_of_retained_features").reset_index(drop=True)
    if int(curve["selected_as_optimal"].sum()) != 1:
        raise RuntimeError("Each fitted RFECV curve must identify exactly one optimum.")
    if int(curve["number_of_retained_features"].min()) != minimum_features:
        raise RuntimeError("RFECV did not evaluate the exact proportional minimum.")
    return curve


def run_nested_cross_validation(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    minimum_features: int = EXPECTED_MINIMUM_FEATURE_COUNT,
    rfecv_step: int | float = DEFAULT_RFECV_STEP,
    inner_cv_splits: int = INNER_CV_SPLITS,
    n_estimators: int = 100,
    estimator_n_jobs: int = -1,
    selector_n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    """Evaluate C1/C2 with RFECV refitted inside every outer training fold."""
    metric_names = (
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "macro_precision",
        "macro_recall",
    )
    rows: list[dict[str, Any]] = []
    curve_tables: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, Any]] = {}
    for configuration_id, configuration in CONFIGURATIONS.items():
        fold_rows: list[dict[str, Any]] = []
        for fold_number, (training_rows, validation_rows) in enumerate(
            outer_folds, start=1
        ):
            pipeline = build_pipeline(
                configuration["class_weight"],
                minimum_features=minimum_features,
                rfecv_step=rfecv_step,
                inner_cv_splits=inner_cv_splits,
                n_estimators=n_estimators,
                estimator_n_jobs=estimator_n_jobs,
                selector_n_jobs=selector_n_jobs,
            )
            pipeline.fit(matrix[training_rows], labels[training_rows])
            selector: RFECV = pipeline.named_steps["feature_selection"]
            classifier: RandomForestClassifier = pipeline.named_steps["classifier"]
            if int(classifier.n_features_in_) != int(selector.n_features_):
                raise RuntimeError(
                    f"{configuration_id} fold {fold_number} selector/classifier mismatch."
                )
            predicted = pipeline.predict(matrix[validation_rows])
            metrics = BASE.calculate_metrics(labels[validation_rows], predicted)
            row = {
                "configuration_id": configuration_id,
                "configuration_label": configuration["label"],
                "class_weight": configuration["class_weight"] or "None",
                "record_type": "fold",
                "fold": fold_number,
                "training_records": len(training_rows),
                "validation_records": len(validation_rows),
                "rfecv_selected_features": int(selector.n_features_),
                **metrics,
                "selected_configuration": False,
            }
            rows.append(row)
            fold_rows.append(row)
            curve_tables.append(
                extract_rfecv_curve(
                    selector,
                    configuration_id=configuration_id,
                    fit_scope="outer_training_fold",
                    outer_fold=fold_number,
                    minimum_features=minimum_features,
                    rfecv_step=rfecv_step,
                )
            )
        summary: dict[str, Any] = {}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in fold_rows], dtype=float)
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1))
        counts = np.asarray(
            [row["rfecv_selected_features"] for row in fold_rows], dtype=int
        )
        summary.update(
            {
                "outer_fold_selected_feature_counts": counts.tolist(),
                "mean_outer_fold_selected_features": float(counts.mean()),
                "minimum_outer_fold_selected_features": int(counts.min()),
                "maximum_outer_fold_selected_features": int(counts.max()),
            }
        )
        summaries[configuration_id] = summary
        for record_type in ("mean", "std"):
            rows.append(
                {
                    "configuration_id": configuration_id,
                    "configuration_label": configuration["label"],
                    "class_weight": configuration["class_weight"] or "None",
                    "record_type": record_type,
                    "fold": "all",
                    "training_records": "",
                    "validation_records": "",
                    "rfecv_selected_features": (
                        float(counts.mean())
                        if record_type == "mean"
                        else float(counts.std(ddof=1))
                    ),
                    **{
                        metric: summary[f"{record_type}_{metric}"]
                        for metric in metric_names
                    },
                    "selected_configuration": False,
                }
            )
    return pd.DataFrame(rows), summaries, pd.concat(curve_tables, ignore_index=True)


def select_configuration(
    summaries: dict[str, dict[str, Any]],
    decimal_places: int = STORED_METRIC_DECIMALS,
) -> str:
    """Select greater outer-CV macro-F1, choosing C1 on a stored-precision tie."""
    c1 = round(float(summaries["C1"]["mean_macro_f1"]), decimal_places)
    c2 = round(float(summaries["C2"]["mean_macro_f1"]), decimal_places)
    return "C1" if c1 >= c2 else "C2"


def compact_rfecv_results(selector: RFECV) -> None:
    """Discard bulky per-feature fold diagnostics that are not needed for prediction."""
    selector.cv_results_ = {
        key: value
        for key, value in selector.cv_results_.items()
        if not key.endswith("_support") and not key.endswith("_ranking")
    }


def dataframe_records_for_json(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas missing values to JSON null without changing CSV output."""
    object_data = data.astype(object).where(pd.notna(data), None)
    return object_data.to_dict(orient="records")


def build_selected_features(
    features: pd.DataFrame,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Map the complete-training RFECV support to registered TF-IDF metadata."""
    selector: RFECV = pipeline.named_steps["feature_selection"]
    classifier: RandomForestClassifier = pipeline.named_steps["classifier"]
    indices = np.flatnonzero(selector.support_)
    importances = np.asarray(classifier.feature_importances_, dtype=float)
    if len(indices) != int(selector.n_features_) or len(importances) != len(indices):
        raise RuntimeError("Final RFECV feature metadata is inconsistent.")
    output = features.iloc[indices].copy()
    output["feature_index"] = output["feature_index"].astype(int)
    output["ngram_length"] = output["ngram_length"].astype(int)
    output["idf_value"] = output["idf_value"].astype(float)
    output = output.rename(columns={"feature_index": "original_tfidf_feature_index"})
    output.insert(0, "model_feature_position", np.arange(1, len(output) + 1))
    output.insert(1, "final_random_forest_importance", importances)
    output.insert(2, "importance_rank", pd.Series(importances).rank(method="first", ascending=False).astype(int).to_numpy())
    output.insert(6, "feature_selection_method", "complete-training RFECV")
    output.insert(7, "rfecv_ranking", selector.ranking_[indices].astype(int))
    return output.sort_values("importance_rank").reset_index(drop=True).loc[
        :,
        [
            "importance_rank",
            "model_feature_position",
            "original_tfidf_feature_index",
            "ngram_text",
            "ngram_length",
            "feature_selection_method",
            "rfecv_ranking",
            "final_random_forest_importance",
            "idf_value",
        ],
    ]


def build_evaluation_outputs(
    configuration_id: str,
    pipeline: Pipeline,
    matrix: sparse.csr_matrix,
    test_index: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one complete-training Model C pipeline on held-out data."""
    configuration = CONFIGURATIONS[configuration_id]
    actual = test_index["reviewer_gold_label"].map(str).to_numpy()
    predicted = pipeline.predict(matrix).astype(str)
    overall = BASE.calculate_metrics(actual, predicted)
    correct = actual == predicted
    overall.update(
        {
            "correct_count": int(correct.sum()),
            "incorrect_count": int((~correct).sum()),
            "correct_proportion": float(correct.mean()),
            "incorrect_proportion": float((~correct).mean()),
            "test_records": int(len(actual)),
        }
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASS_ORDER, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "configuration_id": configuration_id,
            "configuration_label": configuration["label"],
            "class": CLASS_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    values = confusion_matrix(actual, predicted, labels=CLASS_ORDER)
    confusion = pd.DataFrame(
        values, columns=[f"predicted_{label}" for label in CLASS_ORDER]
    )
    confusion.insert(0, "configuration_id", configuration_id)
    confusion.insert(1, "configuration_label", configuration["label"])
    confusion.insert(2, "actual_label", CLASS_ORDER)
    confusion["actual_total"] = values.sum(axis=1)
    predictions = pd.DataFrame(
        {
            "configuration_id": configuration_id,
            "configuration_label": configuration["label"],
            "matrix_row_index": test_index["matrix_row_index"].astype(int),
            "candidate_id": test_index["candidate_id"].map(str),
            "candidate_text_original": test_index["candidate_text_original"].map(str),
            "gold_label": actual,
            "predicted_label": predicted,
            "correct": correct,
        }
    )
    descriptions = {
        "macro_f1": ("principal metric", "Unweighted mean class-specific F1."),
        "weighted_f1": ("secondary metric", "Support-weighted mean class-specific F1."),
        "accuracy": ("secondary metric", "Proportion classified correctly."),
        "macro_precision": ("secondary metric", "Unweighted mean class precision."),
        "macro_recall": ("secondary metric", "Unweighted mean class recall."),
        "correct_count": ("count", "Number of correct held-out predictions."),
        "incorrect_count": ("count", "Number of incorrect held-out predictions."),
        "correct_proportion": ("proportion", "Proportion of correct predictions."),
        "incorrect_proportion": ("proportion", "Proportion of incorrect predictions."),
        "test_records": ("count", "Total held-out statements evaluated."),
    }
    test_metrics = pd.DataFrame(
        [
            {
                "configuration_id": configuration_id,
                "configuration_label": configuration["label"],
                "metric": metric,
                "value": overall[metric],
                "role": role,
                "description": description,
            }
            for metric, (role, description) in descriptions.items()
        ]
    )
    return overall, test_metrics, per_class, confusion, predictions


def load_prior_model_reference(
    database_path: Path,
    vectorization_id: str,
) -> dict[str, Any]:
    """Load authoritative paired A/B CV and held-out results for comparison."""
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        model_b_row = connection.execute(
            """
            SELECT * FROM model_events
            WHERE vectorization_id = ?
              AND model_designation = 'Model B'
              AND model_status = 'Completed'
            ORDER BY created_at DESC, model_id DESC
            LIMIT 1
            """,
            (vectorization_id,),
        ).fetchone()
        if model_b_row is None:
            raise ValueError("No completed Model B event exists for this vectorization.")
        model_b_event = dict(model_b_row)
        b_cv = json.loads(model_b_event["cross_validation_results_json"])
        model_a_id = str(b_cv["model_a_reference_id"])
        model_a_row = connection.execute(
            "SELECT * FROM model_events WHERE model_id = ?", (model_a_id,)
        ).fetchone()
    if model_a_row is None:
        raise ValueError(f"Registered Model A comparison was not found: {model_a_id}")
    model_a_event = dict(model_a_row)
    prior_cv = pd.DataFrame(b_cv["four_model_comparison_records"])
    if set(prior_cv["configuration_id"]) != {"A1", "A2", "B1", "B2"}:
        raise RuntimeError("Registered prior CV comparison is incomplete.")
    held_out: dict[str, dict[str, Any]] = {}
    for event in (model_a_event, model_b_event):
        metrics = json.loads(event["final_test_metrics_json"])
        for configuration_id, values in metrics["configurations"].items():
            held_out[configuration_id] = {
                "source_model_event_id": event["model_id"],
                "configuration_label": values["configuration_label"],
                "overall": values["overall"],
            }
    if set(held_out) != {"A1", "A2", "B1", "B2"}:
        raise RuntimeError("Registered prior held-out comparison is incomplete.")
    return {
        "model_a_id": model_a_id,
        "model_b_id": model_b_event["model_id"],
        "prior_cv_records": prior_cv,
        "held_out": held_out,
        "outer_cv_design": json.loads(model_b_event["cross_validation_design_json"]),
    }


def build_six_model_cv_comparison(
    *,
    prior_reference: dict[str, Any],
    model_c_id: str,
    model_c_results: pd.DataFrame,
    selected_configuration: str,
) -> pd.DataFrame:
    """Combine registered A/B folds with newly computed nested Model C folds."""
    prior = prior_reference["prior_cv_records"].copy()
    current = model_c_results.copy()
    current.insert(0, "source_model_event_id", model_c_id)
    current.insert(
        0,
        "feature_scope",
        "RFECV-selected from all 23,464 TF-IDF features",
    )
    current.insert(0, "model_family", "C")
    current.loc[
        current["configuration_id"].eq(selected_configuration),
        "selected_configuration",
    ] = True
    combined = pd.concat([prior, current], ignore_index=True, sort=False)
    expected = {"A1", "A2", "B1", "B2", "C1", "C2"}
    if set(combined["configuration_id"]) != expected:
        raise RuntimeError("Six-configuration CV comparison is incomplete.")
    fold_rows = combined.loc[combined["record_type"].eq("fold")]
    for fold_number in range(1, OUTER_CV_SPLITS + 1):
        fold = fold_rows.loc[fold_rows["fold"].astype(int).eq(fold_number)]
        if len(fold) != len(expected):
            raise RuntimeError(f"Six-model comparison fold {fold_number} is incomplete.")
        if fold["training_records"].astype(int).nunique() != 1:
            raise RuntimeError(f"Training counts differ in fold {fold_number}.")
        if fold["validation_records"].astype(int).nunique() != 1:
            raise RuntimeError(f"Validation counts differ in fold {fold_number}.")
    return combined


def build_posthoc_test_comparison(
    *,
    prior_reference: dict[str, Any],
    model_c_id: str,
    current_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine registered A/B test metrics with paired C1/C2 post-hoc results."""
    rows: list[dict[str, Any]] = []
    for configuration_id in ("A1", "A2", "B1", "B2"):
        values = prior_reference["held_out"][configuration_id]
        for metric, value in values["overall"].items():
            rows.append(
                {
                    "source_model_event_id": values["source_model_event_id"],
                    "configuration_id": configuration_id,
                    "configuration_label": values["configuration_label"],
                    "evaluation_scope": "registered prior held-out result",
                    "metric": metric,
                    "value": value,
                    "role": "prior comparison",
                    "description": "Previously registered result on the same test partition.",
                }
            )
    current = current_metrics.copy()
    current.insert(0, "evaluation_scope", "post-hoc Model C held-out result")
    current.insert(0, "source_model_event_id", model_c_id)
    rows.extend(current.to_dict(orient="records"))
    comparison = pd.DataFrame(rows)
    if set(comparison["configuration_id"]) != {"A1", "A2", "B1", "B2", "C1", "C2"}:
        raise RuntimeError("Post-hoc test comparison is incomplete.")
    return comparison


def plot_feature_curve(
    curve_data: pd.DataFrame,
    path: Path,
    *,
    minimum_features: int,
    model_a_features: int = MODEL_A_FEATURE_REFERENCE,
) -> None:
    """Plot complete-training RFECV curves without any held-out information."""
    plot_data = curve_data.loc[curve_data["fit_scope"].eq("complete_training")].copy()
    if set(plot_data["configuration_id"]) != set(CONFIGURATIONS):
        raise RuntimeError("Complete-training RFECV curves are missing a configuration.")
    colors = {"C1": "#2F5597", "C2": "#C55A11"}
    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=180)
    optimal_points: list[tuple[str, float, float]] = []
    for configuration_id in CONFIGURATIONS:
        frame = plot_data.loc[
            plot_data["configuration_id"].eq(configuration_id)
        ].sort_values("number_of_retained_features")
        x = frame["number_of_retained_features"].to_numpy(dtype=float)
        mean = frame["mean_inner_cv_macro_f1"].to_numpy(dtype=float)
        std = frame["std_inner_cv_macro_f1"].to_numpy(dtype=float)
        axis.plot(
            x,
            mean,
            color=colors[configuration_id],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"{configuration_id}: {CONFIGURATIONS[configuration_id]['label']}",
        )
        axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=colors[configuration_id],
            alpha=0.14,
            linewidth=0,
        )
        optimum = frame.loc[frame["selected_as_optimal"].astype(bool)].iloc[0]
        optimal_x = float(optimum["number_of_retained_features"])
        optimal_y = float(optimum["mean_inner_cv_macro_f1"])
        axis.scatter(
            [optimal_x],
            [optimal_y],
            s=70,
            color=colors[configuration_id],
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        optimal_points.append((configuration_id, optimal_x, optimal_y))
    axis.axvline(
        minimum_features,
        color="#6B7280",
        linestyle=":",
        linewidth=1.7,
        label=f"Proportional minimum ({minimum_features})",
    )
    axis.axvline(
        model_a_features,
        color="#374151",
        linestyle="--",
        linewidth=1.5,
        label=f"Model A reference ({model_a_features})",
    )
    for point_index, (configuration_id, optimal_x, optimal_y) in enumerate(optimal_points):
        vertical_offset = 18 if point_index == 0 else -30
        axis.annotate(
            f"{configuration_id} optimum: {int(optimal_x):,} features",
            xy=(optimal_x, optimal_y),
            xytext=(22, vertical_offset),
            textcoords="offset points",
            fontsize=9,
            color=colors[configuration_id],
            arrowprops={"arrowstyle": "->", "color": colors[configuration_id], "lw": 1.0},
        )
    axis.set_title("Model C RFECV training curves", fontsize=14, weight="semibold", pad=12)
    axis.set_xlabel("Number of retained TF-IDF features")
    axis.set_ylabel("Mean inner-CV macro-F1")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
    axis.set_xlim(left=0, right=float(plot_data["number_of_retained_features"].max()) * 1.02)
    y_min = float((plot_data["mean_inner_cv_macro_f1"] - plot_data["std_inner_cv_macro_f1"]).min())
    y_max = float((plot_data["mean_inner_cv_macro_f1"] + plot_data["std_inner_cv_macro_f1"]).max())
    padding = max((y_max - y_min) * 0.14, 0.015)
    axis.set_ylim(max(0, y_min - padding), min(1, y_max + padding))
    axis.tick_params(axis="both", labelsize=9)
    axis.legend(loc="best", frameon=True, framealpha=0.95, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def model_summary_table(
    *,
    model_id: str,
    vector_event: dict[str, Any],
    prior_reference: dict[str, Any],
    created_at: str,
    cv_summaries: dict[str, dict[str, Any]],
    selected_configuration: str,
    optimal_feature_counts: dict[str, int],
    test_results: dict[str, dict[str, Any]],
    training_records: int,
    test_records: int,
    vocabulary_size: int,
    minimum_features: int,
    rfecv_step: int | float,
) -> pd.DataFrame:
    """Build Model C's protocol-oriented summary sheet."""
    rows = [
        ("identity", "model_id", model_id, "Append-only Model C event."),
        ("identity", "model_name", MODEL_NAME, "Protocol model name."),
        ("identity", "model_date", created_at[:10], "UTC execution date."),
        ("lineage", "matching_id", vector_event["matching_id"], "Reviewed alignment."),
        ("lineage", "split_id", vector_event["split_id"], "Registered split."),
        ("lineage", "vectorization_id", vector_event["vectorization_id"], "TF-IDF event."),
        ("comparison", "model_a_reference_id", prior_reference["model_a_id"], "Registered paired A source."),
        ("comparison", "model_b_reference_id", prior_reference["model_b_id"], "Registered paired B source."),
        ("data", "training_records", training_records, "Complete training partition."),
        ("data", "test_records", test_records, "Existing held-out test partition."),
        ("features", "initial_tfidf_features", vocabulary_size, "Complete registered representation supplied to RFECV."),
        ("features", "reference_minimum_fraction", REFERENCE_MINIMUM_PROPORTION, "20 / 3,770 = 0.5305%."),
        ("features", "proportional_minimum_features", minimum_features, "Smallest subset RFECV may consider; not a fixed final count."),
        ("features", "current_minimum_fraction", minimum_features / vocabulary_size, "124 / 23,464 approximately 0.5285%."),
        ("features", "rfecv_step", rfecv_step, "Elimination step registered for computational feasibility."),
        ("features", "C1_complete_training_optimum", optimal_feature_counts["C1"], "Maximizes complete-training inner-CV macro-F1 for C1."),
        ("features", "C2_complete_training_optimum", optimal_feature_counts["C2"], "Maximizes complete-training inner-CV macro-F1 for C2."),
        ("cv", "C1_mean_outer_cv_macro_f1", cv_summaries["C1"]["mean_macro_f1"], "Nested outer-CV result."),
        ("cv", "C2_mean_outer_cv_macro_f1", cv_summaries["C2"]["mean_macro_f1"], "Nested outer-CV result."),
        ("cv", "selected_configuration", selected_configuration, "Selected by outer training-only macro-F1."),
        ("test", "held_out_reporting_scope", "C1 and C2", f"Both complete-training configurations are evaluated post hoc; {selected_configuration} remains selected by outer CV."),
    ]
    for configuration_id in CONFIGURATIONS:
        for metric in ("macro_f1", "weighted_f1", "accuracy"):
            rows.append(
                (
                    "test",
                    f"{configuration_id}_test_{metric}",
                    test_results[configuration_id][metric],
                    f"Post-hoc held-out {metric.replace('_', ' ')} for {configuration_id}.",
                )
            )
    rows.extend(
        [
            ("method", "adaptation_basis", "Wroblewska released classifier", "RFECV adaptation of the released feature-selection procedure."),
            ("method", "publication_limit", "minimum 20 not justified in publication", "The released implementation contains the minimum; the publication did not describe or justify it."),
        ]
    )
    return pd.DataFrame(rows, columns=["section", "item", "value", "description"])


def artifact_descriptions_table() -> pd.DataFrame:
    """Return all eight Model C artifact descriptions."""
    files = {
        "fitted_model": MODEL_FILENAME,
        "cross_validation_results": CV_RESULTS_FILENAME,
        "selected_features": SELECTED_FEATURES_FILENAME,
        "test_predictions": TEST_PREDICTIONS_FILENAME,
        "confusion_matrix": CONFUSION_MATRIX_FILENAME,
        "performance_report": PERFORMANCE_REPORT_FILENAME,
        "rfecv_feature_curve": FEATURE_CURVE_FILENAME,
        "rfecv_feature_plot": FEATURE_CURVE_PLOT_FILENAME,
    }
    return pd.DataFrame(
        [
            {
                "filename": files[role],
                "role": role,
                "description": ARTIFACT_DESCRIPTIONS[role],
            }
            for role in files
        ]
    )


def write_performance_workbook(
    path: Path,
    sheets: dict[str, pd.DataFrame],
    feature_curve_plot_path: Path,
) -> None:
    """Write the nine-sheet Model C report and embed the training-only curve plot."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = MODEL_NAME
    workbook.properties.subject = "Nested RFECV cross-validation and held-out evaluation"
    workbook.properties.creator = "IG protocol project"
    table_names = {
        "Model_Summary": "TblModelCSummary",
        "CV_Comparison": "TblSixModelCV",
        "Test_Metrics": "TblModelCTestMetrics",
        "Per_Class_Metrics": "TblModelCPerClass",
        "Confusion_Matrix": "TblModelCConfusion",
        "Selected_Features": "TblModelCFeatures",
        "Test_Predictions": "TblModelCPredictions",
        "RFECV_Feature_Curve": "TblModelCFeatureCurve",
        "Artifact_Descriptions": "TblModelCArtifacts",
    }
    for title, data in sheets.items():
        BASE.add_dataframe_sheet(workbook, title, data, table_names[title])
    curve_sheet = workbook["RFECV_Feature_Curve"]
    chart_column = max(len(sheets["RFECV_Feature_Curve"].columns) + 2, 18)
    chart_anchor = f"{BASE.get_column_letter(chart_column)}2"
    image_buffer = BytesIO(feature_curve_plot_path.read_bytes())
    image = ExcelImage(image_buffer)
    image.width = 900
    image.height = 550
    curve_sheet.add_image(image, chart_anchor)
    curve_sheet.sheet_properties.pageSetUpPr.fitToPage = True
    curve_sheet.page_setup.fitToWidth = 1
    curve_sheet.page_setup.fitToHeight = 0
    try:
        workbook.save(path)
    finally:
        image_buffer.close()


def write_outputs(
    *,
    output_dir: Path,
    model_bundle: dict[str, Any],
    cv_results: pd.DataFrame,
    selected_features: pd.DataFrame,
    predictions: pd.DataFrame,
    confusion: pd.DataFrame,
    curve_data: pd.DataFrame,
    workbook_sheets: dict[str, pd.DataFrame],
    minimum_features: int,
) -> dict[str, Path]:
    """Write all eight Model C artifacts to a staging directory."""
    paths = {
        "fitted_model": output_dir / MODEL_FILENAME,
        "cross_validation_results": output_dir / CV_RESULTS_FILENAME,
        "selected_features": output_dir / SELECTED_FEATURES_FILENAME,
        "test_predictions": output_dir / TEST_PREDICTIONS_FILENAME,
        "confusion_matrix": output_dir / CONFUSION_MATRIX_FILENAME,
        "performance_report": output_dir / PERFORMANCE_REPORT_FILENAME,
        "rfecv_feature_curve": output_dir / FEATURE_CURVE_FILENAME,
        "rfecv_feature_plot": output_dir / FEATURE_CURVE_PLOT_FILENAME,
    }
    joblib.dump(model_bundle, paths["fitted_model"], compress=3)
    cv_results.to_csv(paths["cross_validation_results"], index=False, lineterminator="\n")
    selected_features.to_csv(paths["selected_features"], index=False, lineterminator="\n")
    predictions.to_csv(paths["test_predictions"], index=False, lineterminator="\n")
    confusion.to_csv(paths["confusion_matrix"], index=False, lineterminator="\n")
    curve_data.to_csv(paths["rfecv_feature_curve"], index=False, lineterminator="\n")
    plot_feature_curve(
        curve_data,
        paths["rfecv_feature_plot"],
        minimum_features=minimum_features,
    )
    write_performance_workbook(
        paths["performance_report"],
        workbook_sheets,
        paths["rfecv_feature_plot"],
    )
    return paths


def validate_outputs(
    *,
    paths: dict[str, Path],
    test_matrix: sparse.csr_matrix,
    expected_predictions: pd.DataFrame,
    expected_selected_features: pd.DataFrame,
    expected_confusion: pd.DataFrame,
    expected_curve_data: pd.DataFrame,
    expected_sheet_names: list[str],
    selected_configuration: str,
    minimum_features: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    """Read back and validate every Model C output artifact."""
    bundle = joblib.load(paths["fitted_model"])
    loaded_predictions = pd.read_csv(paths["test_predictions"], dtype=object, keep_default_na=False)
    loaded_features = pd.read_csv(paths["selected_features"], dtype=object, keep_default_na=False)
    loaded_confusion = pd.read_csv(paths["confusion_matrix"], dtype=object, keep_default_na=False)
    loaded_cv = pd.read_csv(paths["cross_validation_results"], keep_default_na=False)
    loaded_curve = pd.read_csv(paths["rfecv_feature_curve"], keep_default_na=False)
    workbook = load_workbook(paths["performance_report"], read_only=False, data_only=True)
    try:
        sheet_names = workbook.sheetnames
        curve_sheet_image_count = len(workbook["RFECV_Feature_Curve"]._images)
    finally:
        workbook.close()
    with PillowImage.open(paths["rfecv_feature_plot"]) as image:
        plot_dimensions = image.size
        plot_format = image.format

    pipelines: dict[str, Pipeline] = bundle["pipelines"]
    selected_pipeline = pipelines[selected_configuration]
    reproduced = {
        configuration_id: pipeline.predict(test_matrix).astype(str).tolist()
        for configuration_id, pipeline in pipelines.items()
    }
    expected = {
        configuration_id: expected_predictions.loc[
            expected_predictions["configuration_id"].eq(configuration_id),
            "predicted_label",
        ].map(str).tolist()
        for configuration_id in CONFIGURATIONS
    }
    selector: RFECV = selected_pipeline.named_steps["feature_selection"]
    confusion_columns = [f"predicted_{label}" for label in CLASS_ORDER]
    numeric_curve_columns = [
        "number_of_retained_features",
        "mean_inner_cv_macro_f1",
        "std_inner_cv_macro_f1",
    ] + [
        column for column in loaded_curve if column.startswith("inner_cv_macro_f1_fold_")
    ]
    group_columns = ["configuration_id", "fit_scope", "outer_fold"]
    optimal_per_fit = (
        loaded_curve.assign(
            selected_bool=loaded_curve["selected_as_optimal"].map(
                lambda value: str(value).strip().lower() in {"true", "1"}
            )
        )
        .groupby(group_columns, dropna=False)["selected_bool"]
        .sum()
    )
    full_curve = loaded_curve.loc[loaded_curve["fit_scope"].eq("complete_training")]
    checks = {
        "all_eight_outputs_exist_and_nonempty": len(paths) == 8
        and all(path.is_file() and path.stat().st_size > 0 for path in paths.values()),
        "bundle_contains_both_complete_training_pipelines": set(pipelines) == set(CONFIGURATIONS),
        "bundle_selected_configuration_matches": bundle["selected_configuration"] == selected_configuration,
        "bundle_class_order_preserved": tuple(bundle["class_order"]) == CLASS_ORDER,
        "both_reloaded_pipelines_reproduce_predictions": reproduced == expected,
        "both_configurations_have_test_predictions": set(loaded_predictions["configuration_id"]) == set(CONFIGURATIONS),
        "every_test_statement_predicted_once_per_configuration": len(loaded_predictions) == len(CONFIGURATIONS) * test_matrix.shape[0],
        "prediction_labels_approved": set(loaded_predictions["predicted_label"]).issubset(CLASS_ORDER),
        "selected_feature_count_matches_pipeline": len(loaded_features) == int(selector.n_features_),
        "selected_feature_indices_match": loaded_features["original_tfidf_feature_index"].astype(int).tolist() == expected_selected_features["original_tfidf_feature_index"].astype(int).tolist(),
        "selected_feature_count_at_least_minimum": int(selector.n_features_) >= minimum_features,
        "selector_started_from_complete_vocabulary": int(selector.n_features_in_) == vocabulary_size,
        "confusion_matches_memory": np.array_equal(loaded_confusion[confusion_columns].astype(int).to_numpy(), expected_confusion[confusion_columns].astype(int).to_numpy()),
        "each_confusion_total_matches_test_rows": all(
            int(
                loaded_confusion.loc[
                    loaded_confusion["configuration_id"].eq(configuration_id),
                    confusion_columns,
                ].astype(int).to_numpy().sum()
            )
            == test_matrix.shape[0]
            for configuration_id in CONFIGURATIONS
        ),
        "cv_contains_six_configurations": set(loaded_cv["configuration_id"]) == {"A1", "A2", "B1", "B2", "C1", "C2"},
        "cv_contains_thirty_fold_rows": int(loaded_cv["record_type"].eq("fold").sum()) == 6 * OUTER_CV_SPLITS,
        "curve_contains_both_configurations": set(loaded_curve["configuration_id"]) == set(CONFIGURATIONS),
        "curve_contains_outer_and_complete_fits": set(loaded_curve["fit_scope"]) == {"outer_training_fold", "complete_training"},
        "curve_has_twelve_fitted_selectors": len(optimal_per_fit) == len(CONFIGURATIONS) * (OUTER_CV_SPLITS + 1),
        "curve_has_one_optimum_per_fit": bool((optimal_per_fit == 1).all()),
        "curve_includes_exact_minimum_per_fit": bool(loaded_curve.groupby(group_columns, dropna=False)["number_of_retained_features"].min().eq(minimum_features).all()),
        "complete_curves_include_full_vocabulary": bool(full_curve.groupby("configuration_id")["number_of_retained_features"].max().eq(vocabulary_size).all()),
        "curve_numeric_values_finite": bool(np.isfinite(loaded_curve[numeric_curve_columns].astype(float).to_numpy()).all()),
        "curve_round_trip_row_count": len(loaded_curve) == len(expected_curve_data),
        "plot_is_readable_png": plot_format == "PNG" and plot_dimensions[0] >= 1200 and plot_dimensions[1] >= 700,
        "workbook_has_expected_sheets_in_order": sheet_names == expected_sheet_names,
        "workbook_curve_sheet_contains_chart_image": curve_sheet_image_count == 1,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Model C output validation failed: " + ", ".join(failed))
    return {"status": "passed", "checks": checks}


def artifact_rows(
    *,
    model_id: str,
    final_dir: Path,
    staged_paths: dict[str, Path],
    created_at: str,
) -> list[dict[str, Any]]:
    """Build artifact registry rows from validated staged files."""
    return [
        {
            "model_id": model_id,
            "artifact_role": role,
            "file_name": path.name,
            "artifact_path": BASE.project_relative(final_dir / path.name),
            "artifact_hash_sha256": BASE.sha256_file(path),
            "size_bytes": path.stat().st_size,
            "artifact_description": ARTIFACT_DESCRIPTIONS[role],
            "created_at": created_at,
        }
        for role, path in staged_paths.items()
    ]


def run_model_c(
    *,
    database_path: Path,
    output_base_dir: Path,
    vectorization_id: str | None = None,
    rfecv_step: int | float = DEFAULT_RFECV_STEP,
) -> dict[str, Any]:
    """Validate, train, evaluate, export, publish, and register Model C."""
    created_at = utc_now_iso()
    schema_migration = ensure_flexible_model_registry_schema(database_path)
    vector_event, vector_artifacts = BASE.load_vectorization_event(database_path, vectorization_id)
    inputs = BASE.load_and_validate_inputs(vector_event, vector_artifacts)
    train_matrix: sparse.csr_matrix = inputs["train_matrix"]
    test_matrix: sparse.csr_matrix = inputs["test_matrix"]
    train_index: pd.DataFrame = inputs["train_index"]
    test_index: pd.DataFrame = inputs["test_index"]
    features: pd.DataFrame = inputs["features"]
    vocabulary_size = int(train_matrix.shape[1])
    if vocabulary_size != EXPECTED_VOCABULARY_SIZE:
        raise ValueError(f"Model C expects {EXPECTED_VOCABULARY_SIZE:,} features; found {vocabulary_size:,}.")
    minimum_features = proportional_minimum_feature_count(vocabulary_size)
    if minimum_features != EXPECTED_MINIMUM_FEATURE_COUNT:
        raise RuntimeError("The proportional minimum did not reproduce 124 features.")

    prior_reference = load_prior_model_reference(database_path, vector_event["vectorization_id"])
    labels = train_index["reviewer_gold_label"].map(str).to_numpy()
    outer_folds, fold_validation = BASE.build_cv_folds(labels)
    c_cv_results, c_cv_summaries, outer_curves = run_nested_cross_validation(
        train_matrix,
        labels,
        outer_folds,
        minimum_features=minimum_features,
        rfecv_step=rfecv_step,
    )
    selected_configuration = select_configuration(c_cv_summaries)
    c_cv_results.loc[c_cv_results["configuration_id"].eq(selected_configuration), "selected_configuration"] = True

    model_id = BASE.next_model_id(database_path)
    combined_cv = build_six_model_cv_comparison(
        prior_reference=prior_reference,
        model_c_id=model_id,
        model_c_results=c_cv_results,
        selected_configuration=selected_configuration,
    )

    final_curves: list[pd.DataFrame] = []
    optimal_feature_counts: dict[str, int] = {}
    final_pipelines: dict[str, Pipeline] = {}
    for configuration_id, configuration in CONFIGURATIONS.items():
        pipeline = build_pipeline(
            configuration["class_weight"],
            minimum_features=minimum_features,
            rfecv_step=rfecv_step,
        )
        pipeline.fit(train_matrix, labels)
        selector: RFECV = pipeline.named_steps["feature_selection"]
        optimal_feature_counts[configuration_id] = int(selector.n_features_)
        final_curves.append(
            extract_rfecv_curve(
                selector,
                configuration_id=configuration_id,
                fit_scope="complete_training",
                outer_fold=None,
                minimum_features=minimum_features,
                rfecv_step=rfecv_step,
            )
        )
        compact_rfecv_results(selector)
        final_pipelines[configuration_id] = pipeline
    selected_pipeline = final_pipelines[selected_configuration]
    curve_data = pd.concat([outer_curves, *final_curves], ignore_index=True)
    curve_data = curve_data.sort_values(
        ["configuration_id", "fit_scope", "outer_fold", "number_of_retained_features"]
    ).reset_index(drop=True)

    test_results: dict[str, dict[str, Any]] = {}
    metric_tables: list[pd.DataFrame] = []
    per_class_tables: list[pd.DataFrame] = []
    confusion_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    for configuration_id in CONFIGURATIONS:
        overall, metrics, classes, matrix_table, configuration_predictions = (
            build_evaluation_outputs(
                configuration_id,
                final_pipelines[configuration_id],
                test_matrix,
                test_index,
            )
        )
        test_results[configuration_id] = overall
        metric_tables.append(metrics)
        per_class_tables.append(classes)
        confusion_tables.append(matrix_table)
        prediction_tables.append(configuration_predictions)
    current_test_metrics = pd.concat(metric_tables, ignore_index=True)
    per_class = pd.concat(per_class_tables, ignore_index=True)
    confusion = pd.concat(confusion_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    selected_features = build_selected_features(features, selected_pipeline)
    test_comparison = build_posthoc_test_comparison(
        prior_reference=prior_reference,
        model_c_id=model_id,
        current_metrics=current_test_metrics,
    )

    output_base_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_base_dir / model_id
    staged_dir = Path(tempfile.mkdtemp(prefix=f".{model_id}_", dir=output_base_dir))
    model_bundle = {
        "artifact_type": "paired_model_c_nested_rfecv_bundle",
        "model_id": model_id,
        "script_version": SCRIPT_VERSION,
        "class_order": list(CLASS_ORDER),
        "initial_feature_count": vocabulary_size,
        "minimum_features_to_select": minimum_features,
        "rfecv_step": rfecv_step,
        "selected_configuration": selected_configuration,
        "complete_training_optimal_feature_counts": optimal_feature_counts,
        "test_scores_used_for_selection": False,
        "pipelines": final_pipelines,
        "pipeline": selected_pipeline,
    }
    summary = model_summary_table(
        model_id=model_id,
        vector_event=vector_event,
        prior_reference=prior_reference,
        created_at=created_at,
        cv_summaries=c_cv_summaries,
        selected_configuration=selected_configuration,
        optimal_feature_counts=optimal_feature_counts,
        test_results=test_results,
        training_records=len(train_index),
        test_records=len(test_index),
        vocabulary_size=vocabulary_size,
        minimum_features=minimum_features,
        rfecv_step=rfecv_step,
    )
    workbook_sheets = {
        "Model_Summary": summary,
        "CV_Comparison": combined_cv,
        "Test_Metrics": test_comparison,
        "Per_Class_Metrics": per_class,
        "Confusion_Matrix": confusion,
        "Selected_Features": selected_features,
        "Test_Predictions": predictions,
        "RFECV_Feature_Curve": curve_data,
        "Artifact_Descriptions": artifact_descriptions_table(),
    }
    try:
        staged_paths = write_outputs(
            output_dir=staged_dir,
            model_bundle=model_bundle,
            cv_results=combined_cv,
            selected_features=selected_features,
            predictions=predictions,
            confusion=confusion,
            curve_data=curve_data,
            workbook_sheets=workbook_sheets,
            minimum_features=minimum_features,
        )
        output_validation = validate_outputs(
            paths=staged_paths,
            test_matrix=test_matrix,
            expected_predictions=predictions,
            expected_selected_features=selected_features,
            expected_confusion=confusion,
            expected_curve_data=curve_data,
            expected_sheet_names=list(workbook_sheets),
            selected_configuration=selected_configuration,
            minimum_features=minimum_features,
            vocabulary_size=vocabulary_size,
        )
        current_selector: RFECV = selected_pipeline.named_steps["feature_selection"]
        correctness_checks = {
            "complete_23464_feature_input": vocabulary_size == 23_464,
            "proportional_minimum_is_124": minimum_features == 124,
            "reference_minimum_percentage_matches": round(100 * 20 / 3_770, 4) == 0.5305,
            "current_minimum_percentage_matches": round(100 * 124 / 23_464, 4) == 0.5285,
            "rfecv_is_inside_pipeline": list(selected_pipeline.named_steps) == ["feature_selection", "classifier"],
            "selected_feature_count_chosen_automatically": int(current_selector.n_features_) == optimal_feature_counts[selected_configuration],
            "selected_feature_count_not_below_minimum": int(current_selector.n_features_) >= minimum_features,
            "outer_cv_refits_ten_selectors": outer_curves.groupby(["configuration_id", "outer_fold"]).ngroups == 10,
            "both_complete_training_curves_exported": set(pd.concat(final_curves)["configuration_id"]) == set(CONFIGURATIONS),
            "both_configurations_reported_post_hoc_on_test": set(predictions["configuration_id"]) == set(CONFIGURATIONS),
            "each_configuration_predicts_all_test_rows": all(
                len(predictions.loc[predictions["configuration_id"].eq(configuration_id)])
                == len(test_index)
                for configuration_id in CONFIGURATIONS
            ),
            "test_scores_not_used_for_selection": True,
            "same_materialized_outer_folds_as_models_a_b": prior_reference["outer_cv_design"]["fold_class_distributions"] == fold_validation["fold_class_distributions"],
            "six_configuration_cv_comparison_complete": set(combined_cv["configuration_id"]) == {"A1", "A2", "B1", "B2", "C1", "C2"},
        }
        correctness_checks = {name: bool(value) for name, value in correctness_checks.items()}
        failed = [name for name, passed in correctness_checks.items() if not passed]
        if failed:
            raise RuntimeError("Model C correctness validation failed: " + ", ".join(failed))
        validation_results = {
            "status": "passed",
            "registry_schema_migration": schema_migration,
            "input_validation": inputs["validation"],
            "outer_cross_validation_fold_validation": fold_validation,
            "output_validation": output_validation,
            "final_correctness_checks": correctness_checks,
        }
        artifacts = artifact_rows(
            model_id=model_id,
            final_dir=final_dir,
            staged_paths=staged_paths,
            created_at=created_at,
        )
        input_artifact_records = [
            {
                "artifact_role": role,
                "file_name": values["file_name"],
                "artifact_path": values["artifact_path"],
                "artifact_hash_sha256": values["artifact_hash_sha256"],
                "size_bytes": int(values["size_bytes"]),
            }
            for role, values in sorted(vector_artifacts.items())
        ]
        configuration_records = {
            configuration_id: {
                **RANDOM_FOREST_PARAMETERS,
                "class_weight": configuration["class_weight"],
                "configuration_label": configuration["label"],
                "feature_selection": "RFECV",
                "minimum_features_to_select": minimum_features,
                "complete_training_optimal_feature_count": optimal_feature_counts[configuration_id],
                "scoring": "f1_macro",
                "inner_cv_splits": INNER_CV_SPLITS,
                "rfecv_step": rfecv_step,
            }
            for configuration_id, configuration in CONFIGURATIONS.items()
        }
        confusion_columns = [f"predicted_{label}" for label in CLASS_ORDER]
        final_metrics_registry = {
            "class_order": list(CLASS_ORDER),
            "held_out_reported_configurations": list(CONFIGURATIONS),
            "cv_selected_primary_configuration": selected_configuration,
            "test_scores_used_for_selection": False,
            "post_hoc_comparison_with_model_a_and_b": True,
            "configurations": {
                configuration_id: {
                    "configuration_label": CONFIGURATIONS[configuration_id]["label"],
                    "overall": test_results[configuration_id],
                    "per_class": per_class.loc[
                        per_class["configuration_id"].eq(configuration_id)
                    ].to_dict(orient="records"),
                    "confusion_matrix": {
                        "rows_are_actual": True,
                        "columns_are_predicted": True,
                        "values": confusion.loc[
                            confusion["configuration_id"].eq(configuration_id),
                            confusion_columns,
                        ].astype(int).values.tolist(),
                    },
                }
                for configuration_id in CONFIGURATIONS
            },
        }
        cv_registry = {
            "stored_numeric_precision_decimal_places": STORED_METRIC_DECIMALS,
            "model_a_reference_id": prior_reference["model_a_id"],
            "model_b_reference_id": prior_reference["model_b_id"],
            "model_c_configurations": c_cv_summaries,
            "model_c_records": c_cv_results.to_dict(orient="records"),
            "six_model_comparison_records": dataframe_records_for_json(combined_cv),
            "rfecv_feature_curve_artifact": FEATURE_CURVE_FILENAME,
            "rfecv_feature_plot_artifact": FEATURE_CURVE_PLOT_FILENAME,
        }
        retention_calculation = (
            f"minimum = round({vocabulary_size} * {REFERENCE_MINIMUM_FEATURE_COUNT} / "
            f"{REFERENCE_VOCABULARY_SIZE}) = {minimum_features}; 20 / 3,770 = "
            f"{100 * REFERENCE_MINIMUM_PROPORTION:.4f}%; 124 / 23,464 = "
            f"{100 * minimum_features / vocabulary_size:.4f}%; final selected "
            f"{optimal_feature_counts[selected_configuration]} by complete-training RFECV"
        )
        event_values = {
            "model_id": model_id,
            "vectorization_id": vector_event["vectorization_id"],
            "split_id": vector_event["split_id"],
            "matching_id": vector_event["matching_id"],
            "segmentation_id": vector_event["segmentation_id"],
            "document_id": vector_event["document_id"],
            "model_date": created_at[:10],
            "created_at": created_at,
            "model_name": MODEL_NAME,
            "model_designation": MODEL_DESIGNATION,
            "model_description": MODEL_DESCRIPTION,
            "script_path": BASE.project_relative(Path(__file__)),
            "script_version": SCRIPT_VERSION,
            "model_tool": "scikit-learn Pipeline, RFECV, and RandomForestClassifier",
            "model_tool_version": sklearn.__version__,
            "input_artifacts_json": BASE.json_text(input_artifact_records),
            "input_artifacts_count": len(input_artifact_records),
            "class_order_json": BASE.json_text(list(CLASS_ORDER)),
            "class_distributions_json": BASE.json_text(inputs["class_distributions"]),
            "training_records_count": len(train_index),
            "test_records_count": len(test_index),
            "reference_vocabulary_size": REFERENCE_VOCABULARY_SIZE,
            "reference_selected_feature_count": REFERENCE_MINIMUM_FEATURE_COUNT,
            "retention_proportion": REFERENCE_MINIMUM_PROPORTION,
            "retention_calculation": retention_calculation,
            "current_vocabulary_size": vocabulary_size,
            "current_selected_feature_count": optimal_feature_counts[selected_configuration],
            "feature_selection_method": "Nested Random-Forest recursive feature elimination with cross-validation (RFECV)",
            "feature_selection_parameters_json": BASE.json_text(
                {
                    "method": "RFECV",
                    "initial_feature_count": vocabulary_size,
                    "reference_vocabulary_size": REFERENCE_VOCABULARY_SIZE,
                    "reference_minimum_feature_count": REFERENCE_MINIMUM_FEATURE_COUNT,
                    "minimum_features_to_select": minimum_features,
                    "minimum_is_lower_bound_not_fixed_count": True,
                    "rfecv_step": rfecv_step,
                    "rfecv_step_interpretation": "fixed fraction of initial feature count per elimination iteration",
                    "scoring": "f1_macro",
                    "inner_cv_splits": INNER_CV_SPLITS,
                    "complete_training_optimal_feature_counts": optimal_feature_counts,
                    "publication_described_or_justified_minimum_20": False,
                    "minimum_20_source": "Wroblewska released classifier implementation",
                }
            ),
            "cross_validation_design_json": BASE.json_text(
                {
                    "type": "nested stratified cross-validation",
                    "outer": {
                        "type": "StratifiedKFold",
                        "n_splits": OUTER_CV_SPLITS,
                        "shuffle": True,
                        "random_state": RANDOM_STATE,
                        "same_folds_as_models_a_and_b": True,
                        "fold_class_distributions": fold_validation["fold_class_distributions"],
                    },
                    "inner": {
                        "type": "StratifiedKFold",
                        "n_splits": INNER_CV_SPLITS,
                        "shuffle": True,
                        "random_state": RANDOM_STATE,
                        "refitted_within_each_outer_training_fold": True,
                        "principal_metric": "macro_f1",
                    },
                    "test_scores_used_for_selection": False,
                    "held_out_reported_configurations": list(CONFIGURATIONS),
                    "held_out_reporting_is_post_hoc": True,
                }
            ),
            "random_forest_configurations_json": BASE.json_text(configuration_records),
            "cross_validation_results_json": BASE.json_text(cv_registry),
            "configuration_selection_rule": (
                "Select the greater mean five-fold outer-CV macro-F1 at 12 decimal "
                "places and select C1 on a tie. RFECV is fitted only within each outer "
                "training fold. After configuration selection, refit RFECV on all "
                "training records and use its inner-CV optimum; never use test scores "
                "for configuration or feature-count selection."
            ),
            "selected_configuration": selected_configuration,
            "final_test_metrics_json": BASE.json_text(final_metrics_registry),
            "validation_results_json": BASE.json_text(validation_results),
            "output_artifacts_count": len(artifacts),
            "model_status": "Completed",
            "notes": (
                "Model C is an RFECV adaptation of the feature-selection procedure "
                "identified in Wroblewska's released classifier. The publication did "
                "not describe or justify the implementation minimum of 20. Scaling "
                "20-of-3,770 to 23,464 gives a lower bound of 124, not a predetermined "
                "final feature count. RFECV step=0.05 creates a registered, reproducible "
                "21-point elimination grid; step=1 would require 23,341 subset levels "
                "per selector fit and is computationally impractical for nested CV. "
                "Both C configurations are compared using outer training-only CV; C1 and "
                "C2 are then evaluated on the existing test partition as post-hoc "
                "comparisons with registered Models A and B. C2 remains the selected "
                "configuration, and neither test result changes selection or tuning."
            ),
        }
        registration, final_validation = BASE.register_and_publish(
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

    print(f"linked_vectorization_id: {vector_event['vectorization_id']}")
    print(f"model_a_cv_reference_id: {prior_reference['model_a_id']}")
    print(f"model_b_cv_reference_id: {prior_reference['model_b_id']}")
    print(f"training_records: {len(train_index)}")
    print(f"test_records: {len(test_index)}")
    print(f"initial_features: {vocabulary_size:,}")
    print(f"minimum_features_to_select: {minimum_features}")
    print(f"rfecv_step: {rfecv_step}")
    for configuration_id in CONFIGURATIONS:
        print(f"{configuration_id}_mean_outer_cv_macro_f1: {c_cv_summaries[configuration_id]['mean_macro_f1']:.6f}")
        print(f"{configuration_id}_complete_training_optimal_features: {optimal_feature_counts[configuration_id]}")
    print(f"model_c_cv_selected_configuration: {selected_configuration}")
    for configuration_id in CONFIGURATIONS:
        result = test_results[configuration_id]
        print(
            f"{configuration_id}_post_hoc_test_metrics: "
            f"macro_f1={result['macro_f1']:.6f}, "
            f"weighted_f1={result['weighted_f1']:.6f}, "
            f"accuracy={result['accuracy']:.6f}"
        )
    print(f"output_directory: {BASE.project_relative(final_dir)}")
    print(f"registered_model_id: {model_id}")
    print("final_validation_status: passed")
    return {
        "model_id": model_id,
        "vectorization_id": vector_event["vectorization_id"],
        "model_a_reference_id": prior_reference["model_a_id"],
        "model_b_reference_id": prior_reference["model_b_id"],
        "output_directory": BASE.project_relative(final_dir),
        "initial_feature_count": vocabulary_size,
        "minimum_features_to_select": minimum_features,
        "optimal_feature_counts": optimal_feature_counts,
        "selected_configuration": selected_configuration,
        "outer_cv": c_cv_summaries,
        "test_metrics": test_results,
        "registration": registration,
        "validation": final_validation,
        "artifacts": artifacts,
    }


def parse_rfecv_step(value: str) -> int | float:
    """Parse a positive integer or a proportional float in (0, 1)."""
    text = value.strip()
    if re.fullmatch(r"\d+", text):
        parsed: int | float = int(text)
    else:
        parsed = float(text)
    if isinstance(parsed, int) and parsed >= 1:
        return parsed
    if isinstance(parsed, float) and 0 < parsed < 1:
        return parsed
    raise argparse.ArgumentTypeError("RFECV step must be an integer >=1 or float in (0,1).")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and register paired nested-RFECV Random Forest Model C."
    )
    parser.add_argument("--database", default=str(DATABASE_PATH), help="Corpus inventory SQLite database.")
    parser.add_argument("--vectorization-id", default=None, help="Registered vectorization ID. Default: latest completed vectorization.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Base directory for append-only MODEL_###### output folders.")
    parser.add_argument(
        "--rfecv-step",
        type=parse_rfecv_step,
        default=DEFAULT_RFECV_STEP,
        help="Elimination step: integer count or fraction in (0,1). Default: 0.05.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_arguments()
    try:
        run_model_c(
            database_path=BASE.resolve_project_path(args.database),
            output_base_dir=BASE.resolve_project_path(args.output_dir),
            vectorization_id=args.vectorization_id,
            rfecv_step=args.rfecv_step,
        )
    except Exception as error:
        raise SystemExit(f"Model C failed: {error}") from error


if __name__ == "__main__":
    main()
