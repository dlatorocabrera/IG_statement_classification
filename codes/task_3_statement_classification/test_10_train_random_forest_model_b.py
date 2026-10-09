"""Tests for 10_train_random_forest_model_b.py using unittest."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_PATH = Path(__file__).with_name("10_train_random_forest_model_b.py")
SPEC = importlib.util.spec_from_file_location("train_random_forest_model_b", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_data() -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    """Create deterministic balanced sparse data with class-specific signals."""
    random = np.random.default_rng(42)
    labels = np.repeat(np.asarray(MODULE.CLASS_ORDER, dtype=object), 10)
    values = random.integers(0, 3, size=(30, 12)).astype(float)
    for class_index in range(3):
        rows = slice(class_index * 10, (class_index + 1) * 10)
        values[rows, class_index * 3 : class_index * 3 + 3] += 4
    features = pd.DataFrame(
        {
            "feature_index": np.arange(12),
            "ngram_text": [f"term {index}" for index in range(12)],
            "ngram_length": [2] * 12,
            "idf_value": np.linspace(1.0, 2.0, 12),
        }
    )
    return sparse.csr_matrix(values), labels, features


def synthetic_index(labels: np.ndarray) -> pd.DataFrame:
    """Build a candidate-aligned synthetic held-out index."""
    return pd.DataFrame(
        {
            "matrix_row_index": np.arange(len(labels)),
            "candidate_id": [f"TEST_{index:03d}" for index in range(len(labels))],
            "candidate_text_original": [f"Synthetic statement {index}" for index in range(len(labels))],
            "reviewer_gold_label": labels,
            "split": "test",
        }
    )


class ModelBTests(unittest.TestCase):
    def test_registry_constraint_migration_preserves_a_general_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            database_path = Path(temp_name) / "registry.sqlite"
            sqlite3.connect(database_path).close()
            MODULE.BASE.ensure_registry_schema(database_path)
            with sqlite3.connect(database_path) as connection:
                before = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name='model_events'"
                ).fetchone()[0]
            self.assertIn("selected_configuration IN ('A1', 'A2')", before)
            result = MODULE.ensure_general_model_registry_schema(database_path)
            self.assertEqual(result["status"], "migrated")
            with sqlite3.connect(database_path) as connection:
                after = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name='model_events'"
                ).fetchone()[0]
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertIn("length(trim(selected_configuration)) > 0", after)

    def test_pipeline_uses_every_feature_and_has_no_selector(self) -> None:
        matrix, labels, _ = synthetic_data()
        pipeline = MODULE.build_pipeline(None)
        self.assertEqual(list(pipeline.named_steps), ["classifier"])
        pipeline.fit(matrix, labels)
        self.assertEqual(pipeline.named_steps["classifier"].n_features_in_, 12)

    def test_B1_and_B2_share_five_stratified_folds(self) -> None:
        matrix, labels, _ = synthetic_data()
        folds, validation = MODULE.BASE.build_cv_folds(labels)
        results, summaries = MODULE.run_cross_validation(matrix, labels, folds)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(set(summaries), {"B1", "B2"})
        fold_rows = results.loc[results["record_type"].eq("fold")]
        self.assertEqual(len(fold_rows), 10)
        for fold_number in range(1, 6):
            rows = fold_rows.loc[fold_rows["fold"].eq(fold_number)]
            self.assertEqual(set(rows["configuration_id"]), {"B1", "B2"})
            self.assertEqual(rows["training_records"].nunique(), 1)
            self.assertEqual(rows["validation_records"].nunique(), 1)

    def test_tie_rule_selects_B1(self) -> None:
        summaries = {
            "B1": {"mean_macro_f1": 0.7000000000001},
            "B2": {"mean_macro_f1": 0.7000000000002},
        }
        self.assertEqual(MODULE.select_configuration(summaries, 12), "B1")
        summaries["B2"]["mean_macro_f1"] = 0.700001
        self.assertEqual(MODULE.select_configuration(summaries, 12), "B2")

    def test_paired_outputs_and_four_model_cv_round_trip(self) -> None:
        matrix, labels, features = synthetic_data()
        train_rows = np.r_[0:8, 10:18, 20:28]
        test_rows = np.r_[8:10, 18:20, 28:30]
        train_matrix, train_labels = matrix[train_rows], labels[train_rows]
        test_matrix, test_labels = matrix[test_rows], labels[test_rows]
        folds, _ = MODULE.BASE.build_cv_folds(labels)
        b_cv, b_summaries = MODULE.run_cross_validation(matrix, labels, folds)
        b_selected = MODULE.select_configuration(b_summaries)
        b_cv.loc[b_cv["configuration_id"].eq(b_selected), "selected_configuration"] = True

        a_cv = b_cv.copy()
        a_cv["configuration_id"] = a_cv["configuration_id"].replace(
            {"B1": "A1", "B2": "A2"}
        )
        a_cv["configuration_label"] = a_cv["configuration_id"].replace(
            {"A1": "unweighted baseline", "A2": "balanced adaptation"}
        )
        a_cv["selected_configuration"] = a_cv["configuration_id"].eq("A1")
        model_a_reference = {
            "model_id": "MODEL_A_TEST",
            "records": a_cv,
            "summaries": {
                "A1": b_summaries["B1"],
                "A2": b_summaries["B2"],
            },
        }
        combined_cv = MODULE.build_four_model_cv_comparison(
            model_a_reference=model_a_reference,
            model_b_id="MODEL_B_TEST",
            model_b_results=b_cv,
            model_b_selected=b_selected,
        )
        self.assertEqual(
            set(combined_cv["configuration_id"]), {"A1", "A2", "B1", "B2"}
        )

        pipelines = {}
        metric_tables = []
        class_tables = []
        confusion_tables = []
        prediction_tables = []
        for configuration_id, configuration in MODULE.CONFIGURATIONS.items():
            pipeline = MODULE.build_pipeline(configuration["class_weight"])
            pipeline.fit(train_matrix, train_labels)
            _, metrics, classes, confusion, predictions = (
                MODULE.build_evaluation_outputs(
                    configuration_id,
                    pipeline,
                    test_matrix,
                    synthetic_index(test_labels),
                )
            )
            pipelines[configuration_id] = pipeline
            metric_tables.append(metrics)
            class_tables.append(classes)
            confusion_tables.append(confusion)
            prediction_tables.append(predictions)
        test_metrics = pd.concat(metric_tables, ignore_index=True)
        per_class = pd.concat(class_tables, ignore_index=True)
        confusion = pd.concat(confusion_tables, ignore_index=True)
        predictions = pd.concat(prediction_tables, ignore_index=True)
        all_features = MODULE.build_all_features(features)
        self.assertEqual(len(all_features), 12)

        workbook_sheets = {
            "Model_Summary": pd.DataFrame(
                [("identity", "model_id", "MODEL_B_TEST", "Synthetic")],
                columns=["section", "item", "value", "description"],
            ),
            "CV_Comparison": combined_cv,
            "Test_Metrics": test_metrics,
            "Per_Class_Metrics": per_class,
            "Confusion_Matrix": confusion,
            "Selected_Features": all_features,
            "Test_Predictions": predictions,
            "Artifact_Descriptions": MODULE.artifact_descriptions_table(),
        }
        bundle = {
            "artifact_type": "paired_model_b_full_feature_bundle",
            "model_id": "MODEL_B_TEST",
            "script_version": MODULE.SCRIPT_VERSION,
            "class_order": list(MODULE.CLASS_ORDER),
            "feature_count": 12,
            "pipelines": pipelines,
        }
        with tempfile.TemporaryDirectory() as temp_name:
            paths = MODULE.write_outputs(
                output_dir=Path(temp_name),
                model_bundle=bundle,
                cv_results=combined_cv,
                all_features=all_features,
                predictions=predictions,
                confusion=confusion,
                workbook_sheets=workbook_sheets,
            )
            validation = MODULE.validate_outputs(
                paths=paths,
                test_matrix=test_matrix,
                expected_predictions=predictions,
                expected_features=all_features,
                expected_confusion=confusion,
                expected_sheet_names=list(workbook_sheets),
            )
        self.assertEqual(validation["status"], "passed")
        self.assertTrue(all(validation["checks"].values()))


if __name__ == "__main__":
    unittest.main()
