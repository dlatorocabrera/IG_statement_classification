"""Tests for 09_train_random_forest_model_a.py using unittest."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif


SCRIPT_PATH = Path(__file__).with_name("09_train_random_forest_model_a.py")
SPEC = importlib.util.spec_from_file_location("train_random_forest_model_a", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_training() -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    """Create a small balanced, deterministic three-class sparse dataset."""
    random = np.random.default_rng(42)
    labels = np.repeat(np.asarray(MODULE.CLASS_ORDER, dtype=object), 10)
    matrix = random.integers(0, 3, size=(len(labels), 12)).astype(float)
    for class_index in range(3):
        rows = slice(class_index * 10, (class_index + 1) * 10)
        matrix[rows, class_index * 3 : class_index * 3 + 3] += 4
    features = pd.DataFrame(
        {
            "feature_index": np.arange(matrix.shape[1]),
            "ngram_text": [f"feature {index}" for index in range(matrix.shape[1])],
            "ngram_length": [2] * matrix.shape[1],
            "idf_value": np.linspace(1.0, 2.0, matrix.shape[1]),
        }
    )
    return sparse.csr_matrix(matrix), labels, features


def synthetic_test_index(labels: np.ndarray) -> pd.DataFrame:
    """Create an index aligned to a synthetic test label array."""
    return pd.DataFrame(
        {
            "matrix_row_index": np.arange(len(labels)),
            "candidate_id": [f"TEST_{index:03d}" for index in range(len(labels))],
            "candidate_text_original": [f"Synthetic statement {index}" for index in range(len(labels))],
            "reviewer_gold_label": labels,
            "split": "test",
        }
    )


class ModelATests(unittest.TestCase):
    def test_registry_json_converts_numpy_scalars(self) -> None:
        encoded = MODULE.json_text(
            {"passed": np.bool_(True), "count": np.int64(436), "score": np.float64(0.5)}
        )
        self.assertEqual(
            json.loads(encoded), {"count": 436, "passed": True, "score": 0.5}
        )

    def test_proportional_retention_calculation_is_exact(self) -> None:
        self.assertAlmostEqual(
            MODULE.REFERENCE_RETENTION_PROPORTION,
            70 / 3770,
            places=15,
        )
        self.assertEqual(MODULE.selected_feature_count(23_464), 436)
        self.assertEqual(MODULE.EXPECTED_SELECTED_FEATURE_COUNT, 436)

    def test_folds_are_stratified_and_shared_by_both_configurations(self) -> None:
        matrix, labels, _ = synthetic_training()
        folds, validation = MODULE.build_cv_folds(labels)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(len(folds), 5)
        results, summaries = MODULE.run_cross_validation(
            matrix,
            labels,
            folds,
            k_features=4,
        )
        fold_rows = results.loc[results["record_type"].eq("fold")]
        self.assertEqual(len(fold_rows), 10)
        self.assertEqual(set(fold_rows["configuration_id"]), {"A1", "A2"})
        self.assertEqual(set(summaries), {"A1", "A2"})
        for fold_number in range(1, 6):
            rows = fold_rows.loc[fold_rows["fold"].eq(fold_number)]
            self.assertEqual(rows["training_records"].nunique(), 1)
            self.assertEqual(rows["validation_records"].nunique(), 1)

    def test_tie_rule_selects_A1_at_stored_precision(self) -> None:
        summaries = {
            "A1": {"mean_macro_f1": 0.8000000000001},
            "A2": {"mean_macro_f1": 0.8000000000002},
        }
        self.assertEqual(MODULE.select_configuration(summaries, 12), "A1")
        summaries["A2"]["mean_macro_f1"] = 0.8000001
        self.assertEqual(MODULE.select_configuration(summaries, 12), "A2")

    def test_final_outputs_round_trip_and_preserve_alignment(self) -> None:
        matrix, labels, features = synthetic_training()
        train_matrix = matrix[:24]
        train_labels = labels[:24]
        # Ensure all three labels are represented in fitting.
        train_rows = np.r_[0:8, 10:18, 20:28]
        train_matrix = matrix[train_rows]
        train_labels = labels[train_rows]
        test_rows = np.r_[8:10, 18:20, 28:30]
        test_matrix = matrix[test_rows]
        test_labels = labels[test_rows]

        pipelines = {}
        selected_tables = {}
        overall_by_configuration = {}
        metric_tables = []
        per_class_tables = []
        confusion_tables = []
        prediction_tables = []
        for configuration_id, configuration in MODULE.CONFIGURATIONS.items():
            pipeline = MODULE.build_pipeline(configuration["class_weight"], k_features=4)
            self.assertEqual(
                list(pipeline.named_steps), ["feature_selection", "classifier"]
            )
            pipeline.fit(train_matrix, train_labels)
            selected_tables[configuration_id] = MODULE.build_selected_features(
                pipeline, features
            )
            overall, metrics, classes, matrix_table, configuration_predictions = (
                MODULE.build_evaluation_outputs(
                    configuration_id,
                    pipeline,
                    test_matrix,
                    synthetic_test_index(test_labels),
                )
            )
            overall_by_configuration[configuration_id] = overall
            metric_tables.append(metrics)
            per_class_tables.append(classes)
            confusion_tables.append(matrix_table)
            prediction_tables.append(configuration_predictions)
            pipeline.named_steps["feature_selection"].score_func = partial(
                mutual_info_classif, random_state=MODULE.RANDOM_STATE
            )
            pipelines[configuration_id] = pipeline

        selected = selected_tables["A1"]
        self.assertEqual(len(selected), 4)
        self.assertTrue(selected.equals(selected_tables["A2"]))
        test_metrics = pd.concat(metric_tables, ignore_index=True)
        per_class = pd.concat(per_class_tables, ignore_index=True)
        confusion = pd.concat(confusion_tables, ignore_index=True)
        predictions = pd.concat(prediction_tables, ignore_index=True)
        self.assertEqual(len(predictions), 12)
        for configuration_id in MODULE.CONFIGURATIONS:
            self.assertEqual(
                int(
                    per_class.loc[
                        per_class["configuration_id"].eq(configuration_id), "support"
                    ].sum()
                ),
                6,
            )
            self.assertEqual(
                int(
                    confusion.loc[
                        confusion["configuration_id"].eq(configuration_id),
                        [f"predicted_{label}" for label in MODULE.CLASS_ORDER],
                    ].to_numpy().sum()
                ),
                6,
            )
            overall = overall_by_configuration[configuration_id]
            self.assertEqual(overall["correct_count"] + overall["incorrect_count"], 6)

        cv_results = pd.DataFrame(
            [
                {
                    "configuration_id": "A1",
                    "configuration_label": "unweighted baseline",
                    "class_weight": "None",
                    "record_type": record_type,
                    "fold": fold,
                    "training_records": 24 if record_type == "fold" else "",
                    "validation_records": 6 if record_type == "fold" else "",
                    "macro_f1": 1.0,
                    "weighted_f1": 1.0,
                    "accuracy": 1.0,
                    "macro_precision": 1.0,
                    "macro_recall": 1.0,
                    "selected_configuration": True,
                }
                for record_type, fold in (
                    [("fold", number) for number in range(1, 6)]
                    + [("mean", "all"), ("std", "all")]
                )
            ]
            + [
                {
                    "configuration_id": "A2",
                    "configuration_label": "balanced adaptation",
                    "class_weight": "balanced",
                    "record_type": record_type,
                    "fold": fold,
                    "training_records": 24 if record_type == "fold" else "",
                    "validation_records": 6 if record_type == "fold" else "",
                    "macro_f1": 0.9,
                    "weighted_f1": 0.9,
                    "accuracy": 0.9,
                    "macro_precision": 0.9,
                    "macro_recall": 0.9,
                    "selected_configuration": False,
                }
                for record_type, fold in (
                    [("fold", number) for number in range(1, 6)]
                    + [("mean", "all"), ("std", "all")]
                )
            ]
        )
        summary = pd.DataFrame(
            [("identity", "model_id", "MODEL_TEST", "Synthetic test")],
            columns=["section", "item", "value", "description"],
        )
        workbook_sheets = {
            "Model_Summary": summary,
            "CV_Comparison": cv_results,
            "Test_Metrics": test_metrics,
            "Per_Class_Metrics": per_class,
            "Confusion_Matrix": confusion,
            "Selected_Features": selected,
            "Test_Predictions": predictions,
            "Artifact_Descriptions": MODULE.artifact_descriptions_table(),
        }
        model_bundle = {
            "artifact_type": "paired_model_a_configuration_bundle",
            "model_id": "MODEL_TEST",
            "script_version": MODULE.SCRIPT_VERSION,
            "class_order": list(MODULE.CLASS_ORDER),
            "cv_selected_primary_configuration": "A1",
            "held_out_reported_configurations": list(MODULE.CONFIGURATIONS),
            "test_scores_used_for_selection": False,
            "pipelines": pipelines,
        }
        with tempfile.TemporaryDirectory() as temp_name:
            paths = MODULE.write_outputs(
                output_dir=Path(temp_name),
                model_bundle=model_bundle,
                cv_results=cv_results,
                selected_features=selected,
                predictions=predictions,
                confusion=confusion,
                workbook_sheets=workbook_sheets,
            )
            validation = MODULE.validate_outputs(
                paths=paths,
                test_matrix=test_matrix,
                expected_predictions=predictions,
                expected_selected_features=selected,
                expected_confusion=confusion,
                expected_sheet_names=list(workbook_sheets),
                k_features=4,
            )
        self.assertEqual(validation["status"], "passed")
        self.assertTrue(all(validation["checks"].values()))


if __name__ == "__main__":
    unittest.main()
