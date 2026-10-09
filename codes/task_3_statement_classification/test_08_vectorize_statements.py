"""Tests for 08_vectorize_statements.py using unittest."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).with_name("08_vectorize_statements.py")
SPEC = importlib.util.spec_from_file_location("vectorize_statements", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    training_rows = [
        (1, "regulative", "La autoridad deberá presentar el informe."),
        (2, "regulative", "El organismo podrá revisar la solicitud."),
        (3, "constitutive", "Se crea el registro público ambiental."),
        (4, "constitutive", "Para estos efectos se define área verde."),
        (5, "non_institutional", "El estudio describe la situación actual."),
        (6, "non_institutional", "La contaminación aumentó durante el invierno."),
    ]
    test_rows = [
        (7, "regulative", "La autoridad deberá resolver oportunamente."),
        (8, "constitutive", "Se establece una categoría institucional."),
        (9, "non_institutional", "El informe presenta antecedentes históricos."),
    ]

    def frame(rows: list[tuple[int, str, str]], partition: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "review_record_id": f"REV_{number:04d}",
                    "candidate_id": f"DOC_STMT_{number:06d}",
                    "candidate_text_original": text,
                    "reviewer_gold_label": label,
                    "split": partition,
                }
                for number, label, text in rows
            ]
        )

    return frame(training_rows, "training"), frame(test_rows, "test")


class VectorizerTests(unittest.TestCase):
    def test_protocol_tokenizer_retains_only_approved_single_characters(self) -> None:
        vectorizer = MODULE.build_vectorizer()
        analyzer = vectorizer.build_analyzer()
        terms = set(analyzer("a e o u y x ñ 1 casa árbol"))
        self.assertTrue(set(MODULE.SPANISH_SINGLE_CHARACTER_TOKENS).issubset(terms))
        self.assertFalse({"x", "ñ", "1"} & terms)
        self.assertIn("casa", terms)
        self.assertIn("árbol", terms)
        self.assertEqual(vectorizer.ngram_range, (1, 3))
        self.assertIsNone(vectorizer.stop_words)
        self.assertIsNone(vectorizer.strip_accents)

    def test_vocabulary_and_idf_are_fitted_on_training_text_only(self) -> None:
        vectorizer = MODULE.build_vectorizer()
        training_matrix = vectorizer.fit_transform(
            ["árbol arbol a y término entrenamiento"]
        )
        test_matrix = vectorizer.transform(["palabraexclusivaprueba árbol"])
        vocabulary = set(vectorizer.get_feature_names_out())
        self.assertIn("árbol", vocabulary)
        self.assertIn("arbol", vocabulary)
        self.assertNotIn("palabraexclusivaprueba", vocabulary)
        self.assertEqual(training_matrix.shape[1], test_matrix.shape[1])

    def test_output_round_trip_validation(self) -> None:
        training, test = make_partitions()
        vectorizer = MODULE.build_vectorizer()
        training_matrix = vectorizer.fit_transform(
            training["candidate_text_original"].tolist()
        ).tocsr()
        test_matrix = vectorizer.transform(
            test["candidate_text_original"].tolist()
        ).tocsr()
        training_index = MODULE.build_index(training)
        test_index = MODULE.build_index(test)
        features = MODULE.build_features(vectorizer)
        with tempfile.TemporaryDirectory() as temp_name:
            paths = MODULE.write_outputs(
                output_dir=Path(temp_name),
                training_matrix=training_matrix,
                test_matrix=test_matrix,
                training_index=training_index,
                test_index=test_index,
                features=features,
                vectorizer=vectorizer,
            )
            validation = MODULE.validate_outputs(
                paths=paths,
                training=training,
                test=test,
                training_matrix=training_matrix,
                test_matrix=test_matrix,
                training_index=training_index,
                test_index=test_index,
                features=features,
                vectorizer=vectorizer,
            )
        self.assertEqual(validation["status"], "passed")
        self.assertTrue(all(validation["checks"].values()))

    def test_full_run_registers_event_and_six_artifacts(self) -> None:
        training, test = make_partitions()
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            training_path = temp_dir / "training.csv"
            test_path = temp_dir / "test.csv"
            database_path = temp_dir / "corpus_inventory.sqlite"
            output_dir = temp_dir / "vectorization"
            training.to_csv(training_path, index=False)
            test.to_csv(test_path, index=False)

            distributions = {
                "training": MODULE.class_counts(training),
                "test": MODULE.class_counts(test),
            }
            distributions["eligible"] = {
                label: distributions["training"][label] + distributions["test"][label]
                for label in MODULE.VALID_LABELS
            }
            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE document_inventory (document_id TEXT PRIMARY KEY);
                    CREATE TABLE segmentation_events (segmentation_id TEXT PRIMARY KEY);
                    CREATE TABLE matching_events (
                        matching_id TEXT PRIMARY KEY,
                        segmentation_id TEXT NOT NULL,
                        document_id TEXT NOT NULL
                    );
                    CREATE TABLE split_events (
                        split_id TEXT PRIMARY KEY,
                        matching_id TEXT NOT NULL,
                        segmentation_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        split_status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        training_csv_path TEXT NOT NULL,
                        training_csv_hash_sha256 TEXT NOT NULL,
                        test_csv_path TEXT NOT NULL,
                        test_csv_hash_sha256 TEXT NOT NULL,
                        training_records_count INTEGER NOT NULL,
                        test_records_count INTEGER NOT NULL,
                        class_counts_json TEXT NOT NULL
                    );
                    CREATE TABLE split_assignments (
                        split_id TEXT NOT NULL,
                        review_record_id TEXT NOT NULL,
                        candidate_id TEXT NOT NULL,
                        partition TEXT NOT NULL,
                        gold_label TEXT NOT NULL
                    );
                    INSERT INTO document_inventory VALUES ('DOC_000001');
                    INSERT INTO segmentation_events VALUES ('SEG_000001');
                    INSERT INTO matching_events VALUES (
                        'MATCH_000001', 'SEG_000001', 'DOC_000001'
                    );
                    """
                )
                connection.execute(
                    """
                    INSERT INTO split_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "SPLIT_000001",
                        "MATCH_000001",
                        "SEG_000001",
                        "DOC_000001",
                        "Completed",
                        "2026-08-04T12:00:00+00:00",
                        str(training_path),
                        MODULE.sha256_file(training_path),
                        str(test_path),
                        MODULE.sha256_file(test_path),
                        len(training),
                        len(test),
                        MODULE.json_text(distributions),
                    ),
                )
                assignment_rows = []
                for data, partition in ((training, "train"), (test, "test")):
                    assignment_rows.extend(
                        (
                            "SPLIT_000001",
                            row.review_record_id,
                            row.candidate_id,
                            partition,
                            row.reviewer_gold_label,
                        )
                        for row in data.itertuples(index=False)
                    )
                connection.executemany(
                    "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?)",
                    assignment_rows,
                )

            result = MODULE.run_vectorization(
                database_path=database_path,
                output_base_dir=output_dir,
                split_id="SPLIT_000001",
            )
            self.assertEqual(result["vectorization_id"], "VECT_000001")
            self.assertEqual(result["validation_status"], "passed")
            self.assertTrue((output_dir / "VECT_000001").is_dir())
            with sqlite3.connect(database_path) as connection:
                event = connection.execute(
                    """
                    SELECT vectorization_date, split_id, vocabulary_size,
                           vectorization_status
                    FROM vectorization_events
                    """
                ).fetchone()
                artifact_count = connection.execute(
                    "SELECT COUNT(*) FROM vectorization_artifacts"
                ).fetchone()[0]
                foreign_key_violations = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
            self.assertEqual(event[1], "SPLIT_000001")
            self.assertGreater(event[2], 0)
            self.assertEqual(event[3], "Completed")
            self.assertEqual(artifact_count, 6)
            self.assertEqual(foreign_key_violations, [])


if __name__ == "__main__":
    unittest.main()
