"""Tests for 07_create_train_test_split.py using only unittest."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
import unicodedata
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from sklearn.model_selection import train_test_split


SCRIPT_PATH = Path(__file__).with_name("07_create_train_test_split.py")
SPEC = importlib.util.spec_from_file_location("create_train_test_split", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_row(
    number: int,
    *,
    label: str = "regulative",
    include: str = "yes",
    text: str | None = None,
    candidate_id: str | None = None,
) -> dict[str, object]:
    """Create one source-like review row."""
    return {
        "review_record_id": f"REV_{number:04d}",
        "candidate_id": (
            f"CL_TEST_STMT_{number:06d}" if candidate_id is None else candidate_id
        ),
        "alignment_group_id": f"ALIGN_{number % 7:04d}",
        "article": f"Article {number}",
        "scope": "body",
        "source_parent_id": f"PARENT_{number:06d}",
        "match_type": "matched",
        "candidate_text_original": (
            f"Original reviewed statement {number}." if text is None else text
        ),
        "reviewer_gold_label": label,
        "include_in_final_set": include,
        "proposed_gold_label": label,
        "review_status": "confirmed",
        "_source_excel_row": number + 5,
    }


def make_balanced_eligible(rows_per_class: int = 20) -> pd.DataFrame:
    """Create balanced, unique-text eligible rows."""
    rows: list[dict[str, object]] = []
    number = 1
    for label in MODULE.VALID_LABELS:
        for _ in range(rows_per_class):
            rows.append(make_row(number, label=label))
            number += 1
    return pd.DataFrame(rows)


class EligibilityTests(unittest.TestCase):
    def test_only_fully_eligible_records_are_selected(self) -> None:
        rows = [
            make_row(1, label="regulative"),
            make_row(2, label="constitutive"),
            make_row(3, label="non_institutional"),
            make_row(4, include="no"),
            make_row(5, include="hold"),
            make_row(6, label="other"),
            make_row(7, text="   "),
            make_row(8, candidate_id="   "),
        ]
        eligible, excluded, summary = MODULE.select_eligible_records(
            pd.DataFrame(rows)
        )

        self.assertEqual(
            set(eligible["candidate_id"]),
            {
                "CL_TEST_STMT_000001",
                "CL_TEST_STMT_000002",
                "CL_TEST_STMT_000003",
            },
        )
        self.assertEqual(set(eligible["reviewer_gold_label"]), set(MODULE.VALID_LABELS))
        self.assertEqual(len(excluded), 5)
        self.assertEqual(summary["reason_counts"]["include_in_final_set_no"], 1)
        self.assertEqual(summary["reason_counts"]["include_in_final_set_hold"], 1)
        self.assertEqual(summary["reason_counts"]["invalid_reviewer_gold_label"], 1)
        self.assertEqual(summary["reason_counts"]["blank_candidate_text_original"], 1)
        self.assertEqual(summary["reason_counts"]["blank_candidate_id"], 1)

    def test_source_total_mismatches_are_detected(self) -> None:
        rows: list[dict[str, str]] = []
        number = 1
        for label, count in MODULE.EXPECTED_SOURCE_COUNTS[
            "included_reviewer_gold_label"
        ].items():
            for _ in range(count):
                rows.append(
                    {
                        "include_in_final_set": "yes",
                        "reviewer_gold_label": label,
                        "row": str(number),
                    }
                )
                number += 1
        rows.extend(
            {"include_in_final_set": "no", "reviewer_gold_label": "regulative"}
            for _ in range(43)
        )
        rows.extend(
            {"include_in_final_set": "hold", "reviewer_gold_label": "constitutive"}
            for _ in range(22)
        )
        source = pd.DataFrame(rows)
        self.assertEqual(MODULE.validate_source_counts(source)["status"], "passed")

        source.loc[0, "include_in_final_set"] = "no"
        with self.assertRaisesRegex(ValueError, "Source count validation failed"):
            MODULE.validate_source_counts(source)
        overridden = MODULE.validate_source_counts(
            source, allow_count_mismatch=True
        )
        self.assertEqual(overridden["status"], "overridden")

    def test_duplicate_nonblank_candidate_ids_raise(self) -> None:
        data = make_balanced_eligible(2)
        data.loc[data.index[1], "candidate_id"] = data.loc[data.index[0], "candidate_id"]
        with self.assertRaisesRegex(ValueError, "candidate_id must be unique"):
            MODULE.verify_candidate_id_uniqueness(data)

    def test_authorized_702_record_eligibility_counts(self) -> None:
        rows: list[dict[str, object]] = []
        number = 1
        for label, count in {
            "regulative": 339,
            "constitutive": 155,
            "non_institutional": 209,
        }.items():
            for _ in range(count):
                rows.append(make_row(number, label=label))
                number += 1
        blank_index = next(
            index
            for index, row in enumerate(rows)
            if row["reviewer_gold_label"] == "constitutive"
        )
        rows[blank_index]["review_record_id"] = "REV_0768"
        rows[blank_index]["candidate_id"] = ""
        rows[blank_index]["candidate_text_original"] = ""

        eligible, excluded, _ = MODULE.select_eligible_records(pd.DataFrame(rows))
        validation = MODULE.assess_eligibility_counts(eligible)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["observed"]["eligible_records"], 702)
        self.assertEqual(
            validation["observed"]["reviewer_gold_label"],
            {"regulative": 339, "constitutive": 154, "non_institutional": 209},
        )
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded.iloc[0]["review_record_id"], "REV_0768")

        mismatch = MODULE.assess_eligibility_counts(eligible.iloc[:-1])
        self.assertEqual(mismatch["status"], "mismatch")
        self.assertTrue(mismatch["mismatches"])

    def test_workbook_comparison_ignores_only_excel_float_noise(self) -> None:
        expected = pd.DataFrame(
            [{"metric": "test_share", "value": 68 / 141, "text": "Exact text"}]
        )
        excel_round_trip = pd.DataFrame(
            [{"metric": "test_share", "value": 0.482269503546099, "text": "Exact text"}]
        )
        self.assertEqual(
            MODULE._canonical_frame_rows(expected, expected.columns),
            MODULE._canonical_frame_rows(excel_round_trip, expected.columns),
        )
        changed_text = excel_round_trip.copy()
        changed_text.loc[0, "text"] = "Changed text"
        self.assertNotEqual(
            MODULE._canonical_frame_rows(expected, expected.columns),
            MODULE._canonical_frame_rows(changed_text, expected.columns),
        )

    def test_authorized_review_adjustments_are_record_bound(self) -> None:
        rows = [
            make_row(283, label="regulative", text="Identical reviewed text."),
            make_row(752, label="regulative", text="Identical reviewed text."),
            make_row(
                768,
                label="constitutive",
                include="yes",
                text="",
                candidate_id="",
            ),
        ]
        validation = MODULE.validate_authorized_review_adjustments(
            pd.DataFrame(rows)
        )
        self.assertEqual(validation["status"], "passed")
        self.assertTrue(all(validation["checks"].values()))

        rows[1]["reviewer_gold_label"] = "constitutive"
        with self.assertRaisesRegex(
            ValueError, "REV_0752_label_regulative"
        ):
            MODULE.validate_authorized_review_adjustments(pd.DataFrame(rows))


class DuplicateTests(unittest.TestCase):
    def test_workbook_style_expands_long_wrapped_text(self) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Training"
        worksheet.append(["review_record_id", "candidate_text_original"])
        worksheet.append(["REV_0001", "Long reviewed text. " * 120])
        MODULE._style_workbook(workbook)
        self.assertEqual(worksheet.column_dimensions["B"].width, 90)
        self.assertGreater(worksheet.row_dimensions[2].height, 48)
        self.assertTrue(worksheet["B2"].alignment.wrap_text)

    def test_duplicate_label_conflicts_raise(self) -> None:
        decomposed = unicodedata.normalize("NFD", "CAFÉ VERDE")
        data = pd.DataFrame(
            [
                make_row(1, label="regulative", text="  Café   Verde "),
                make_row(2, label="constitutive", text=f"{decomposed}\t"),
            ]
        )
        eligible, _, _ = MODULE.select_eligible_records(data)
        with self.assertRaisesRegex(
            MODULE.DuplicateLabelConflictError,
            "conflicting reviewer_gold_label",
        ):
            MODULE.assign_duplicate_groups(eligible)

    def test_duplicate_group_ids_are_stable_and_text_is_unchanged(self) -> None:
        original = "  Árbol\tVERDE  "
        data = pd.DataFrame(
            [
                make_row(1, text=original),
                make_row(2, text="árbol verde"),
                make_row(3, text="A separate text."),
            ]
        )
        eligible, _, _ = MODULE.select_eligible_records(data)
        grouped = MODULE.assign_duplicate_groups(eligible)
        reversed_grouped = MODULE.assign_duplicate_groups(
            eligible.iloc[::-1].copy()
        )

        self.assertEqual(grouped.iloc[0]["candidate_text_original"], original)
        mapping = dict(zip(grouped["candidate_id"], grouped["duplicate_group_id"]))
        reversed_mapping = dict(
            zip(reversed_grouped["candidate_id"], reversed_grouped["duplicate_group_id"])
        )
        self.assertEqual(mapping, reversed_mapping)
        self.assertEqual(mapping["CL_TEST_STMT_000001"], mapping["CL_TEST_STMT_000002"])

    def test_near_duplicates_are_reported_but_not_regrouped(self) -> None:
        data = pd.DataFrame(
            [
                make_row(1, text="The agency shall submit report within ten days."),
                make_row(2, text="The agency shall submit the report within ten days."),
            ]
        )
        eligible, _, _ = MODULE.select_eligible_records(data)
        grouped = MODULE.assign_duplicate_groups(eligible)
        self.assertEqual(grouped["duplicate_group_id"].nunique(), 2)
        audit, summary = MODULE.build_duplicate_audit(
            grouped, near_duplicate_threshold=0.85
        )
        self.assertGreaterEqual(summary["near_duplicate_text_group_pairs"], 1)
        self.assertIn("possible_near_duplicate", set(audit["audit_type"]))


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eligible = make_balanced_eligible(20)
        self.grouped = MODULE.assign_duplicate_groups(self.eligible)

    def test_singleton_branch_matches_sklearn_membership(self) -> None:
        expected_training, expected_test = train_test_split(
            self.grouped,
            test_size=0.20,
            stratify=self.grouped["reviewer_gold_label"],
            random_state=42,
        )
        training, test, metadata = MODULE.create_stratified_split(
            self.grouped, 0.20, 42
        )
        self.assertEqual(metadata["method"], "sklearn_train_test_split")
        self.assertEqual(set(training["candidate_id"]), set(expected_training["candidate_id"]))
        self.assertEqual(set(test["candidate_id"]), set(expected_test["candidate_id"]))

    def test_seed_42_is_reproducible(self) -> None:
        first_training, first_test, _ = MODULE.create_stratified_split(
            self.grouped, 0.20, 42
        )
        second_training, second_test, _ = MODULE.create_stratified_split(
            self.grouped, 0.20, 42
        )
        self.assertEqual(
            list(first_training["candidate_id"]),
            list(second_training["candidate_id"]),
        )
        self.assertEqual(list(first_test["candidate_id"]), list(second_test["candidate_id"]))

    def test_another_seed_can_change_membership(self) -> None:
        _, seed_42_test, _ = MODULE.create_stratified_split(self.grouped, 0.20, 42)
        _, seed_43_test, _ = MODULE.create_stratified_split(self.grouped, 0.20, 43)
        self.assertNotEqual(
            set(seed_42_test["candidate_id"]),
            set(seed_43_test["candidate_id"]),
        )

    def test_split_has_no_overlap_and_every_class_in_both(self) -> None:
        training, test, _ = MODULE.create_stratified_split(self.grouped, 0.20, 42)
        validation = MODULE.validate_split(
            self.grouped, training, test, 0.20
        )
        self.assertFalse(set(training["candidate_id"]) & set(test["candidate_id"]))
        self.assertTrue(validation["checks"]["every_class_appears_in_both_sets"])
        self.assertTrue(validation["checks"]["candidate_text_original_unchanged"])

    def test_exact_duplicates_remain_in_one_partition(self) -> None:
        duplicated = self.eligible.copy()
        duplicated.loc[duplicated.index[1], "candidate_text_original"] = duplicated.loc[
            duplicated.index[0], "candidate_text_original"
        ]
        # Deliberately give the identical statements different alignment groups;
        # only normalized statement text controls split grouping.
        duplicated.loc[duplicated.index[1], "alignment_group_id"] = "ALIGN_DIFFERENT"
        grouped = MODULE.assign_duplicate_groups(duplicated)
        training, test, metadata = MODULE.create_stratified_split(grouped, 0.20, 42)
        self.assertEqual(
            metadata["method"],
            "deterministic_weighted_duplicate_group_allocation",
        )
        target_group = grouped.iloc[0]["duplicate_group_id"]
        in_training = target_group in set(training["duplicate_group_id"])
        in_test = target_group in set(test["duplicate_group_id"])
        self.assertNotEqual(in_training, in_test)
        MODULE.validate_split(grouped, training, test, 0.20)

    def test_output_columns_and_round_trip_export(self) -> None:
        training, test, split_metadata = MODULE.create_stratified_split(
            self.grouped, 0.20, 42
        )
        excluded = self.eligible.iloc[0:0].copy()
        excluded["exclusion_reasons"] = pd.Series(dtype=object)
        duplicate_audit, near_summary = MODULE.build_duplicate_audit(self.grouped)
        duplicate_summary = MODULE.duplicate_statistics(self.grouped)
        summary = MODULE.build_split_summary(
            self.grouped,
            training,
            test,
            excluded,
            duplicate_summary,
            near_summary,
        )
        readme = MODULE.build_readme_sheet(
            review_path=Path("review.xlsx"),
            summary_path=Path("summary.xlsx"),
            run_id="SPLIT_000001",
            created_at="2026-08-03T00:00:00+00:00",
            test_size=0.20,
            random_state=42,
            split_method=split_metadata["method"],
            near_duplicate_threshold=0.90,
        )
        manifest = {
            "task_id": MODULE.TASK_ID,
            "script_id": MODULE.SCRIPT_ID,
            "script_name": MODULE.SCRIPT_NAME,
            "run_id": "SPLIT_000001",
            "matching_id": "MATCH_000001",
            "split_date": "2026-08-03",
            "created_at": "2026-08-03T00:00:00+00:00",
            "source": {
                "eligibility_count_validation": {"status": "passed"},
                "review_adjustment_validation": {"status": "passed"},
            },
            "eligibility_rules": [],
            "eligibility_policy": {
                "expected_eligible_counts": MODULE.EXPECTED_ELIGIBLE_COUNTS,
                "note": MODULE.ELIGIBILITY_POLICY_NOTE,
            },
            "exclusions": {},
            "split": {},
            "counts": {
                "eligible_total": len(training) + len(test),
                "training_total": len(training),
                "test_total": len(test),
            },
            "duplicate_audit": {},
            "training_review_record_ids": training["review_record_id"].tolist(),
            "training_candidate_ids": training["candidate_id"].tolist(),
            "test_review_record_ids": test["review_record_id"].tolist(),
            "test_candidate_ids": test["candidate_id"].tolist(),
            "quality_checks": {},
            "modeling_steps_performed": [],
            "execution_status": "Completed",
        }
        with tempfile.TemporaryDirectory() as temp_name:
            paths, verification = MODULE.export_outputs(
                output_dir=Path(temp_name),
                training=training,
                test=test,
                excluded=excluded,
                duplicate_audit=duplicate_audit,
                split_summary=summary,
                readme=readme,
                manifest=manifest,
            )
            self.assertEqual(set(paths), {
                MODULE.TRAINING_FILENAME,
                MODULE.TEST_FILENAME,
                MODULE.WORKBOOK_FILENAME,
                MODULE.MANIFEST_FILENAME,
            })
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertTrue(verification["candidate_text_round_trip_exact"])
            self.assertEqual(
                verification["workbook_sheets"],
                ["Training", "Test", "Excluded", "Duplicate Check", "Split Summary", "README"],
            )
            payload = json.loads(
                paths[MODULE.MANIFEST_FILENAME].read_text(encoding="utf-8")
            )
            payload["output_files"][MODULE.TRAINING_FILENAME]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Manifest hash mismatch"):
                MODULE.validate_manifest_payload(
                    payload,
                    training=training,
                    test=test,
                    output_paths={
                        MODULE.TRAINING_FILENAME: paths[MODULE.TRAINING_FILENAME],
                        MODULE.TEST_FILENAME: paths[MODULE.TEST_FILENAME],
                        MODULE.WORKBOOK_FILENAME: paths[MODULE.WORKBOOK_FILENAME],
                    },
                )

    def test_split_registry_records_dated_event_and_lightweight_membership(self) -> None:
        grouped = MODULE.assign_duplicate_groups(make_balanced_eligible(4))
        training, test, split_metadata = MODULE.create_stratified_split(
            grouped, 0.20, 42
        )
        created_at = "2026-08-04T12:00:00+00:00"
        review_hash = "a" * 64
        summary_hash = "b" * 64

        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            database_path = temp_dir / "corpus_inventory.sqlite"
            output_paths = {
                MODULE.TRAINING_FILENAME: temp_dir / MODULE.TRAINING_FILENAME,
                MODULE.TEST_FILENAME: temp_dir / MODULE.TEST_FILENAME,
                MODULE.WORKBOOK_FILENAME: temp_dir / MODULE.WORKBOOK_FILENAME,
                MODULE.MANIFEST_FILENAME: temp_dir / MODULE.MANIFEST_FILENAME,
            }
            for filename, path in output_paths.items():
                path.write_text(f"test artifact: {filename}\n", encoding="utf-8")

            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE document_inventory (document_id TEXT PRIMARY KEY);
                    CREATE TABLE segmentation_events (segmentation_id TEXT PRIMARY KEY);
                    CREATE TABLE candidate_statements (
                        segmentation_id TEXT NOT NULL,
                        statement_id TEXT NOT NULL,
                        PRIMARY KEY (segmentation_id, statement_id)
                    );
                    CREATE TABLE matching_events (
                        matching_id TEXT PRIMARY KEY,
                        segmentation_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        alignment_review_hash_sha256 TEXT NOT NULL,
                        alignment_summary_hash_sha256 TEXT NOT NULL
                    );
                    INSERT INTO document_inventory VALUES ('DOC_000001');
                    INSERT INTO segmentation_events VALUES ('SEG_000001');
                    INSERT INTO matching_events VALUES (
                        'MATCH_000001', 'SEG_000001', 'DOC_000001',
                        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                        'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
                    );
                    """
                )
                connection.executemany(
                    "INSERT INTO candidate_statements VALUES (?, ?)",
                    [
                        ("SEG_000001", str(candidate_id))
                        for candidate_id in grouped["candidate_id"]
                    ],
                )

            class_counts = {
                "eligible_by_class": MODULE._counts_by_class(grouped),
                "training_by_class": MODULE._counts_by_class(training),
                "test_by_class": MODULE._counts_by_class(test),
            }
            manifest = {
                "run_id": "SPLIT_000001",
                "matching_id": "MATCH_000001",
                "split_date": "2026-08-04",
                "created_at": created_at,
                "script_version": MODULE.SCRIPT_VERSION,
                "source": {
                    "review_workbook": {
                        "path": "review.xlsx",
                        "sha256": review_hash,
                    },
                    "summary_workbook": {
                        "path": "summary.xlsx",
                        "sha256": summary_hash,
                    },
                },
                "eligibility_rules": ["included and reviewed"],
                "exclusions": {"excluded_records": 0},
                "split": {
                    "test_proportion_requested": 0.20,
                    "random_seed": 42,
                    "stratification_field": "reviewer_gold_label",
                    "grouping_field": "duplicate_group_id",
                    "method": split_metadata["method"],
                },
                "counts": {
                    "eligible_total": len(grouped),
                    "training_total": len(training),
                    "test_total": len(test),
                    **class_counts,
                },
                "duplicate_audit": {
                    "normalization": "Unicode NFKC; lowercase; collapse whitespace; trim",
                    "near_duplicate_threshold": 0.90,
                },
                "quality_checks": {
                    "observed_test_proportion": len(test) / len(grouped),
                    "checks": {"duplicate_groups_do_not_cross_partitions": True},
                },
                "execution_status": "Completed",
                "output_files": {
                    filename: {
                        "path": path.name,
                        "sha256": MODULE.sha256_file(path),
                    }
                    for filename, path in output_paths.items()
                    if filename != MODULE.MANIFEST_FILENAME
                },
            }
            output_paths[MODULE.MANIFEST_FILENAME].write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            matching_event = MODULE.resolve_matching_event(
                database_path,
                review_hash_sha256=review_hash,
                summary_hash_sha256=summary_hash,
            )
            registration = MODULE.register_split_run(
                database_path=database_path,
                matching_event=matching_event,
                manifest=manifest,
                training=training,
                test=test,
                output_paths=output_paths,
            )
            self.assertEqual(registration["split_date"], "2026-08-04")
            self.assertEqual(registration["assignment_records"], len(grouped))

            with sqlite3.connect(database_path) as connection:
                event = connection.execute(
                    "SELECT split_date, matching_id FROM split_events"
                ).fetchone()
                assignments = connection.execute(
                    "SELECT partition, COUNT(*) FROM split_assignments GROUP BY partition"
                ).fetchall()
                foreign_key_violations = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
            self.assertEqual(event, ("2026-08-04", "MATCH_000001"))
            self.assertEqual(dict(assignments), {
                "train": len(training),
                "test": len(test),
            })
            self.assertEqual(foreign_key_violations, [])
            self.assertEqual(
                MODULE.next_split_run_id(temp_dir / "fresh_outputs", database_path),
                "SPLIT_000002",
            )
            with self.assertRaisesRegex(ValueError, "append-only"):
                MODULE.register_split_run(
                    database_path=database_path,
                    matching_event=matching_event,
                    manifest=manifest,
                    training=training,
                    test=test,
                    output_paths=output_paths,
                )


if __name__ == "__main__":
    unittest.main()
