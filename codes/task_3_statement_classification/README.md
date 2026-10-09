# Task 3: statement-classification train/test split

`07_create_train_test_split.py` converts the completed alignment review into
raw, reproducible statement-level training and test datasets. It does not fit a
vectorizer or classifier.

## Install

From the project root:

```bash
ig_env/bin/python -m pip install -r requirements.txt
```

Task 3 adds the pinned `scikit-learn` dependency. Tests use the Python standard
library's `unittest`; no separate test dependency is required.

## Run

```bash
ig_env/bin/python codes/task_3_statement_classification/07_create_train_test_split.py \
  --input outputs/reports/IG_statement_alignment_review.xlsx \
  --output-dir outputs/packages/task_3_statement_classification \
  --database data/database/corpus_inventory.sqlite \
  --sheet "Alignment Review" \
  --test-size 0.20 \
  --random-state 42 \
  --allow-count-mismatch
```

The matched `IG_statement_alignment_summary.xlsx` is loaded automatically from
the review workbook's folder. Use `--summary-input` only when the matched file is
elsewhere. Project-relative and absolute paths are supported.

`--allow-count-mismatch` permits a deliberately reviewed workbook version with
different expected raw or eligible totals. It never bypasses candidate-ID
uniqueness, review/summary lineage mismatches, blank-field exclusion, or
conflicting reviewed labels within an exact normalized-text group.

## Rules

- Eligibility requires `include_in_final_set=yes`, an approved reviewed label,
  and nonblank `candidate_id` and `candidate_text_original` values.
- The corrected reviewed source has 703 included rows: 339 regulative,
  155 constitutive, and 209 non-institutional. This reflects the authorized
  correction of `REV_0752` from constitutive to regulative so that it matches
  exact-text duplicate `REV_0283`.
- The authorized final eligibility target is 702 statements: 339 regulative,
  154 constitutive, and 209 non-institutional. `REV_0768` is a workbook-only gap,
  not a candidate statement, and is excluded because its candidate ID and text
  are blank.
- `candidate_text_original` is exported unchanged and is the only modeling text.
- `reviewer_gold_label` is the target; `proposed_gold_label` is never used.
- Unicode NFKC normalization, lowercase conversion, whitespace collapse, and
  trimming are used only for duplicate detection.
- Exact duplicates receive a stable content-hash group ID and cannot cross
  partitions. `alignment_group_id` is audit metadata, not a split group.
- Exact duplicates with conflicting reviewed labels stop execution.
- Possible near duplicates are reported with `difflib.SequenceMatcher` and are
  not automatically excluded or grouped.
- If every text is unique, the script uses the specified scikit-learn
  `train_test_split` call. Otherwise it selects a deterministic, label-stratified
  weighted group allocation closest to the requested record and class counts.

## Outputs

Successful runs are staged and verified before publication to
`outputs/packages/task_3_statement_classification/`:

- `IG_statement_training_set.csv`
- `IG_statement_test_set.csv`
- `IG_statement_train_test_split.xlsx`
- `IG_statement_split_manifest.json`

The same validated files are copied to
`outputs/packages/task_3_statement_classification/history/SPLIT_######/`.
SQLite points to this immutable run-specific copy, while the files at the
package root remain the convenient latest version for downstream modeling.

The workbook contains `Training`, `Test`, `Excluded`, `Duplicate Check`,
`Split Summary`, and `README`. The manifest records source and output hashes,
the `SPLIT_######` run ID, the matching-event ID, split date, UTC creation time,
rules, exclusions, duplicate handling, split membership, QA results, and
execution status.

After all files and validations pass, the script registers the run in
`data/database/corpus_inventory.sqlite`:

- `split_events` stores one dated, append-only process record per split run,
  including parameters, counts, class distributions, validation results, paths,
  and hashes.
- `split_assignments` stores only the review/candidate identifiers, partition,
  reviewed label, and duplicate-group membership for each eligible record. It
  does not duplicate statement text or model features in SQLite.

When the reviewed corpus grows, a new `SPLIT_######` event and dated membership
snapshot are added. Prior split records are never overwritten. A split can be
registered only when the exact review and summary hashes match a previously
registered `matching_events` row.

## Tests

```bash
ig_env/bin/python -m unittest discover \
  -s codes/task_3_statement_classification \
  -p 'test_07_create_train_test_split.py' -v
```

The suite covers eligibility, valid labels, blank fields, source-total checks,
candidate-ID uniqueness, conflicting duplicate labels, stable duplicate IDs,
near-duplicate reporting, direct and group-aware splitting, partition isolation,
class coverage, seed reproducibility, original-text preservation, and CSV/XLSX
round-trip verification.

## Production validation (2026-08-03)

The supplied workbooks retain 768 review rows and the 703/43/22 include/no/hold
totals. The corrected included-label distribution is 339/155/209 because
`REV_0752` changed from constitutive to regulative. Under the authorized policy,
`REV_0768` is intentionally excluded and the expected modelable dataset contains
702 statements with a 339/154/209 label distribution.

The original task baseline of 338/156/209 remains in source validation, so the
production command uses `--allow-count-mismatch`. The manifest records the
reviewed +1 regulative / -1 constitutive adjustment as an explicit override.
Record-level preflight checks also bind those notes to REV_0752/REV_0283's
matching regulative text and REV_0768's blank ID/text condition.

- Source-pair validation passed across 768 records with zero differing cells.
- Eligibility validation passed at 702 records (339/154/209); `REV_0768` is in
  `Excluded` for blank candidate text and blank candidate ID.
- `SPLIT_000005`, dated 2026-08-04, contains 561 training records
  (271/123/167) and 141 test records (68/31/42). It is linked to
  `MATCH_000001` in `split_events`, with all 702 lightweight memberships in
  `split_assignments`.
- The duplicate audit found 28 multi-record exact-text groups covering 63
  records, with zero label conflicts and zero groups crossing partitions.
- Seed 42 reproduced identical membership; seed 43 changed membership while
  retaining the same partition sizes.
- All 17 automated tests passed, including dated database registration,
  append-only IDs, membership counts, and foreign-key validation.

The source workbooks are never modified by this script.

## Stage 08: TF-IDF vectorization

`08_vectorize_statements.py` creates reusable sparse TF-IDF representations
without fitting or evaluating a classifier. It reads the immutable training and
test CSV paths registered for a completed split and fits the vectorizer on
training text only.

```bash
ig_env/bin/python codes/task_3_statement_classification/08_vectorize_statements.py \
  --database data/database/corpus_inventory.sqlite \
  --split-id SPLIT_000005 \
  --output-dir outputs/task_3_statement_classification/vectorization
```

Each run writes an append-only `VECT_######/` directory containing the sparse
training and test matrices, row-index CSVs, feature/IDF table, and fitted
vectorizer. No vectorization JSON manifest is created: `vectorization_events`
is the authoritative process record and `vectorization_artifacts` registers the
six external files, hashes, sizes, and descriptions. Matrices, complete texts,
and fitted binaries remain outside SQLite.

The configured word analyzer retains unigrams, bigrams, and trigrams; preserves
accents and stop words; and deliberately retains the Spanish one-character
tokens `a`, `e`, `o`, `u`, and `y`. It performs no stemming, lemmatization,
supervised feature selection, classifier fitting, or fixed 70-feature limit.

```bash
ig_env/bin/python -m unittest \
  codes/task_3_statement_classification/test_08_vectorize_statements.py -v
```

The production run `VECT_000001` was created on 2026-08-04 from
`SPLIT_000005`. It produced a 23,464-feature vocabulary, a sparse training
matrix with shape 561 × 23,464, and a sparse test matrix with shape
141 × 23,464. All input, read-back, finite-value, row-order, vocabulary/IDF,
artifact-hash, database-lineage, and foreign-key validations passed.

## Stage 09: proportional Random Forest Model A

`09_train_random_forest_model_a.py` compares an unweighted Random Forest (A1)
with an otherwise identical class-balanced adaptation (A2). Each model uses a
`SelectKBest` mutual-information selector inside the fitted pipeline, so feature
selection is repeated using only the training observations in every one of the
five stratified cross-validation folds. Both configurations use the same folds.

The selector retains the Wróblewska-inspired vocabulary proportion rather than
the reference absolute count: `round(23,464 × 70 / 3,770) = 436` features. The
configuration with the greatest mean cross-validation macro-F1 is retained as
the primary CV reference; A1 is the predefined winner of a stored-precision
tie. Both predefined configurations are then fitted on all training records and
reported on the held-out partition. Test scores are descriptive and are not
used for feature, weight, parameter, or model selection.

```bash
ig_env/bin/python codes/task_3_statement_classification/09_train_random_forest_model_a.py \
  --database data/database/corpus_inventory.sqlite \
  --vectorization-id VECT_000001 \
  --output-dir outputs/task_3_statement_classification/models
```

Every run writes six files to an append-only `MODEL_######/` directory. The
model artifact is a bundle containing both final pipelines, and prediction and
confusion-matrix files use long form with an explicit configuration ID. SQLite
tables `model_events` and `model_artifacts` are the authoritative process and
artifact registry; no separate model JSON manifest is created. Sparse inputs,
candidate-level predictions, statement text, fitted binaries, and workbooks
remain outside SQLite.

```bash
ig_env/bin/python -m unittest discover \
  -s codes/task_3_statement_classification \
  -p 'test_*.py' -v
```

`MODEL_000001`, dated 2026-08-04, remains the immutable original winner-only
report. The revised production run `MODEL_000002`, dated 2026-08-05, is linked
through `MATCH_000001 → SPLIT_000005 → VECT_000001`. A1 remains the primary CV
reference with mean macro-F1 0.727893 (A2: 0.722988). On the 141-record held-out
partition, A1 obtained macro-F1 0.764813, weighted F1 0.782467, and accuracy
0.787234; A2 obtained macro-F1 0.786942, weighted F1 0.804608, and accuracy
0.808511. Because A1's held-out result was known before the reporting scope was
expanded, the paired held-out comparison is explicitly registered as post-hoc
relative to the initial run and is not used to select or tune a configuration.
All 26 available Task 3 tests and all model input, fold, output, reload,
artifact-hash, database-integrity, and foreign-key validations passed.

## Stage 10: full-feature Random Forest Model B

`10_train_random_forest_model_b.py` is the direct full-vocabulary counterpart
to Model A. B1 is unweighted and B2 uses `class_weight="balanced"`; all other
Random Forest parameters match A1/A2. Neither Model B pipeline performs feature
selection: both classifiers receive all 23,464 registered TF-IDF columns.

Model B reuses the exact five stratified folds generated from the unchanged 561
training labels. Its CV artifact and workbook combine the registered A1/A2
results from `MODEL_000002` with newly computed B1/B2 results, providing one
four-configuration comparison. Both B configurations are then fitted on the
complete training partition and reported on the held-out partition without
using test scores for selection or tuning.

```bash
ig_env/bin/python codes/task_3_statement_classification/10_train_random_forest_model_b.py \
  --database data/database/corpus_inventory.sqlite \
  --vectorization-id VECT_000001 \
  --output-dir outputs/task_3_statement_classification/models
```

The production event `MODEL_000003`, dated 2026-08-05, is linked through
`MATCH_000001 → SPLIT_000005 → VECT_000001` and uses `MODEL_000002` as its
registered A-family comparison source. Mean CV macro-F1 values were 0.727893
for A1, 0.722988 for A2, 0.686734 for B1, and 0.689120 for B2. B2 was retained
as Model B's primary CV reference. On the 141-record held-out partition, B1
obtained macro-F1 0.665397, weighted F1 0.702519, and accuracy 0.723404; B2
obtained macro-F1 0.702890, weighted F1 0.734171, and accuracy 0.744681.

The stage generalized the existing `model_events.selected_configuration`
constraint from A1/A2-only to any nonblank configuration ID. The transactional
migration preserved both prior model events and all prior artifact records.
All 31 available Task 3 tests and all Model B feature-count, fold, prediction,
model-reload, workbook, hash, database-integrity, and foreign-key validations
passed.

## Stage 11: nested-RFECV Random Forest Model C

`11_train_random_forest_model_c.py` adapts the feature-selection procedure
identified in Wróblewska's released classifier to Random-Forest recursive
feature elimination with cross-validation (`RFECV`). The publication itself did
not describe or justify the released implementation's minimum of 20 features.
The minimum is therefore registered transparently and scaled to the current
vocabulary: `round(23,464 × 20 / 3,770) = 124`. The corresponding proportions
are `20 / 3,770 = 0.5305%` and `124 / 23,464 ≈ 0.5285%`.

The value 124 is only the smallest subset RFECV may evaluate. It is not the
final feature count. Each selector begins with all 23,464 TF-IDF features and
chooses its optimum by maximizing inner-CV macro-F1 using training records
only. C1 is unweighted and C2 uses `class_weight="balanced"`; the selector's
forest and final classifier otherwise use the same Random Forest specification
as Models A and B.

Model C uses nested validation. The unchanged five stratified outer folds
estimate C1/C2 performance, while a new five-fold stratified RFECV is fitted
inside each outer training fold. Thus, feature elimination never sees the
corresponding outer validation fold. The registered `step=0.05` removes a fixed
5% of the initial vocabulary (1,173 features) per iteration and evaluates 21
subset sizes, including the exact 124-feature minimum and the complete 23,464-
feature representation. This predeclared grid makes nested RFECV operational;
the one-feature default would require 23,341 subset levels per selector fit.

```bash
ig_env/bin/python codes/task_3_statement_classification/11_train_random_forest_model_c.py \
  --database data/database/corpus_inventory.sqlite \
  --vectorization-id VECT_000001 \
  --output-dir outputs/task_3_statement_classification/models \
  --rfecv-step 0.05
```

The production event `MODEL_000004`, dated 2026-08-05, is linked through
`MATCH_000001 → SPLIT_000005 → VECT_000001` and references `MODEL_000002` and
`MODEL_000003` for the A/B comparison. Mean nested outer-CV macro-F1 was
0.695625 for C1 and 0.722149 for C2, so C2 was selected without using the test
partition. Complete-training RFECV selected 124 features for C1 and 3,523 for
C2. The initial version reported only selected C2 on the existing 141-record
test partition: macro-F1 0.730301, weighted F1 0.759094, and accuracy 0.773050.

The append-only reporting revision `MODEL_000005`, also dated 2026-08-05,
preserves `MODEL_000004` and repeats the same training-only procedure. C2
remains the selected configuration because the outer-CV results and feature
counts reproduce exactly. Version 1.1 retains both complete-training pipelines
and reports both predefined configurations post hoc on the unchanged test set.
C1 obtained macro-F1 0.812732, weighted F1 0.826495, and accuracy 0.829787;
C2 reproduced macro-F1 0.730301, weighted F1 0.759094, and accuracy 0.773050.
The better C1 test score is descriptive and does not retroactively change the
CV selection or tune either configuration.

Each Model C run writes eight registered artifacts. In version 1.1, the fitted
bundle contains both complete-training pipelines, while test predictions and
confusion matrices contain explicit C1/C2 configuration identifiers. The
selected-features artifact continues to describe selected C2. Together with the
six-model CV table and performance workbook, the stage exports
`model_c_rfecv_feature_curve.csv` and
`model_c_macro_f1_vs_features.png`. The curve CSV includes all 12 RFECV fits
(five outer-fold fits plus one complete-training fit for each configuration),
every evaluated feature count, mean/standard-deviation macro-F1, and all five
inner-fold scores. The plot and embedded `RFECV_Feature_Curve` worksheet chart
use only complete-training RFECV results and contain no test performance.

The stage transactionally generalized the model registry from exactly six
artifacts and six enumerated artifact roles to any positive artifact count and
any nonblank role. All prior model and artifact records were preserved. All 38
Task 3 tests, model reload/prediction checks, curve and plot checks, workbook
readback, artifact hashes, database integrity, and foreign-key validations
passed.
