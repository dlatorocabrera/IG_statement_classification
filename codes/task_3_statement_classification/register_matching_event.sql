PRAGMA foreign_keys = ON;

BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS matching_events (
    matching_id TEXT PRIMARY KEY,
    segmentation_id TEXT NOT NULL,
    structure_id TEXT NOT NULL,
    normalization_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    matching_date TEXT NOT NULL,
    matching_tool TEXT NOT NULL,
    matching_tool_version TEXT NOT NULL,
    matching_version TEXT NOT NULL,
    reference_file_name TEXT NOT NULL,
    reference_file_path TEXT NOT NULL,
    reference_drive_id TEXT,
    reference_drive_url TEXT,
    reference_hash_sha256 TEXT NOT NULL,
    candidate_csv_path TEXT NOT NULL,
    candidate_drive_id TEXT,
    candidate_drive_url TEXT,
    candidate_hash_sha256 TEXT NOT NULL,
    candidate_statements_count INTEGER NOT NULL,
    reference_statements_count INTEGER NOT NULL,
    alignment_groups_count INTEGER NOT NULL,
    review_records_count INTEGER NOT NULL,
    linked_candidate_records_count INTEGER NOT NULL,
    unmatched_candidate_records_count INTEGER NOT NULL,
    outside_scope_candidate_records_count INTEGER NOT NULL,
    workbook_only_records_count INTEGER NOT NULL,
    matching_method TEXT NOT NULL,
    normalization_rules_json TEXT NOT NULL,
    similarity_formula TEXT NOT NULL,
    alignment_parameters_json TEXT NOT NULL,
    label_mapping_json TEXT NOT NULL,
    match_type_counts_json TEXT NOT NULL,
    review_outcome_counts_json TEXT NOT NULL,
    alignment_review_path TEXT NOT NULL,
    alignment_review_hash_sha256 TEXT NOT NULL,
    alignment_summary_path TEXT NOT NULL,
    alignment_summary_hash_sha256 TEXT NOT NULL,
    matching_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    CHECK (candidate_statements_count >= 0),
    CHECK (reference_statements_count >= 0),
    CHECK (alignment_groups_count >= 0),
    CHECK (review_records_count >= 0),
    CHECK (linked_candidate_records_count >= 0),
    CHECK (unmatched_candidate_records_count >= 0),
    CHECK (outside_scope_candidate_records_count >= 0),
    CHECK (workbook_only_records_count >= 0),
    CHECK (json_valid(normalization_rules_json)),
    CHECK (json_valid(alignment_parameters_json)),
    CHECK (json_valid(label_mapping_json)),
    CHECK (json_valid(match_type_counts_json)),
    CHECK (json_valid(review_outcome_counts_json)),
    FOREIGN KEY (segmentation_id)
        REFERENCES segmentation_events(segmentation_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (structure_id)
        REFERENCES structure_events(structure_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (normalization_id)
        REFERENCES normalization_events(normalization_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    FOREIGN KEY (document_id)
        REFERENCES document_inventory(document_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_matching_events_document_created
    ON matching_events(document_id, created_at);

CREATE INDEX IF NOT EXISTS idx_matching_events_segmentation
    ON matching_events(segmentation_id);

INSERT INTO matching_events (
    matching_id,
    segmentation_id,
    structure_id,
    normalization_id,
    document_id,
    matching_date,
    matching_tool,
    matching_tool_version,
    matching_version,
    reference_file_name,
    reference_file_path,
    reference_drive_id,
    reference_drive_url,
    reference_hash_sha256,
    candidate_csv_path,
    candidate_drive_id,
    candidate_drive_url,
    candidate_hash_sha256,
    candidate_statements_count,
    reference_statements_count,
    alignment_groups_count,
    review_records_count,
    linked_candidate_records_count,
    unmatched_candidate_records_count,
    outside_scope_candidate_records_count,
    workbook_only_records_count,
    matching_method,
    normalization_rules_json,
    similarity_formula,
    alignment_parameters_json,
    label_mapping_json,
    match_type_counts_json,
    review_outcome_counts_json,
    alignment_review_path,
    alignment_review_hash_sha256,
    alignment_summary_path,
    alignment_summary_hash_sha256,
    matching_status,
    created_at,
    notes
) VALUES (
    'MATCH_000001',
    'SEG_000003',
    'STRUCT_000010',
    'NORM_000004',
    'CL_MMA_DEC_000001',
    '2026-08-03',
    'ChatGPT-assisted semiautomatic workflow',
    'not recorded',
    '1.0',
    'PLAN_INSUMO.xlsx',
    'documents/goal/PLAN_INSUMO.xlsx',
    '161G8kTJ08pJnQLZ9Kgi7-IX7aXFHUrcE',
    'https://drive.google.com/file/d/161G8kTJ08pJnQLZ9Kgi7-IX7aXFHUrcE',
    'fd9f8a74973ccb669479fa9769f1a337e2e9e8a8a64829e99a2b8530d6c7387b',
    'tables/statement_segmentation/CL_MMA_DEC_000001/STRUCT_000010_SEG_000003_candidate_statements.csv',
    '14uCTUkCtvRmEUvSOFNY8iyXxa57EQHgS',
    'https://drive.google.com/file/d/14uCTUkCtvRmEUvSOFNY8iyXxa57EQHgS',
    '48c408712cd241b198aa1aed19f16ee1b125829179ba9beb4847f5fdf796d5ea',
    767,
    365,
    665,
    768,
    438,
    51,
    278,
    1,
    'Order-constrained dynamic alignment using normalized textual similarity, allowing 1:1, split, merge, and complex blocks; original source texts were preserved for human review.',
    '{"matching_only":true,"remove_accents":true,"lowercase":true,"standardize_degree_and_number_symbols":true,"remove_leading_article_label":true,"remove_bracketed_text":true,"remove_quotes":true,"replace_other_punctuation_with_spaces":true,"collapse_whitespace":true,"preserve_original_source_texts":true}',
    '0.34*token_containment + 0.31*token_Dice + 0.28*token_bigram_Dice + 0.07*normalized_character_length_ratio',
    '{"order_constraint":"document order","candidate_block_max":4,"reference_block_max":3,"article_gap_greater_than_one_disallowed":true,"minimum_similarity_considered":0.14,"skip_penalty":-0.34,"block_penalty":"applied; numeric value not recorded","article_gap_penalty":"applied; numeric value not recorded","allowed_relationships":["1:1","split","merge","complex"]}',
    '{"constitutiva":"constitutive","named_regulative_rule_types":"regulative","outside_coverage":"semantic proposal requiring human review","proposed_gold_label_is_final":false}',
    '{"exact":228,"fuzzy":23,"split":154,"merge":23,"complex":10,"unmatched_candidate":51,"outside_workbook_scope":278,"workbook_only":1}',
    '{"include_in_final_set":{"yes":703,"no":43,"hold":22},"reviewer_gold_label":{"regulative":347,"constitutive":163,"non_institutional":211,"exclude":16,"undetermined":31},"downstream_eligible_candidate_records":{"total":702,"regulative":339,"constitutive":154,"non_institutional":209},"downstream_eligibility_rule":"candidate_id present AND include_in_final_set=yes AND reviewer_gold_label in the three target classes"}',
    'outputs/reports/IG_statement_alignment_review.xlsx',
    '7e88d9f58f0ce487659f8c9459499f628502a39727d422c7711d962d68e1861c',
    'outputs/reports/IG_statement_alignment_summary.xlsx',
    'badfde4056769faba3a085b36b94a05e74a1ed7336207a7601b2942bbb5c36fc',
    'Completed with human review decisions recorded',
    '2026-08-04T16:34:04+00:00',
    'The review workbook was created with ChatGPT; the exact model/tool version and numeric block/article-gap penalties were not recorded, so they are explicitly marked as unavailable. The review_status field retains the original review-queue categories. REV_0768 is the single workbook-only gap: it has no candidate_id and is therefore excluded from downstream split eligibility.'
);

COMMIT;
