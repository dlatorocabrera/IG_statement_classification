# Institutional Grammar Protocol Automation

This project implements a reproducible Python pipeline for preparing legal documents, identifying their legal structure, segmenting candidate Institutional Grammar statements, and building statement-classification models. It retains the source documents, structured research data, database, review tables, and model outputs needed to audit the workflow.

## Methodological protocol

**A Computational Protocol for Statement Classification within the Institutional Grammar Framework**  
**Version 1.0, October 2026**

The protocol covers text preparation, candidate-statement segmentation, and statement classification within the Institutional Grammar framework. Its objective is to classify candidate statements as regulative, constitutive, or non-institutional. The classified statements provide a basis for future development of a broader Institutional Grammar coding process.

### Authors

- **Karina Arias-Yurisch** — Escuela de Gobierno, Pontificia Universidad Católica de Chile.
- **Daniel Toro-Cabrera** — Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile.

### Suggested citation

Arias-Yurisch, K., & Toro-Cabrera, D. (2026). *A Computational Protocol for Statement Classification within the Institutional Grammar Framework* (Version 1.0). https://github.com/dlatorocabrera/IG_protocol_project

### Funding

This work was supported by the Concurso de Investigación y Creación Avanza UC 2025 (grant AV25365), Dirección de Investigación, Vicerrectoría de Investigación y Postgrado, Pontificia Universidad Católica de Chile.

### License

© 2026 the authors. The methodological protocol and original project documentation are licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [LICENSE.md](LICENSE.md) for the scope of this notice.

## Authoritative protocol documents

The methodological protocol is available in English and Spanish, accompanied by a technical specification. These documents should be preserved and updated deliberately rather than replaced by this README:

- **[English methodological protocol — Version 1.0](documents/protocol/IG_Automation_Methodological_Protocol.docx)**: explains the methodological workflow, analytical decisions, and quality controls.
- **[Protocolo metodológico en español — Versión 1.0](documents/protocol/IG_Statement_Classification_Protocol_v1.0_ES.docx)**: traducción completa de la versión inglesa, con las mismas tablas, referencias y alcance metodológico. Los identificadores técnicos se mantienen para corresponder con el código y la base de datos.
- **[Technical Database and Data-Structure Specification](documents/protocol/IG_Technical_Database_and_Data_Specification.docx)**: defines the technical implementation, database structure, fields, identifiers, and data-management rules.

Earlier drafts under `documents/protocol/archive/` are retained only as historical records.

## Directory structure

- `codes/`: shared configuration and numbered pipeline scripts.
- `data/`: machine-readable intermediate data and the canonical SQLite database.
- `documents/`: protocol files, source PDFs, references, and project inputs.
- `tables/`: CSV/JSON inventories and review/report tables.
- `outputs/`: task packages, reports, vectorizations, and fitted models.
- `memos_avance/`: project progress memoranda.

Canonical locations used by the code are:

- SQLite database: `data/database/corpus_inventory.sqlite`
- Inventory CSV and JSON: `tables/inventory/`
- Legal-structure JSON: `data/legal_structure/`
- Legal-structure CSV reports: `tables/legal_structure/`

All project paths are derived from `codes/config.py`; the project can therefore be moved or cloned without editing a personal absolute path.

## Python setup

The dependency set is tested with Python 3.11. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The segmentation stage requires the Spanish medium spaCy pipeline. spaCy language models are distributed separately from normal PyPI dependencies, so install the model after `requirements.txt`:

```bash
python -m spacy download es_core_news_md
python -m spacy validate
```

The environment verified for this repository uses `es_core_news_md==3.8.0`, which is the compatible model selected for the pinned `spacy==3.8.14` release.

Virtual environments are machine-specific build artifacts. Never commit `.venv/`, `ig_env/`, or another environment directory.

## Running the pipeline

Run commands from the project root. Create any missing configured folders with:

```bash
python codes/config.py
```

The text-preparation and segmentation stages are:

```bash
python codes/task_1_text_preparation/01_build_document_inventory.py \
  --pdf documents/source_pdfs/DTO-31_24-Establece-plan-de-prevencion-y-descontaminacion-atmosferica-para-la-Region-Metropolitana-de-santiago.pdf

python codes/task_1_text_preparation/02_extract_pdf.py \
  --document-id CL_MMA_DEC_000001

python codes/task_1_text_preparation/03_assess_extraction_quality.py \
  --document-id CL_MMA_DEC_000001

python codes/task_1_text_preparation/04_normalize_text.py \
  --document-id CL_MMA_DEC_000001

python codes/task_1_text_preparation/05_identify_legal_structure.py \
  --document-id CL_MMA_DEC_000001

python codes/task_2_statement_segmentation/06_segment_candidate_statements.py \
  --document-id CL_MMA_DEC_000001
```

Stage 01 writes CSV/JSON to `tables/inventory/` and rebuilds `data/database/corpus_inventory.sqlite`; use its `--out-dir` and `--database` options only when an intentional override is needed. Later stages read the canonical database by default.

Task 3 uses the completed review workbooks and registered database events:

```bash
python codes/task_3_statement_classification/07_create_train_test_split.py
python codes/task_3_statement_classification/08_vectorize_statements.py
python codes/task_3_statement_classification/09_train_random_forest_model_a.py
python codes/task_3_statement_classification/10_train_random_forest_model_b.py
python codes/task_3_statement_classification/11_train_random_forest_model_c.py
```

Use `python <script> --help` to inspect selectors and optional path overrides before rerunning a completed stage. Several Task 3 stages create append-only run directories and register events in SQLite.

## Tests and verification

Run the complete automated test suite with the clean environment:

```bash
python -m unittest discover -s codes/task_3_statement_classification -p 'test_*.py' -v
```

Compile-check all Python files with:

```bash
python -m compileall -q codes
```

The research database, source PDFs, protocol documents, tables, and final results are intentional repository content and are not excluded by `.gitignore`. Before publishing, recheck file sizes and use Git LFS for any binary artifact that approaches the hosting limit.
