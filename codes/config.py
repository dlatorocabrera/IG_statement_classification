from pathlib import Path

# Project root:
# IG_protocol_project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main folders
CODES_DIR = PROJECT_ROOT / "codes"
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
TABLES_DIR = PROJECT_ROOT / "tables"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Specific folders
SOURCE_PDF_DIR = DOCUMENTS_DIR / "source_pdfs"
PROTOCOL_DIR = DOCUMENTS_DIR / "protocol"

DATABASE_DIR = DATA_DIR / "database"
RAW_TEXT_DIR = DATA_DIR / "raw_text"
QUALITY_CHECKED_DIR = DATA_DIR / "quality_checked"
NORMALIZED_DIR = DATA_DIR / "normalized"
LEGAL_STRUCTURE_DIR = DATA_DIR / "legal_structure"
STATEMENT_SEGMENTATION_DIR = DATA_DIR / "statement_segmentation"

INVENTORY_TABLE_DIR = TABLES_DIR / "inventory"
QUALITY_REPORT_DIR = TABLES_DIR / "quality_reports"
NORMALIZATION_REPORT_DIR = TABLES_DIR / "normalization_reports"
LEGAL_STRUCTURE_REPORT_DIR = TABLES_DIR / "legal_structure"
STATEMENT_SEGMENTATION_REPORT_DIR = TABLES_DIR / "statement_segmentation"

REPORTS_DIR = OUTPUTS_DIR / "reports"
PACKAGES_DIR = OUTPUTS_DIR / "packages"

# Database file
DATABASE_PATH = DATABASE_DIR / "corpus_inventory.sqlite"


def create_project_directories() -> None:
    """Create all expected project directories if they do not exist."""
    directories = [
        SOURCE_PDF_DIR,
        PROTOCOL_DIR,
        DATABASE_DIR,
        RAW_TEXT_DIR,
        QUALITY_CHECKED_DIR,
        NORMALIZED_DIR,
        LEGAL_STRUCTURE_DIR,
        STATEMENT_SEGMENTATION_DIR,
        INVENTORY_TABLE_DIR,
        QUALITY_REPORT_DIR,
        NORMALIZATION_REPORT_DIR,
        LEGAL_STRUCTURE_REPORT_DIR,
        STATEMENT_SEGMENTATION_REPORT_DIR,
        REPORTS_DIR,
        PACKAGES_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print("Project directories created successfully.")
