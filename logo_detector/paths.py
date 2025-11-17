"""Centralized filesystem paths used across the project."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

# Intermediate and derived artifacts
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PATCH_DIR = PROCESSED_DIR / "patches"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ANNOTATIONS_CSV = INTERIM_DIR / "annotations.csv"
PATCH_MANIFEST = PROCESSED_DIR / "patches_manifest.csv"


def ensure_structure() -> None:
    """Create the directories that store derived artifacts if needed."""
    for path in (
        INTERIM_DIR,
        PROCESSED_DIR,
        PATCH_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
