"""Paths shared by preprocessing, weighting, and publication notebooks.

Set LONGCOVID_HOME to a secure study workspace before invoking a command.
Individual reference inputs may be overridden through the corresponding variables.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME_DIR = os.environ.get("LONGCOVID_HOME", str(REPO_ROOT))
DATA_PATH = os.environ.get("LONGCOVID_DATA", os.path.join(HOME_DIR, "data"))
RAW_DATA_PATH = os.environ.get(
    "LONGCOVID_RAW_DATA", os.path.join(DATA_PATH, "raw_data")
)
OUTPUT_DIR = os.environ.get("LONGCOVID_OUTPUTS", os.path.join(HOME_DIR, "outputs"))
BASELINE_DATA_PATH = os.path.join(DATA_PATH, "featurized")
RAW_DATA_BATCH_FOLDER = "batches_main"
FINISHED_INDICATOR_FILE = "finished.txt"

DIAGNOSES_TO_CODES_PROCESSED_PATH = os.environ.get(
    "LONGCOVID_PROCESSED_DIAGNOSES",
    os.path.join(RAW_DATA_PATH, "diagnosis_codes_to_categories_processed.csv"),
)
PROCESSED_DEP_INDEX_PATH = os.environ.get(
    "LONGCOVID_ADI", os.path.join(RAW_DATA_PATH, "dep_index.csv")
)
CENSUS_DATA_PATH = os.environ.get("LONGCOVID_CENSUS", os.path.join(DATA_PATH, "census"))

# A bundled font supplies a portable default; the study fonts can be supplied externally.
from matplotlib import font_manager

HELVETICA_PATH = os.environ.get("LONGCOVID_FONT", font_manager.findfont("DejaVu Sans"))
HELVETICA_LIGHT_PATH = os.environ.get("LONGCOVID_LIGHT_FONT", HELVETICA_PATH)
