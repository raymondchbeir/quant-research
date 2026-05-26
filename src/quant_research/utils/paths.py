from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ALPACA_RAW_DIR = RAW_DATA_DIR / "alpaca"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        ALPACA_RAW_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
