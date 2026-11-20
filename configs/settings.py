
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Location: Lahore, Pakistan
LATITUDE = 31.5580
LONGITUDE = 74.3507
TIMEZONE = "Asia/Karachi"

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"

# Create data directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
