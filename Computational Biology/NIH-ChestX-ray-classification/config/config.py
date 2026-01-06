import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
with open(BASE_DIR / "data/lookup.json") as f:
    LOOKUP = json.load(f)