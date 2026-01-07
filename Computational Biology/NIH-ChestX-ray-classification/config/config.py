import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
with open(BASE_DIR / "data/lookup.json") as f:
    LOOKUP = json.load(f)

CLASSES = {
    "Atelectasis": 0,
    "Consolidation": 1,
    "Infiltration": 2,
    "Pneumothorax": 3,
    "Edema": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Effusion": 7,
    "Pneumonia": 8,
    "Pleural_Thickening": 9,
    "Cardiomegaly": 10,
    "Nodule": 11,
    "Hernia": 12,
    "Mass": 13,
    "No Finding": 14,
}
