import pandas as pd
import torch
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # root dir
DATA_DIR = BASE_DIR / "data"

data = pd.read_csv(DATA_DIR / "Data_Entry_2017.csv")

indices = {
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

num_rows = data.shape[0]
num_pos = [0] * 14
for i in range(num_rows):

    label = data.iloc[i]["Finding Labels"]
    findings = label.split("|")
    for f in findings:
        if f == "No Finding":
            continue
        num_pos[indices[f]] += 1

num_neg = [num_rows - x for x in num_pos]
pos_weight = [neg / (pos + 1e-8) for neg, pos in zip(num_neg, num_pos)]
pos_weight = torch.tensor(pos_weight)
torch.save(pos_weight, DATA_DIR / "pos_weight.pt")
