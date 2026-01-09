import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from config.config import LOOKUP, CLASSES
from pathlib import Path


class ChestXRayDataset(Dataset):

    def __init__(self, csv, transform):

        self.df = pd.read_csv(csv)
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = LOOKUP[
            self.df.iloc[idx]["Image Index"]
        ]  # ie. data/images/images_001/images/00029464_010.png

        label = [0] * 15
        findings = self.df.iloc[idx]["Finding Labels"].split("|")
        for f in findings:
            label[CLASSES[f]] = 1

        sample = {
            "image": self.transform(
                (torchvision.io.read_image(self.BASE_DIR / path))
            ),
            "label": torch.tensor(label, dtype=float32),
        }

        return sample
