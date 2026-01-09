import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from config.config import LOOKUP, CLASSES
from pathlib import Path

# USAGE: ie. 
# training_set = ChestXRayDataset(csv_path, train=True, transforms)
# training_generator = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True, num_workers=6)

class ChestXRayDataset(Dataset):

    # TODO: update to take transform as a param
    def __init__(self, csv, train=True):

        self.df = pd.read_csv(csv)
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.transform = (
            transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(0.1, 0.1)
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5], std=[0.5]
                    ),  # TODO: calculate from dataset
                ]
            )
            if train
            else transforms.Compose(
                [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
            )
        )

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
