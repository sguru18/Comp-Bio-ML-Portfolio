import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from config.config import LOOKUP
from pathlib import Path


class ChestXRayDataset(Dataset):

    def __init__(self, csv, images, transforms):

        self.df = pd.read_csv(csv)
        self.image_dir = images
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        BASE_DIR = Path(__file__).resolve().parent.parent
        path = LOOKUP[self.df.iloc[idx]["Image Index"]] #ie. data/images/images_0{num}/images/00029464_010.png
        sample = {
            "image": (torchvision.io.read_image(BASE_DIR / path) / 255.) ,
            "label": self.df.iloc[idx]["Finding Labels"] # TODO: have to split by | ?? how to handle multi labels
        }

        
