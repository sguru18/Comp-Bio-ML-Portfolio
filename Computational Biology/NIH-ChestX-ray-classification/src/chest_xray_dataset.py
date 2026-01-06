import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class ChestXRayDataset(Dataset):

    def __init__(self, csv, images, transforms):

        self.df = pd.read_csv(csv)
        self.image_dir = images
        self.transforms = transforms

        
