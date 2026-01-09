import torch
from models.BaselineCNN import BaselineCNN

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

model = BaselineCNN().to(device)
