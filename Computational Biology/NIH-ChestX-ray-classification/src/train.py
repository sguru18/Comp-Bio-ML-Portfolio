import torch
from src.models.BaselineCNN import BaselineCNN
from src.datasets.ChestXRayDataset import ChestXRayDataset
from src.transforms import train_transform, val_transform
from pathlib import Path
import yaml

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"using {device}")

ROOT_PATH = Path(__file__).resolve().parent.parent # root dir

with open(ROOT_PATH / "config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = BaselineCNN().to(device)
criterion = nn.BCEWithLogistLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])

training_set = ChestXRayDataset(ROOT_PATH / "data/train.csv", train_transform)
training_generator = torch.utils.data.DataLoader(training_set, config["train"]["batch_size"], shuffle=True, num_workers=6)

val_set = ChestXRayDataset(ROOT_PATH / "data/val.csv", val_transform)
val_generator = torch.utils.data.DataLoader(val_set, config["val"]["batch_size"], shuffle=False, num_workers=6)

for epoch in range(config["train"]["epochs"]):
    print("-------------EPOCH {epoch}-------------")
    
    # training
    model.train()
    for batch in training_generator:
        # transfer to gpu
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        predictions = model(images) # [16, 14]
        loss = criterion(prediction, labels)
        print(f"training loss: {loss}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # validation
    model.eval()
    with torch.set_grad_enabled(False): # what is this???
        for batch in val_generator:
            # transfer to gpu
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
        
            # model computations...
