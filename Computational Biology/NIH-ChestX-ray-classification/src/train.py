import torch
import torch.nn as nn
from src.models.BaselineCNN import BaselineCNN
from src.datasets.ChestXRayDataset import ChestXRayDataset
from src.transforms import train_transform, val_transform
from pathlib import Path
import yaml
from tqdm import tqdm

if __name__ == "__main__":
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"using {device}")

    ROOT_PATH = Path(__file__).resolve().parent.parent  # root dir
    DATA_DIR = ROOT_PATH / "data"

    with open(ROOT_PATH / "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = BaselineCNN().to(device)
    pos_weight = torch.load(DATA_DIR / "pos_weight.pt", map_location=device)
    # TODO: make lr, min, max, batch_size, epochs come from config but as ints
    pos_weight = torch.clamp(pos_weight, min=1, max=30)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    training_set = ChestXRayDataset(ROOT_PATH / "data/train.csv", train_transform)
    training_generator = torch.utils.data.DataLoader(
        training_set, batch_size=32, shuffle=True, num_workers=6
    )

    val_set = ChestXRayDataset(ROOT_PATH / "data/val.csv", val_transform)
    val_generator = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False, num_workers=6
    )

    for epoch in range(1):
        print("-------------EPOCH {epoch}-------------")

        # training
        model.train()
        num_batches = 0
        running_loss = 0
        for batch in tqdm(training_generator):
            # transfer to gpu
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            predictions = model(images)  # [16, 15]
            loss = criterion(predictions, labels)
            running_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"avg train loss: {running_loss / num_batches}")

        # validation
        model.eval()
        with torch.set_grad_enabled(False):
            num_batches = 0
            running_loss = 0
            for batch in tqdm(val_generator):
                # transfer to gpu
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                # model computations...
                predictions = model(images)
                loss = criterion(predictions, labels)
                running_loss += loss.item()
                num_batches += 1
            print(f"avg val loss: {running_loss / num_batches}")
