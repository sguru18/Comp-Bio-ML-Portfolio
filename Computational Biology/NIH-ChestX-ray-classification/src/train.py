import torch
import torch.nn as nn
from src.models.BaselineCNN import BaselineCNN
from src.datasets.ChestXRayDataset import ChestXRayDataset
from src.transforms import train_transform, val_transform
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from config.config import CLASSES

if __name__ == "__main__":
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"using {device}")

    ROOT_PATH = Path(__file__).resolve().parent.parent  # root dir
    DATA_DIR = ROOT_PATH / "data"
    MODELS_DIR = ROOT_PATH / "src/models"

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

    best_val_auc = 0
    for epoch in range(10):
        print(f"-------------EPOCH {epoch}-------------")

        # training
        model.train()
        num_batches = 0
        running_loss = 0
        for batch in tqdm(training_generator):
            # transfer to gpu
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            predictions = model(images)  # [16, 14]
            loss = criterion(predictions, labels)
            running_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = running_loss / num_batches
        print(f"avg train loss: {avg_train_loss}")

        # validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.set_grad_enabled(False):
            num_batches = 0
            running_loss = 0
            for batch in tqdm(val_generator):
                # transfer to gpu
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                all_labels.append(labels)

                predictions = model(images)
                all_preds.append(predictions)
                loss = criterion(predictions, labels)
                running_loss += loss.item()
                num_batches += 1

            avg_val_loss = running_loss / num_batches
            print(f"avg val loss: {avg_val_loss}")

            # concatenate predictions and labels and move to cpu to calculate auc-roc
            all_preds = torch.cat(all_preds, dim=0)  # [_, 14]
            all_preds = all_preds.cpu().numpy()
            all_labels = torch.cat(all_labels, dim=0)  # [_, 14]
            all_labels = all_labels.cpu().numpy()

            individual_auc_roc = roc_auc_score(all_labels, all_preds, average=None)
            total_sensitivity = 0
            total_specificity = 0
            for key, value in CLASSES.items():  # value is the index 0 - 13
                print(f"{key}   roc_auc: {individual_auc_roc[value]:.3f}", end=" ")
                
                # calculate sensitivity and specificity
                y_true = all_labels[:, value]  # alreadys 1s and 0s
                # TODO: 0.5 threshold to config.yaml
                y_pred_binary = (all_preds[:, value] > 0.5).astype(int)
                # TODO: move metric calculation from y_true and y_pred to src_shared/metrics
                c = confusion_matrix(y_true, y_pred_binary)
                TN, FP, FN, TP = c[0][0], c[0][1], c[1][0], c[1][1]
                
                if (TP + FN) != 0:
                    sensitivity = TP / (TP + FN)  # AKA recall AKA true positive rate
                    total_sensitivity += sensitivity
                    print(f"recall: {sensitivity:.3f}", end=" ")
                if (TN + FP) != 0:
                    specificity = TN / (TN + FP)  # AKA true negative rate
                    total_specificity += specificity
                    print(f"specificity: {specificity:.3f}", end=" ")

                print("\n")
                print("\n")
                

            single_auc_roc = roc_auc_score(all_labels, all_preds, average="macro")
            macro_recall = total_sensitivity / 14
            macro_specificity = total_specificity / 14
            print(f"MACRO ROC_AUC: {single_auc_roc:.3f}")
            # TODO: move 15 num_classes to config.yaml
            print(f"MACRO RECALL: {macro_recall:.3f}")
            print(f"MACRO SPECIFICITY: {macro_specificity:.3f}")

            if single_auc_roc > best_val_auc:
                best_val_auc = single_auc_roc
                print("saving model to checkpoint.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "macro_auc_roc": single_auc_roc,
                        "avg_val_loss": avg_val_loss,
                        "macro_recall": macro_recall,
                        "macro_specificity": macro_specificity,
                    },
                    MODELS_DIR / "checkpoint.pt",
                )
            else:
                print("val auc_roc not lower than previous epoch")
