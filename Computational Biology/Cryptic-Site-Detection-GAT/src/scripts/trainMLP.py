import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from MLP import MLP
from ResidueDataset import ResidueDataset

if __name__ == "__main__":
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"using {device}")

    ROOT_PATH = Path(__file__).parent.parent.parent
    FOLDS_DIR = ROOT_PATH.resolve() / "data/cryptobench/cryptobench-dataset/folds"
    MODELS_DIR = ROOT_PATH.resolve() / "checkpoints"

    model = MLP().to(device)
    # TODO: pos_weight, how to make dynamic?
    # pos_weight = torch.load(DATA_DIR / "pos_weight.pt", map_location=device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCEWithLogitsLoss()
    # TODO: move hyperparams to config
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    all_folds = [FOLDS_DIR / f"train-fold-{i}.json" for i in range(4)]

    # just using fold 3 as val for now
    # TODO: implement full 4-fold cross-val and average metrics for a better baseline

    train_folds = all_folds[:3]
    training_set = ResidueDataset(train_folds)
    training_generator = torch.utils.data.DataLoader(
        training_set, batch_size=2048, shuffle=True, num_workers=0
    )

    num_pos = len(training_set.positive_set)
    num_neg = len(training_set) - num_pos
    # pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    pos_weight = torch.tensor([math.sqrt(num_neg / num_pos)], dtype=torch.float32).to(
        device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_fold = [all_folds[3]]
    val_set = ResidueDataset(val_fold)
    val_generator = torch.utils.data.DataLoader(
        val_set, batch_size=2048, shuffle=False, num_workers=0
    )

    best_val_auprc = 0
    best_val_auroc = 0
    # TODO: move
    NUM_EPOCHS = 50
    for epoch in range(NUM_EPOCHS):
        print(f"-------------EPOCH {epoch}-------------")

        # training
        model.train()
        num_batches = 0
        running_loss = 0
        for batch in tqdm(training_generator):
            # transfer to gpu
            embeddings = batch["embedding_vector"].to(device)
            labels = batch["label"].to(device)  # [batch_size]

            predictions = model(embeddings)  # [batch_size, 1]
            predictions = predictions.squeeze(
                1
            )  # get rid of the nested 1 dimension to match dimensions of labels
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
                embeddings = batch["embedding_vector"].to(device)
                labels = batch["label"].to(device)
                all_labels.append(labels)

                predictions = model(
                    embeddings
                )  # these are the raw logits, 1 per residue
                predictions = predictions.squeeze(1)
                all_preds.append(predictions)
                loss = criterion(predictions, labels)
                running_loss += loss.item()
                num_batches += 1

            avg_val_loss = running_loss / num_batches
            print(f"avg val loss: {avg_val_loss}")

            # concatenate predictions and labels and move to cpu for metric calculations
            all_preds = torch.cat(all_preds, dim=0)
            all_preds = all_preds.cpu().numpy()
            all_labels = torch.cat(all_labels, dim=0)
            all_labels = all_labels.cpu().numpy()

            auprc = average_precision_score(all_labels, all_preds)
            auroc = roc_auc_score(all_labels, all_preds)
            print(f"AUPRC: {auprc:.3f}")
            print(f"AUROC: {auroc:.3f}")
            if auprc > best_val_auprc:
                best_val_auprc = auprc
                best_val_auroc = auroc
                print("saving model to checkpoint.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                        "auprc": auprc,
                        "auroc": auroc,
                    },
                    MODELS_DIR / f"MLPcheckpoint_{best_val_auprc:.3f}.pt",
                )
            else:
                print("val auprc not higher than previous epoch")
