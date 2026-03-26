from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import LensDataset, Sample, build_train_test_samples
from src.model import build_resnet18_binary
from src.utils import configure_logging, set_seed


import logging


LOGGER = logging.getLogger("lens_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 for gravitational lens classification")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Root containing train/test class folders")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of training set used for validation")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def build_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # reduce rotation
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def build_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def split_train_validation(samples: List[Sample], val_split: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be in the range (0, 1)")

    labels = [sample.label for sample in samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(samples)),
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


from torch.utils.data import WeightedRandomSampler

def build_dataloaders(args: argparse.Namespace):

    train_samples, test_samples = build_train_test_samples(args.data_root)
    train_samples, val_samples = split_train_validation(
        train_samples, args.val_split, args.seed
    )

    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_dataset = LensDataset(train_samples, transform=train_transform)
    val_dataset = LensDataset(val_samples, transform=eval_transform)
    test_dataset = LensDataset(test_samples, transform=eval_transform)

    # 🔥 CLASS IMBALANCE FIX
    train_labels = torch.tensor([s.label for s in train_samples])
    class_counts = torch.bincount(train_labels, minlength=2)

    # 🔥 sampler ke liye (balanced sampling)
    weights = 1.0 / class_counts.float()
    sample_weights = weights[train_labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    

    # 🔥 TRAIN LOADER (no shuffle now)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    LOGGER.info(
        "Samples | train=%d val=%d test=%d | class_counts(train)=%s",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        class_counts.tolist(),
    )

    return train_loader, val_loader, test_loader
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, np.ndarray | float]:

    model.eval()
    all_labels = []
    all_scores = []
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        running_loss += loss.item() * images.size(0)
        all_labels.append(labels.detach().cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

    results: Dict[str, np.ndarray | float] = {
        "y_true": y_true,
        "y_score": y_score,
        "loss": running_loss / len(loader.dataset),
        "auc": float("nan"),
        "fpr": np.array([]),
        "tpr": np.array([]),
    }

    # CORRECTLY INSIDE FUNCTION
    try:
        if len(np.unique(y_true)) > 1:
            results["auc"] = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            results["fpr"] = fpr
            results["tpr"] = tpr
        else:
            results["auc"] = float("nan")
            results["fpr"] = np.array([])
            results["tpr"] = np.array([])
    except Exception:
        results["auc"] = float("nan")
        results["fpr"] = np.array([])
        results["tpr"] = np.array([])

    return results


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ResNet18 (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Gravitational Lens Classification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "val_auc"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "training.log")
    set_seed(args.seed)

    # CORRECT PLACE (inside main)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_num_threads(1)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LOGGER.info("Using device: %s", device)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = build_resnet18_binary(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, device, criterion)
        val_loss = float(val_results["loss"])
        val_auc = float(val_results["auc"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            }
        )
        LOGGER.info(
            "Epoch %02d/%02d | train_loss=%.6f | val_loss=%.6f | val_auc=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_auc,
        )

    eval_results = evaluate(model, test_loader, device, criterion)
    test_loss = float(eval_results["loss"])
    auc = eval_results["auc"]
    LOGGER.info("Test metrics | loss=%.6f | roc_auc=%.6f", test_loss, float(auc))

    history_path = args.output_dir / "history.csv"
    save_history_csv(history, history_path)
    model_path = args.output_dir / "resnet18_lens.pt"
    torch.save(model.state_dict(), model_path)

    fpr = eval_results["fpr"]
    tpr = eval_results["tpr"]
    if len(fpr) > 0 and len(tpr) > 0:
        roc_path = args.output_dir / "roc_curve.png"
        plot_roc(fpr, tpr, float(auc), roc_path)
        LOGGER.info("Saved ROC curve to: %s", roc_path)

    LOGGER.info("Saved history to: %s", history_path)
    LOGGER.info("Saved model weights to: %s", model_path)


if __name__ == "__main__":
    main()