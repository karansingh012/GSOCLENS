from __future__ import annotations

import argparse
import copy
import csv
import logging
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.data import LensDataset, Sample, build_train_test_samples
from src.model import DomainCritic, build_classifier, build_feature_extractor, build_resnet18_binary
from src.utils import configure_logging, set_seed


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
    parser.add_argument("--use-wdgrl", action="store_true", help="Enable WDGRL domain adaptation")
    parser.add_argument("--use-adda", action="store_true", help="Enable ADDA")
    parser.add_argument("--lambda-wd", type=float, default=0.1)
    parser.add_argument("--lambda-adda", type=float, default=0.1)
    parser.add_argument("--improved-adda", action="store_true", help="Enable improved ADDA path (keeps legacy ADDA unchanged)")
    parser.add_argument("--adda-target-loader", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--adda-cls-warmup-epochs", type=int, default=2)
    parser.add_argument("--adda-warmup-epochs", type=int, default=1)
    parser.add_argument("--adda-ramp-epochs", type=int, default=5)
    parser.add_argument("--adda-disc-steps", type=int, default=2)
    parser.add_argument("--adda-encoder-steps", type=int, default=1)
    parser.add_argument("--adda-label-smoothing", type=float, default=0.1)
    parser.add_argument("--adda-grad-clip", type=float, default=5.0)
    parser.add_argument("--adda-cls-weight", type=float, default=1.0)
    parser.add_argument("--adda-adv-weight", type=float, default=1.0)
    parser.add_argument("--adda-finetune-classifier", type=str, choices=["frozen", "full"], default="full")
    parser.add_argument("--adda-target-lr", type=float, default=0.0)
    parser.add_argument("--adda-disc-lr", type=float, default=0.0)
    parser.add_argument("--adda-cls-lr", type=float, default=0.0)
    return parser.parse_args()


def build_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            
            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_robustness_transforms(mode="clean"):
    transforms_list = [
        transforms.Resize((224, 224)),
    ]

    if mode == "blur":
        transforms_list.append(transforms.GaussianBlur(kernel_size=5))

    elif mode == "low_light":
        transforms_list.append(transforms.ColorJitter(brightness=0.3))

    transforms_list.append(transforms.ToTensor())

    if mode == "noise":
        transforms_list.append(
            transforms.Lambda(lambda x: torch.clamp(x + 0.1 * torch.randn_like(x), 0, 1))
        )

    transforms_list.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    return transforms.Compose(transforms_list)

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


def build_dataloaders(args: argparse.Namespace):
    train_samples, test_samples = build_train_test_samples(args.data_root)
    train_samples, val_samples = split_train_validation(train_samples, args.val_split, args.seed)

    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_dataset = LensDataset(train_samples, transform=train_transform)
    val_dataset = LensDataset(val_samples, transform=eval_transform)
    test_dataset = LensDataset(test_samples, transform=eval_transform)

    train_labels = torch.tensor([s.label for s in train_samples])
    class_counts = torch.bincount(train_labels, minlength=2)

    weights = 1.0 / class_counts.float()
    sample_weights = weights[train_labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
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


def train_one_epoch_wdgrl(
    feature_extractor,
    classifier,
    critic,
    loader,
    optimizer,
    criterion,
    device,
    lambda_wd,
):
    feature_extractor.train()
    classifier.train()
    # classifier.eval() 
    critic.train()

    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_extractor(images)

        logits = classifier(features)
        cls_loss = criterion(logits, labels)

        half = images.size(0) // 2
        if half == 0:
            continue

        source_feat = features[:half]
        target_feat = features[half:]

        critic_source = critic(source_feat).mean()
        critic_target = critic(target_feat).mean()

        wd_loss = critic_source - critic_target

        loss = cls_loss + lambda_wd * wd_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def train_one_epoch_adda(
    source_encoder,
    target_encoder,
    classifier,
    discriminator,
    loader,
    optimizer_target,
    optimizer_discriminator,
    criterion_cls,
    criterion_adv,
    device,
    lambda_adda,
):
    source_encoder.eval()
    target_encoder.train()
    classifier.train()
    # classifier.eval()
    discriminator.train()

    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        perm = torch.randperm(images.size(0))
        half = images.size(0) // 2


        if half == 0:
            continue

        source_idx = perm[:half]
        target_idx = perm[half:]

        source_images = images[source_idx]
        source_labels = labels[source_idx]
        target_images = images[target_idx]
        
        

        with torch.no_grad():
            source_features = source_encoder(source_images)

        target_features_detached = target_encoder(target_images).detach()

        src_domain_logits = discriminator(source_features)
        tgt_domain_logits = discriminator(target_features_detached)

        src_domain_labels = torch.ones_like(src_domain_logits)
        tgt_domain_labels = torch.zeros_like(tgt_domain_logits)

        d_loss = criterion_adv(src_domain_logits, src_domain_labels) + criterion_adv(tgt_domain_logits, tgt_domain_labels)

        optimizer_discriminator.zero_grad(set_to_none=True)
        d_loss.backward()
        optimizer_discriminator.step()

        target_features = target_encoder(target_images)
        source_logits = classifier(source_features)
        cls_loss = criterion_cls(source_logits, source_labels)

        fool_domain_logits = discriminator(target_features)
        fool_domain_labels = torch.ones_like(fool_domain_logits)
        adv_loss = criterion_adv(fool_domain_logits, fool_domain_labels)

        loss = cls_loss + lambda_adda * adv_loss

        optimizer_target.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_target.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _linear_lambda_schedule(
    epoch: int,
    max_lambda: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_lambda
    progress = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    return max_lambda * progress


def train_source_classifier_warmup_epoch(
    source_encoder,
    classifier,
    source_loader,
    optimizer_classifier,
    criterion_cls,
    device,
):
    source_encoder.eval()
    classifier.train()

    running_loss = 0.0

    for source_images, source_labels in source_loader:
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)

        with torch.no_grad():
            source_features = source_encoder(source_images)

        source_logits = classifier(source_features)
        cls_loss = criterion_cls(source_logits, source_labels)

        optimizer_classifier.zero_grad(set_to_none=True)
        cls_loss.backward()
        optimizer_classifier.step()

        running_loss += cls_loss.item() * source_images.size(0)

    return running_loss / len(source_loader.dataset)


def train_one_epoch_adda_improved(
    source_encoder,
    target_encoder,
    classifier,
    discriminator,
    source_loader,
    target_loader,
    optimizer_target,
    optimizer_discriminator,
    criterion_cls,
    criterion_adv,
    device,
    lambda_adda,
    disc_steps,
    encoder_steps,
    label_smoothing,
    grad_clip,
    cls_weight,
    adv_weight,
):
    source_encoder.eval()
    target_encoder.train()
    classifier.train()
    discriminator.train()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_adv_loss = 0.0
    running_disc_loss = 0.0

    target_iter = cycle(target_loader)

    smooth = min(max(label_smoothing, 0.0), 0.49)
    src_domain_value = 1.0 - smooth
    tgt_domain_value = smooth

    for source_images, source_labels in source_loader:
        target_images, _ = next(target_iter)

        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        with torch.no_grad():
            source_features = source_encoder(source_images)

        disc_loss_batch = 0.0
        for _ in range(max(1, disc_steps)):
            with torch.no_grad():
                target_features_detached = target_encoder(target_images).detach()

            src_domain_logits = discriminator(source_features)
            tgt_domain_logits = discriminator(target_features_detached)

            src_domain_labels = torch.full_like(src_domain_logits, src_domain_value)
            tgt_domain_labels = torch.full_like(tgt_domain_logits, tgt_domain_value)

            d_loss = 0.5 * (
                criterion_adv(src_domain_logits, src_domain_labels)
                + criterion_adv(tgt_domain_logits, tgt_domain_labels)
            )

            optimizer_discriminator.zero_grad(set_to_none=True)
            d_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
            optimizer_discriminator.step()

            disc_loss_batch += d_loss.item()

        disc_loss_batch /= float(max(1, disc_steps))

        enc_total = 0.0
        enc_cls = 0.0
        enc_adv = 0.0
        for _ in range(max(1, encoder_steps)):
            target_features = target_encoder(target_images)
            source_logits = classifier(source_features)
            cls_loss = criterion_cls(source_logits, source_labels)

            fool_domain_logits = discriminator(target_features)
            fool_domain_labels = torch.full_like(fool_domain_logits, src_domain_value)
            adv_loss = criterion_adv(fool_domain_logits, fool_domain_labels)

            total_loss = (cls_weight * cls_loss) + (lambda_adda * adv_weight * adv_loss)

            optimizer_target.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(target_encoder.parameters(), grad_clip)
                classifier_params = [p for p in classifier.parameters() if p.requires_grad]
                if classifier_params:
                    nn.utils.clip_grad_norm_(classifier_params, grad_clip)
            optimizer_target.step()

            enc_total += total_loss.item()
            enc_cls += cls_loss.item()
            enc_adv += adv_loss.item()

        enc_total /= float(max(1, encoder_steps))
        enc_cls /= float(max(1, encoder_steps))
        enc_adv /= float(max(1, encoder_steps))

        running_total_loss += enc_total * source_images.size(0)
        running_cls_loss += enc_cls * source_images.size(0)
        running_adv_loss += enc_adv * source_images.size(0)
        running_disc_loss += disc_loss_batch * source_images.size(0)

    denom = len(source_loader.dataset)
    return {
        "total_loss": running_total_loss / denom,
        "cls_loss": running_cls_loss / denom,
        "adv_loss": running_adv_loss / denom,
        "disc_loss": running_disc_loss / denom,
    }


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

    try:
        if len(np.unique(y_true)) > 1:
            results["auc"] = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            results["fpr"] = fpr
            results["tpr"] = tpr
    except Exception:
        pass

    return results


@torch.no_grad()
def evaluate_wdgrl(
    feature_extractor,
    classifier,
    loader,
    device,
    criterion,
):
    feature_extractor.eval()
    classifier.eval()

    all_labels = []
    all_scores = []
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_extractor(images)
        logits = classifier(features)

        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        running_loss += loss.item() * images.size(0)

        all_labels.append(labels.detach().cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

    results = {
        "loss": running_loss / len(loader.dataset),
        "auc": float("nan"),
        "fpr": np.array([]),
        "tpr": np.array([]),
    }

    try:
        if len(np.unique(y_true)) > 1:
            results["auc"] = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            results["fpr"] = fpr
            results["tpr"] = tpr
    except Exception:
        pass

    return results



def evaluate_robustness_all(
    model_type,
    model,
    feature_extractor,
    classifier,
    args,
    device,
):
    modes = ["clean", "noise", "blur", "low_light"]
    results = {}

    _, test_samples = build_train_test_samples(args.data_root)

    for mode in modes:
        print(f"\n🔍 Testing: {mode}")

        transform = build_robustness_transforms(mode)
        dataset = LensDataset(test_samples, transform=transform)

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        if model_type == "baseline":
            res = evaluate(model, loader, device, nn.CrossEntropyLoss())
        else:
            res = evaluate_wdgrl(feature_extractor, classifier, loader, device, nn.CrossEntropyLoss())

        results[mode] = res["auc"]
        print(f"{mode} AUC: {res['auc']:.4f}")

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

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_num_threads(1)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LOGGER.info("Using device: %s", device)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    if args.improved_adda and not args.use_adda:
        LOGGER.warning("--improved-adda is set but --use-adda is not enabled. Ignoring improved ADDA options.")

    if args.improved_adda:
        if args.adda_cls_warmup_epochs < 0:
            raise ValueError("--adda-cls-warmup-epochs must be >= 0")
        if args.adda_warmup_epochs < 0:
            raise ValueError("--adda-warmup-epochs must be >= 0")
        if args.adda_ramp_epochs < 0:
            raise ValueError("--adda-ramp-epochs must be >= 0")
        if args.adda_disc_steps <= 0:
            raise ValueError("--adda-disc-steps must be > 0")
        if args.adda_encoder_steps <= 0:
            raise ValueError("--adda-encoder-steps must be > 0")

    adda_target_loader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }[args.adda_target_loader]

    if args.use_wdgrl and args.use_adda:
        raise ValueError("Use either --use-wdgrl or --use-adda, not both.")

    criterion = nn.CrossEntropyLoss()
    criterion_adv = nn.BCEWithLogitsLoss()
    history: List[Dict[str, float]] = []

    # =========================
    # 🔥 MODEL SETUP
    # =========================
    if args.use_wdgrl:
        feature_extractor = build_feature_extractor().to(device)
        classifier = build_classifier().to(device)
        critic = DomainCritic().to(device)

        params = list(feature_extractor.parameters()) + list(classifier.parameters())
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    elif args.use_adda:
        source_encoder = build_feature_extractor().to(device)
        classifier = build_classifier().to(device)

        checkpoint = torch.load("outputs/model_baseline.pt", map_location=device)

        # encoder load
        source_dict = source_encoder.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in source_dict}
        source_dict.update(pretrained_dict)
        source_encoder.load_state_dict(source_dict)

        # ✅ classifier load (CORRECT)
        classifier_dict = classifier.state_dict()

        # ONLY load last layer safely
        if "fc.weight" in checkpoint and "fc.bias" in checkpoint:
            try:
                classifier_dict["2.weight"] = checkpoint["fc.weight"]
                classifier_dict["2.bias"] = checkpoint["fc.bias"]
            except KeyError:
                print("⚠️ Layer index mismatch — check classifier architecture")

        classifier.load_state_dict(classifier_dict, strict=False)        


        # copy to target
        target_encoder = copy.deepcopy(source_encoder).to(device)

        discriminator = DomainCritic().to(device)

        # freeze source
        for p in source_encoder.parameters():
            p.requires_grad = False
        source_encoder.eval()

        # # freeze classifier
        # for p in classifier.parameters():
        #     p.requires_grad = False
        # classifier.eval()

        optimizer_target = Adam(
            target_encoder.parameters(),
            lr=args.lr,
            #lr=args.lr * 0.1,
            weight_decay=args.weight_decay,
        )

        optimizer_discriminator = Adam(
            discriminator.parameters(),
            #lr=args.lr,
            lr=args.lr * 0.1,
            weight_decay=args.weight_decay,
        )

        optimizer_classifier_warmup = None

        if args.improved_adda:
            if args.adda_finetune_classifier == "frozen":
                _set_trainable(classifier, False)
                classifier.eval()
            else:
                _set_trainable(classifier, True)

            target_params = list(target_encoder.parameters())
            if args.adda_finetune_classifier != "frozen":
                target_params += list(classifier.parameters())

            target_lr = args.adda_target_lr if args.adda_target_lr > 0 else args.lr
            disc_lr = args.adda_disc_lr if args.adda_disc_lr > 0 else args.lr * 0.1

            optimizer_target = Adam(
                target_params,
                lr=target_lr,
                weight_decay=args.weight_decay,
            )

            optimizer_discriminator = Adam(
                discriminator.parameters(),
                lr=disc_lr,
                weight_decay=args.weight_decay,
            )

            if args.adda_cls_warmup_epochs > 0:
                cls_lr = args.adda_cls_lr if args.adda_cls_lr > 0 else args.lr
                optimizer_classifier_warmup = Adam(
                    classifier.parameters(),
                    lr=cls_lr,
                    weight_decay=args.weight_decay,
                )

    else:
        model = build_resnet18_binary(pretrained=not args.no_pretrained).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    


        # =========================
    # ⚡ LOAD MODEL (ADD HERE)
    # =========================
    if args.epochs == 0:
        print("⚡ Loading pretrained model for evaluation...")

        if args.use_wdgrl:
            checkpoint = torch.load("outputs_wd/model_wd0.1.pt", map_location=device)
            feature_extractor.load_state_dict(checkpoint["feature_extractor"])
            classifier.load_state_dict(checkpoint["classifier"])

        elif args.use_adda:
            checkpoint = torch.load("outputs_adda/model_adda0.1.pt", map_location=device)
            target_encoder.load_state_dict(checkpoint["target_encoder"])
            classifier.load_state_dict(checkpoint["classifier"])

        else:
            model.load_state_dict(
                torch.load("outputs_baseline/model_baseline.pt", map_location=device)
            )



    # =========================
    # 🔥 TRAINING LOOP
    # =========================
    for epoch in range(1, args.epochs + 1):

        if args.use_wdgrl:
            train_loss = train_one_epoch_wdgrl(
                feature_extractor,
                classifier,
                critic,
                train_loader,
                optimizer,
                criterion,
                device,
                lambda_wd=args.lambda_wd,
            )
            val_results = evaluate_wdgrl(
                feature_extractor,
                classifier,
                val_loader,
                device,
                criterion,
            )

        elif args.use_adda:
            if args.improved_adda:
                warmup_loss = 0.0
                if epoch <= args.adda_cls_warmup_epochs and optimizer_classifier_warmup is not None:
                    # Warmup keeps class boundaries meaningful before adversarial alignment.
                    warmup_loss = train_source_classifier_warmup_epoch(
                        source_encoder,
                        classifier,
                        train_loader,
                        optimizer_classifier_warmup,
                        criterion,
                        device,
                    )

                lambda_epoch = _linear_lambda_schedule(
                    epoch=epoch,
                    max_lambda=args.lambda_adda,
                    warmup_epochs=args.adda_warmup_epochs,
                    ramp_epochs=args.adda_ramp_epochs,
                )

                adda_metrics = train_one_epoch_adda_improved(
                    source_encoder,
                    target_encoder,
                    classifier,
                    discriminator,
                    source_loader=train_loader,
                    target_loader=adda_target_loader,
                    optimizer_target=optimizer_target,
                    optimizer_discriminator=optimizer_discriminator,
                    criterion_cls=criterion,
                    criterion_adv=criterion_adv,
                    device=device,
                    lambda_adda=lambda_epoch,
                    disc_steps=args.adda_disc_steps,
                    encoder_steps=args.adda_encoder_steps,
                    label_smoothing=args.adda_label_smoothing,
                    grad_clip=args.adda_grad_clip,
                    cls_weight=args.adda_cls_weight,
                    adv_weight=args.adda_adv_weight,
                )
                train_loss = float(adda_metrics["total_loss"])

                LOGGER.info(
                    "ADDA(improved) epoch=%d | warmup_cls=%.6f | lambda=%.6f | cls=%.6f | adv=%.6f | disc=%.6f",
                    epoch,
                    warmup_loss,
                    lambda_epoch,
                    float(adda_metrics["cls_loss"]),
                    float(adda_metrics["adv_loss"]),
                    float(adda_metrics["disc_loss"]),
                )
            else:
                train_loss = train_one_epoch_adda(
                    source_encoder,
                    target_encoder,
                    classifier,
                    discriminator,
                    train_loader,
                    optimizer_target,
                    optimizer_discriminator,
                    criterion,
                    criterion_adv,
                    device,
                    lambda_adda=args.lambda_adda,
                )
            val_results = evaluate_wdgrl(
                target_encoder,
                classifier,
                val_loader,
                device,
                criterion,
            )

        else:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
            val_results = evaluate(
                model,
                val_loader,
                device,
                criterion,
            )

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

    # =========================
    # 🔥 FINAL EVALUATION (LOOP KE BAAD)
    # =========================
    if args.use_wdgrl:
        eval_results = evaluate_wdgrl(
            feature_extractor,
            classifier,
            test_loader,
            device,
            criterion,
        )
    elif args.use_adda:
        eval_results = evaluate_wdgrl(
            target_encoder,
            classifier,
            test_loader,
            device,
            criterion,
        )
    else:
        eval_results = evaluate(
            model,
            test_loader,
            device,
            criterion,
        )

    test_loss = float(eval_results["loss"])
    auc = eval_results["auc"]

    LOGGER.info("Test metrics | loss=%.6f | roc_auc=%.6f", test_loss, float(auc))

    history_path = args.output_dir / "history.csv"
    save_history_csv(history, history_path)

    # model_path = args.output_dir / "resnet18_lens.pt"
    if args.use_wdgrl:
        model_path = args.output_dir / f"model_wd{args.lambda_wd}.pt"
    elif args.use_adda:
        model_path = args.output_dir / f"model_adda{args.lambda_adda}.pt"
    else:
        model_path = args.output_dir / "model_baseline.pt"

    if args.use_wdgrl:
        torch.save(
            {
                "feature_extractor": feature_extractor.state_dict(),
                "classifier": classifier.state_dict(),
            },
            model_path,
        )
    elif args.use_adda:
        torch.save(
            {
                "source_encoder": source_encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "discriminator": discriminator.state_dict(),
            },
            model_path,
        )
    else:
        torch.save(model.state_dict(), model_path)

    fpr = eval_results["fpr"]
    tpr = eval_results["tpr"]

    if len(fpr) > 0:
        if args.use_wdgrl:
            roc_path = args.output_dir / f"roc_curve_wd{args.lambda_wd}.png"
        elif args.use_adda:
            roc_path = args.output_dir / f"roc_curve_adda{args.lambda_adda}.png"
        else:
            roc_path = args.output_dir / "roc_curve_baseline.png"

        plot_roc(fpr, tpr, float(auc), roc_path)

    # =========================
    # 🔥 LOGGING
    # =========================
    LOGGER.info("Saved history to: %s", history_path)
    LOGGER.info("Saved model weights to: %s", model_path)

    # =========================
    # 🔥 ROBUSTNESS TESTING
    # =========================
    print("\n🔥 ROBUSTNESS TESTING STARTED")

    if args.use_wdgrl:
        wdgrl_results = evaluate_robustness_all(
            model_type="wdgrl",
            model=None,
            feature_extractor=feature_extractor,
            classifier=classifier,
            args=args,
            device=device,
        )
        print("WDGRL Results:", wdgrl_results)

    elif not args.use_adda:
    # else:
        baseline_results = evaluate_robustness_all(
            model_type="baseline",
            model=model,
            feature_extractor=None,
            classifier=None,
            args=args,
            device=device,
        )
        print("Baseline Results:", baseline_results)


if __name__ == "__main__":
    main()