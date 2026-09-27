#!/usr/bin/env python3
"""Paired 3D volume classification with a shared-weight Swin Transformer.

This public version contains no dataset-specific paths, identifiers, cohort sizes,
project names, institutions, or outcome descriptions. Data locations and split
assignments are supplied at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import Swin3D_T_Weights, swin3d_t
from tqdm import tqdm


@dataclass
class Settings:
    roi_depth: int = 32
    roi_size: int = 64
    in_channels: int = 2
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 100
    lr: float = 3e-4
    warmup_epochs: int = 5
    weight_decay: float = 5e-4
    grad_clip: float = 2.0
    early_stop_patience: int = 12
    seed: int = 2024
    dropout: float = 0.6
    unfreeze_after_epoch: int = 10
    internal_n_splits: int = 5
    internal_fold_index: int = 0
    use_kinetics_weights: bool = True


CFG = Settings()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def amp_context():
    if AMP_ENABLED:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt).pow(self.gamma) * ce).mean()


class GatedFeatureFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )
        self.norm = nn.LayerNorm(dim * 3)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        weights = self.gate(torch.cat([first, second], dim=1))
        fused = first * weights[:, 0:1] + second * weights[:, 1:2]
        return self.norm(torch.cat([fused, first, second], dim=1))


def classifier_head(in_dim: int, hidden_dims: Iterable[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for hidden in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        prev = hidden
    layers.append(nn.Linear(prev, 2))
    return nn.Sequential(*layers)


class Paired3DSwinClassifier(nn.Module):
    def __init__(self, use_pretrained: bool = True):
        super().__init__()
        weights = Swin3D_T_Weights.KINETICS400_V1 if use_pretrained else None
        self.backbone = swin3d_t(weights=weights)
        self.feature_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.input_projection = nn.Conv3d(CFG.in_channels, 3, kernel_size=1)
        self.fusion = GatedFeatureFusion(self.feature_dim)
        self.first_head = classifier_head(self.feature_dim, [128, 64], CFG.dropout)
        self.second_head = classifier_head(self.feature_dim, [128, 64], CFG.dropout)
        self.fusion_head = classifier_head(self.feature_dim * 3, [256, 128], CFG.dropout)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> Dict[str, torch.Tensor]:
        first_feat = self.backbone(self.input_projection(first))
        second_feat = self.backbone(self.input_projection(second))
        fused_feat = self.fusion(first_feat, second_feat)
        return {
            "first": self.first_head(first_feat),
            "second": self.second_head(second_feat),
            "fusion": self.fusion_head(fused_feat),
            "first_feat": first_feat,
            "second_feat": second_feat,
            "fusion_feat": fused_feat,
        }


class PairedVolumeDataset(Dataset):
    """Dataset for two co-registered 3D volumes and their binary masks.

    Required metadata columns: instance_id, subject_id, label, split.
    Each data directory is expected to contain <instance_id>.nii.gz.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        first_volume_dir: Path,
        first_mask_dir: Path,
        second_volume_dir: Path,
        second_mask_dir: Path,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.dirs = {
            "first_volume": first_volume_dir,
            "first_mask": first_mask_dir,
            "second_volume": second_volume_dir,
            "second_mask": second_mask_dir,
        }
        self._validate_files()

    def _path(self, key: str, instance_id: str) -> Path:
        return self.dirs[key] / f"{instance_id}.nii.gz"

    def _validate_files(self) -> None:
        missing = []
        for instance_id in self.frame["instance_id"].astype(str):
            for key in self.dirs:
                path = self._path(key, instance_id)
                if not path.exists():
                    missing.append(str(path))
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"Missing {len(missing)} required input files. Examples:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        return torch.from_numpy(nib.load(str(path)).get_fdata().astype(np.float32))

    @staticmethod
    def _resize(volume: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        return F.interpolate(
            volume[None, None], size=size, mode="trilinear", align_corners=False
        )[0, 0]

    def _crop(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.nonzero(mask > 0.5, as_tuple=False)
        d, h, w = image.shape
        if coords.numel() == 0:
            center = torch.tensor([d // 2, h // 2, w // 2])
        else:
            center = ((coords.min(dim=0).values + coords.max(dim=0).values) // 2).cpu()

        target = (CFG.roi_depth, CFG.roi_size, CFG.roi_size)
        starts = [
            max(0, int(center[0]) - target[0] // 2),
            max(0, int(center[1]) - target[1] // 2),
            max(0, int(center[2]) - target[2] // 2),
        ]
        ends = [min(d, starts[0] + target[0]), min(h, starts[1] + target[1]), min(w, starts[2] + target[2])]
        cropped_image = image[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
        cropped_mask = mask[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
        return self._resize(cropped_image, target), self._resize(cropped_mask, target)

    @staticmethod
    def _normalize(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = image[mask > 0.5]
        if values.numel() > 1 and values.std() > 1e-6:
            mean, std = values.mean(), values.std()
        else:
            mean, std = image.mean(), image.std()
            if std <= 1e-6:
                std = torch.tensor(1.0, dtype=image.dtype)
        image = (image - mean) / std
        return torch.nan_to_num(image, nan=0.0, posinf=10.0, neginf=-10.0)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        instance_id = str(row["instance_id"])

        first_image = self._load(self._path("first_volume", instance_id))
        first_mask = self._load(self._path("first_mask", instance_id))
        second_image = self._load(self._path("second_volume", instance_id))
        second_mask = self._load(self._path("second_mask", instance_id))

        first_image, first_mask = self._crop(first_image, first_mask)
        second_image, second_mask = self._crop(second_image, second_mask)
        first_image = self._normalize(first_image, first_mask)
        second_image = self._normalize(second_image, second_mask)
        first_mask = (first_mask > 0.5).float()
        second_mask = (second_mask > 0.5).float()

        first = torch.stack([first_image, first_mask], dim=0)
        second = torch.stack([second_image, second_mask], dim=0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return first, second, label, instance_id


def normalize_split(value: object) -> str:
    text = str(value).strip().lower()
    mapping = {
        "0": "train", "train": "train", "training": "train",
        "1": "validation", "val": "validation", "valid": "validation", "validation": "validation",
        "2": "test", "test": "test", "testing": "test",
    }
    if text not in mapping:
        raise ValueError(f"Unrecognized split value: {value!r}")
    return mapping[text]


def load_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"instance_id", "subject_id", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["instance_id"] = frame["instance_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("label must be binary (0/1).")
    frame["split_name"] = frame["split"].map(normalize_split)
    if frame["instance_id"].duplicated().any():
        raise ValueError("instance_id must be unique.")
    split_per_subject = frame.groupby("subject_id")["split_name"].nunique()
    if (split_per_subject > 1).any():
        raise ValueError("A subject appears in more than one data split.")
    return frame


def class_weight(frame: pd.DataFrame) -> torch.Tensor:
    positive = int(frame["label"].sum())
    negative = len(frame) - positive
    ratio = min(negative / max(positive, 1), 5.0)
    return torch.tensor([1.0, ratio], dtype=torch.float32, device=DEVICE)


def make_optimizer(model: nn.Module, after_unfreeze: bool = False):
    if after_unfreeze:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr * 0.08, weight_decay=CFG.weight_decay)
        warmup = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)
        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[1])
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CFG.lr,
            weight_decay=CFG.weight_decay,
        )
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=CFG.warmup_epochs)
        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[CFG.warmup_epochs])
    return optimizer, scheduler


def freeze_early_backbone(model: Paired3DSwinClassifier) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not (
            "backbone" in name and ("features.0" in name or "features.1" in name)
        )


def auc_value(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else math.nan


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def metric_row(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": auc_value(y_true, y_prob),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "threshold": float(threshold),
    }


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    labels_all: List[int] = []
    probs = {"first": [], "second": [], "fusion": []}

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for _batch_index, (first, second, labels, _ids) in enumerate(tqdm(loader, leave=False)):
            first, second, labels = first.to(DEVICE), second.to(DEVICE), labels.to(DEVICE)
            with amp_context():
                out = model(first, second)
                loss = (
                    criterion(out["first"], labels)
                    + criterion(out["second"], labels)
                    + 1.5 * criterion(out["fusion"], labels)
                )
            if training:
                if scaler is not None and AMP_ENABLED:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu())
            labels_all.extend(labels.detach().cpu().tolist())
            for branch in probs:
                probs[branch].extend(torch.softmax(out[branch], dim=1)[:, 1].detach().cpu().tolist())

    y = np.asarray(labels_all, dtype=int)
    p = {k: np.asarray(v, dtype=float) for k, v in probs.items()}
    return total_loss / max(len(loader), 1), y, p


def make_loader(frame: pd.DataFrame, args, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = PairedVolumeDataset(
        frame,
        Path(args.first_volume_dir),
        Path(args.first_mask_dir),
        Path(args.second_volume_dir),
        Path(args.second_mask_dir),
    )
    return DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def internal_split(train_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(
        n_splits=CFG.internal_n_splits, shuffle=True, random_state=CFG.seed
    )
    splits = list(
        splitter.split(
            train_frame,
            y=train_frame["label"].to_numpy(),
            groups=train_frame["subject_id"].to_numpy(),
        )
    )
    fit_idx, dev_idx = splits[CFG.internal_fold_index]
    return train_frame.iloc[fit_idx].reset_index(drop=True), train_frame.iloc[dev_idx].reset_index(drop=True)


def new_model() -> Paired3DSwinClassifier:
    model = Paired3DSwinClassifier(CFG.use_kinetics_weights).to(DEVICE)
    freeze_early_backbone(model)
    return model


def select_epoch(train_frame: pd.DataFrame, args) -> int:
    fit_frame, dev_frame = internal_split(train_frame)
    fit_loader = make_loader(fit_frame, args, shuffle=True, drop_last=True)
    dev_loader = make_loader(dev_frame, args, shuffle=False, drop_last=False)
    model = new_model()
    criterion = FocalLoss(weight=class_weight(fit_frame))
    optimizer, scheduler = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    best_auc = -np.inf
    best_epoch = 1
    stale = 0
    for epoch in range(CFG.epochs):
        if epoch == CFG.unfreeze_after_epoch:
            for p in model.parameters():
                p.requires_grad = True
            optimizer, scheduler = make_optimizer(model, after_unfreeze=True)
        train_loss, _, _ = run_epoch(model, fit_loader, criterion, optimizer, scaler)
        _, y_dev, p_dev = run_epoch(model, dev_loader, criterion)
        scheduler.step()
        current = auc_value(y_dev, p_dev["fusion"])
        print(f"selection epoch={epoch + 1:03d} loss={train_loss:.4f} dev_auc={current:.4f}")
        if np.isfinite(current) and current > best_auc:
            best_auc = current
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= CFG.early_stop_patience:
                break
    return best_epoch


def fit_locked_model(train_frame: pd.DataFrame, selected_epoch: int, args) -> Tuple[Paired3DSwinClassifier, FocalLoss]:
    loader = make_loader(train_frame, args, shuffle=True, drop_last=True)
    model = new_model()
    criterion = FocalLoss(weight=class_weight(train_frame))
    optimizer, scheduler = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    for epoch in range(selected_epoch):
        if epoch == CFG.unfreeze_after_epoch:
            for p in model.parameters():
                p.requires_grad = True
            optimizer, scheduler = make_optimizer(model, after_unfreeze=True)
        loss, _, _ = run_epoch(model, loader, criterion, optimizer, scaler)
        scheduler.step()
        print(f"fit epoch={epoch + 1:03d}/{selected_epoch:03d} loss={loss:.4f}")
    return model, criterion


@torch.no_grad()
def infer(model, loader):
    model.eval()
    rows = []
    feature_rows = []
    for first, second, labels, instance_ids in tqdm(loader, leave=False):
        first, second = first.to(DEVICE), second.to(DEVICE)
        with amp_context():
            out = model(first, second)
        batch_probs = {
            branch: torch.softmax(out[branch], dim=1)[:, 1].detach().cpu().numpy()
            for branch in ("first", "second", "fusion")
        }
        for i, instance_id in enumerate(instance_ids):
            rows.append(
                {
                    "instance_id": instance_id,
                    "label": int(labels[i]),
                    "first_prob": float(batch_probs["first"][i]),
                    "second_prob": float(batch_probs["second"][i]),
                    "fusion_prob": float(batch_probs["fusion"][i]),
                }
            )
            feature_rows.append(
                (
                    instance_id,
                    out["first_feat"][i].detach().cpu().numpy(),
                    out["second_feat"][i].detach().cpu().numpy(),
                    out["fusion_feat"][i].detach().cpu().numpy(),
                )
            )
    return pd.DataFrame(rows), feature_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--first-volume-dir", required=True)
    parser.add_argument("--first-mask-dir", required=True)
    parser.add_argument("--second-volume-dir", required=True)
    parser.add_argument("--second-mask-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/stage1")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CFG.use_kinetics_weights = not args.no_pretrained
    set_seed(CFG.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_metadata(Path(args.metadata))
    train = frame[frame["split_name"] == "train"].reset_index(drop=True)
    validation = frame[frame["split_name"] == "validation"].reset_index(drop=True)
    test = frame[frame["split_name"] == "test"].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("train, validation, and test splits must all be non-empty.")

    selected_epoch = select_epoch(train, args)
    model, _criterion = fit_locked_model(train, selected_epoch, args)
    torch.save(model.state_dict(), output_dir / "locked_model.pt")

    prediction_tables = []
    features_by_split = {}
    for split_name, split_frame in (("train", train), ("validation", validation), ("test", test)):
        loader = make_loader(split_frame, args, shuffle=False, drop_last=False)
        pred, feature_rows = infer(model, loader)
        pred["split"] = split_name
        prediction_tables.append(pred)
        features_by_split[split_name] = feature_rows

    predictions = pd.concat(prediction_tables, ignore_index=True)
    train_pred = predictions[predictions["split"] == "train"]
    thresholds = {
        branch: choose_threshold(train_pred["label"].to_numpy(), train_pred[f"{branch}_prob"].to_numpy())
        for branch in ("first", "second", "fusion")
    }

    metric_rows = []
    for split_name in ("train", "validation", "test"):
        subset = predictions[predictions["split"] == split_name]
        for branch in ("first", "second", "fusion"):
            row = {"split": split_name, "branch": branch}
            row.update(
                metric_row(
                    subset["label"].to_numpy(),
                    subset[f"{branch}_prob"].to_numpy(),
                    thresholds[branch],
                )
            )
            metric_rows.append(row)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_epoch": int(selected_epoch),
                "thresholds_selected_from": "train",
                "thresholds": thresholds,
                "holdout_splits_used_for_model_selection": False,
            },
            f,
            indent=2,
        )

    for split_name, rows in features_by_split.items():
        ids = np.asarray([x[0] for x in rows], dtype=str)
        first = np.stack([x[1] for x in rows])
        second = np.stack([x[2] for x in rows])
        fusion = np.stack([x[3] for x in rows])
        np.savez_compressed(
            output_dir / f"features_{split_name}.npz",
            instance_id=ids,
            first=first,
            second=second,
            fusion=fusion,
        )

    print(f"Completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
