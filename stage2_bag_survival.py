#!/usr/bin/env python3
"""Variable-length multiple-instance survival modeling.

This public version is deliberately dataset-agnostic. It contains no study name,
private path, institution, fixed cohort size, disease label, or project-specific
outcome terminology.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class Settings:
    hidden_dim: int = 64
    dropout: float = 0.3
    attention_dim: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 120
    early_stopping_patience: int = 30
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    rank_margin: float = 0.10
    entropy_weight: float = 0.05
    risk_penalty_weight: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42


CFG = Settings()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def concordance_index(risk_scores, events, times) -> float:
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    order = np.argsort(times)
    risk_scores, events, times = risk_scores[order], events[order], times[order]

    concordant = 0.0
    permissible = 0
    for i in range(len(times)):
        if events[i] != 1:
            continue
        for j in range(i + 1, len(times)):
            if times[i] < times[j]:
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
    return concordant / permissible if permissible else 0.5


def bootstrap_cindex(risk_scores, events, times, n_bootstraps: int = 1000, seed: int = 42) -> Tuple[float, float]:
    risk_scores = np.asarray(risk_scores, dtype=float)
    events = np.asarray(events, dtype=int)
    times = np.asarray(times, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(risk_scores), len(risk_scores), replace=True)
        if events[idx].sum() > 0 and len(np.unique(times[idx])) > 1:
            values.append(concordance_index(risk_scores[idx], events[idx], times[idx]))
    if not values:
        return 0.5, 0.5
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def margin_ranking_loss(risk, times, events, margin: float) -> torch.Tensor:
    risk_matrix = risk.unsqueeze(1) - risk.unsqueeze(0)
    time_matrix = times.unsqueeze(1) - times.unsqueeze(0)
    valid_pairs = (time_matrix < 0) & (events.unsqueeze(1) == 1)
    losses = torch.relu(margin - risk_matrix)[valid_pairs]
    return losses.mean() if losses.numel() else risk.sum() * 0.0


def cox_breslow_loss(risk, times, events) -> torch.Tensor:
    risk = risk.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1)
    event_mask = events > 0.5
    n_events = event_mask.sum()
    if n_events.item() == 0:
        return risk.sum() * 0.0

    loglik = risk.new_tensor(0.0)
    for time_value in torch.unique(times[event_mask]):
        event_at_time = (times == time_value) & event_mask
        at_risk = times >= time_value
        d_t = event_at_time.sum().to(risk.dtype)
        loglik = loglik + risk[event_at_time].sum() - d_t * torch.logsumexp(risk[at_risk], dim=0)
    return -loglik / n_events.to(risk.dtype)


class BagDataset(Dataset):
    def __init__(self, bags, global_features, times, events, subject_ids):
        self.bags = bags
        self.global_features = global_features
        self.times = times
        self.events = events
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        return (
            torch.tensor(self.bags[index], dtype=torch.float32),
            torch.tensor(self.global_features[index], dtype=torch.float32),
            torch.tensor(self.times[index], dtype=torch.float32),
            torch.tensor(self.events[index], dtype=torch.float32),
            self.subject_ids[index],
        )


class GatedAttentionSurvival(nn.Module):
    def __init__(self, input_dim: int, global_dim: int):
        super().__init__()
        h = CFG.hidden_dim
        a = CFG.attention_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.1),
            nn.Dropout(CFG.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.1),
            nn.Dropout(CFG.dropout),
        )
        self.attn_v = nn.Sequential(nn.Linear(h, a), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(h, a), nn.Sigmoid())
        self.attn_w = nn.Linear(a, 1)
        self.fusion = nn.Sequential(
            nn.Linear(h + global_dim, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.1),
            nn.Dropout(CFG.dropout),
        )
        self.risk_head = nn.Linear(h, 1)

    def forward(self, bag: torch.Tensor, global_features: torch.Tensor):
        encoded = self.encoder(bag.squeeze(0))
        logits = self.attn_w(self.attn_v(encoded) * self.attn_u(encoded))
        attention = torch.softmax(logits.transpose(1, 0), dim=1)
        entropy = -torch.sum(attention * torch.log(attention + 1e-8)) / attention.size(1)
        pooled = torch.mm(attention, encoded)
        fused = self.fusion(torch.cat([pooled, global_features], dim=1))
        risk = self.risk_head(fused)
        return risk, attention, entropy


def load_inputs(feature_path: Path, subject_path: Path, metadata_path: Path, feature_prefix: str):
    features = pd.read_csv(feature_path)
    subjects = pd.read_csv(subject_path)
    metadata = pd.read_csv(metadata_path)

    required_features = {"instance_id", "subject_id"}
    required_subjects = {"subject_id", "time", "event", "split"}
    required_metadata = {"instance_id", "subject_id", "location", "region_group"}
    if not required_features.issubset(features.columns):
        raise ValueError(f"Feature table must contain: {sorted(required_features)}")
    if not required_subjects.issubset(subjects.columns):
        raise ValueError(f"Subject table must contain: {sorted(required_subjects)}")
    if not required_metadata.issubset(metadata.columns):
        raise ValueError(f"Instance metadata must contain: {sorted(required_metadata)}")

    for frame in (features, subjects, metadata):
        frame["subject_id"] = frame["subject_id"].astype(str)
    features["instance_id"] = features["instance_id"].astype(str)
    metadata["instance_id"] = metadata["instance_id"].astype(str)
    subjects["split_name"] = subjects["split"].map(normalize_split)
    subjects["time"] = pd.to_numeric(subjects["time"], errors="raise")
    subjects["event"] = pd.to_numeric(subjects["event"], errors="raise").astype(int)

    if subjects["subject_id"].duplicated().any():
        raise ValueError("subject_id must be unique in the subject table.")
    if features["instance_id"].duplicated().any() or metadata["instance_id"].duplicated().any():
        raise ValueError("instance_id must be unique in instance-level tables.")
    if not subjects["event"].isin([0, 1]).all():
        raise ValueError("event must be binary (0/1).")
    if (subjects["time"] <= 0).any():
        raise ValueError("time must be positive.")

    if feature_prefix:
        feature_cols = [c for c in features.columns if c.startswith(feature_prefix)]
    else:
        feature_cols = [c for c in features.columns if c not in {"instance_id", "subject_id"}]
    if not feature_cols:
        raise ValueError("No feature columns found. Check --feature-prefix.")

    merged = features[["instance_id", "subject_id"] + feature_cols].merge(
        metadata[["instance_id", "subject_id", "location", "region_group"]],
        on=["instance_id", "subject_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(metadata):
        raise ValueError("Feature and metadata tables do not contain the same instance set.")

    merged = merged.merge(
        subjects[["subject_id", "split_name"]],
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )
    return merged, subjects, feature_cols


def prepare_data(instances: pd.DataFrame, subjects: pd.DataFrame, feature_cols: List[str], output_dir: Path):
    instances = instances.copy()
    instances[feature_cols] = instances[feature_cols].replace([np.inf, -np.inf], np.nan)
    train_mask = instances["split_name"] == "train"
    if train_mask.sum() == 0:
        raise ValueError("No training instances found.")

    medians = instances.loc[train_mask, feature_cols].median(axis=0)
    if medians.isna().any():
        bad = medians[medians.isna()].index.tolist()
        raise ValueError(f"Features completely missing in training data: {bad[:10]}")
    instances[feature_cols] = instances[feature_cols].fillna(medians)

    feature_scaler = StandardScaler()
    feature_scaler.fit(instances.loc[train_mask, feature_cols].to_numpy())
    instances.loc[:, feature_cols] = feature_scaler.transform(instances[feature_cols].to_numpy())
    joblib.dump(feature_scaler, output_dir / "instance_feature_scaler.joblib")

    global_rows = []
    for subject_id, group in instances.groupby("subject_id", sort=False):
        global_rows.append(
            {
                "subject_id": subject_id,
                "instance_count": float(len(group)),
                "unique_location_count": float(group["location"].nunique()),
                "multi_region": float(group["region_group"].nunique() > 1),
            }
        )
    global_frame = pd.DataFrame(global_rows)
    global_cols = ["instance_count", "unique_location_count", "multi_region"]

    bags: Dict[str, np.ndarray] = {
        subject_id: group[feature_cols].to_numpy(dtype=np.float32)
        for subject_id, group in instances.groupby("subject_id", sort=False)
    }
    final = subjects.merge(global_frame, on="subject_id", how="inner", validate="one_to_one")
    final["bag"] = final["subject_id"].map(bags)
    if len(final) != len(subjects):
        missing = sorted(set(subjects["subject_id"]) - set(final["subject_id"]))
        raise ValueError(f"Subjects without complete instance data: {missing[:10]}")

    global_scaler = StandardScaler()
    train_subject_mask = final["split_name"] == "train"
    global_scaler.fit(final.loc[train_subject_mask, global_cols].to_numpy())
    final.loc[:, global_cols] = global_scaler.transform(final[global_cols].to_numpy())
    joblib.dump(global_scaler, output_dir / "global_feature_scaler.joblib")
    return instances, final, global_cols


def make_loader(frame: pd.DataFrame, global_cols: List[str]) -> DataLoader:
    ds = BagDataset(
        bags=frame["bag"].tolist(),
        global_features=frame[global_cols].to_numpy(dtype=np.float32),
        times=frame["time"].to_numpy(dtype=np.float32),
        events=frame["event"].to_numpy(dtype=np.float32),
        subject_ids=frame["subject_id"].astype(str).tolist(),
    )
    return DataLoader(ds, batch_size=1, shuffle=False)


def train_model(model, train_loader, train_frame, model_path: Path):
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CFG.scheduler_factor, patience=CFG.scheduler_patience
    )
    best_cindex = -np.inf
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, CFG.max_epochs + 1):
        model.train()
        risks, times, events, entropies = [], [], [], []
        for bag, global_features, time_value, event_value, _subject_id in train_loader:
            risk, _attention, entropy = model(bag.to(DEVICE), global_features.to(DEVICE))
            risks.append(risk.reshape(-1))
            times.append(time_value.reshape(-1).to(DEVICE))
            events.append(event_value.reshape(-1).to(DEVICE))
            entropies.append(entropy)

        risk_tensor = torch.cat(risks)
        time_tensor = torch.cat(times)
        event_tensor = torch.cat(events)
        cox = cox_breslow_loss(risk_tensor, time_tensor, event_tensor)
        rank = margin_ranking_loss(risk_tensor, time_tensor, event_tensor, CFG.rank_margin)
        entropy = torch.stack(entropies).mean()
        risk_penalty = CFG.risk_penalty_weight * torch.mean(risk_tensor.pow(2))
        loss = cox + rank + CFG.entropy_weight * entropy + risk_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        optimizer.step()

        model.eval()
        train_risk = []
        with torch.no_grad():
            for bag, global_features, _time, _event, _subject_id in train_loader:
                risk, _attention, _entropy = model(bag.to(DEVICE), global_features.to(DEVICE))
                train_risk.append(float(risk.item()))
        cindex = concordance_index(train_risk, train_frame["event"].to_numpy(), train_frame["time"].to_numpy())
        scheduler.step(cindex)

        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "cox_loss": float(cox.detach().cpu()),
                "ranking_loss": float(rank.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "train_cindex": float(cindex),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.5f} train_cindex={cindex:.4f}")

        if cindex > best_cindex:
            best_cindex = cindex
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "train_cindex": best_cindex,
                },
                model_path,
            )
        else:
            stale += 1
            if stale >= CFG.early_stopping_patience:
                break
    return pd.DataFrame(history)


@torch.no_grad()
def infer(model, loader):
    risks: List[float] = []
    subject_ids: List[str] = []
    attentions: List[np.ndarray] = []
    model.eval()
    for bag, global_features, _time, _event, subject_id in loader:
        risk, attention, _entropy = model(bag.to(DEVICE), global_features.to(DEVICE))
        risks.append(float(risk.item()))
        subject_ids.append(subject_id[0])
        attentions.append(attention.detach().cpu().numpy().reshape(-1))
    return np.asarray(risks), subject_ids, attentions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-features", required=True)
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--instance-metadata", required=True)
    parser.add_argument("--feature-prefix", default="feat_")
    parser.add_argument("--output-dir", default="outputs/stage2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(CFG.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances, subjects, feature_cols = load_inputs(
        Path(args.instance_features),
        Path(args.subjects),
        Path(args.instance_metadata),
        args.feature_prefix,
    )
    instances, final, global_cols = prepare_data(instances, subjects, feature_cols, output_dir)

    splits = {
        name: final[final["split_name"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    if any(len(frame) == 0 for frame in splits.values()):
        raise ValueError("train, validation, and test splits must all be non-empty.")

    loaders = {name: make_loader(frame, global_cols) for name, frame in splits.items()}
    model = GatedAttentionSurvival(input_dim=len(feature_cols), global_dim=len(global_cols)).to(DEVICE)
    model_path = output_dir / "locked_model.pt"
    history = train_model(model, loaders["train"], splits["train"], model_path)
    history.to_csv(output_dir / "training_history.csv", index=False)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = {
        "input_feature_count": len(feature_cols),
        "global_feature_count": len(global_cols),
        "model_selection_split": "train",
        "holdout_splits_used_for_training_or_tuning": False,
        "cox_ties": "Breslow",
        "splits": {},
    }
    score_tables = []

    for split_name in ("train", "validation", "test"):
        frame = splits[split_name]
        raw_risk, ids, attention_rows = infer(model, loaders[split_name])
        if ids != frame["subject_id"].astype(str).tolist():
            raise RuntimeError(f"Subject order mismatch in {split_name} inference.")

        cindex = concordance_index(raw_risk, frame["event"].to_numpy(), frame["time"].to_numpy())
        low, high = bootstrap_cindex(raw_risk, frame["event"].to_numpy(), frame["time"].to_numpy(), seed=CFG.seed)
        score = 1.0 / (1.0 + np.exp(-raw_risk))
        score_frame = frame[["subject_id", "time", "event"]].copy()
        score_frame["split"] = split_name
        score_frame["raw_risk"] = raw_risk
        score_frame["score"] = score
        score_tables.append(score_frame)
        score_frame.to_csv(output_dir / f"scores_{split_name}.csv", index=False)

        attention_records = []
        for subject_id, weights in zip(ids, attention_rows):
            instance_ids = instances.loc[
                instances["subject_id"] == subject_id, "instance_id"
            ].astype(str).tolist()
            if len(instance_ids) != len(weights):
                raise RuntimeError(f"Attention length mismatch for subject {subject_id}.")
            attention_records.extend(
                {
                    "subject_id": subject_id,
                    "instance_id": instance_id,
                    "attention_weight": float(weight),
                    "split": split_name,
                }
                for instance_id, weight in zip(instance_ids, weights)
            )
        pd.DataFrame(attention_records).to_csv(
            output_dir / f"attention_{split_name}.csv", index=False
        )
        metrics["splits"][split_name] = {
            "n": int(len(frame)),
            "events": int(frame["event"].sum()),
            "c_index": float(cindex),
            "c_index_95ci": [float(low), float(high)],
        }

    pd.concat(score_tables, ignore_index=True).to_csv(output_dir / "scores_all.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
