"""Quantum-inspired qubit and qutrit neural networks.

The Scientific Reports qubit/qutrit paper compares a classical ANN with
qubit-based and qutrit-based neural networks for financial forecasting. The
paper describes the learning flow and state representations, but it does not
provide executable code. This module implements a reproducible approximation:

- ANN: ordinary feed-forward binary direction classifier.
- QQBN: each feature is represented by a two-amplitude qubit-style map.
- QQTN: each feature is represented by a three-amplitude qutrit-style map.

The models are intentionally small so they can run during interim demos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ANNClassifier(nn.Module):
    """Classical feed-forward baseline used beside QQBN and QQTN."""

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QQBNClassifier(nn.Module):
    """Qubit-inspired classifier using a two-state feature map."""

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def encode(x: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(x, 0.0, 1.0)
        theta = 0.5 * torch.pi * clipped
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(x)).squeeze(-1)


class QQTNClassifier(nn.Module):
    """Qutrit-inspired classifier using a three-state feature map."""

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def encode(x: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(x, 0.0, 1.0)
        theta = 0.5 * torch.pi * clipped
        phi = torch.pi * clipped
        amp0 = torch.cos(theta)
        amp1 = torch.sin(theta) * torch.cos(phi)
        amp2 = torch.sin(theta) * torch.sin(phi)
        return torch.cat([amp0, amp1, amp2], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(x)).squeeze(-1)


@dataclass(frozen=True)
class TrainingHistory:
    model_name: str
    losses: list[float]
    train_losses: list[float]
    device: str


def default_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_binary_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 20,
    learning_rate: float = 0.01,
    seed: int = 42,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    patience: int = 10,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    device: torch.device | None = None,
    balance_classes: bool = True,
) -> TrainingHistory:
    """Train a small binary classifier with BCE loss and Adam."""

    torch.manual_seed(seed)
    device = default_device() if device is None else device
    model = model.to(device)
    x_all = np.asarray(train_x, dtype=np.float32).copy()
    y_all = np.asarray(train_y, dtype=np.float32).copy().reshape(-1)
    if len(x_all) != len(y_all):
        raise ValueError("train_x and train_y must contain the same number of samples.")
    split_idx = max(1, int(len(x_all) * (1.0 - val_ratio)))
    split_idx = min(split_idx, len(x_all) - 1) if len(x_all) > 2 else len(x_all)
    x_train = torch.as_tensor(x_all[:split_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_all[:split_idx], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x_all[split_idx:], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_all[split_idx:], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    if balance_classes:
        positives = float(y_train.sum().item())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    train_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_count = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.arange(len(x_train), device=device)
        batch_losses: list[float] = []
        for start in range(0, len(permutation), batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        model.eval()
        with torch.no_grad():
            if len(x_val) > 0:
                val_logits = model(x_val)
                val_loss = float(loss_fn(val_logits, y_val).detach().cpu())
            else:
                val_loss = train_loss

        train_losses.append(train_loss)
        losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingHistory(model.__class__.__name__, losses, train_losses, str(device))


def predict_binary_classifier(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class labels and positive-class probabilities."""

    device = next(model.parameters()).device if device is None else device
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(features, dtype=np.float32).copy(), device=device)
        probabilities = torch.sigmoid(model(x)).cpu().numpy()
    return (probabilities >= 0.5).astype(int), probabilities


__all__ = [
    "ANNClassifier",
    "QQBNClassifier",
    "QQTNClassifier",
    "TrainingHistory",
    "count_trainable_parameters",
    "default_device",
    "predict_binary_classifier",
    "train_binary_classifier",
]
