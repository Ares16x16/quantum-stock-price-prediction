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


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_binary_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 20,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> TrainingHistory:
    """Train a small binary classifier with BCE loss and Adam."""

    torch.manual_seed(seed)
    model.train()
    x = torch.as_tensor(np.asarray(train_x, dtype=np.float32).copy())
    y = torch.as_tensor(np.asarray(train_y, dtype=np.float32).copy())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return TrainingHistory(model.__class__.__name__, losses)


def predict_binary_classifier(model: nn.Module, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return class labels and positive-class probabilities."""

    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(features, dtype=np.float32).copy())
        probabilities = torch.sigmoid(model(x)).cpu().numpy()
    return (probabilities >= 0.5).astype(int), probabilities


__all__ = [
    "ANNClassifier",
    "QQBNClassifier",
    "QQTNClassifier",
    "TrainingHistory",
    "count_trainable_parameters",
    "predict_binary_classifier",
    "train_binary_classifier",
]
